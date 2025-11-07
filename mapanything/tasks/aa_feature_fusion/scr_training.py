"""Train a scene coordinate regression head using ACE buffers and AA fusion."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from mapanything.tasks.aa_feature_fusion.builder import build_pipeline_from_cfg
from mapanything.tasks.aa_feature_fusion.common import (
    extract_pointmap_like_outputs,
    instantiate_dataset,
    prepare_view,
)


class ACETrainingBuffer:
    """Simple FIFO buffer that mirrors ACE's cached feature workflow."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.features: deque[torch.Tensor] = deque(maxlen=capacity)
        self.targets: deque[torch.Tensor] = deque(maxlen=capacity)

    def add_batch(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        for feat, tgt in zip(features, targets):
            self.features.append(feat.detach().cpu())
            self.targets.append(tgt.detach().cpu())

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.features)

    def as_dataloader(self, batch_size: int, shuffle: bool) -> DataLoader:
        if not self.features:
            raise RuntimeError("The ACE training buffer is empty; add samples before training.")
        feature_stack = torch.stack(list(self.features))
        target_stack = torch.stack(list(self.targets))
        dataset = TensorDataset(feature_stack, target_stack)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class SCRRegressionHead(nn.Module):
    """Lightweight convolutional regressor mapping dense features to XYZ maps."""

    def __init__(self, in_channels: int, hidden_dim: int, output_channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.net(x)


def _iter_training_indices(cfg: DictConfig, dataset_length: int) -> Iterable[int]:
    if cfg.training.sample_indices is not None:
        return cfg.training.sample_indices
    return range(min(cfg.training.num_samples, dataset_length))


def _collect_buffer(
    cfg: DictConfig,
    pipeline,
    device: torch.device,
    dataset,
) -> Tuple[ACETrainingBuffer, int]:
    buffer = ACETrainingBuffer(capacity=cfg.buffer.capacity)

    for idx in _iter_training_indices(cfg, len(dataset)):
        views = dataset[idx]
        if not views:
            continue
        single_view = views[0]
        prepared = prepare_view(
            single_view,
            device=device,
            include_intrinsics=pipeline.include_intrinsics,
            include_depth=pipeline.include_depth,
            include_pose=pipeline.include_pose,
            include_scale=pipeline.include_scale,
        )

        with torch.no_grad():
            fused_tokens = pipeline.fuse(prepared)
            feature_map = pipeline.fusion_module.tokens_to_feature_map(fused_tokens)
            scale_token = pipeline.fusion_module.get_additional_token(
                device=device, dtype=feature_map.dtype
            )
            if scale_token is None:
                scale_token = torch.zeros(
                    feature_map.shape[0],
                    feature_map.shape[1],
                    1,
                    device=device,
                    dtype=feature_map.dtype,
                )
            dense_outputs, pose_outputs, scale_output = pipeline.base_model.downstream_head(
                dense_head_inputs=feature_map,
                scale_head_inputs=scale_token,
                img_shape=feature_map.shape[-2:],
                memory_efficient_inference=cfg.training.memory_efficient_inference,
            )

        reconstruction = extract_pointmap_like_outputs(
            pipeline, dense_outputs, pose_outputs, scale_output
        )
        buffer.add_batch(dense_outputs.value, reconstruction["pts3d"])

        if len(buffer) >= cfg.training.num_samples:
            break

    if not buffer:
        raise RuntimeError("Failed to populate ACE buffer; confirm dataset indices and configuration.")

    sample_channels = buffer.features[0].shape[0]
    return buffer, sample_channels


def run_training(cfg: DictConfig) -> Dict[str, str]:
    if cfg.fusion.stored_feature_file is None:
        raise ValueError("Provide fusion.stored_feature_file to reuse AA memory during training.")

    requested_device = cfg.training.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU for SCR training.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    pipeline = build_pipeline_from_cfg(cfg)
    pipeline.base_model.to(device)
    pipeline.base_model.eval()
    pipeline.fusion_module.to(device)
    pipeline.fusion_module.eval()

    dataset = instantiate_dataset(cfg.dataset.dataset_str)

    buffer, in_channels = _collect_buffer(cfg, pipeline, device, dataset)

    dataloader = buffer.as_dataloader(
        batch_size=cfg.training.batch_size,
        shuffle=cfg.buffer.shuffle,
    )

    head = SCRRegressionHead(
        in_channels=in_channels,
        hidden_dim=cfg.scr_head.hidden_dim,
        output_channels=cfg.scr_head.output_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    criterion = nn.SmoothL1Loss()

    output_dir = Path(cfg.training.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(cfg.training.max_epochs):
        head.train()
        for batch_features, batch_targets in dataloader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = head(batch_features)
            loss = criterion(preds, batch_targets)
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % cfg.training.log_interval == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "loss": float(loss.detach().cpu()),
                            "buffer_size": len(buffer),
                        }
                    )
                )

    checkpoint_path = output_dir / "scr_head.pt"
    torch.save({"state_dict": head.state_dict()}, checkpoint_path)

    return {"checkpoint": str(checkpoint_path), "buffer_size": len(buffer)}


@hydra.main(
    version_base=None,
    config_path="../../../configs/tasks/aa_feature_fusion",
    config_name="scr_train",
)
def main(cfg: DictConfig) -> None:
    info = run_training(cfg)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

