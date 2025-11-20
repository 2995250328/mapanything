"""Query-only ACE training entry that skips intermediate feature fusion."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from mapanything.datasets import SevenScenesWAI
from mapanything.datasets.base.base_dataset import ForcedRandomDataLoader
from mapanything.models import init_model
from mapanything.tasks.ace import ACEHead_Pointwise_Decoupled_WithScale, ACEHead_Pointwise_FiLM, ReproLoss
from mapanything.tasks.train import (
    BufferDataset,
    FeatureReplayBuffer,
    _loss_fn,
    _prepare_targets,
    _resolve_intrinsics,
    _resolve_pose,
)


def _collect_buffer_query_only(
    cfg: DictConfig,
    model,
    upsampler,
    dataset,
    device: torch.device,
) -> Tuple[FeatureReplayBuffer, int]:
    """Fill the replay buffer using only query features (no memory fusion)."""

    buffer: FeatureReplayBuffer | None = None
    capacity = int(getattr(cfg.training, "buffer_size", getattr(cfg.training, "buffer_capacity", 0)))
    if capacity <= 0:
        raise ValueError("training.buffer_size / training.buffer_capacity 未正确设置为正整数。")

    pbar = tqdm(total=capacity, desc="Filling training buffer", unit="sample", leave=False)
    try:
        for views in dataset:
            batch = views
            for view in batch:
                if "idx" in view:
                    view["idx"] = view["idx"][2:]

            ignore_keys = {"dataset", "label", "instance", "idx", "true_shape", "rng", "data_norm_type"}
            for view in batch:
                for name in list(view.keys()):
                    if name in ignore_keys:
                        continue
                    view[name] = view[name].to(device, non_blocking=True)

            with torch.no_grad():
                memory_tokens = [None] * getattr(model.info_sharing, "depth", 24)
                scale_token = model.scale_token.view(1, -1, 1).to(device)
                fused_feature, fused_token, _, _, _ = model.forward_with_memory_dense_feature(
                    query_view=batch,
                    device=str(device),
                    memory_tokens_per_block=memory_tokens,
                    additional_tokens=scale_token,
                    memory_keep_ratio=1.0,
                    memory_efficient_inference=cfg.training.memory_efficient_inference,
                )
                dense_feat_up = upsampler(batch[0]["img"], fused_feature)

            view = batch[0]
            target_world, valid_mask = _prepare_targets(view, device)
            intrinsics = _resolve_intrinsics(view, device).to(torch.float32)
            c2w = _resolve_pose(view, device).to(torch.float32)

            if buffer is None:
                in_channels = int(dense_feat_up.shape[1])
                buffer = FeatureReplayBuffer(capacity, in_channels, device=torch.device("cpu"))

            added = buffer.add_view(
                dense_feat_up.squeeze(0).detach(),
                fused_token.squeeze(0).detach(),
                target_world.squeeze(0).detach(),
                valid_mask.squeeze(0).detach(),
                intrinsics.squeeze(0).detach(),
                c2w.squeeze(0).detach(),
                cfg.training.samples_per_view,
            )
            if added > 0:
                pbar.update(added)
            if buffer.is_full:
                break

        if buffer is None or buffer.size == 0:
            raise RuntimeError("No training samples were collected. Check dataset or configuration.")
        return buffer, buffer.storage.features.shape[1]
    finally:
        pbar.close()


def run_training_query_only(cfg: DictConfig) -> Dict[str, str]:
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    model = init_model(cfg.model.model_str, cfg.model.model_config, torch_hub_force_reload=False)
    model.to(device).eval()
    if cfg.model.pretrained:
        ckpt = torch.load(cfg.model.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    hub_dir = torch.hub.get_dir()
    repo_dir_name = "wimmerth_anyup_main"
    local_cache_path = os.path.join(hub_dir, repo_dir_name)
    if os.path.exists(local_cache_path):
        print(f"Loading AnyUp from local cache: {local_cache_path}")
        upsampler = torch.hub.load(local_cache_path, "anyup", source="local", trust_repo=True)
    else:
        print("Downloading AnyUp from GitHub...")
        upsampler = torch.hub.load("wimmerth/anyup", "anyup", trust_repo=True)
    upsampler.to(device).eval()

    iterations = cfg.training.epochs * (cfg.training.buffer_size // cfg.training.batch_size)
    repro_loss = ReproLoss(
        total_iterations=iterations,
        soft_clamp=cfg.loss.repro_loss_soft_clamp,
        soft_clamp_min=cfg.loss.repro_loss_soft_clamp_min,
        type=cfg.loss.repro_loss_type,
        circle_schedule=(cfg.loss.repro_loss_schedule == "circle"),
    )

    if isinstance(cfg.dataset.train_dataset, str):
        dataset = eval(cfg.dataset.train_dataset)
    dataloader = ForcedRandomDataLoader(dataset=dataset, batch_size=1)

    buffer, in_channels = _collect_buffer_query_only(cfg, model, upsampler, dataloader, device)
    bufferloader = DataLoader(
        BufferDataset(buffer),
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle,
        num_workers=cfg.training.num_workers,
    )

    if cfg.model.head_mode == "film":
        head = ACEHead_Pointwise_FiLM(in_channels=in_channels, hidden_dim=cfg.head.hidden_dim).to(device)
    else:
        head = ACEHead_Pointwise_Decoupled_WithScale(
            in_channels=in_channels,
            hidden_dim=cfg.head.hidden_dim,
            token_dim=in_channels,
        ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    output_dir = Path(cfg.training.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(cfg.training.epochs):
        head.train()
        for batch in bufferloader:
            features = batch["features"].to(device)
            scale = batch["scale_token"].to(device)
            preds = head(features, scale)

            loss, metrics = _loss_fn(preds, batch, repro_loss, global_step, cfg.loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % cfg.training.log_interval == 0:
                print(json.dumps({"epoch": epoch, "step": global_step, "metrics": metrics}, ensure_ascii=False))

    try:
        task_name = HydraConfig.get().runtime.choices.get("model/task", "default")
    except Exception:
        task_name = "unknown"
    ckpt_name = (
        f"ace_"
        f"task-{task_name}_"
        f"head-{cfg.model.head_mode}_"
        f"loss-{cfg.loss.mode}_"
        f"scale-{'on' if cfg.loss.scale_reg.enabled else 'off'}_"
        f"var-{cfg.loss.scale_reg.variant}_"
        f"ep{cfg.training.epochs}_"
        f"buf{cfg.training.buffer_size}"
        ".pt"
    )
    ckpt_path = output_dir / ckpt_name
    torch.save({"state_dict": head.state_dict(), "in_channels": in_channels}, ckpt_path)
    return {"checkpoint": str(ckpt_path), "steps": str(global_step), "in_channels": str(in_channels)}


@hydra.main(version_base=None, config_path="../../configs", config_name="ace_train")
def main(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    info = run_training_query_only(cfg)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()