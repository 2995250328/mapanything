"""Inference & evaluation entrypoint for ACE-style regression heads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from mapanything.models import init_model
from mapanything.tasks.aa_feature_fusion.common import instantiate_dataset
from mapanything.tasks.ace import ACERegressionHead, load_memory_features, move_view_to_device
from mapanything.tasks.ace.train import (
    _loss_fn,
    _prepare_targets,
    _resolve_intrinsics,
    _resolve_pose,
)


def _load_head(cfg: DictConfig, in_channels: int, device: torch.device) -> ACERegressionHead:
    head = ACERegressionHead(in_channels=in_channels, hidden_dim=cfg.head.hidden_dim).to(device)
    if cfg.head.checkpoint:
        payload = torch.load(cfg.head.checkpoint, map_location="cpu")
        state_dict = payload.get("state_dict", payload)
        inferred_channels = payload.get("in_channels")
        if inferred_channels is not None and inferred_channels != in_channels:
            head = ACERegressionHead(in_channels=inferred_channels, hidden_dim=cfg.head.hidden_dim).to(device)
        head.load_state_dict(state_dict, strict=False)
    return head


def run_eval(cfg: DictConfig) -> Dict[str, Any]:
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model = init_model(cfg.model.model_str, cfg.model.model_config, torch_hub_force_reload=False)
    model.to(device).eval()

    if cfg.model.pretrained:
        ckpt = torch.load(cfg.model.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    memory_feats, memory_token = load_memory_features(cfg.fusion.stored_feature_file, device)
    dataset = instantiate_dataset(cfg.dataset.dataset_str)

    # Prepare one sample to infer channel count
    sample_view = dataset[0][0]
    prepared = move_view_to_device(sample_view, device)
    with torch.no_grad():
        sample_feature, _ = model.forward_with_memory_dense_feature(
            query_view=[prepared],
            device=str(device),
            memory_feats=memory_feats,
            additional_tokens=memory_token,
            memory_keep_ratio=cfg.fusion.memory_keep_ratio,
            memory_efficient_inference=cfg.training.memory_efficient_inference,
        )
    head = _load_head(cfg, sample_feature.shape[1], device)

    output_dir = Path(cfg.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics: List[Dict[str, float]] = []
    saved: List[Dict[str, Any]] = []
    head.eval()
    for idx in range(min(cfg.eval.max_samples, len(dataset))):
        view = dataset[idx][0]
        prepared = move_view_to_device(view, device)
        with torch.no_grad():
            dense_feature, _ = model.forward_with_memory_dense_feature(
                query_view=[prepared],
                device=str(device),
                memory_feats=memory_feats,
                additional_tokens=memory_token,
                memory_keep_ratio=cfg.fusion.memory_keep_ratio,
                memory_efficient_inference=cfg.training.memory_efficient_inference,
            )
            preds = head(dense_feature)

        target_world, valid_mask = _prepare_targets(view, dense_feature.shape[-2:], device)
        batch = {
            "target_world": target_world,
            "valid_mask": valid_mask,
            "intrinsics": _resolve_intrinsics(view, device),
            "c2w": _resolve_pose(view, device),
        }
        _, batch_metrics = _loss_fn(preds, batch, cfg.loss)
        metrics.append(batch_metrics)

        save_path = output_dir / f"sample_{idx:05d}.pt"
        torch.save({
            "coords_world": preds[:, :3].detach().cpu(),
            "confidence": torch.sigmoid(preds[:, 3:4]).detach().cpu(),
            "scene": view.get("label") or view.get("scene_name") or str(idx),
        }, save_path)
        saved.append({"index": idx, "path": str(save_path)})

    if metrics:
        mean_metrics = {
            key: float(sum(m[key] for m in metrics) / len(metrics))
            for key in metrics[0].keys()
        }
    else:
        mean_metrics = {}

    summary = {
        "outputs": saved,
        "metrics": mean_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


@hydra.main(version_base=None, config_path="../../configs/ace", config_name="test")
def main(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    summary = run_eval(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
