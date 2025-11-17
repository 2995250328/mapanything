"""Inference & evaluation entrypoint for ACE-style regression heads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from mapanything.datasets import SevenScenesWAI
from mapanything.datasets.base.base_dataset import ForcedRandomDataLoader
from mapanything.models import init_model
from mapanything.tasks.aa_feature_fusion.common import instantiate_dataset
from mapanything.tasks.ace import ACEHead_Pointwise_Decoupled_WithScale, ACEHead_Pointwise_FiLM, load_memory_features, move_view_to_device, load_regression_head
from mapanything.tasks.train import (
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

    # 创建dataset
    dataset = SevenScenesWAI(
        num_views=cfg.dataset.num_views,
        split="train",
        covisibility_thres=0.025,
        ROOT="/mnt/storage/xwh/mapanything-dataset/wai_data/7scenes",
        dataset_metadata_dir="/mnt/storage/xwh/map-anything/mapanything_dataset_metadata",
        sample_specific_scene=True,
        specific_scene_name='chess_train',
        resolution=(518, 392),
        transform="imgnorm",
        data_norm_type="dinov2",
        seed=777
    )
    # 创建无限读取器
    dataloader = ForcedRandomDataLoader(
        dataset=dataset,
        batch_size=1  # 你想要的 batch size
    )

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
    upsampler = torch.hub.load('wimmerth/anyup', 'anyup')  # 如需可：, trust_repo=True
    upsampler.eval()  # 推理模式

    metrics: List[Dict[str, float]] = []
    saved: List[Dict[str, Any]] = []
    head.eval()
    for idx in range(min(cfg.eval.max_samples, len(dataset))):
        view = dataset[idx][0]
        prepared = move_view_to_device(view, device)
        with torch.no_grad():
            fused_feature, fused_token, dense_feat, final_pose, final_scale = model.forward_with_memory_dense_feature(
                query_view=[prepared],
                device=str(device),
                memory_feats=memory_feats,
                additional_tokens=memory_token,
                memory_keep_ratio=cfg.fusion.memory_keep_ratio,
                memory_efficient_inference=cfg.training.memory_efficient_inference,
            )
            dense_feat_up = upsampler(batch[0]["img"], fused_feature)
            preds = head(dense_feat_up,fused_token)

        target_world, valid_mask = _prepare_targets(view, device)
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
