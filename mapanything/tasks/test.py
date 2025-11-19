"""Inference & evaluation entrypoint for ACE-style regression heads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from mapanything.datasets import SevenScenesWAI
from mapanything.datasets.base.base_dataset import ForcedRandomDataLoader
from mapanything.models import init_model
from mapanything.tasks.ace import ACEHead_Pointwise_Decoupled_WithScale, ACEHead_Pointwise_FiLM, load_memory_features, move_view_to_device, load_regression_head
from mapanything.tasks.train import (
    _loss_fn,
    _prepare_targets,
    _resolve_intrinsics,
    _resolve_pose,
)

def run_eval(cfg: DictConfig, _logger=None) -> Dict[str, Any]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = init_model(cfg.model.model_str, cfg.model.model_config, torch_hub_force_reload=False)
    model.to(device).eval()

    if cfg.model.pretrained:
        ckpt = torch.load(cfg.model.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    head = ACEHead_Pointwise_FiLM(in_channels=cfg.head.in_channels, hidden_dim=cfg.head.hidden_dim)
    head_ckpt = getattr(cfg.head, "checkpoint", None)
    if head_ckpt:
        report = load_regression_head(
            head,
            head_ckpt,
            device="cpu",  # 建议先在 CPU 过一遍
            strict=False,  # 不要求严格，允许部分缺失
            allowed_prefixes=("", "module.", "head.", "reg_head.", "ace_head."),
            rename_map=None,  # 如需兼容旧命名，传映射表
        )
        _logger.info(
            "Loaded regression head from %s | used=%d, total_in_ckpt=%d, missing=%d, unexpected=%d, skipped_shape=%d",
            head_ckpt, report["used_from_ckpt"], report["total_in_ckpt"],
            len(report["missing"]), len(report["unexpected"]), len(report["skipped_shape"])
        )
        if report["skipped_shape"]:
            _logger.debug("Skipped (shape mismatch): %s", report["skipped_shape"][:10])
    head.eval().to(device)

    memory_feats, memory_token = load_memory_features(cfg.fusion.stored_feature_file, device)

    # 创建dataset
    dataset = None
    if isinstance(cfg.dataset.train_dataset, str):
        # 注意：eval 在生产环境中存在风险，确保配置来源可信
        dataset = eval(cfg.dataset.train_dataset)
    if dataset is None:
        raise ValueError("Dataset not initialized properly.")

        # 创建顺序读取的 Dataloader
    if cfg.dataset.sequential_view_mode:
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,  # 关键：顺序读取，不打乱
            num_workers=4,  # 适当增加 worker 加速数据读取
            collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
        )
    else:
        # 如果没有开启顺序模式，为了兼容下方循环，也创建一个临时的 loader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

        # =========================================
        # 2. 准备推理组件
        # =========================================
    output_dir = Path(cfg.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载上采样器 (用于将 dense feature 恢复到原图分辨率)
    upsampler = torch.hub.load('wimmerth/anyup', 'anyup', trust_repo=True)
    upsampler.eval().to(device)

    model.eval()
    head.eval()

    metrics: List[Dict[str, float]] = []
    saved_metadata: List[Dict[str, Any]] = []

    print(f"Starting inference on {len(dataset)} frames...")

    # =========================================
    # 3. 主推理循环
    # =========================================
    # 使用 dataloader 进行遍历
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Evaluating")):
        # 限制测试数量 (可选)
        if cfg.eval.max_samples is not None and batch_idx >= cfg.eval.max_samples:
            break

        batch = batch_data
        # 移除不必要的键
        for view in batch:
            if "idx" in view:
                view["idx"] = view["idx"][2:]
        # 转到 GPU（忽略无关键）
        ignore_keys = {
            "dataset", "label", "instance", "idx",
            "true_shape", "rng", "data_norm_type",
        }
        for view in batch:
            for name in list(view.keys()):
                if name in ignore_keys:
                    continue
                view[name] = view[name].to(device, non_blocking=True)

        # -------------------------------------------------
        # A. 模型推理 (Model + Head)
        # -------------------------------------------------
        with torch.no_grad():
            # 1. 提取特征
            # 注意：query_view 通常期望是一个 list of dicts
            fused_feature, fused_token, dense_feat, final_pose, final_scale = model.forward_with_memory_dense_feature(
                query_view=batch,
                device=str(device),
                memory_feats=memory_feats,
                additional_tokens=memory_token,
                memory_keep_ratio=cfg.fusion.memory_keep_ratio,
                memory_efficient_inference=cfg.fusion.memory_efficient_inference,
            )
            dense_feat_up = upsampler(batch[0]["img"], fused_feature)
            preds = head(dense_feat_up, fused_token)

        scale = None
        if isinstance(preds, (tuple, list)):
            preds, scale = preds
        elif isinstance(preds, dict):
            scale = preds.get("scale", None)
            preds = preds["preds"]
        # flatten BCHW 情况（通常你的缓冲区已是 [N,4]；这里容错）
        if preds.dim() == 4:  # [B,4,H,W] -> [N,4]
            B, C, H, W = preds.shape
            preds = preds.permute(0, 2, 3, 1).reshape(-1, C)
        coords_pred_in = preds[:, :3]  # 若带 scale，这是 XYZ_unit；否则是有尺度 XYZ
        raw = preds[:, 3]  # 异方差 raw
        # 如果提供了 scale，则这是 Decoupled 情况：先把无尺度坐标乘回尺度
        if scale is not None:
            scale = scale.to(device).view(-1)  # [N]
            coords_unit = coords_pred_in  # for scale regularization('unit')
            coords = coords_unit * scale.unsqueeze(-1)  # 有尺度坐标，进入主监督
        else:
            coords_unit = None
            pred_coords = coords_pred_in  # FiLM 情况：已是有尺度

        view = batch[0]
        target_world, valid_mask = _prepare_targets(view, device)

        # 计算 L2 距离 (Euclidean Distance)
        diff = pred_coords - target_world
        dist = torch.norm(diff, dim=-1)  # (H, W)
        # 仅在有效区域计算
        valid_dist = dist[valid_mask]

        batch_metrics = {}
        if valid_dist.numel() > 0:
            mae = valid_dist.mean().item()

            # 记录指标 (单位通常是米，取决于数据集)
            batch_metrics = {
                "MAE": mae,
                "Acc_5cm": (valid_dist < 0.05).float().mean().item() * 100,
                "Acc_10cm": (valid_dist < 0.10).float().mean().item() * 100,
            }
            metrics.append(batch_metrics)

        # 保存预测的坐标云用于后续可视化或对齐
        scene_name = view.get("label") or view.get("scene_name") or f"scene_{batch_idx}"
        file_name = Path(view.get("instance", f"{batch_idx}")).stem  # 获取原始文件名
        save_path = output_dir / f"{scene_name}_{file_name}_pred.pt"

        # 压缩保存以节省空间
        torch.save({
            "coords_world": pred_coords.half().cpu(),  # 转半精度保存
            "confidence": torch.sigmoid(preds[0, :, :, 3:4]).half().cpu(),
            "metrics": batch_metrics,
            "scene": scene_name
        }, save_path)

        saved_metadata.append({
            "index": batch_idx,
            "scene": scene_name,
            "path": str(save_path),
            **batch_metrics
        })

    if metrics:
        keys = metrics[0].keys()
        mean_metrics = {k: float(np.mean([m[k] for m in metrics])) for k in keys}
        print("\nGlobal Metrics:")
        for k, v in mean_metrics.items():
            print(f"  {k}: {v:.4f}")
    else:
        mean_metrics = {}

    summary = {
        "outputs": saved_metadata,
        "global_metrics": mean_metrics,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary

@hydra.main(version_base=None, config_path="../../configs", config_name="ace_test")
def main(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    summary = run_eval(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":  # pragma: no cover
    main()
