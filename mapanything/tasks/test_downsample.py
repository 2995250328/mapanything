"""Inference & evaluation entrypoint for ACE-style regression heads (Downsample/Native Resolution Version)."""

from __future__ import annotations

import json
import logging
import os
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from mapanything.utils.debugprinter import DebugPrinter

import dsacstar
from mapanything.datasets.base.base_dataset import ForcedRandomDataLoader
from mapanything.models import init_model
from mapanything.tasks.ace import (
    ACEHead_Pointwise_Decoupled_WithScale,
    ACEHead_Pointwise_FiLM,
    load_memory_features,
    load_regression_head,
)
# 注意：这里需要确保从正确的路径导入 _prepare_targets 等函数
# 如果你在 train_downsample.py 里定义了修复版，请从那里导入，或者直接把修复版辅助函数复制到这里
from mapanything.tasks.train import _prepare_targets, _resolve_pose, BlockwiseAggregator

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

printer = DebugPrinter()

def _build_head(cfg: DictConfig) -> torch.nn.Module:
    head_type = getattr(cfg.head, "type", "film").lower()
    if head_type == "decoupled":
        return ACEHead_Pointwise_Decoupled_WithScale(
            in_channels=cfg.head.in_channels,
            token_dim=getattr(cfg.head, "token_dim", cfg.head.in_channels),
            hidden_dim=cfg.head.hidden_dim,
        )
    return ACEHead_Pointwise_FiLM(
        in_channels=cfg.head.in_channels,
        hidden_dim=cfg.head.hidden_dim,
    )

def _get_eval_attr(cfg: DictConfig, name: str, default: Any) -> Any:
    eval_cfg = getattr(cfg, "eval", None)
    if eval_cfg is not None and hasattr(eval_cfg, name):
        return getattr(eval_cfg, name)
    return getattr(cfg, name, default)

def run_eval(cfg: DictConfig, _logger=None) -> Dict[str, Any]:
    logger = _logger or logging.getLogger(__name__)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 1. 初始化模型
    model = init_model(cfg.model.model_str, cfg.model.model_config, torch_hub_force_reload=False)
    model.to(device).eval()

    if cfg.model.pretrained:
        ckpt = torch.load(cfg.model.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    # 2. 初始化 Head
    head = _build_head(cfg)
    head_ckpt = getattr(cfg.head, "checkpoint", None)
    if head_ckpt:
        # 使用你之前的 load_regression_head 逻辑
        report = load_regression_head(
            head,
            head_ckpt,
            device="cpu",
            strict=False,
            allowed_prefixes=("", "module.", "head.", "reg_head.", "ace_head."),
            rename_map=None,
        )
        logger.info(
            "Loaded regression head from %s | used=%d, missing=%d",
            head_ckpt, report["used_from_ckpt"], len(report["missing"])
        )
    head.eval().to(device)

    # 3. 加载 Memory 并聚合 (适配 25 层或其他层数)
    # 即使训练时没用 Memory，推理时如果想用，可以在这里加载；
    # 如果想复刻训练时的“单视图模式”，可以将 memory_tokens_per_block 设为 [None]*24
    if cfg.fusion.stored_feature_file and os.path.exists(cfg.fusion.stored_feature_file):
        logger.info(f"Loading memory features from {cfg.fusion.stored_feature_file}")
        memory_feats, memory_token = load_memory_features(cfg.fusion.stored_feature_file, device)

        # 自动适配层数
        num_blocks = len(memory_feats)
        aggregator = BlockwiseAggregator(
            num_blocks=num_blocks,
            tokens_per_block=256,
            kmeans_iters=6,
            metric="cosine",
            pre_cap=8192,
            weight_scale=True,
        ).eval()

        with torch.no_grad():
            memory_tokens_per_block = aggregator(memory_feats)
    else:
        logger.info("No memory file provided or file not found. Running in Single-View Mode.")
        num_blocks = getattr(model.info_sharing, "depth", 24)
        memory_tokens_per_block = [None] * num_blocks
        memory_token = None

    # 4. 创建 Dataset
    dataset = None
    if isinstance(cfg.dataset.test_dataset, str):
        dataset = eval(cfg.dataset.test_dataset)
    if dataset is None:
        raise ValueError("Dataset not initialized properly.")

    if bool(getattr(cfg.dataset, "sequential_view_mode", False)):
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=int(getattr(cfg.dataset, "num_workers", 4)),
            collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
        )
    else:
        dataloader = ForcedRandomDataLoader(dataset=dataset, batch_size=1)

    output_dir = Path(cfg.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # [修改] 移除 AnyUp 上采样器的加载和初始化
    logger.info("Upsampler (AnyUp) disabled. Using native feature resolution.")

    # 5. 推理参数
    hyps = int(_get_eval_attr(cfg, "hypotheses", 64))
    threshold = float(_get_eval_attr(cfg, "threshold", 10.0))
    inlieralpha = float(_get_eval_attr(cfg, "inlieralpha", 100.0))
    maxpixelerror = float(_get_eval_attr(cfg, "maxpixelerror", 100.0))
    output_subsample = int(getattr(head, "OUTPUT_SUBSAMPLE", 1)) # DSAC 的下采样率

    saved_metadata = []
    rErrs = []
    tErrs_cm = []
    pct10_5 = pct5 = pct2 = pct1 = 0
    total_time = 0.0
    num_frames = 0

    logger.info("Starting inference on %d frames...", len(dataset))

    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Evaluating")):
        batch = batch_data
        # 数据解包与移动到 GPU (保持原逻辑)
        for view in batch:
            str_keys = ["dataset", "label", "instance", "scene_name", "idx"]
            for k in str_keys:
                if k in view and isinstance(view[k], (list, tuple)):
                    view[k] = view[k][0]
            if "idx" in view:
                view["idx"] = view["idx"][2:]
            ignore_keys = {"dataset", "label", "instance", "idx", "true_shape", "rng", "data_norm_type", "scene_name"}
            for name in list(view.keys()):
                if name in ignore_keys: continue
                if isinstance(view[name], torch.Tensor):
                    view[name] = view[name].to(device, non_blocking=True)

        start_time = time.time()
        with torch.no_grad():
            # 获取原始图像尺寸
            img_h, img_w = batch[0]["img"].shape[-2:]

            # 1. 提取特征 (28x37)
            fused_feature, fused_token, _, _, _ = model.forward_with_memory_dense_feature(
                query_view=batch,
                device=str(device),
                memory_tokens_per_block=memory_tokens_per_block,
                additional_tokens=memory_token,
                memory_keep_ratio=cfg.fusion.memory_keep_ratio,
                memory_efficient_inference=cfg.fusion.memory_efficient_inference,
            )

            # [修改] 直接将低分辨率特征送入 Head
            # fused_feature: [1, C, 28, 37]
            preds = head(fused_feature, fused_token) # preds: [1, 4, 28, 37]

            # 处理多 Head 输出
            scale = None
            if isinstance(preds, (tuple, list)):
                preds, scale = preds
            elif isinstance(preds, dict):
                scale = preds.get("scale", None)
                preds = preds["preds"]

            if preds.dim() == 2: # [N, 4] -> [B, 4, H, W]
                # 这种情况理论上在 dense inference 不会出现，除非 head 写错了
                # 假设 head 输出已经是 feature map 形状
                pass

            coords_raw = preds[:, :3] # [1, 3, 28, 37]
            conf_logits = preds[:, 3:4]

            # 应用 Scale
            if scale is not None:
                scale = scale.to(device)
                while scale.dim() < coords_raw.dim():
                    scale = scale.unsqueeze(-1)
                pred_coords = coords_raw * scale
            else:
                pred_coords = coords_raw

            # [关键步骤] 将预测的坐标图上采样回原图分辨率
            # 为什么这样做？因为 DSAC* 需要在图像坐标系下采点 (PnP)。
            # 直接把 28x37 变回 392x518，相当于告诉 DSAC：
            # "原图中这 14x14 的区域都对应同一个预测坐标 (即 Patch 中心的预测)"
            # 使用 'bilinear' 可以让坐标变化更平滑，有助于 RANSAC；'nearest' 则严格遵循训练时的假设。
            # 推荐使用 'bilinear' 以获得更好的亚像素精度表现。
            pred_coords_up = F.interpolate(
                pred_coords,
                size=(img_h, img_w),
                mode='bilinear',
                align_corners=False
            ) # [1, 3, H, W]

        # 准备数据给 DSAC*
        view = batch[0]
        # target_world, _ = _prepare_targets(view, device) # 仅用于 check，非必须

        # 获取场景坐标图 (CPU)
        scene_coordinates = pred_coords_up[0].float().cpu() # [3, H, W]

        # 初始化输出 Pose
        out_pose = torch.zeros((4, 4))

        # 获取内参
        intrinsics = torch.as_tensor(view["camera_intrinsics"], dtype=torch.float32)
        if intrinsics.dim() == 3: intrinsics = intrinsics[0]
        focal_length = intrinsics[0, 0].item()
        ppX = intrinsics[0, 2].item()
        ppY = intrinsics[1, 2].item()

        # [DSAC* Inference]
        # 注意：这里我们传入了全分辨率的 map (pred_coords_up)
        # DSAC* 会根据 output_subsample (通常为 8) 再次下采样进行 PnP
        # 这样流程就和标准 ACE 只有分辨率的区别了
        inlier_count = dsacstar.forward_rgb(
            scene_coordinates.unsqueeze(0),
            out_pose,
            hyps,
            threshold,
            focal_length,
            ppX,
            ppY,
            inlieralpha,
            maxpixelerror,
            output_subsample,
        )

        # 计算误差
        gt_pose = _resolve_pose(view, device=torch.device("cpu"))[0]
        t_err_m = float(torch.norm(gt_pose[0:3, 3] - out_pose[0:3, 3]))

        gt_R = gt_pose[0:3, 0:3].numpy()
        out_R = out_pose[0:3, 0:3].numpy()
        r_err_mat = np.matmul(out_R, np.transpose(gt_R))
        r_err = cv2.Rodrigues(r_err_mat)[0]
        r_err = np.linalg.norm(r_err) * 180 / math.pi

        logger.info(
            "Frame %d | R: %.2fdeg, T: %.1fcm, Inliers: %d",
            batch_idx, r_err, t_err_m * 100, inlier_count,
        )

        # 统计指标
        rErrs.append(r_err)
        tErrs_cm.append(t_err_m * 100)
        num_frames += 1

        if r_err < 5 and t_err_m < 0.1: pct10_5 += 1
        if r_err < 5 and t_err_m < 0.05: pct5 += 1
        if r_err < 2 and t_err_m < 0.02: pct2 += 1
        if r_err < 1 and t_err_m < 0.01: pct1 += 1

        elapsed = time.time() - start_time
        total_time += elapsed

        # 保存结果
        scene_name = view.get("label") or view.get("scene_name") or f"scene_{batch_idx}"
        file_name = Path(view.get("instance", f"{batch_idx}")).stem
        save_path = output_dir / f"{scene_name}_{file_name}_pred.pt"

        saved_metadata.append({
            "index": batch_idx,
            "scene": scene_name,
            "rotation_error_deg": float(r_err),
            "translation_error_cm": float(tErrs_cm[-1]),
            "inlier_count": int(inlier_count),
        })

    # 计算最终统计
    if num_frames > 0:
        rErrs.sort()
        tErrs_cm.sort()
        median_rErr = rErrs[len(rErrs) // 2]
        median_tErr = tErrs_cm[len(tErrs_cm) // 2]
        avg_time = total_time / num_frames

        logger.info("================ Summary ================")
        logger.info(f"Median Error: {median_rErr:.2f}deg, {median_tErr:.2f}cm")
        logger.info(f"Acc (10cm/5deg): {pct10_5/num_frames*100:.1f}%")
        logger.info(f"Acc (5cm/5deg):  {pct5/num_frames*100:.1f}%")
        logger.info(f"Acc (1cm/1deg):  {pct1/num_frames*100:.1f}%")
    else:
        median_rErr = median_tErr = avg_time = 0.0

    summary = {
        "metrics": {
            "median_rot": median_rErr,
            "median_trans": median_tErr,
            "pct_10cm_5deg": pct10_5 / max(1, num_frames),
        },
        "details": saved_metadata
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    return summary

@hydra.main(version_base=None, config_path="../../configs", config_name="ace_test")
def main(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    run_eval(cfg)

if __name__ == "__main__":
    main()