"""Inference & evaluation entrypoint for ACE-style regression heads (Downsample/Native Resolution Version)."""

from __future__ import annotations

import json
import logging
import os
import math
import time
from pathlib import Path
from typing import Any, Dict, List
from mapanything.datasets import SevenScenesWAI

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
    ACEHead_Homogeneous_Mean,
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
        elif isinstance(obj, (np.floating, float)):  # np.floating 是所有浮点类型的基类
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

printer = DebugPrinter()


def _build_head(cfg: DictConfig, in_channels: int, scene_mean: torch.Tensor = None) -> torch.nn.Module:
    # 优先读取 cfg.model.head_mode，如果不存在则回退到 cfg.head.type
    head_mode = getattr(cfg.model, "head_mode", getattr(cfg.head, "type", "film")).lower()

    print(f"Building Regression Head: Mode={head_mode}, In-Channels={in_channels}")

    if head_mode == "ace_homogeneous":
        if scene_mean is None:
            # 如果没有均值，通常回退到 0，或者抛出错误
            print("Warning: ACEHead_Homogeneous_Mean requested but scene_mean is None. Using zeros.")
            scene_mean = torch.zeros(3)

        return ACEHead_Homogeneous_Mean(
            in_channels=in_channels,
            hidden_dim=cfg.head.hidden_dim,
            mean=scene_mean,  # 传入场景均值
            depth=getattr(cfg.head, "depth", 8)
        )

    elif head_mode == "decoupled":
        return ACEHead_Pointwise_Decoupled_WithScale(
            in_channels=in_channels,
            token_dim=getattr(cfg.head, "token_dim", in_channels),
            hidden_dim=cfg.head.hidden_dim,
        )

    else:  # Default to FiLM
        return ACEHead_Pointwise_FiLM(
            in_channels=in_channels,
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

    dataset = None
    if isinstance(cfg.dataset.test_dataset, str):
        dataset = eval(cfg.dataset.test_dataset)
    if dataset is None:
        raise ValueError("Dataset not initialized properly.")

    # 获取场景均值 (Homogeneous Head 需要)
    scene_mean = None
    if hasattr(dataset, "mean_cam_center"):
        scene_mean = dataset.mean_cam_center.to(device)
        print(f"Loaded Scene Mean from Dataset: {scene_mean.tolist()}")
    else:
        print("Warning: Dataset does not provide mean_cam_center. Using Zero Mean.")
        scene_mean = torch.zeros(3, device=device)

    # 1. 初始化模型
    model = init_model(cfg.model.model_str, cfg.model.model_config, torch_hub_force_reload=False)
    model.to(device).eval()

    if cfg.model.pretrained:
        ckpt = torch.load(cfg.model.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    head_ckpt_path = getattr(cfg.head, "checkpoint", None)
    in_channels = cfg.head.in_channels  # 默认从 config 读取

    if head_ckpt_path and os.path.exists(head_ckpt_path):
        try:
            # 仅加载元数据，不加载权重
            payload = torch.load(head_ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "in_channels" in payload:
                ckpt_channels = int(payload["in_channels"])
                print(f"Override: Found 'in_channels' in checkpoint: {ckpt_channels}")
                in_channels = ckpt_channels
        except Exception as e:
            print(f"Warning: Could not probe checkpoint for metadata: {e}")
    head = _build_head(cfg, in_channels, scene_mean)

    if head_ckpt_path:
        print(f"Loading regression head weights from {head_ckpt_path}...")
        # 使用你提供的 load_regression_head 逻辑
        report = load_regression_head(
            head,
            head_ckpt_path,
            device=str(device),  # 建议在 CPU 或指定 Device 加载
            strict=False,
            # ace_homogeneous 有时会被包装在 ace_head 下，增加前缀容错
            allowed_prefixes=("", "module.", "head.", "reg_head.", "ace_head."),
            rename_map=None,
            verbose=True
        )
        # 打印加载报告
        print(f"Head Loaded | used={report['used_from_ckpt']}, missing={len(report['missing'])}, "
              f"unexpected={len(report['unexpected'])}, shape_skip={len(report['skipped_shape'])}")

    head.eval().to(device)

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
    empty_memory = [None] * 24
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
            # 获取原始图像尺寸 (例如 518x518 或 480x640)
            img_h, img_w = batch[0]["img"].shape[-2:]
            # 1. 提取特征 (例如 37x37 或 28x37)
            # 注意：这里使用的是下采样后的特征
            )

            # 2. Head 预测 (输出尺寸为低分辨率，例如 [1, 4, 37, 37])
            preds = head(fused_feature, fused_token)
            # 处理多 Head 输出
            scale = None
            if isinstance(preds, (tuple, list)):
                preds, scale = preds
            elif isinstance(preds, dict):
                scale = preds.get("scale", None)
                preds = preds["preds"]
            coords_raw = preds[:, :3]  # [1, 3, H_feat, W_feat]
            # conf_logits = preds[:, 3:4] # 如果需要置信度

            # 3. 应用 Scale (如果有)
            if scale is not None:
                scale = scale.to(device)
                while scale.dim() < coords_raw.dim():
                    scale = scale.unsqueeze(-1)
                pred_coords = coords_raw * scale
            else:
                pred_coords = coords_raw
            # ==============================================================================
            # [DEBUG] 3D Coordinate Verification Block (Fixed Shapes)
            # ==============================================================================
            # 1. 准备 Ground Truth (Robust Shape Handling)
            # ------------------------------------------------------------------------------
            # 假设 batch[0] 是当前帧的数据字典
            gt_pts3d_raw = batch[0]["pts3d"].to(device)

            # 自动纠正维度：确保变为 [B, 3, H, W]
            if gt_pts3d_raw.dim() == 3:  # [H, W, 3] or [3, H, W]
                if gt_pts3d_raw.shape[-1] == 3:  # HWC -> CHW -> 1CHW
                    gt_pts3d_full = gt_pts3d_raw.permute(2, 0, 1).unsqueeze(0)
                else:  # CHW -> 1CHW
                    gt_pts3d_full = gt_pts3d_raw.unsqueeze(0)
            elif gt_pts3d_raw.dim() == 4:  # [B, H, W, 3] or [B, 3, H, W]
                if gt_pts3d_raw.shape[-1] == 3:  # BHWC -> BCHW
                    gt_pts3d_full = gt_pts3d_raw.permute(0, 3, 1, 2)
                else:  # BCHW
                    gt_pts3d_full = gt_pts3d_raw
            else:
                raise ValueError(f"Unknown pts3d shape: {gt_pts3d_raw.shape}")

            # 处理 Mask (同理处理 HW/BHW/BHW1)
            if "valid_mask" in batch[0]:
                mask_raw = batch[0]["valid_mask"].to(device)
                if mask_raw.dim() == 2:  # HW -> 11HW
                    gt_mask_full = mask_raw.unsqueeze(0).unsqueeze(0)
                elif mask_raw.dim() == 3:  # BHW -> B1HW
                    gt_mask_full = mask_raw.unsqueeze(1)
                elif mask_raw.dim() == 4 and mask_raw.shape[-1] == 1:  # BHW1 -> B1HW
                    gt_mask_full = mask_raw.permute(0, 3, 1, 2)
                else:
                    gt_mask_full = mask_raw
            else:
                gt_mask_full = torch.ones_like(gt_pts3d_full[:, :1])

            # 2. 执行下采样 (Patch Center Sampling)
            # ------------------------------------------------------------------------------
            H_feat, W_feat = pred_coords.shape[-2:]
            H_img, W_img = gt_pts3d_full.shape[-2:]

            # 构建采样网格 (align_corners=True 对应 Patch Center)
            i = torch.arange(H_feat, device=device)
            j = torch.arange(W_feat, device=device)
            yy, xx = torch.meshgrid(i, j, indexing="ij")

            patch_h = H_img / H_feat
            patch_w = W_img / W_feat

            u_c = (xx + 0.5) * patch_w
            v_c = (yy + 0.5) * patch_h

            x_norm = (u_c / (W_img - 1)) * 2 - 1
            y_norm = (v_c / (H_img - 1)) * 2 - 1
            grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # [1, Hf, Wf, 2]

            # 对 GT 坐标进行采样 (此时 gt_pts3d_full 已经是正确的 1x3xHxW)
            gt_pts3d_down = F.grid_sample(gt_pts3d_full, grid, mode='bilinear', align_corners=True)  # -> [1, 3, Hf, Wf]
            gt_mask_down = F.grid_sample(gt_mask_full.float(), grid, mode='nearest', align_corners=True).bool()

            # 3. 计算直接 3D 误差
            # ------------------------------------------------------------------------------
            # 确保 pred_coords 加上了 Mean (根据你的情况调整)
            # 如果是 Homogeneous Head 且正确加载了 Mean，这里 pred_coords 已经是世界坐标
            # 如果发现误差依然巨大且是偏移，尝试取消注释下面这行：
            # if scene_mean is not None and not hasattr(head, "mean"):
            #     pred_coords = pred_coords + scene_mean.view(1, 3, 1, 1)

            diff = pred_coords - gt_pts3d_down
            dist = diff.norm(dim=1, p=2)  # [1, Hf, Wf]

            valid_dist = dist[gt_mask_down.squeeze(1)]

            if valid_dist.numel() > 0:
                print(f"\n[DEBUG Frame {batch_idx}] 3D Coordinate Check:")
                print(f"  > Mean Error:   {valid_dist.mean().item():.4f} m")
                print(f"  > Median Error: {valid_dist.median().item():.4f} m")
                print(f"  > Min Error:    {valid_dist.min().item():.4f} m")

                # 抽查中心点数值
                c_h, c_w = H_feat // 2, W_feat // 2
                p_val = pred_coords[0, :, c_h, c_w].detach().cpu().numpy()
                g_val = gt_pts3d_down[0, :, c_h, c_w].detach().cpu().numpy()
                print(f"  > Center Pred: {p_val}")
                print(f"  > Center GT:   {g_val}")
            else:
                print(f"[DEBUG Frame {batch_idx}] No valid GT points.")
            # ==============================================================================
            # [关键] 动态计算下采样倍率 (Stride)
            # 例如：原图 518，特征图 37 -> stride = 14
            feat_h, feat_w = pred_coords.shape[-2:]
            output_subsample = int(round(img_h / feat_h))

            # 简单校验一下宽高比是否一致，确保 stride 计算正确
            # assert int(round(img_w / feat_w)) == output_subsample, "Width/Height stride mismatch!"

        # 准备数据给 DSAC*
        view = batch[0]
        # 获取场景坐标图 (CPU) [3, H_feat, W_feat]
        scene_coordinates = pred_coords[0].float().cpu()
        # 初始化输出 Pose
        out_pose = torch.zeros((4, 4))

        # 获取内参
        intrinsics = torch.as_tensor(view["camera_intrinsics"], dtype=torch.float32)
        if intrinsics.dim() == 3: intrinsics = intrinsics[0]
        focal_length = intrinsics[0, 0].item()
        ppX = intrinsics[0, 2].item()
        ppY = intrinsics[1, 2].item()

        # [DSAC* Inference]
        # 这里直接传入低分辨率的 scene_coordinates
        # 关键点：传入计算出的 output_subsample (例如 14)
        # DSAC* 内部逻辑：像素坐标 (u, v) -> 对应原图 (u * subsample, v * subsample)
        offset = output_subsample / 2.0
        ppX_corrected = ppX - offset
        ppY_corrected = ppY - offset
        inlier_count = dsacstar.forward_rgb(
            scene_coordinates.unsqueeze(0),  # shape: [1, 3, H_feat, W_feat]
            out_pose,
            hyps,
            threshold,
            focal_length,
            ppX,
            ppY,
            inlieralpha,
            maxpixelerror,
            output_subsample,  # <--- 这里传入 14 (或实际的 stride)
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