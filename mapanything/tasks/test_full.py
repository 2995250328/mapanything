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
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

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

    # ------------------------------------------------------------------
    # 1. 基础配置与设备
    # ------------------------------------------------------------------
    # 修复 device 读取逻辑
    device_str = getattr(cfg, "device", None)
    if device_str is None:
        device_str = getattr(getattr(cfg, "training", None), "device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 2. 动态输出路径生成 (基于权重文件名)
    # ------------------------------------------------------------------
    # 基础目录
    base_output_dir_str = getattr(cfg, "output_dir", None)
    if base_output_dir_str is None:
        base_output_dir_str = getattr(getattr(cfg, "training", None), "output_dir", "outputs/eval")
    base_output_dir = Path(base_output_dir_str).expanduser()

    # 获取权重文件名作为子目录名
    head_ckpt_path = getattr(cfg.head, "checkpoint", None)
    if head_ckpt_path:
        ckpt_name = Path(head_ckpt_path).stem  # e.g., "ace_model_ep64"
    else:
        ckpt_name = "random_weights"

    # 最终输出目录: outputs/eval/{ckpt_name}/
    final_output_dir = base_output_dir / ckpt_name
    final_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Results will be saved to: {final_output_dir}")

    # ------------------------------------------------------------------
    # 3. 数据集与模型初始化
    # ------------------------------------------------------------------
    if isinstance(cfg.dataset.test_dataset, str):
        dataset = eval(cfg.dataset.test_dataset)
    else:
        raise ValueError("Dataset not initialized.")

    scene_mean = getattr(dataset, "mean_cam_center", torch.zeros(3)).to(device)

    model = init_model(cfg.model.model_str, cfg.model.model_config, torch_hub_force_reload=False)
    model.to(device).eval()
    if cfg.model.pretrained:
        ckpt = torch.load(cfg.model.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    # 加载 Head
    in_channels = getattr(cfg.head, "in_channels", 1024)
    if head_ckpt_path and os.path.exists(head_ckpt_path):
        try:
            payload = torch.load(head_ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "in_channels" in payload:
                in_channels = int(payload["in_channels"])
        except: pass

    head = _build_head(cfg, in_channels, scene_mean)
    if head_ckpt_path:
        load_regression_head(head, head_ckpt_path, device=str(device), strict=False, verbose=True)
    head.eval().to(device)

    # Memory
    memory_file = getattr(cfg.fusion, "stored_feature_file", None)
    if memory_file and os.path.exists(memory_file):
        memory_feats, memory_token = load_memory_features(memory_file, device)
        aggregator = BlockwiseAggregator(len(memory_feats), 256).to(device).eval()
        with torch.no_grad(): memory_tokens_per_block = aggregator(memory_feats)
    else:
        memory_tokens_per_block = [None] * getattr(model.info_sharing, "depth", 24)
        memory_token = None

    # DataLoader
    if getattr(cfg.dataset, "sequential_view_mode", False):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    else:
        dataloader = ForcedRandomDataLoader(dataset=dataset, batch_size=1)

    # DSAC* Params
    hyps = int(_get_eval_attr(cfg, "hypotheses", 64))
    threshold = float(_get_eval_attr(cfg, "threshold", 10.0))
    inlieralpha = float(_get_eval_attr(cfg, "inlieralpha", 100.0))
    maxpixelerror = float(_get_eval_attr(cfg, "maxpixelerror", 100.0))

    # 统计变量
    saved_metadata = []
    large_error_samples = [] # 存储误差 > 5cm 的 instance
    rErrs, tErrs_cm = [], []
    pct10_5 = pct5 = pct2 = pct1 = 0
    total_time = 0.0
    num_frames = 0

    # 自动探测场景名称 (用于命名输出文件)
    detected_scene_name = "unknown_scene"

    logger.info("Starting inference...")

    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Evaluating")):
        if isinstance(batch_data, list): batch = batch_data
        else: batch = [batch_data]

        for view in batch:
            if "idx" in view and isinstance(view["idx"], torch.Tensor): view["idx"] = view["idx"][2:]
            for k, v in view.items():
                if isinstance(v, torch.Tensor) and k not in ["dataset", "label", "instance"]:
                    view[k] = v.to(device, non_blocking=True)

        start_time = time.time()
        with torch.no_grad():
            img_h, _ = batch[0]["img"].shape[-2:]
            fused_feature, fused_token = model.forward_dense_feats(batch, getattr(cfg, "memory_efficient_inference", False))
            preds = head(fused_feature, fused_token)

            scale = None
            if isinstance(preds, (tuple, list)): preds, scale = preds
            elif isinstance(preds, dict):
                scale = preds.get("scale", None)
                preds = preds["preds"]

            coords_raw = preds[:, :3]
            if scale is not None:
                while scale.dim() < coords_raw.dim(): scale = scale.unsqueeze(-1)
                pred_coords = coords_raw * scale
            else:
                pred_coords = coords_raw

            # DSAC* Pose Estimation
            view = batch[0]
            feat_h, _ = pred_coords.shape[-2:]
            output_subsample = int(round(img_h / feat_h))

            intrinsics = torch.as_tensor(view.get("camera_intrinsics"), dtype=torch.float32)
            if intrinsics.dim() == 3: intrinsics = intrinsics[0]

            out_pose = torch.zeros((4, 4), dtype=torch.float32)
            inlier_count = dsacstar.forward_rgb(
                pred_coords.cpu(), out_pose, hyps, threshold,
                intrinsics[0, 0].item(), intrinsics[0, 2].item(), intrinsics[1, 2].item(),
                inlieralpha, maxpixelerror, output_subsample
            )

            # Metrics
            if "camera_pose" in view:
                gt_pose = view["camera_pose"].float().cpu()
                if gt_pose.dim() == 3: gt_pose = gt_pose[0]
            else:
                gt_pose = torch.eye(4)

            t_err_m = float(torch.norm(gt_pose[0:3, 3] - out_pose[0:3, 3]))
            r_err_mat = np.matmul(out_pose[0:3, 0:3].numpy(), np.transpose(gt_pose[0:3, 0:3].numpy()))
            r_err = np.arccos(np.clip((np.trace(r_err_mat) - 1) / 2, -1.0, 1.0)) * 180 / math.pi
            logger.info(
                "Frame %d | R: %.2fdeg, T: %.1fcm, Inliers: %d",
                batch_idx, r_err, t_err_m * 100, inlier_count,)

            total_time += (time.time() - start_time)
            rErrs.append(r_err)
            tErrs_cm.append(t_err_m * 100)
            num_frames += 1

            if r_err < 5 and t_err_m < 0.1: pct10_5 += 1
            if r_err < 5 and t_err_m < 0.05: pct5 += 1
            if r_err < 2 and t_err_m < 0.02: pct2 += 1
            if r_err < 1 and t_err_m < 0.01: pct1 += 1

            # -------------------------------------------------------
            # 数据收集 (Data Collection)
            # -------------------------------------------------------
            # 1. 获取标识符
            raw_instance = view.get('instance', view.get('file_path', view.get('name', f"idx_{batch_idx}")))
            instance_name = str(raw_instance)
            if isinstance(raw_instance, (str, Path)): instance_name = Path(raw_instance).name # 只保留文件名

            # 2. 探测场景名称 (用于最终命名文件，取第一帧的场景名即可)
            scene_val = view.get("label") or view.get("scene_name") or f"scene"
            current_scene_name = str(scene_val).replace("/", "_")
            if batch_idx == 0:
                detected_scene_name = current_scene_name

            # 3. 记录坏样本 (>5cm)
            if t_err_m > 0.05:
                large_error_samples.append(instance_name)

            # 4. 记录元数据 (不保存 .pt 文件!)
            saved_metadata.append({
                "index": batch_idx,
                "instance": instance_name,
                "r_err": r_err,
                "t_err": t_err_m * 100, # cm
                "inliers": int(inlier_count)
            })

    # ------------------------------------------------------------------
    # 4. 保存最终结果
    # ------------------------------------------------------------------
    if num_frames > 0:
        median_rErr = np.median(rErrs)
        median_tErr = np.median(tErrs_cm)

        logger.info("================ Summary ================")
        logger.info(f"Scene: {detected_scene_name}")
        logger.info(f"Median: {median_rErr:.2f}deg, {median_tErr:.2f}cm")
        logger.info(f"Acc 5cm/5deg: {pct5/num_frames*100:.1f}%")
        logger.info(f"Acc 2cm/2deg: {pct2/num_frames*100:.1f}%")
        logger.info(f"Acc 1cm/1deg: {pct1/num_frames*100:.1f}%")
    else:
        median_rErr = median_tErr = 0.0

    # 构造文件名：{Scene}_{Metric}.json
    # 例如: chess_summary.json, chess_bad_cases.txt
    summary_file = final_output_dir / f"{detected_scene_name}_summary.json"
    bad_cases_file = final_output_dir / f"{detected_scene_name}_bad_cases.txt"

    # 1. 保存 Bad Cases
    if large_error_samples:
        with open(bad_cases_file, "w") as f:
            for name in large_error_samples:
                f.write(f"{name}\n")
        logger.info(f"Saved {len(large_error_samples)} bad cases to: {bad_cases_file.name}")

    # 2. 保存 Summary JSON
    summary = {
        "checkpoint": ckpt_name,
        "scene": detected_scene_name,
        "metrics": {
            "median_rot": median_rErr,
            "median_trans": median_tErr,
            "pct_5cm_5deg": pct5 / max(1, num_frames),
            "pct_2cm_2deg": pct2 / max(1, num_frames),
            "pct_1cm_1deg": pct1 / max(1, num_frames),
        },
        "large_error_instances": large_error_samples, # 包含在 JSON 中方便读取
        "details": saved_metadata # 包含每一帧的详细数据
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    print("\n" + "="*60)
    print(f" [SUCCESS] Evaluation Finished!")
    print(f" > Output Directory: {final_output_dir.resolve()}")
    print(f" > Summary File:     {summary_file.name}")
    print(f" > Bad Cases File:   {bad_cases_file.name if large_error_samples else 'None'}")
    print("="*60 + "\n")

    return summary

@hydra.main(version_base=None, config_path="../../configs", config_name="ace_test")
def main(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    run_eval(cfg)

if __name__ == "__main__":
    main()