"""Inference & evaluation entrypoint for ACE-style regression heads."""

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
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from mapanything.utils.debugprinter import DebugPrinter

import dsacstar
from mapanything.datasets import SevenScenesWAI
from mapanything.datasets.base.base_dataset import ForcedRandomDataLoader
from mapanything.models import init_model
from mapanything.tasks.ace import (
    ACEHead_Pointwise_Decoupled_WithScale,
    ACEHead_Pointwise_FiLM,
    load_memory_features,
    load_regression_head,
    move_view_to_device,
)
from mapanything.tasks.train import _prepare_targets, _resolve_pose, BlockwiseAggregator
import json
import numpy as np

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
    model = init_model(cfg.model.model_str, cfg.model.model_config, torch_hub_force_reload=False)
    model.to(device).eval()

    if cfg.model.pretrained:
        ckpt = torch.load(cfg.model.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    head = _build_head(cfg)
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
        logger.info(
            "Loaded regression head from %s | used=%d, total_in_ckpt=%d, missing=%d, unexpected=%d, skipped_shape=%d",
            head_ckpt, report["used_from_ckpt"], report["total_in_ckpt"],
            len(report["missing"]), len(report["unexpected"]), len(report["skipped_shape"])
        )
        if report["skipped_shape"]:
            logger.debug("Skipped (shape mismatch): %s", report["skipped_shape"][:10])
    head.eval().to(device)

    memory_feats, memory_token = load_memory_features(cfg.fusion.stored_feature_file, device)
    aggregator = BlockwiseAggregator(
        num_blocks=24,
        tokens_per_block=256,  # 或者 [256,256,...] 按块自定义
        kmeans_iters=6,
        metric="cosine",
        pre_cap=8192,
        weight_scale=True,
    ).eval()
    with torch.no_grad():
        memory_tokens_per_block = aggregator(memory_feats)  # len=24, 每个 [K_b, C]
    printer.print(memory_tokens_per_block, "memory_tokens_per_block")

    # 创建dataset
    dataset = None
    print(cfg.dataset.test_dataset)
    if isinstance(cfg.dataset.test_dataset, str):
        # 注意：eval 在生产环境中存在风险，确保配置来源可信
        dataset = eval(cfg.dataset.test_dataset)
    if dataset is None:
        raise ValueError("Dataset not initialized properly.")

    # 创建顺序读取的 Dataloader
    if bool(getattr(cfg.dataset, "sequential_view_mode", False)):
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,  # 关键：顺序读取，不打乱
            num_workers=int(getattr(cfg.dataset, "num_workers", 4)),
            collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
        )
    else:
        dataloader = ForcedRandomDataLoader(dataset=dataset, batch_size=1)

    output_dir = Path(cfg.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用anyup进行特征图上采样
    hub_dir = torch.hub.get_dir()
    repo_dir_name = 'wimmerth_anyup_main'  # GitHub 仓库下载后的默认文件夹名
    local_cache_path = os.path.join(hub_dir, repo_dir_name)
    if os.path.exists(local_cache_path):
        print(f"Loading AnyUp from local cache: {local_cache_path}")
        # source='local' 时，第一个参数必须是本地的绝对路径
        upsampler = torch.hub.load(local_cache_path, 'anyup', source='local', trust_repo=True)
    else:
        print("Downloading AnyUp from GitHub...")
        # 第一次下载，或缓存不存在时使用默认方式
        upsampler = torch.hub.load('wimmerth/anyup', 'anyup', trust_repo=True)
    upsampler.to(device).eval()  # 推理模式

    model.eval()
    head.eval()

    saved_metadata: List[Dict[str, Any]] = []
    rErrs: List[float] = []
    tErrs_cm: List[float] = []
    pct10_5 = pct5 = pct2 = pct1 = 0
    total_time = 0.0
    num_frames = 0

    logger.info("Starting inference on %d frames...", len(dataset))

    hyps = int(_get_eval_attr(cfg, "hypotheses", 64))
    threshold = float(_get_eval_attr(cfg, "threshold", 10.0))
    inlieralpha = float(_get_eval_attr(cfg, "inlieralpha", 100.0))
    maxpixelerror = float(_get_eval_attr(cfg, "maxpixelerror", 100.0))
    output_subsample = int(getattr(head, "OUTPUT_SUBSAMPLE", 1))

    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Evaluating")):
        batch = batch_data
        # 遍历 batch 中的每个 view (batch_size=1 时 batch 通常是包含 view 字典的列表)
        for view in batch:
            # === 1. 关键修复：解包字符串字段 ===
            # DataLoader 默认会将字符串 (dataset, label, instance) 打包成 list (例如 ['chess'])
            # 必须将它们还原为字符串，否则后续 Path(view['instance']) 会报错 TypeError
            str_keys = ["dataset", "label", "instance", "scene_name", "idx"]
            for k in str_keys:
                if k in view and isinstance(view[k], (list, tuple)):
                    view[k] = view[k][0]  # 取出列表中的第一个元素还原为字符串
            # === 2. 原有的 idx 处理逻辑 ===
            # 现在 view["idx"] 已经是字符串了，可以安全切片
            if "idx" in view:
                view["idx"] = view["idx"][2:]
            # === 3. 转到 GPU (忽略非 Tensor 键) ===
            ignore_keys = {
                "dataset", "label", "instance", "idx",
                "true_shape", "rng", "data_norm_type",
                "scene_name"
            }
            for name in list(view.keys()):
                if name in ignore_keys:
                    continue
                # 仅对 Tensor 执行 .to()
                if isinstance(view[name], torch.Tensor):
                    view[name] = view[name].to(device, non_blocking=True)

        start_time = time.time()
        with torch.no_grad():
            # 注意：这里假设 batch[0]["img"] 已经通过上面的循环被移动到了 device
            memory_tokens = [None] * getattr(model.info_sharing, "depth", 24)
            scale_token = nn.Parameter(torch.zeros(model.encoder.enc_embed_dim))
            torch.nn.init.trunc_normal_(scale_token, std=0.02)
            scale_token = scale_token.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1).to(device)
            fused_feature, fused_token, _, _, _ = model.forward_with_memory_dense_feature(
                query_view=batch,
                device=str(device),
                memory_tokens_per_block=memory_tokens,
                additional_tokens=scale_token,
                memory_keep_ratio=cfg.fusion.memory_keep_ratio,
                memory_efficient_inference=cfg.fusion.memory_efficient_inference,
            )

            # 确保 upsampler 输入在正确设备
            dense_feat_up = upsampler(batch[0]["img"], fused_feature)
            preds = head(dense_feat_up, fused_token)

        scale = None
        if isinstance(preds, (tuple, list)):
            preds, scale = preds
        elif isinstance(preds, dict):
            scale = preds.get("scale", None)
            preds = preds["preds"]

        if preds.dim() == 2:
            preds = preds.view(preds.shape[0], preds.shape[1], 1, 1)

        coords_raw = preds[:, :3]
        conf_logits = preds[:, 3:4]
        if scale is not None:
            scale = scale.to(device)
            while scale.dim() < coords_raw.dim():
                scale = scale.unsqueeze(-1)
            pred_coords = coords_raw * scale
        else:
            pred_coords = coords_raw

        view = batch[0]
        target_world, _ = _prepare_targets(view, device)
        pred_coords = pred_coords.to(target_world.dtype)

        # Run DSAC++ to recover pose from predicted scene coordinates
        scene_coordinates = pred_coords[0].float().cpu()
        out_pose = torch.zeros((4, 4))

        intrinsics = torch.as_tensor(view["camera_intrinsics"], dtype=torch.float32)
        if intrinsics.dim() == 3:
            intrinsics = intrinsics[0]
        focal_length = intrinsics[0, 0].item()
        ppX = intrinsics[0, 2].item()
        ppY = intrinsics[1, 2].item()

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

        gt_pose = _resolve_pose(view, device=torch.device("cpu"))[0]
        t_err_m = float(torch.norm(gt_pose[0:3, 3] - out_pose[0:3, 3]))

        gt_R = gt_pose[0:3, 0:3].numpy()
        out_R = out_pose[0:3, 0:3].numpy()
        r_err_mat = np.matmul(out_R, np.transpose(gt_R))
        r_err = cv2.Rodrigues(r_err_mat)[0]
        r_err = np.linalg.norm(r_err) * 180 / math.pi

        logger.info(
            "Frame %d | Rotation Error: %.2fdeg, Translation Error: %.1fcm, Inliers: %d",
            batch_idx,
            r_err,
            t_err_m * 100,
            inlier_count,
        )

        rErrs.append(r_err)
        t_err_cm = t_err_m * 100
        tErrs_cm.append(t_err_cm)
        num_frames += 1

        if r_err < 5 and t_err_m < 0.1:
            pct10_5 += 1
        if r_err < 5 and t_err_m < 0.05:
            pct5 += 1
        if r_err < 2 and t_err_m < 0.02:
            pct2 += 1
        if r_err < 1 and t_err_m < 0.01:
            pct1 += 1

        elapsed = time.time() - start_time
        total_time += elapsed

        scene_name = view.get("label") or view.get("scene_name") or f"scene_{batch_idx}"
        file_name = Path(view.get("instance", f"{batch_idx}")).stem
        save_path = output_dir / f"{scene_name}_{file_name}_pred.pt"

        saved_metadata.append(
            {
                "index": batch_idx,
                "scene": scene_name,
                "path": str(save_path),
                "rotation_error_deg": float(r_err),
                "translation_error_cm": float(t_err_cm),
                "inlier_count": int(inlier_count),
            }
        )

    assert len(rErrs) == num_frames
    rErrs.sort()
    tErrs_cm.sort()
    median_idx = len(rErrs) // 2 if rErrs else 0
    median_rErr = rErrs[median_idx] if rErrs else 0.0
    median_tErr = tErrs_cm[median_idx] if tErrs_cm else 0.0

    pct10_5 = pct10_5 / num_frames * 100 if num_frames else 0.0
    pct5 = pct5 / num_frames * 100 if num_frames else 0.0
    pct2 = pct2 / num_frames * 100 if num_frames else 0.0
    pct1 = pct1 / num_frames * 100 if num_frames else 0.0
    avg_time = total_time / num_frames if num_frames else 0.0

    logger.info("===================================================")
    logger.info("Test complete.")
    logger.info("Accuracy:")
    logger.info("\t10cm/5deg: %.1f%%", pct10_5)
    logger.info("\t5cm/5deg: %.1f%%", pct5)
    logger.info("\t2cm/2deg: %.1f%%", pct2)
    logger.info("\t1cm/1deg: %.1f%%", pct1)
    logger.info("Median Error: %.1fdeg, %.1fcm", median_rErr, median_tErr)
    logger.info("Avg. processing time: %4.1fms", avg_time * 1000)

    summary = {
        "outputs": saved_metadata,
        "global_metrics": {
            "median_rotation_error_deg": median_rErr,
            "median_translation_error_cm": median_tErr,
            "pct_10cm_5deg": pct10_5,
            "pct_5cm_5deg": pct5,
            "pct_2cm_2deg": pct2,
            "pct_1cm_1deg": pct1,
            "avg_time_ms": avg_time * 1000,
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    )
    return summary

@hydra.main(version_base=None, config_path="../../configs", config_name="ace_test")
def main(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    summary = run_eval(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":  # pragma: no cover
    main()