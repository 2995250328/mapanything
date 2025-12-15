"""ACE 风格的回归头训练入口（支持 Chunk Buffer 机制与置信度采样）。"""

from __future__ import annotations

import json
import os
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import hydra
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn, autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mapanything.datasets import SevenScenesWAI
from mapanything.datasets.base.base_dataset import ForcedRandomDataLoader
from mapanything.models import init_model
from mapanything.tasks.ace import ACEHead_Pointwise_Decoupled_WithScale, ACEHead_Pointwise_FiLM, load_memory_features, ReproLoss
from mapanything.utils.debugprinter import DebugPrinter

# 引入之前拆分的模块和新的置信度模块
from mapanything.tasks.ace.buffer import BufferDataset, _collect_buffer
from mapanything.tasks.ace.loss_utils import _loss_fn, save_batch_error_histograms
# [修改] 导入带有 Hook 功能的加载函数
from mapanything.tasks.ace.confidence import load_confidence_predictor_and_hook

plt.switch_backend('Agg')
printer = DebugPrinter()

# ---------------------------------------------------------------------------
# Buffer Scheduling Helpers (Mimicking ACE Trainer)
# ---------------------------------------------------------------------------
def _recommend_buffer_chunk_size(total_size: int, batch_size: int, target_features: int) -> int:
    """
    启发式计算 Chunk 大小。
    Args:
        target_features: 每次 Buffer 刷新期望包含的样本数 (默认 2M)。
    """
    if total_size <= target_features:
        return total_size

    approx_chunks = max(1, math.ceil(total_size / target_features))
    chunk_size = math.ceil(total_size / approx_chunks)

    # 确保 chunk_size 是 batch_size 的倍数
    chunk_size = max(batch_size, math.ceil(chunk_size / batch_size) * batch_size)

    return min(chunk_size, total_size)

def _build_buffer_chunk_schedule(total_size: int, batch_size: int, preferred_chunk_size: int | None, target_features: int) -> List[int]:
    """构建 Buffer 填充计划。"""

    # 如果配置中没有强制指定 buffer_chunk_size，则根据 target_features 自动推荐
    if preferred_chunk_size is None or preferred_chunk_size <= 0:
        preferred_chunk_size = _recommend_buffer_chunk_size(total_size, batch_size, target_features)

    preferred_chunk_size = int(preferred_chunk_size)
    schedule = []

    if preferred_chunk_size >= total_size:
        schedule.append(total_size)
        return schedule

    chunk_multiple = max(preferred_chunk_size // batch_size, 1) * batch_size
    remaining = total_size
    while remaining > chunk_multiple:
        schedule.append(chunk_multiple)
        remaining -= chunk_multiple

    if remaining > 0:
        if remaining < batch_size and schedule:
            schedule[-1] += remaining
        else:
            schedule.append(remaining)

    return schedule

def _get_feature_dim_dummy(cfg, model, dataset, device):
    """运行一次 Dummy 推理来确定特征维度，用于初始化 Head。"""
    # 注意：这里传入的是 dataset，而不是 dataloader
    dl = ForcedRandomDataLoader(dataset=dataset, batch_size=1)
    batch = next(iter(dl))

    ignore_keys = {"dataset", "label", "instance", "idx", "true_shape", "rng", "data_norm_type"}
    for view in batch:
        if "idx" in view: view["idx"] = view["idx"][2:]
        for name in list(view.keys()):
            if name in ignore_keys: continue
            view[name] = view[name].to(device, non_blocking=True)

    with torch.no_grad():
        fused_feature, _ = model.forward_dense_feats(
            batch,
            cfg.training.memory_efficient_inference
        )
    return fused_feature.shape[1]

# ---------------------------------------------------------------------------
# 训练入口
# ---------------------------------------------------------------------------
def run_training(cfg: DictConfig) -> Dict[str, str]:
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    use_half = getattr(cfg.training, "use_half", False)

    # 1. 加载模型
    model = init_model(cfg.model.model_str, cfg.model.model_config, torch_hub_force_reload=False)
    model.to(device).eval()
    if cfg.model.pretrained:
        print(f"Loading pretrained backbone: {cfg.model.pretrained}")
        ckpt = torch.load(cfg.model.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    # 2. [修改] 加载置信度预测器并注册 Hook
    confidence_predictor = None
    confidence_hook = None
    use_confidence = getattr(cfg.model, "use_confidence_sampling", False)

    if use_confidence:
        print("Initializing Confidence Predictor & Hooks...")
        ckpt_path = getattr(cfg.model, "confidence_checkpoint", "/path/to/checkpoint.pth")
        conf_cfg_path = getattr(cfg.model, "confidence_config", "/path/to/config.json")
        try:
            # 需要传入 backbone，通常是 model.encoder.model
            # 这里的路径取决于 MapAnything 的具体实现结构
            if hasattr(model, "encoder") and hasattr(model.encoder, "model"):
                backbone = model.encoder.model
            else:
                raise AttributeError("Could not find 'encoder.model' in MapAnything.")

            confidence_predictor, confidence_hook = load_confidence_predictor_and_hook(
                ckpt_path, conf_cfg_path, device, backbone
            )
            print("Confidence System Ready.")
        except Exception as e:
            print(f"[Warning] Failed to setup confidence system: {e}. Fallback to random.")
            confidence_predictor = None
            confidence_hook = None

    upsampler = None

    # 3. 准备数据集与 Memory
    memory_feats, memory_token = load_memory_features(cfg.fusion.stored_feature_file, device)
    del memory_feats

    if isinstance(cfg.dataset.train_dataset, str):
        dataset = eval(cfg.dataset.train_dataset)

    # 注意：这里创建 dataloader 是为了给 _collect_buffer 用的，它需要一个 DataLoader 实例
    dataloader = ForcedRandomDataLoader(dataset=dataset, batch_size=1)

    if hasattr(dataset, "mean_cam_center"):
        scene_mean = dataset.mean_cam_center.to(device)
    else:
        scene_mean = torch.zeros(3, device=device)

    # 4. Buffer Scheduling Config (含稳健性防护)
    total_buffer_samples = int(getattr(cfg.training, "buffer_size", getattr(cfg.training, "buffer_capacity", 0)))
    batch_size = cfg.training.batch_size

    # 获取 target_features_per_chunk 并进行稳健性检查
    DEFAULT_TARGET_FEATURES = 2_000_000
    raw_target_features = getattr(cfg.training, "target_features_per_chunk", None)

    target_features = DEFAULT_TARGET_FEATURES
    if raw_target_features is not None:
        try:
            val = int(raw_target_features)
            if val <= 0:
                print(f"[Config Warning] 'target_features_per_chunk' must be > 0. Got {val}. Fallback to default.")
            else:
                target_features = val
        except (ValueError, TypeError):
            print(f"[Config Warning] Invalid 'target_features_per_chunk'. Fallback to default.")

    preferred_chunk_size = getattr(cfg.training, "buffer_chunk_size", None)
    chunk_schedule = _build_buffer_chunk_schedule(total_buffer_samples, batch_size, preferred_chunk_size, target_features)

    # 计算总迭代次数 (Steps/Epoch = 所有 Chunk 的 Batch 数之和)
    steps_per_epoch = sum(chunk // batch_size for chunk in chunk_schedule)
    total_iterations = cfg.training.epochs * steps_per_epoch

    print(f"Buffer Scheduling: Total Size={total_buffer_samples}, Batch={batch_size}")
    print(f"Auto-Refill Target: {target_features} samples/chunk")
    print(f"Chunks per Epoch: {len(chunk_schedule)} | Schedule: {chunk_schedule}")
    print(f"Total Iterations: {total_iterations} (Steps/Epoch: {steps_per_epoch})")

    warmup_steps = getattr(cfg.training, "warmup_steps", 2000)

    repro_loss = ReproLoss(
        total_iterations=total_iterations,
        soft_clamp=cfg.loss.repro_loss_soft_clamp,
        soft_clamp_min=cfg.loss.repro_loss_soft_clamp_min,
        type=cfg.loss.repro_loss_type,
        circle_schedule=(cfg.loss.repro_loss_schedule == 'circle')
    )

    empty_memory = [None] * 24

    # 5. Initialize Head
    print("Running dummy inference to determine input channels...")
    # [FIX] 传入 dataset 而不是 dataloader
    in_channels = _get_feature_dim_dummy(cfg, model, dataset, device)
    print(f"Initializing Regression Head with input dim: {in_channels}")

    if cfg.model.head_mode == "ace_homogeneous":
        from mapanything.tasks.ace.regression_head import ACEHead_Homogeneous_Mean
        head = ACEHead_Homogeneous_Mean(
            in_channels=in_channels,
            hidden_dim=cfg.head.hidden_dim,
            mean=scene_mean,
            depth=getattr(cfg.head, "depth", 8)
        ).to(device)
    elif cfg.model.head_mode == "film":
        head = ACEHead_Pointwise_FiLM(in_channels=in_channels, hidden_dim=cfg.head.hidden_dim).to(device)
    else:
        head = ACEHead_Pointwise_Decoupled_WithScale(in_channels=in_channels, hidden_dim=cfg.head.hidden_dim,
                                                     token_dim=in_channels).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.training.lr, epochs=cfg.training.epochs,
        steps_per_epoch=steps_per_epoch, cycle_momentum=False
    )
    scaler = GradScaler(enabled=use_half)

    output_dir = Path(cfg.training.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    debug_log_path = output_dir / "training_debug1.txt"
    with open(debug_log_path, "w") as f:
        f.write("Step\tLoss\tPxErr\t3D_Mean\t3D_Med\t3D_Min\tMode\tSample_Pred\tSample_GT\n")

    # -----------------------------------------------------------------------
    # 6. Main Training Loop (Refactored: Chunk -> Epoch)
    # -----------------------------------------------------------------------
    global_step = 0
    training_start = time.time()

    print("Starting training loop...")
    print(f"Strategy: Reuse each collected buffer for {cfg.training.epochs} epochs.")

    # 外层循环：Buffer Refill (Chunk)
    for chunk_idx, chunk_size in enumerate(chunk_schedule):

        chunk_start_time = time.time()

        # --- A. Fill Buffer Chunk (Only once per chunk cycle) ---
        print(f"\n[Refill Round {chunk_idx + 1}/{len(chunk_schedule)}] Collecting buffer (Size: {chunk_size})...")

        buffer, _ = _collect_buffer(
            cfg, model, upsampler, dataloader,
            empty_memory, memory_token, device,
            capacity_override=chunk_size,
            confidence_predictor=confidence_predictor,
            confidence_hook=confidence_hook # [新增] 传入 Hook
        )

        bufferloader = DataLoader(
            BufferDataset(buffer),
            batch_size=cfg.training.batch_size,
            shuffle=True, # Shuffle inside the chunk
            num_workers=cfg.training.num_workers,
            drop_last=True
        )

        # --- B. Train on this Buffer for N Epochs ---
        for epoch in range(cfg.training.epochs):
            head.train()

            # 记录本地 epoch loss
            epoch_loss_sum = 0.0
            num_batches = 0

            for batch in bufferloader:
                features = batch["features"].to(device)
                scale = batch["scale_token"].to(device)

                with open_dict(cfg.loss):
                    if global_step < warmup_steps:
                        cfg.loss.mode = "xyz"
                        if "scale_reg" in cfg.loss: cfg.loss.scale_reg.enabled = False
                    else:
                        cfg.loss.mode = "reproj"
                        if "scale_reg" in cfg.loss: cfg.loss.scale_reg.enabled = True

                with torch.autocast(device_type=cfg.training.device, dtype=torch.float16, enabled=use_half):
                    preds = head(features, scale)
                    loss, metrics = _loss_fn(preds, batch, repro_loss, global_step, cfg.loss)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                global_step += 1
                epoch_loss_sum += metrics['loss']
                num_batches += 1

                if global_step % cfg.training.log_interval == 0:
                    time_since_start = time.time() - training_start
                    current_mode = metrics.get('mode', 'unknown')
                    log_msg = (f"Refill:{chunk_idx+1}/{len(chunk_schedule)} | Ep:{epoch+1}/{cfg.training.epochs} | "
                               f"Iter:{global_step:6d} | Loss:{metrics['loss']:.4f} ({current_mode}) | "
                               f"PxErr:{metrics.get('err_mean_px', 0):.1f} | "
                               f"3DMed:{metrics.get('err_3d_median', -1):.3f}m")
                    print(log_msg)

                    with open(debug_log_path, "a") as f:
                        f.write(f"Step {global_step} [Refill {chunk_idx} Ep {epoch}]:\n")
                        f.write(f"  Mode: {current_mode}\n")
                        f.write(f"  Loss: {metrics['loss']:.6f}\n")
                        if 'err_mean_px' in metrics: f.write(f"  PxErr: {metrics['err_mean_px']:.4f}\n")
                        f.write("-" * 40 + "\n")

                    if global_step % (cfg.training.log_interval * 20) == 0:
                        dist_3d_np = metrics.get("raw_dist_3d", None)
                        reproj_err_np = metrics.get("raw_reproj_err", None)
                        save_batch_error_histograms(global_step, dist_3d_np, reproj_err_np, plot_dir)

            # Local Epoch finished
            avg_epoch_loss = epoch_loss_sum / max(1, num_batches)
            print(f"  --> Chunk {chunk_idx+1} | Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}")

        # End of Chunk Refill Round
        print(f"Finished Refill Round {chunk_idx+1}. Cleaning up buffer...")
        del buffer
        del bufferloader
        torch.cuda.empty_cache()

        print(f"Chunk cycle processed in {time.time() - chunk_start_time:.1f}s")

    # 7. 保存 Checkpoint
    try:
        task_name = HydraConfig.get().runtime.choices.get("model/task", "default")
    except Exception:
        task_name = "unknown"

    ckpt_name = (
        f"ace-downsample-chunked_task-{task_name}_head-{cfg.model.head_mode}_loss-{cfg.loss.mode}_"
        f"scale-{'on' if cfg.loss.scale_reg.enabled else 'off'}_"
        f"ep{cfg.training.epochs}_buf{total_buffer_samples}.pt"
    )
    ckpt_path = output_dir / ckpt_name
    torch.save({"state_dict": head.state_dict(), "in_channels": in_channels}, ckpt_path)
    return {"checkpoint": str(ckpt_path), "steps": str(global_step), "in_channels": str(in_channels)}

@hydra.main(version_base=None, config_path="../../configs", config_name="ace_train")
def main(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    info = run_training(cfg)
    print(json.dumps(info, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()