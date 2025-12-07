"""Query-only ACE training entry that skips intermediate feature fusion."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Mapping

import hydra
import imageio
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
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
from mapanything.utils.debugprinter import DebugPrinter
printer = DebugPrinter()

def _sanity_check_buffer_reprojection(
    buffer: FeatureReplayBuffer,
    device: torch.device,
    max_samples: int = 4096,
    num_workers: int = 0,
):
    """
    从 FeatureReplayBuffer 中随机抽一批样本：
      - 使用 target_world (世界坐标), intrinsics, c2w
        进行 world->cam->image 投影；
      - 对比投影得到的 (û, ṽ) 与 buffer 中存的像素 (u', v') 的误差。
    如果误差很大，说明特征与真值/像素坐标存在对齐问题。
    """
    from torch.utils.data import DataLoader

    tmp_dataset = BufferDataset(buffer)
    if len(tmp_dataset) == 0:
        print("[SanityCheck] Buffer 为空，跳过重投影检查。")
        return

    batch_size = min(max_samples, len(tmp_dataset))
    tmp_loader = DataLoader(
        tmp_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    batch = next(iter(tmp_loader))

    # 期望 BufferDataset __getitem__ 返回以下字段：
    #   - batch["target_world"]: [N,3]
    #   - batch["intrinsics"]:  [N,3,3]
    #   - batch["c2w"]:         [N,4,4] 或 [N,3,4]
    #   - batch["pixels"]:      [N,2]   (u', v')
    target_world = batch["target_world"].to(device).to(torch.float64)  # (N,3)
    intrinsics = batch["intrinsics"].to(device).to(torch.float64)     # (N,3,3)
    c2w = batch["c2w"].to(device).to(torch.float64)                   # (N,*,*)
    pixels = batch["pixels"].to(device).to(torch.float64)             # (N,2)

    N = target_world.shape[0]

    # ---- 世界坐标 -> 齐次坐标 ----
    ones = torch.ones(N, 1, dtype=torch.float64, device=device)
    pts_w_h = torch.cat([target_world, ones], dim=-1)   # (N,4)

    # ---- 统一 c2w 为 (N,4,4)，再求 w2c ----
    if c2w.shape[-2:] == (3, 4):
        # pad 成 4x4
        pad_row = torch.tensor([0, 0, 0, 1], dtype=torch.float64, device=device)
        pad_row = pad_row.view(1, 1, 4).expand(N, 1, 4)  # (N,1,4)
        c2w_4x4 = torch.cat([c2w, pad_row], dim=-2)      # (N,4,4)
    elif c2w.shape[-2:] == (4, 4):
        c2w_4x4 = c2w
    else:
        raise ValueError(f"[SanityCheck] Unexpected c2w shape: {c2w.shape}")

    w2c = torch.inverse(c2w_4x4)                         # (N,4,4)

    # ---- world -> cam ----
    pts_c_h = torch.bmm(w2c, pts_w_h.unsqueeze(-1)).squeeze(-1)  # (N,4)
    Xc = pts_c_h[:, 0]
    Yc = pts_c_h[:, 1]
    Zc = pts_c_h[:, 2]

    # ---- 用 intrinsics 投影到像素 ----
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    Z_safe = Zc.clamp(min=1e-6)
    u_proj = fx * (Xc / Z_safe) + cx
    v_proj = fy * (Yc / Z_safe) + cy

    # ---- 与 buffer 中的采样像素比较 ----
    u_ref = pixels[:, 0]
    v_ref = pixels[:, 1]

    du = u_proj - u_ref
    dv = v_proj - v_ref
    err = torch.sqrt(du * du + dv * dv)  # 像素 L2 误差

    mean_err = err.mean().item()
    med_err = err.median().item()
    q90_err = err.quantile(0.9).item()
    max_err = err.max().item()

    print("[SanityCheck] Buffer reprojection error (target_world -> cam -> image):")
    print(f"  N samples    = {N}")
    print(f"  mean  pix err = {mean_err:.4f}")
    print(f"  median pix err = {med_err:.4f}")
    print(f"  90%   pix err = {q90_err:.4f}")
    print(f"  max    pix err = {max_err:.4f}")

    # 你也可以在这里加一个简单的 guard，比如误差过大时直接报错：
    # if mean_err > 3.0:
    #     raise RuntimeError(f"[SanityCheck] Mean reprojection error {mean_err:.2f} > 3px, 对齐可能有问题。")

from typing import Mapping

def _eval_head_on_buffer_once(
    head: nn.Module,
    buffer: FeatureReplayBuffer,
    repro_loss: ReproLoss,
    loss_cfg: DictConfig,
    device: torch.device,
    global_step: int,
    max_samples: int = 4096,
    num_workers: int = 0,
) -> Dict[str, float]:
    """
    从 FeatureReplayBuffer 中随机抽一批样本，在当前 head 上做一次前向：
      1. 用 _loss_fn + ReproLoss 计算 loss 和 xyz 误差（err_mean_m 等）
      2. 额外从 head 的输出中抽出 pred_world（世界坐标）,
         结合 intrinsics + c2w 计算重投影误差（像素）
    返回一个扁平化的 dict，key 带 'eval_' 前缀。
    """
    if buffer.size == 0:
        print("[EvalOnBuffer] Buffer 为空，跳过评估。")
        return {}

    ds = BufferDataset(buffer)
    batch_size = min(max_samples, len(ds))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    batch = next(iter(loader))  # 抽一批样本

    # 和训练时保持一致
    features = batch["features"].to(device)
    scale = batch["scale_token"].to(device)

    head_was_training = head.training
    head.eval()
    with torch.no_grad():
        preds = head(features, scale)
        # 先让原来的 _loss_fn 做一遍，拿到 xyz 相关的 metrics
        loss, metrics = _loss_fn(preds, batch, repro_loss, global_step, loss_cfg)

        # ====== 额外计算：pred_world 的重投影误差 ======
        # 1) 解包出 pred_world (N,3)
        if isinstance(preds, torch.Tensor):
            # 兼容 FiLM / Decoupled: 假设前3维是 xyz
            pred_world = preds[..., :3]
        elif isinstance(preds, (list, tuple)):
            # (preds, scale) 这种形式
            main_pred = preds[0]
            pred_world = main_pred[..., :3]
        elif isinstance(preds, dict):
            main_pred = preds.get("preds", None)
            if main_pred is None:
                raise ValueError("[EvalOnBuffer] preds 是 dict 但不含 'preds' 键，无法解析 pred_world")
            pred_world = main_pred[..., :3]
        else:
            raise TypeError(f"[EvalOnBuffer] Unsupported preds type: {type(preds)}")

        # 2) 拿到 GT 像素坐标 & 内参 & 位姿
        #    期望 BufferDataset __getitem__ 已经返回这些字段
        device_d = device
        pixels = batch.get("pixels", None)
        if pixels is None:
            raise KeyError("[EvalOnBuffer] batch 中没有 'pixels' 字段，无法计算重投影误差。")
        pixels = pixels.to(device_d).to(torch.float64)           # (N,2)

        intrinsics = batch.get("intrinsics", None)
        if intrinsics is None:
            intrinsics = batch.get("camera_intrinsics", None)
        if intrinsics is None:
            raise KeyError("[EvalOnBuffer] batch 中没有 'intrinsics' / 'camera_intrinsics' 字段。")
        intrinsics = intrinsics.to(device_d).to(torch.float64)   # (N,3,3)

        c2w = batch.get("c2w", None)
        if c2w is None:
            c2w = batch.get("camera_pose", None)
        if c2w is None:
            raise KeyError("[EvalOnBuffer] batch 中没有 'c2w' / 'camera_pose' 字段。")
        c2w = c2w.to(device_d).to(torch.float64)                 # (N,4,4) or (N,3,4)

        # 3) 世界坐标 -> 像素
        N = pred_world.shape[0]
        pred_world = pred_world.to(device_d, dtype=torch.float64)  # (N,3)

        ones = torch.ones(N, 1, dtype=torch.float64, device=device_d)
        pts_w_h = torch.cat([pred_world, ones], dim=-1)            # (N,4)

        # 统一 c2w 为 (N,4,4)，再求 w2c
        if c2w.shape[-2:] == (3, 4):
            pad_row = torch.tensor([0, 0, 0, 1], dtype=torch.float64, device=device_d)
            pad_row = pad_row.view(1, 1, 4).expand(N, 1, 4)        # (N,1,4)
            c2w_4x4 = torch.cat([c2w, pad_row], dim=-2)            # (N,4,4)
        elif c2w.shape[-2:] == (4, 4):
            c2w_4x4 = c2w
        else:
            raise ValueError(f"[EvalOnBuffer] Unexpected c2w shape: {c2w.shape}")

        w2c = torch.inverse(c2w_4x4)                               # (N,4,4)

        # world -> cam
        pts_c_h = torch.bmm(w2c, pts_w_h.unsqueeze(-1)).squeeze(-1)  # (N,4)
        Xc = pts_c_h[:, 0]
        Yc = pts_c_h[:, 1]
        Zc = pts_c_h[:, 2]

        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]

        Z_safe = Zc.clamp(min=1e-6)
        u_proj = fx * (Xc / Z_safe) + cx
        v_proj = fy * (Yc / Z_safe) + cy

        u_ref = pixels[:, 0]
        v_ref = pixels[:, 1]

        du = u_proj - u_ref
        dv = v_proj - v_ref
        reproj_err = torch.sqrt(du * du + dv * dv)  # (N,)

        reproj_mean = reproj_err.mean().item()
        reproj_med = reproj_err.median().item()
        reproj_q90 = reproj_err.quantile(0.9).item()
        reproj_max = reproj_err.max().item()

    if head_was_training:
        head.train()

    # ====== 汇总所有 eval 指标 ======
    out: Dict[str, float] = {}
    out["eval_total_loss"] = float(loss.item())
    if isinstance(metrics, Mapping):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item()
            try:
                out[f"eval_{k}"] = float(v)
            except Exception:
                pass

    # 追加重投影相关指标（像素单位）
    out["eval_reproj_err_px_mean"] = reproj_mean
    out["eval_reproj_err_px_median"] = reproj_med
    out["eval_reproj_err_px_q90"] = reproj_q90
    out["eval_reproj_err_px_max"] = reproj_max

    return out

def _collect_buffer_query_only(
    cfg: DictConfig,
    model,
    upsampler,
    dataset,
    device: torch.device,
) -> Tuple[FeatureReplayBuffer, int]:
    """Fill the replay buffer using only query features (no memory fusion)."""
    buffer: FeatureReplayBuffer | None = None
    # 兼容两种命名：buffer_size / buffer_capacity
    capacity = int(getattr(cfg.training, "buffer_size",
                    getattr(cfg.training, "buffer_capacity", 0)))
    if capacity <= 0:
        raise ValueError("training.buffer_size / training.buffer_capacity 未正确设置为正整数。")

    # === Debug: 只保存前 N 个 batch 的视图图像，用于检查是否重复 ===
    debug_save_first_n = 5
    debug_dir = Path("/home/xwh/project/tmp")
    debug_dir.mkdir(parents=True, exist_ok=True)
    capacity = int(getattr(cfg.training, "buffer_size", getattr(cfg.training, "buffer_capacity", 0)))
    if capacity <= 0:
        raise ValueError("training.buffer_size / training.buffer_capacity 未正确设置为正整数。")

    pbar = tqdm(total=capacity, desc="Filling training buffer", unit="sample", leave=False)
    try:
        for batch_id, views in enumerate(dataset):
            if batch_id < debug_save_first_n:
                view0 = views[0]
                img = view0["img"]  # 可能是 [1,3,H,W] 或 [3,H,W]

                # 先搬到 CPU，再做后处理
                img = img.detach().cpu()

                # 如果是 [1,3,H,W]，去掉 batch 维
                if img.dim() == 4:
                    img = img[0]  # [3,H,W]

                # [3,H,W] -> [H,W,3]
                img_np = img.permute(1, 2, 0).numpy()

                # 如果你的图像是 0~1 浮点，这样处理；如果是别的范围也没关系，你只是想看“是否相同”
                img_np = np.clip(img_np, 0.0, 1.0)
                img_uint8 = (img_np * 255.0).astype(np.uint8)

                out_path = debug_dir / f"view_batch{batch_id:03d}_view0.png"
                imageio.imwrite(out_path, img_uint8)
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
                scale_token = nn.Parameter(torch.zeros(model.encoder.enc_embed_dim))
                torch.nn.init.trunc_normal_(scale_token, std=0.02)
                scale_token = scale_token.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1).to(device)
                # scale_token = model.scale_token.view(1, -1, 1).to(device)
                # fused_feature, fused_token, _, _, _ = model.forward_with_memory_dense_feature(
                #     query_view=batch,
                #     device=str(device),
                #     memory_tokens_per_block=memory_tokens,
                #     additional_tokens=scale_token,
                #     memory_keep_ratio=1.0,
                #     memory_efficient_inference=cfg.training.memory_efficient_inference,
                # )
                fused_feature, fused_token = model.forward_dense_feats(
                    batch,
                    cfg.training.memory_efficient_inference
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
    # ==== 在这里做一次重投影 sanity check ====
    if getattr(cfg.training, "enable_reproj_sanity_check", True):
        print("\n[SanityCheck] 开始对 FeatureReplayBuffer 做世界坐标重投影检查...")
        _sanity_check_buffer_reprojection(
            buffer=buffer,
            device=device,
            max_samples=getattr(cfg.training, "reproj_check_max_samples", 4096),
            num_workers=0,
        )
        print("[SanityCheck] 完成。\n")
    # ===========================================
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
                # 在 buffer 上做一次评估（GT vs 预测的 xyz / reprojection）
                if global_step % (cfg.training.log_interval * 20) == 0:
                    eval_metrics = _eval_head_on_buffer_once(
                        head=head,
                        buffer=buffer,
                        repro_loss=repro_loss,
                        loss_cfg=cfg.loss,
                        device=device,
                        global_step=global_step,
                        max_samples=getattr(cfg.training, "eval_max_samples", 4096),
                        num_workers=getattr(cfg.training, "num_workers", 0),
                    )
                    log_obj = {
                        "epoch": epoch,
                        "step": global_step,
                        "train": metrics,  # 当前 batch 的训练 metrics
                        "eval": eval_metrics,  # buffer 上整体评估 metrics（含 xyz/reproj 误差）
                    }
                else:
                    log_obj = {
                        "epoch": epoch,
                        "step": global_step,
                        "train": metrics,  # 当前 batch 的训练 metrics
                    }
                print(json.dumps(log_obj, ensure_ascii=False))

    try:
        task_name = HydraConfig.get().runtime.choices.get("model/task", "default")
    except Exception:
        task_name = "unknown"
    ckpt_name = (
        f"ace-noninfo-queryonly_"
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