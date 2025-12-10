"""ACE 风格的回归头训练入口。"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import hydra
import imageio
import math
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
from mapanything.utils.geometry import quaternion_to_rotation_matrix
plt.switch_backend('Agg')
printer = DebugPrinter()

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class BufferTensors:
    features: torch.Tensor
    scale_token: torch.Tensor
    target_world: torch.Tensor
    pixels: torch.Tensor
    intrinsics: torch.Tensor
    c2w: torch.Tensor

class FeatureReplayBuffer:
    """仿照 ACE，把随机采样的像素特征/位姿放入固定大小的缓冲区。"""

    def __init__(self, buffer_size: int, feat_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.size = 0
        self.storage = BufferTensors(
            features=torch.empty((buffer_size, feat_dim), device=device, dtype=torch.float32),
            scale_token=torch.empty((buffer_size,feat_dim), device=device, dtype=torch.float32),
            target_world=torch.empty((buffer_size, 3), device=device, dtype=torch.float32),
            pixels=torch.empty((buffer_size, 2), device=device, dtype=torch.float32),
            intrinsics=torch.empty((buffer_size, 3, 3), device=device, dtype=torch.float32),
            c2w=torch.empty((buffer_size, 4, 4), device=device, dtype=torch.float32),
        )

    @property
    def is_full(self) -> bool:
        return self.size >= self.buffer_size

    def _remaining(self) -> int:
        return max(0, self.buffer_size - self.size)

    def add_view(
            self,
            fused_feature: torch.Tensor,  # [1,C,H,W] 或 [C,H,W]
            scale_token: torch.Tensor,
            target_world: torch.Tensor,  # [H,W,3]
            valid_mask: torch.Tensor,  # [H,W]
            intrinsics: torch.Tensor,  # [3,3]
            c2w: torch.Tensor,  # [4,4]
            max_per_view: int,
            pixel_grid: Optional[torch.Tensor] = None, # [H, W, 2] 新增参数，用于直接传入下采样后的中心坐标
    ) -> int:
        """从特征图随机采样 max_per_view 个有效像素，写入缓冲区（不超过剩余容量）。"""
        if self.is_full:
            return 0
        # -------- 形状归一：features -> [C,H,W] --------
        if fused_feature.dim() == 4:
            b, c, h, w = fused_feature.shape
            assert b == 1, f"add_view expects single-view tensor, got batch={b}"
            feat_CHW = fused_feature[0]
        elif fused_feature.dim() == 3:
            c, h, w = fused_feature.shape
            feat_CHW = fused_feature
        else:
            raise ValueError(f"Unexpected fused_feature shape: {tuple(fused_feature.shape)}")
        # -------- 形状归一：scale_token -> [C]（本视角共享一个 token）--------
        st = scale_token
        if st.dim() == 1:
            scale_vec = st
        elif st.dim() == 2 and 1 in st.shape:
            scale_vec = st.view(-1)
        elif st.dim() == 3 and (st.shape[-1] == 1 or st.shape[-2] == 1):
            scale_vec = st.view(-1)
        elif st.dim() == 4 and st.shape[0] == 1 and st.shape[-2:] == (1, 1):
            scale_vec = st.view(-1)
        else:
            raise ValueError(f"Unexpected scale_token shape for a per-view token: {tuple(st.shape)}")
        if scale_vec.numel() != c:
            raise ValueError(
                f"Scale token length ({scale_vec.numel()}) != feature dim ({c}). "
                f"Make sure your token size matches channel dim."
            )
        scale_vec = scale_vec.to(feat_CHW.device).contiguous()  # [C]

        # -------- target_world -> [H,W,3] --------
        if target_world.dim() == 4:  # [1,3,H,W]
            assert target_world.shape[0] == 1
            tw = target_world[0]
            if tw.shape[0] == 3:
                tw_HWC = tw.permute(1, 2, 0).contiguous()
            else:
                raise ValueError(f"Unexpected target_world shape: {tuple(target_world.shape)}")
        elif target_world.dim() == 3:
            if target_world.shape[0] == 3:  # [3,H,W]
                tw_HWC = target_world.permute(1, 2, 0).contiguous()
            elif target_world.shape[2] == 3:  # [H,W,3]
                tw_HWC = target_world.contiguous()
            else:
                raise ValueError(f"Unexpected target_world shape: {tuple(target_world.shape)}")
        else:
            raise ValueError(f"Unexpected target_world shape: {tuple(target_world.shape)}")

        # -------- valid_mask -> [H,W] (bool) --------
        if valid_mask is None:
            vm_HW = torch.ones((h, w), dtype=torch.bool, device=feat_CHW.device)
        else:
            if valid_mask.dim() == 4:  # [1,1,H,W]
                vm_HW = valid_mask[0, 0].to(torch.bool)
            elif valid_mask.dim() == 3:  # [1,H,W]
                vm_HW = valid_mask[0].to(torch.bool)
            elif valid_mask.dim() == 2:  # [H,W]
                vm_HW = valid_mask.to(torch.bool)
            else:
                raise ValueError(f"Unexpected valid_mask shape: {tuple(valid_mask.shape)}")

        # -------- 尺寸检查 --------
        assert tw_HWC.shape[0] == h and tw_HWC.shape[1] == w, \
            f"Size mismatch: features ({h},{w}) vs target_world {tuple(tw_HWC.shape[:2])}"
        assert vm_HW.shape[0] == h and vm_HW.shape[1] == w, \
            f"Size mismatch: features ({h},{w}) vs valid_mask {tuple(vm_HW.shape)}"

        # -------- 展平并构建像素坐标 --------
        feats_flat = feat_CHW.view(c, -1).t()  # [N,C]
        targets_flat = tw_HWC.view(-1, 3)  # [N,3]
        mask_flat = vm_HW.view(-1)  # [N]
        if not mask_flat.any().item():
            return 0

        # [修改] 使用传入的 pixel_grid 或 默认 meshgrid
        if pixel_grid is not None:
            # pixel_grid: [H, W, 2]
            assert pixel_grid.shape[0] == h and pixel_grid.shape[1] == w
            pixels_flat = pixel_grid.to(feat_CHW.device).view(-1, 2)
        else:
            ys, xs = torch.meshgrid(
                torch.arange(h, device=feat_CHW.device),
                torch.arange(w, device=feat_CHW.device),
                indexing="ij"
            )
            pixels_flat = torch.stack([xs, ys], dim=-1).view(-1, 2)  # [N,2]  (u,v)

        # -------- 采样（仅有效，且不超过剩余容量）--------
        valid_idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            return 0
        remaining = self._remaining() if hasattr(self, "_remaining") else (self.capacity - self.size)
        if remaining <= 0:
            return 0
        num_to_take = min(max_per_view, valid_idx.numel(), remaining)
        perm = torch.randperm(valid_idx.numel(), device=feat_CHW.device)
        chosen = valid_idx[perm[:num_to_take]]
        if chosen.numel() == 0:
            return 0

        # -------- 写入缓冲区 --------
        slot = slice(self.size, self.size + chosen.numel())
        self.storage.features[slot] = feats_flat[chosen].to(self.device)
        self.storage.target_world[slot] = targets_flat[chosen].to(self.device)
        self.storage.pixels[slot] = pixels_flat[chosen].to(self.device)  # (u,v)
        # 视角常量：intrinsics / c2w
        self.storage.intrinsics[slot] = intrinsics.to(self.device)
        self.storage.c2w[slot] = c2w.to(self.device)
        # 每个采样都使用同一个 per-view scale token（广播）
        self.storage.scale_token[slot] = scale_vec.unsqueeze(0).expand(chosen.numel(), -1).to(self.device)
        self.size += chosen.numel()

        return int(chosen.numel())

class BufferDataset(Dataset):
    """薄包装，直接从缓冲区张量读取样本。"""

    def __init__(self, buffer: FeatureReplayBuffer):
        self.buffer = buffer

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.buffer.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # pragma: no cover - thin wrapper
        return {
            "features": self.buffer.storage.features[idx],
            "scale_token": self.buffer.storage.scale_token[idx],
            "target_world": self.buffer.storage.target_world[idx],
            "pixels": self.buffer.storage.pixels[idx],
            "intrinsics": self.buffer.storage.intrinsics[idx],
            "c2w": self.buffer.storage.c2w[idx],
        }

# ---------------------------------------------------------------------------
# 基础几何工具
# ---------------------------------------------------------------------------
def _project_world_points(
    points_world: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """稠密投影版本，兼容 Bx3xHxW 的输入。"""

    batch, _, height, width = points_world.shape
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if intrinsics.size(0) == 1 and batch > 1:
        intrinsics = intrinsics.expand(batch, -1, -1)
    if c2w.dim() == 2:
        c2w = c2w.unsqueeze(0)
    if c2w.size(0) == 1 and batch > 1:
        c2w = c2w.expand(batch, -1, -1)

    points_flat = points_world.view(batch, 3, -1)
    rotation = c2w[:, :3, :3]
    translation = c2w[:, :3, 3:4]
    cam_points = torch.bmm(rotation.transpose(1, 2), points_flat - translation)

    z = cam_points[:, 2:3, :]
    valid = z > 1e-6
    pixels = torch.bmm(intrinsics, cam_points)
    pixels = pixels[:, :2, :] / z.clamp(min=1e-6)
    return pixels.view(batch, 2, height, width), valid.view(batch, 1, height, width)

def _resolve_intrinsics(view: Dict[str, Any], device: torch.device) -> torch.Tensor:
    if "camera_intrinsics" in view:
        intr = torch.as_tensor(view["camera_intrinsics"], device=device, dtype=torch.float32)
    elif "intrinsics" in view:
        intr = torch.as_tensor(view["intrinsics"], device=device, dtype=torch.float32)
    else:
        raise KeyError("Camera intrinsics not available in view dictionary.")

    # 统一到 [B, 3, 3]
    if intr.dim() == 2:  # [3, 3]
        intr = intr.unsqueeze(0)
    return intr

def _resolve_pose(view: Dict[str, Any], device: torch.device) -> torch.Tensor:
    if "camera_pose" in view:
        pose = torch.as_tensor(view["camera_pose"], device=device, dtype=torch.float32)
    elif "camera_pose_quats" in view and "camera_pose_trans" in view:
        # 合成 pose
        quats = torch.as_tensor(view["camera_pose_quats"], device=device, dtype=torch.float32)
        trans = torch.as_tensor(view["camera_pose_trans"], device=device, dtype=torch.float32)
        # 注意：这里假设 quaternion_to_rotation_matrix 支持 batch
        # 如果不支持，可能需要 reshape
        rot = quaternion_to_rotation_matrix(quats)  # [B, 3, 3] or [3, 3]
        if rot.dim() == 2: rot = rot.unsqueeze(0)
        if trans.dim() == 1: trans = trans.unsqueeze(0)

        B = rot.shape[0]
        pose = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)
        pose[:, :3, :3] = rot
        pose[:, :3, 3] = trans.view(B, 3)
    else:
        raise KeyError("Camera pose not available in view dictionary.")

    # 统一到 [B, 4, 4]
    if pose.dim() == 2:
        pose = pose.unsqueeze(0)
    return pose

def _prepare_targets(view: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and preprocess ground truth 3D points and valid mask.
    Handles inputs with or without batch dimension.
    Returns:
        pts3d: [B, 3, H, W]
        valid_mask: [B, 1, H, W]
    """
    if "pts3d" not in view:
        # 尝试从 'depth' 或 'depthmap' 恢复，或者报错
        # 这里假设必须有 pts3d (ACE 训练通常需要世界坐标)
        raise KeyError("View is missing dense world coordinates (pts3d).")

    pts3d = torch.as_tensor(view["pts3d"], device=device, dtype=torch.float32)

    # 统一维度到 [B, 3, H, W]
    # Case A: [H, W, 3] -> [1, 3, H, W]
    if pts3d.dim() == 3 and pts3d.shape[-1] == 3:
        pts3d = pts3d.permute(2, 0, 1).unsqueeze(0)
    # Case B: [B, H, W, 3] -> [B, 3, H, W]  (DataLoader batch_size=1 常见情况)
    elif pts3d.dim() == 4 and pts3d.shape[-1] == 3:
        pts3d = pts3d.permute(0, 3, 1, 2)
    # Case C: [B, 3, H, W] -> 保持不变
    elif pts3d.dim() == 4 and pts3d.shape[1] == 3:
        pass
    else:
        raise ValueError(f"Unsupported pts3d shape: {pts3d.shape}")

    # 处理 Mask
    if "valid_mask" in view:
        mask = torch.as_tensor(view["valid_mask"], device=device, dtype=torch.float32)
        # Case A: [H, W] -> [1, 1, H, W]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        # Case B: [B, H, W] -> [B, 1, H, W]
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        # Case C: [B, H, W, 1] -> [B, 1, H, W]
        elif mask.dim() == 4 and mask.shape[-1] == 1:
            mask = mask.permute(0, 3, 1, 2)
        valid_mask = mask
    else:
        # 默认全有效
        valid_mask = torch.ones((pts3d.shape[0], 1, pts3d.shape[2], pts3d.shape[3]),
                                device=device, dtype=torch.float32)

    return pts3d, valid_mask

# ---------------------------------------------------------------------------
# 缓冲区构建与损失
# ---------------------------------------------------------------------------
def _collect_buffer(
        cfg: DictConfig,
        model,
        upsampler,  # 这里的 upsampler 可以是 None
        dataset,
        memory_feats,  # 这里的 memory_feats 即使传进来，我们也可以选择不用
        memory_token,
        device: torch.device,
) -> Tuple[FeatureReplayBuffer, int]:
    buffer: FeatureReplayBuffer | None = None

    capacity = int(getattr(cfg.training, "buffer_size", getattr(cfg.training, "buffer_capacity", 0)))
    if capacity <= 0:
        raise ValueError("training.buffer_size 未正确设置。")

    pbar = tqdm(total=capacity, desc="Filling training buffer", unit="sample", leave=False)

    # 构造一个空的 Memory 列表，用于强制模型进入单视图模式
    # 假设模型有 24 层 block，我们传入 24 个 None
    # 这样 forward_query_with_memory 会跳过所有 Cross-Attention
    num_blocks = getattr(model.info_sharing, "depth", 24)
    empty_memory = [None] * num_blocks
    debug_dir = Path("/home/xwh/project/tmp")
    debug_dir.mkdir(parents=True, exist_ok=True)
    try:
        for batch_id, views in enumerate(dataset):
            if batch_id < 5:
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
                if "idx" in view: view["idx"] = view["idx"][2:]

            # 转到 GPU
            ignore_keys = {"dataset", "label", "instance", "idx", "true_shape", "rng", "data_norm_type"}
            for view in batch:
                for name in list(view.keys()):
                    if name in ignore_keys: continue
                    view[name] = view[name].to(device, non_blocking=True)

            with torch.no_grad():
                # [关键修改]：传入 empty_memory 而不是 memory_feats
                # 这样模型只提取 Query 自身的特征，不做 Memory 融合
                # fused_feature, fused_query_feature_noinfo, fused_token, dense_feat, final_pose, final_scale = model.forward_with_memory_dense_feature(
                #     query_view=batch,
                #     device=str(device),
                #     memory_tokens_per_block=empty_memory,  # <--- 强制不用 Memory
                #     additional_tokens=memory_token,
                #     memory_keep_ratio=cfg.fusion.memory_keep_ratio,
                #     memory_efficient_inference=cfg.training.memory_efficient_inference,
                # )
                fused_feature, fused_token = model.forward_dense_feats(
                    batch,
                    cfg.training.memory_efficient_inference
                )

                # [关键修改]：直接使用下采样的特征 (28x37)，不进行上采样
                # fused_feature shape: [1, C, 28, 37]
                feat_down = fused_feature

            view = batch[0]
            # 获取全分辨率的 GT (392x518)
            target_world_full, valid_mask_full = _prepare_targets(view, device)

            # --- 2D-3D 对齐逻辑：Patch 中心采样 ---
            H0, W0 = target_world_full.shape[-2:]  # 原图尺寸
            Hf, Wf = feat_down.shape[-2:]  # 特征尺寸

            # 1. 构建特征网格索引
            i = torch.arange(Hf, device=device)
            j = torch.arange(Wf, device=device)
            yy, xx = torch.meshgrid(i, j, indexing="ij")  # yy: [Hf, Wf], xx: [Hf, Wf]

            # 2. 计算 Patch 大小
            patch_h = H0 / Hf
            patch_w = W0 / Wf

            # 3. 计算 Patch 中心在原图中的精确像素坐标
            u_c = (xx + 0.5) * patch_w  # [Hf, Wf]
            v_c = (yy + 0.5) * patch_h  # [Hf, Wf]

            # 4. 归一化到 [-1, 1] 用于 grid_sample (align_corners=True)
            x_norm = (u_c / (W0 - 1)) * 2 - 1
            y_norm = (v_c / (H0 - 1)) * 2 - 1
            grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # [1, Hf, Wf, 2]

            # 5. 采样 GT 3D 坐标
            # target_world_full: [1, 3, H0, W0] -> [1, 3, Hf, Wf]
            target_world_down = F.grid_sample(target_world_full, grid, mode='nearest', align_corners=True)
            target_world_for_buffer = target_world_down.squeeze(0).permute(1, 2, 0)  # [Hf, Wf, 3]

            # 6. 采样 Mask (使用 nearest 避免无效值扩散)
            valid_mask_down = F.grid_sample(valid_mask_full.float(), grid, mode='nearest', align_corners=True)
            valid_mask_for_buffer = valid_mask_down.squeeze(0).squeeze(0).bool()  # [Hf, Wf]

            # 7. 准备 Buffer 用的像素坐标 (用于 Loss 重投影)
            # 必须使用精确计算的 u_c, v_c，而不是整数索引
            pixel_grid_for_buffer = torch.stack([u_c, v_c], dim=-1)  # [Hf, Wf, 2]

            intrinsics = _resolve_intrinsics(view, device).to(torch.float32)
            c2w = _resolve_pose(view, device).to(torch.float32)

            if buffer is None:
                in_channels = int(feat_down.shape[1])
                buffer = FeatureReplayBuffer(capacity, in_channels, device=torch.device("cpu"))

            # 8. 写入 Buffer
            added = buffer.add_view(
                feat_down.squeeze(0).detach(),  # [C, 28, 37]
                fused_token.squeeze(0).detach(),  # [C]
                target_world_for_buffer.detach(),  # [28, 37, 3]
                valid_mask_for_buffer.detach(),  # [28, 37]
                intrinsics.squeeze(0).detach(),  # [3, 3]
                c2w.squeeze(0).detach(),  # [4, 4]
                cfg.training.samples_per_view,
                pixel_grid=pixel_grid_for_buffer.detach()  # [28, 37, 2] 传入精确坐标
            )

            if added > 0:
                pbar.update(added)
            if buffer.is_full:
                break

        if buffer is None or buffer.size == 0:
            raise RuntimeError("Buffer empty.")
        return buffer, buffer.storage.features.shape[1]

    finally:
        pbar.close()

def _invert_c2w_to_w2c(c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    统一的位姿求逆函数。
    支持输入形状: [..., 3, 4] 或 [..., 4, 4]
    Args:
        c2w: 相机到世界坐标系的变换矩阵 (Camera-to-World)
    Returns:
        R: 世界到相机的旋转矩阵 (World-to-Camera Rotation), shape [..., 3, 3]
        t: 世界到相机的平移向量 (World-to-Camera Translation), shape [..., 3, 1]
    """
    # 检查最后两个维度
    if c2w.shape[-2:] == (4, 4):
        # 情况 1: 4x4 齐次矩阵
        # 使用 torch.inverse 对整体求逆比较数值稳定
        w2c = torch.inverse(c2w)
        R = w2c[..., :3, :3]
        t = w2c[..., :3, 3:4]
    elif c2w.shape[-2:] == (3, 4):
        # 情况 2: 3x4 仿射矩阵 [R|t]
        # 利用几何性质求逆: R_inv = R^T, t_inv = -R^T * t
        R_c2w = c2w[..., :3, :3]
        t_c2w = c2w[..., :3, 3:4]

        R = R_c2w.transpose(-1, -2)
        t = -R @ t_c2w
    else:
        raise ValueError(f"Unexpected c2w shape: {tuple(c2w.shape)}. Expected (..., 3, 4) or (..., 4, 4).")

    return R, t

def _project_world_points_sparse(
    Xw_N3: torch.Tensor,
    K_N33: torch.Tensor,
    c2w_N: torch.Tensor,
    depth_min: float = 1e-4
) -> torch.Tensor:
    """
    Xw_N3 -> pixels_N2  (batched, no mask; z clamped to depth_min)
    """
    R, t = _invert_c2w_to_w2c(c2w_N)                      # [N,3,3], [N,3,1]
    Xw = Xw_N3.unsqueeze(-1)                              # [N,3,1]
    Xc = R @ Xw + t                                       # [N,3,1]
    Xc_z = Xc[:, 2:3, :]                                  # [N,1,1]
    Xc_z = Xc_z.clamp(min=depth_min)
    uvd = K_N33 @ Xc                                      # [N,3,1]
    uv = uvd[:, :2, :] / Xc_z                             # [N,2,1]
    return uv.squeeze(-1)                                 # [N,2]

def _compute_scale_reg(z_flat, target_world, c2w, loss_cfg, device):
    """计算尺度正则化 Loss"""
    sr = getattr(loss_cfg, "scale_reg", None)
    if not (sr and getattr(sr, "enabled", False)):
        return torch.tensor(0.0, device=device), 0.0

    variant = getattr(sr, "variant", "prior")
    weight = float(getattr(sr, "weight", 1e-3))
    depth_min = float(getattr(loss_cfg, "depth_min", 0.1))
    eps = float(getattr(loss_cfg, "eps", 1e-8))

    if weight <= 0 or z_flat.numel() == 0:
        return torch.tensor(0.0, device=device), 0.0

    med_pred = torch.median(z_flat)
    reg = torch.tensor(0.0, device=device)

    if variant == "match_to_gt":
        if target_world is not None and c2w is not None:
            R, t = _invert_c2w_to_w2c(c2w)
            Xw_gt = target_world.unsqueeze(-1)
            # 计算 GT 在相机系下的 Z
            Zg = (R @ Xw_gt + t)[:, 2, 0].clamp_min(depth_min)
            med_gt = torch.median(Zg)
            reg = weight * torch.abs(torch.log((med_pred + eps) / (med_gt + eps)))
    elif variant == "unit":
        reg = weight * torch.abs(torch.log(med_pred + eps))
    elif variant == "prior":
        depth_prior = getattr(sr, "depth_prior", 1.0)
        reg = weight * torch.abs(torch.log((med_pred + eps) / (float(depth_prior) + eps)))

    return reg, float(reg.detach().cpu())

def save_batch_error_histograms(step, dist_3d_np, reproj_err_np, save_dir):
    """绘制并保存当前 Batch 的误差分布直方图"""
    save_path = save_dir / f"step_{step:06d}_dist.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. 3D Error Histogram
    if dist_3d_np is not None and len(dist_3d_np) > 0:
        # 过滤掉极端的异常值以便绘图更好看 (显示 99% 分位数以内)
        limit_3d = np.percentile(dist_3d_np, 99) if len(dist_3d_np) > 100 else dist_3d_np.max()
        axes[0].hist(dist_3d_np, bins=50, range=(0, limit_3d), color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_title(f'3D Error Dist (m)\nMedian: {np.median(dist_3d_np):.3f}m')
        axes[0].set_xlabel('Error (m)')
        axes[0].set_ylabel('Count')
    else:
        axes[0].text(0.5, 0.5, 'No 3D Data', ha='center')

    # 2. Reproj Error Histogram
    if reproj_err_np is not None and len(reproj_err_np) > 0:
        limit_px = np.percentile(reproj_err_np, 99) if len(reproj_err_np) > 100 else reproj_err_np.max()
        axes[1].hist(reproj_err_np, bins=50, range=(0, limit_px), color='salmon', edgecolor='black', alpha=0.7)
        axes[1].set_title(f'Reproj Error Dist (px)\nMedian: {np.median(reproj_err_np):.2f}px')
        axes[1].set_xlabel('Error (px)')
    else:
        axes[1].text(0.5, 0.5, 'No Reproj Data', ha='center')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
# 尺度正则默认关闭，需在 loss.scale_reg.enabled=true 才会生效。
# match_to_gt：需要 batch 中提供 target_world 与 c2w。
# unit：用于无尺度预测（解耦方案），把中位数拉向 1。
# prior：无 GT 时让深度中位数靠近 depth_prior。
# 正则权重建议从 1e-4 ~ 1e-3 起步，防止盖过主监督信号。
def _loss_fn(
        preds,
        batch: Dict[str, torch.Tensor],
        repro_loss,
        global_step,
        loss_cfg: DictConfig,
):
    device = preds.device
    mode = getattr(loss_cfg, "mode", "reproj")
    conf_mode = getattr(loss_cfg, "conf_mode", "log_sigma")

    # 阈值
    sigma_min = float(getattr(loss_cfg, "sigma_min", 1e-4))
    sigma_max = float(getattr(loss_cfg, "sigma_max", 10.0))
    depth_min = float(getattr(loss_cfg, "depth_min", 0.1))
    depth_max = float(getattr(loss_cfg, "depth_max", 50.0))
    eps = float(getattr(loss_cfg, "eps", 1e-8))

    SANITY_PIXEL_ERR = 50000.0
    SANITY_COORD_VAL = 10000.0

    # 预处理 Preds
    scale = None
    if isinstance(preds, (tuple, list)):
        preds, scale = preds
    elif isinstance(preds, dict):
        scale = preds.get("scale", None)
        preds = preds["preds"]

    if preds.dim() == 4:
        preds = preds.permute(0, 2, 3, 1).reshape(-1, preds.shape[1])

    if torch.isnan(preds).any() or torch.isinf(preds).any():
        print(f"[CRITICAL] Iter {global_step}: Input preds contain NaN/Inf! Returning zero loss.")
        return preds.sum() * 0.0, {"loss": 0.0, "mode": "nan_skipped"}

    coords_pred_in = preds[:, :3]
    raw = preds[:, 3]

    # 不确定性处理 (Confidence / Sigma)
    if conf_mode == "confidence":
        p = torch.sigmoid(raw)
        sigma = (1.0 - p) * sigma_max + p * sigma_min
    else:
        sigma = F.softplus(raw) + eps
        sigma = torch.clamp(sigma, min=sigma_min, max=sigma_max)

    # 应用 Scale
    if scale is not None:
        coords = coords_pred_in * scale.to(device).view(-1).unsqueeze(-1)
    else:
        coords = coords_pred_in

    # -------------------------------------------------------------------------
    # 统一计算几何指标 (无论什么模式都算，方便统计)
    # -------------------------------------------------------------------------
    target_world = batch.get("target_world", None)
    if target_world is not None:
        target_world = target_world.to(device)
        diff_3d = coords - target_world
        dist_3d = torch.norm(diff_3d, dim=1, p=2)  # [N]
    else:
        dist_3d = None

    nuclear_mask = torch.zeros(coords.shape[0], dtype=torch.bool, device=device)
    repro_err = None
    loss_val = 0.0
    metrics = {}

    # -------------------------------------------------------------------------
    # 分支逻辑
    # -------------------------------------------------------------------------
    if mode == "xyz":
        # === Branch 1: XYZ Loss (Warmup) ===
        if target_world is None:
            raise ValueError("XYZ mode requires 'target_world' in batch.")

        # 使用 Huber Loss 提高鲁棒性
        loss_huber = F.huber_loss(coords, target_world, delta=1.0, reduction='none').sum(dim=1)
        total = loss_huber.mean() * 100.0  # 放大 Loss 以匹配 Reproj 量级

        metrics["loss"] = float(total.detach().cpu())
        metrics["mode"] = "xyz-huber"

        # 为了统计图表，构造 dummy reproj_err
        repro_err = torch.zeros_like(dist_3d)
    else:
        # === Branch 2: Reprojection Loss ===
        K = batch["intrinsics"].to(device)
        c2w = batch["c2w"].to(device)
        px = batch["pixels"].to(device)

        R, t = _invert_c2w_to_w2c(c2w)
        Xw = coords.unsqueeze(-1)
        Xc = R @ Xw + t
        z = Xc[:, 2:3, :]
        z_safe = z.clamp(min=1e-3)
        z_flat = z[:, 0, 0]

        uvh = K @ Xc
        uv = (uvh[:, :2, :] / z_safe).squeeze(-1)
        repro_err = (uv - px).abs().sum(dim=1)

        # 熔断检测
        nuclear_mask = (repro_err > SANITY_PIXEL_ERR) | \
                       (torch.abs(coords).max(dim=1)[0] > SANITY_COORD_VAL) | \
                       (torch.isnan(repro_err)) | \
                       (torch.isinf(repro_err))

        if nuclear_mask.any():
            metrics["nuclear_cnt"] = float(nuclear_mask.sum().cpu())

        # Mask 生成
        invalid_min_depth = (z_flat < depth_min)
        invalid_max_depth = (z_flat > depth_max)
        repro_clamp = float(getattr(loss_cfg, "repro_loss_hard_clamp", 100.0))
        invalid_repro = (repro_err > repro_clamp)

        base_invalid_mask = invalid_min_depth | invalid_repro | invalid_max_depth
        valid_mask = (~base_invalid_mask) & (~nuclear_mask)
        invalid_mask = base_invalid_mask & (~nuclear_mask)

        # Loss 计算
        loss_valid = torch.tensor(0.0, device=device)
        loss_invalid = torch.tensor(0.0, device=device)

        if valid_mask.any():
            loss_valid = repro_loss.compute(repro_err[valid_mask], global_step)

        if invalid_mask.any():
            invK = batch.get("intrinsics_inv", torch.inverse(K)).to(device)
            uv1 = torch.cat([px, torch.ones_like(px[:, :1])], dim=1).unsqueeze(-1)
            Xc_tgt = float(getattr(loss_cfg, "depth_target", 10.0)) * (invK @ uv1)
            dist_error = (Xc_tgt - Xc).abs().sum(dim=1).squeeze()
            loss_invalid = dist_error[invalid_mask].sum()

        total = (loss_valid + loss_invalid) / coords.shape[0]

        # 尺度正则化
        scale_reg_val = 0.0
        safe_indices = ~nuclear_mask
        if safe_indices.any():
            c2w_safe = c2w[safe_indices] if target_world is not None else None
            reg_loss, scale_reg_val = _compute_scale_reg(
                z_flat[safe_indices],
                target_world[safe_indices] if target_world is not None else None,
                c2w_safe, loss_cfg, device
            )
            total = total + reg_loss

        metrics["loss"] = float(total.detach().cpu())
        metrics["mode"] = "reproj-safe"
        metrics["scale_reg"] = scale_reg_val

    # -------------------------------------------------------------------------
    # 统一 Debug 统计 (Data Export)
    # -------------------------------------------------------------------------
    # 导出 Raw Tensor 供外部绘图使用 (转到 CPU)
    # 只有非 Nuclear 的点才值得统计
    safe_indices = ~nuclear_mask
    if safe_indices.any():
        # 1. 3D Error Stats
        if dist_3d is not None:
            d_safe = dist_3d[safe_indices]
            metrics["err_3d_mean"] = float(d_safe.mean().detach().cpu())
            metrics["err_3d_median"] = float(d_safe.median().detach().cpu())
            metrics["err_3d_min"] = float(d_safe.min().detach().cpu())
            metrics["err_3d_max"] = float(d_safe.max().detach().cpu())

            # Export for Histogram
            # 为了防止数据过大，可以随机采样或者全部返回
            metrics["raw_dist_3d"] = d_safe.detach().cpu().numpy()

            # Sample Print
            sample_idx = torch.nonzero(safe_indices, as_tuple=True)[0][0]
            metrics["dbg_pred"] = coords[sample_idx].detach().cpu().tolist()
            metrics["dbg_gt"] = target_world[sample_idx].detach().cpu().tolist()
        else:
            metrics["err_3d_mean"] = -1.0
            metrics["raw_dist_3d"] = np.array([])

        # 2. Reproj Error Stats
        if repro_err is not None:
            r_safe = repro_err[safe_indices]
            metrics["err_mean_px"] = float(r_safe.mean().detach().cpu())
            metrics["raw_reproj_err"] = r_safe.detach().cpu().numpy()
        else:
            metrics["err_mean_px"] = 0.0
            metrics["raw_reproj_err"] = np.array([])
    else:
        metrics["err_3d_mean"] = -1.0
        metrics["nuclear_cnt"] = float(nuclear_mask.sum().cpu())

    return total, metrics
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

    upsampler = None

    # 计算迭代
    total_buffer_samples = int(getattr(cfg.training, "buffer_size", getattr(cfg.training, "buffer_capacity", 0)))
    steps_per_epoch = total_buffer_samples // cfg.training.batch_size
    total_iterations = cfg.training.epochs * steps_per_epoch

    # Warmup 设置
    warmup_steps = getattr(cfg.training, "warmup_steps", 2000)
    print(f"Total Iterations: {total_iterations}, Warmup: {warmup_steps}")

    repro_loss = ReproLoss(
        total_iterations=total_iterations,
        soft_clamp=cfg.loss.repro_loss_soft_clamp,
        soft_clamp_min=cfg.loss.repro_loss_soft_clamp_min,
        type=cfg.loss.repro_loss_type,
        circle_schedule=(cfg.loss.repro_loss_schedule == 'circle')
    )

    # 3. Memory & Dataset
    memory_feats, memory_token = load_memory_features(cfg.fusion.stored_feature_file, device)
    del memory_feats

    if isinstance(cfg.dataset.train_dataset, str):
        dataset = eval(cfg.dataset.train_dataset)
    dataloader = ForcedRandomDataLoader(dataset=dataset, batch_size=1)

    if hasattr(dataset, "mean_cam_center"):
        scene_mean = dataset.mean_cam_center.to(device)
    else:
        scene_mean = torch.zeros(3, device=device)

    # 4. Buffer
    empty_memory = [None] * 24
    # 请确保引入了正确版本的 _collect_buffer, _prepare_targets 等
    from mapanything.tasks.train_downsample import _collect_buffer
    buffer, in_channels = _collect_buffer(cfg, model, upsampler, dataloader, empty_memory, memory_token, device)

    bufferloader = DataLoader(
        BufferDataset(buffer),
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle,
        num_workers=cfg.training.num_workers,
        drop_last=True
    )

    # 5. Head
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
        steps_per_epoch=len(bufferloader), cycle_momentum=False
    )
    scaler = GradScaler(enabled=use_half)

    output_dir = Path(cfg.training.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建图片保存目录
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    debug_log_path = output_dir / "training_debug1.txt"
    with open(debug_log_path, "w") as f:
        f.write("Step\tLoss\tPxErr\t3D_Mean\t3D_Med\t3D_Min\tMode\tSample_Pred\tSample_GT\n")

    # 6. Training Loop
    global_step = 0
    training_start = time.time()

    for epoch in range(cfg.training.epochs):
        head.train()
        for batch in bufferloader:
            features = batch["features"].to(device)
            scale = batch["scale_token"].to(device)

            # 策略切换
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

            if global_step % cfg.training.log_interval == 0:
                time_since_start = time.time() - training_start
                current_mode = metrics.get('mode', 'unknown')

                # --- 控制台打印 ---
                log_msg = (f"Iter: {global_step:6d} | Loss: {metrics['loss']:.4f} ({current_mode}) | "
                           f"PxErr: {metrics.get('err_mean_px', 0):.1f} | "
                           f"3DMed: {metrics.get('err_3d_median', -1):.3f}m")
                print(log_msg)

                # --- 文本日志 ---
                with open(debug_log_path, "a") as f:
                    pred_l = metrics.get('dbg_pred', [])
                    gt_l = metrics.get('dbg_gt', [])
                    pred_str = "[" + ",".join([f"{x:.2f}" for x in (pred_l if pred_l else [])]) + "]"
                    gt_str = "[" + ",".join([f"{x:.2f}" for x in (gt_l if gt_l else [])]) + "]"

                    f.write(f"Step {global_step}:\n")
                    f.write(f"  Mode: {current_mode}\n")
                    f.write(f"  Loss: {metrics['loss']:.6f}\n")
                    # [新增] 记录重投影误差 (仅在 Reproj 阶段有意义，但XYZ阶段为0也无妨)
                    if 'err_mean_px' in metrics and metrics['err_mean_px'] > 0:
                        f.write(f"  Px Error (px): {metrics['err_mean_px']:.4f}\n")
                    # [新增] 记录尺度正则项
                    if 'scale_reg' in metrics and metrics['scale_reg'] > 0:
                        f.write(f"  Scale Reg    : {metrics['scale_reg']:.6f}\n")

                    f.write(
                        f"  3D Error (m): Mean={metrics.get('err_3d_mean', -1):.3f}, Median={metrics.get('err_3d_median', -1):.3f}, Min={metrics.get('err_3d_min', -1):.3f}\n")
                    f.write(f"  Sample Val  : Pred={pred_str}  vs  GT={gt_str}\n")
                    f.write(f"  Nuclear Cnt : {metrics.get('nuclear_cnt', 0)}\n")
                    f.write("-" * 40 + "\n")

                # --- 绘图逻辑 (每 20 个 log interval 绘制一次) ---
                if global_step % (cfg.training.log_interval * 20) == 0:
                    dist_3d_np = metrics.get("raw_dist_3d", None)
                    reproj_err_np = metrics.get("raw_reproj_err", None)
                    save_batch_error_histograms(global_step, dist_3d_np, reproj_err_np, plot_dir)

            # 7. 保存
    try:
        task_name = HydraConfig.get().runtime.choices.get("model/task", "default")
    except Exception:
        task_name = "unknown"

    ckpt_name = (
        f"ace-downsample-debug1_task-{task_name}_head-{cfg.model.head_mode}_loss-{cfg.loss.mode}_"
        f"scale-{'on' if cfg.loss.scale_reg.enabled else 'off'}_"
        f"ep{cfg.training.epochs}_buf{cfg.training.buffer_size}.pt"
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