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
from omegaconf import DictConfig, OmegaConf
from torch import nn, autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mapanything.datasets import SevenScenesWAI
from mapanything.datasets.base.base_dataset import ForcedRandomDataLoader
from mapanything.models import init_model
from mapanything.tasks.ace import ACEHead_Pointwise_Decoupled_WithScale, ACEHead_Pointwise_FiLM, load_memory_features, ReproLoss
from mapanything.utils.debugprinter import DebugPrinter
from mapanything.utils.geometry import quaternion_to_rotation_matrix


printer = DebugPrinter()
# -----------------------------
# 1) 轻量 k-means：把大量 token 聚成 K 个中心（作为 memory token）
# -----------------------------
@torch.no_grad()
def kmeans_merge(x: torch.Tensor, k: int, iters: int = 6, metric: str = "cosine"):
    """
    x: [N, C]  (已加过位置编码的 token 序列)
    返回:
      centers: [k, C]
      assign_weights: [k, 1] 每个簇的权重 (sqrt(count))，可用于后续加权
    """
    assert x.dim() == 2
    n, c = x.shape
    k = min(k, n)
    if k == n:
        return x.clone(), torch.ones(k, 1, device=x.device, dtype=x.dtype)

    # 归一化对齐 cosine 距离
    if metric == "cosine":
        x_norm = F.normalize(x, dim=-1)
    else:
        x_norm = x

    # 初始化：随机选 k 个点
    idx = torch.randperm(n, device=x.device)[:k]
    centers = x_norm[idx].clone()  # 用于分配
    true_centers = x[idx].clone()  # 用于输出（不丢失原值缩放）

    for _ in range(iters):
        # 计算距离并分配
        if metric == "cosine":
            # 最大相似度 -> 最近中心
            sim = torch.matmul(x_norm, centers.T)  # [N, k]
            labels = sim.argmax(dim=1)
        else:
            dist = torch.cdist(x_norm, centers, p=2)  # [N, k]
            labels = dist.argmin(dim=1)

        # 聚合得到新中心（用原始 x 求均值，避免累积归一化误差）
        counts = torch.bincount(labels, minlength=k).clamp_(min=1).to(x.dtype)  # [k]
        sums = torch.zeros(k, c, device=x.device, dtype=x.dtype)
        sums.index_add_(0, labels, x)
        new_centers = sums / counts.unsqueeze(1)  # [k, C]

        # 收敛性检查（可选）
        shift = (new_centers - true_centers).pow(2).mean()
        true_centers = new_centers
        if shift < 1e-6:
            break

        # 用于下一轮分配的“归一化中心”
        centers = F.normalize(true_centers, dim=-1) if metric == "cosine" else true_centers

    assign_weights = torch.sqrt(counts).unsqueeze(1)  # [k, 1]，权重 = sqrt(簇大小)
    return true_centers, assign_weights

# -----------------------------
# 2) 按块聚合器：对 24 个块分别把 N 视图的 token 聚成 K_b 个记忆 token
# -----------------------------
class BlockwiseAggregator(nn.Module):
    def __init__(
        self,
        num_blocks: int = 24,
        tokens_per_block: Union[int, List[int]] = 256,
        kmeans_iters: int = 6,
        metric: str = "cosine",
        pre_cap: Optional[int] = 8192,
        weight_scale: bool = True,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        if isinstance(tokens_per_block, int):
            self.tokens_per_block = [tokens_per_block] * num_blocks
        else:
            assert len(tokens_per_block) == num_blocks
            self.tokens_per_block = tokens_per_block
        self.kmeans_iters = kmeans_iters
        self.metric = metric
        self.pre_cap = pre_cap
        self.weight_scale = weight_scale

    @staticmethod
    def _to_tokens(feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 2:
            return feat.contiguous()
        elif feat.dim() == 3:
            C, H, W = feat.shape
            return feat.permute(1, 2, 0).reshape(H * W, C).contiguous()
        elif feat.dim() == 4:
            B, C, H, W = feat.shape
            return feat.permute(0, 2, 3, 1).reshape(B * H * W, C).contiguous()
        else:
            raise ValueError(f"Unsupported feat dim={feat.dim()}, expected 2/3/4.")
    @torch.no_grad()
    def forward(self, memory_feats: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        assert len(memory_feats) == self.num_blocks, \
            f"期望 {self.num_blocks} 个块，收到 {len(memory_feats)}"

        memory_tokens_per_block: List[torch.Tensor] = []

        for b in range(self.num_blocks):
            views = memory_feats[b]
            assert len(views) > 0, f"block {b} 为空"
            first_valid = None
            for t in views:
                if t is not None:
                    first_valid = t
                    break
            if first_valid is None:
                raise ValueError(f"block {b} 全为 None")

            dev = first_valid.device
            dtype = first_valid.dtype

            tokens_b = []
            for v, feat in enumerate(views):
                if feat is None:
                    continue
                if feat.device != dev:
                    feat = feat.to(dev, non_blocking=True)
                tb = self._to_tokens(feat)  # [T_vb, C]
                assert tb.dim() == 2, f"block {b}, view {v} 转换失败，得到维度 {tb.dim()}"
                tokens_b.append(tb)

            if len(tokens_b) == 0:
                memory_tokens_per_block.append(
                    torch.empty(0, first_valid.shape[-3] if first_valid.dim() >= 3 else first_valid.shape[-1],
                                device=dev, dtype=dtype)
                )
                continue

            x = torch.cat(tokens_b, dim=0)  # [T_total_b, C]
            if self.pre_cap is not None and x.shape[0] > self.pre_cap:
                idx = torch.randperm(x.shape[0], device=dev)[: self.pre_cap]
                x = x.index_select(0, idx)

            Kb = int(self.tokens_per_block[b])
            centers, weights = kmeans_merge(x, k=Kb, iters=self.kmeans_iters, metric=self.metric)  # [Kb,C], [Kb,1]
            if self.weight_scale:
                centers = centers * weights  # 频次加权

            memory_tokens_per_block.append(centers.to(device=dev, dtype=dtype))

        return memory_tokens_per_block

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
    c2w: [N,3,4] or [N,4,4]
    return Rcw^T, -Rcw^T * tcw   (world->cam: Xc = Rwc * Xw + twc)
    """
    if c2w.shape[-2:] == (4, 4):
        w2c = torch.inverse(c2w)  # [N,4,4]
        R = w2c[:, :3, :3]
        t = w2c[:, :3, 3:4]
        return R, t
    elif c2w.shape[-2:] == (3, 4):
        Rcw = c2w[:, :3, :3]                  # cam->world
        tcw = c2w[:, :3, 3:4]
        R = Rcw.transpose(-1, -2)             # world->cam
        t = -R @ tcw
        return R, t
    else:
        raise ValueError(f"Unexpected c2w shape: {tuple(c2w.shape)}")

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

# 尺度正则默认关闭，需在 loss.scale_reg.enabled=true 才会生效。
# match_to_gt：需要 batch 中提供 target_world 与 c2w。
# unit：用于无尺度预测（解耦方案），把中位数拉向 1。
# prior：无 GT 时让深度中位数靠近 depth_prior。
# 正则权重建议从 1e-4 ~ 1e-3 起步，防止盖过主监督信号。
def _loss_fn(
        preds,  # Tensor [N,4] 或 (Tensor[N,4], scale[N]) 或 {"preds":..., "scale":...}
        batch: DictConfig | dict,
        repro_loss,
        global_step,
        loss_cfg: DictConfig,
):
    """
    兼容两种回归头：
      - FiLM: preds = Tensor[N,4]  (已是有尺度 XYZ + raw)
      - Decoupled+Scale: preds = (Tensor[N,4], scale[N]) 或 dict{"preds":Tensor[N,4], "scale":Tensor[N]}

    Required in batch:
      - intrinsics: [N,3,3]
      - c2w: [N,3,4] or [N,4,4]
      - pixels: [N,2]       (if mode='reproj')
      - target_world: [N,3] (if mode='xyz' 或 scale_reg.match_to_gt)

    loss_cfg keys:
      - mode: 'reproj' | 'xyz'                      (default 'reproj')
      - conf_mode: 'log_sigma' | 'confidence'       (default 'log_sigma')
      - sigma_min/sigma_max/depth_min/depth_max/eps
      - repro_loss_hard_clamp, depth_target
      - scale_reg: {enabled, variant: 'match_to_gt'|'unit'|'prior', weight, depth_prior}
    """
    # ---------- unpack two head cases ----------
    scale = None
    if isinstance(preds, (tuple, list)):
        preds, scale = preds
    elif isinstance(preds, dict):
        scale = preds.get("scale", None)
        preds = preds["preds"]

    device = preds.device
    mode = getattr(loss_cfg, "mode", "reproj")
    conf_mode = getattr(loss_cfg, "conf_mode", "log_sigma")
    sigma_min = float(getattr(loss_cfg, "sigma_min", 1e-4))
    sigma_max = float(getattr(loss_cfg, "sigma_max", 10.0))
    depth_min = float(getattr(loss_cfg, "depth_min", 0.1))  # 建议设为 0.1
    depth_max = float(getattr(loss_cfg, "depth_max", 50.0))
    eps = float(getattr(loss_cfg, "eps", 1e-8))
    sqrt2 = math.sqrt(2.0)

    # [新增] 熔断阈值
    SANITY_PIXEL_ERR = 50000.0  # 像素误差超过 5万
    SANITY_COORD_VAL = 10000.0  # 坐标值超过 1万米

    # flatten BCHW 情况（通常你的缓冲区已是 [N,4]；这里容错）
    if preds.dim() == 4:  # [B,4,H,W] -> [N,4]
        B, C, H, W = preds.shape
        preds = preds.permute(0, 2, 3, 1).reshape(-1, C)

    # [新增] 输入层面的 NaN 检查
    if torch.isnan(preds).any() or torch.isinf(preds).any():
        print(f"[CRITICAL] Iter {global_step}: Input preds contain NaN/Inf! returning zero loss.")
        # 返回带梯度的 0，避免训练崩溃
        return preds.sum() * 0.0, {"loss": 0.0, "mode": "nan_skipped"}

    coords_pred_in = preds[:, :3]  # 若带 scale，这是 XYZ_unit；否则是有尺度 XYZ
    raw = preds[:, 3]  # 异方差 raw

    # -------- map raw -> sigma (>0) --------
    if conf_mode == "confidence":
        p = torch.sigmoid(raw)
        sigma = (1.0 - p) * sigma_max + p * sigma_min
    else:  # 'log_sigma'
        sigma = F.softplus(raw) + eps
        sigma = torch.clamp(sigma, min=sigma_min, max=sigma_max)
    log_sigma = torch.log(sigma)

    # 如果提供了 scale，则这是 Decoupled 情况：先把无尺度坐标乘回尺度
    if scale is not None:
        scale = scale.to(device).view(-1)  # [N]
        coords_unit = coords_pred_in  # for scale regularization('unit')
        coords = coords_unit * scale.unsqueeze(-1)  # 有尺度坐标，进入主监督
    else:
        coords_unit = None
        coords = coords_pred_in  # FiLM 情况：已是有尺度

    # =============== helpers ===============
    def _invert_c2w_to_w2c(c2w: torch.Tensor):
        if c2w.shape[-2:] == (4, 4):
            w2c = torch.inverse(c2w)
            R, t = w2c[:, :3, :3], w2c[:, :3, 3:4]
        else:
            Rcw, tcw = c2w[:, :3, :3], c2w[:, :3, 3:4]
            R = Rcw.transpose(-1, -2)
            t = -R @ tcw
        return R, t

    def _scale_reg_add(total_loss, z_flat: torch.Tensor,
                       *, variant: str, weight: float,
                       c2w=None, target_world=None, depth_prior=None):
        """对 z 的中位数做轻微尺度正则；返回 total_loss+reg, reg_value"""
        if weight <= 0 or z_flat.numel() == 0:
            return total_loss, 0.0
        med_pred = torch.median(z_flat)

        if variant == "match_to_gt":
            if target_world is None or c2w is None:
                return total_loss, 0.0
            R, t = _invert_c2w_to_w2c(c2w.to(z_flat.device))
            Xw_gt = target_world.to(z_flat.device).unsqueeze(-1)  # [N,3,1]
            Zg = (R @ Xw_gt + t)[:, 2, 0].clamp_min(depth_min)
            med_gt = torch.median(Zg)
            reg = weight * torch.abs(torch.log((med_pred + eps) / (med_gt + eps)))

        elif variant == "unit":
            # 让 median(z_unit) ≈ 1
            reg = weight * torch.abs(torch.log(med_pred + eps))

        elif variant == "prior":
            if depth_prior is None:
                return total_loss, 0.0
            reg = weight * torch.abs(torch.log((med_pred + eps) / (float(depth_prior) + eps)))
        else:
            return total_loss, 0.0

        return total_loss + reg, float(reg.detach().cpu())

    # =========================================================================
    # Branch 1: XYZ Loss (Warmup)
    # =========================================================================
    if mode == "xyz":
        target_world = batch["target_world"].to(device)  # [N,3]

        # 1. 计算 Loss
        # 使用 Huber Loss 增强鲁棒性
        loss_huber = F.huber_loss(coords, target_world, delta=1.0, reduction='none').sum(dim=1)
        total = loss_huber.mean()*100

        # 2. 计算 Debug 统计量 (关键修复：在这里直接计算并返回)
        diff_3d = coords - target_world
        dist_3d = torch.norm(diff_3d, dim=1, p=2)  # [N]

        metrics = {
            "loss": float(total.detach().cpu()),
            "mode": "xyz-huber",
            # [修复] 补全 Debug Key
            "err_3d_mean": float(dist_3d.mean().detach().cpu()),
            "err_3d_median": float(dist_3d.median().detach().cpu()),
            "err_3d_min": float(dist_3d.min().detach().cpu()),
            "err_3d_max": float(dist_3d.max().detach().cpu()),
            "dbg_pred": coords[0].detach().cpu().tolist(),
            "dbg_gt": target_world[0].detach().cpu().tolist(),
            "nuclear_cnt": 0.0,  # XYZ 模式默认无熔断
            "err_mean_px": 0.0  # XYZ 模式无重投影误差
        }
        return total, metrics

    else:  # 'reproj'
        K = batch["intrinsics"].to(device)  # [N,3,3]
        c2w = batch["c2w"].to(device)  # [N,3,4] or [N,4,4]
        px = batch["pixels"].to(device)  # [N,2]
        # === [DEBUG START] 验证 GT 数据的自洽性 ===
        if "target_world" in batch:
            gt_xyz = batch["target_world"].to(device)  # [N, 3]
            # 使用你的辅助函数计算
            R_gt, t_gt = _invert_c2w_to_w2c(c2w)
            Xc_gt = R_gt @ gt_xyz.unsqueeze(-1) + t_gt
            uvh_gt = K @ Xc_gt
            z_gt = Xc_gt[:, 2:3, :]
            uv_gt_proj = (uvh_gt[:, :2, :] / (z_gt + 1e-6)).squeeze(-1)
            # 计算 GT 的重投影误差
            gt_reproj_err = (uv_gt_proj - px).norm(dim=1).mean().item()

            # 如果这个值很大 (比如 > 1.0)，说明 Pose/K/Pixel 不匹配！
            if gt_reproj_err > 5.0:
                print(f"[CRITICAL ALARM] GT Mapping Error: {gt_reproj_err:.4f} px")
                print("这意味着 buffer 中的 (Intrinsics + Pose + 3D点) 无法投影回 (Pixel)！")
                print("请检查：1. Intrinsics 是否对应原图尺寸？ 2. c2w 是否需要转置？ 3. Pixel offset 是否正确？")
        # === [DEBUG END] ===

        R, t = _invert_c2w_to_w2c(c2w)
        Xw = coords.unsqueeze(-1)  # [N,3,1]
        Xc = R @ Xw + t  # [N,3,1]

        z = Xc[:, 2:3, :]

        # [修改] z_clamped 仅用于计算 reprojection，原 z 用于判断有效性
        # 使用 epsilon 1e-3 防止除零
        z_safe = z.clamp(min=1e-3)
        z_flat = z[:, 0, 0]  # 原始深度

        uvh = K @ Xc
        uv = (uvh[:, :2, :] / z_safe).squeeze(-1)  # [N,2]

        repro_err = (uv - px).abs().sum(dim=1)  # L1, [N]

        # =========================================================
        # [新增] 熔断机制 (Circuit Breaker)
        # =========================================================
        nuclear_mask = (repro_err > SANITY_PIXEL_ERR) | \
                       (torch.abs(coords).max(dim=1)[0] > SANITY_COORD_VAL) | \
                       (torch.isnan(repro_err)) | \
                       (torch.isinf(repro_err))

        if nuclear_mask.any():
            num_nuclear = nuclear_mask.sum().item()
            # 打印前 3 个异常点用于 Debug
            bad_indices = torch.nonzero(nuclear_mask, as_tuple=True)[0][:3]
            print(f"\n[CRITICAL] Iter {global_step}: Found {num_nuclear} NUCLEAR samples! Excluding them.")
            for idx in bad_indices:
                print(f"  -> Sample {idx.item()}: "
                      f"Z_cam={z_flat[idx].item():.4f}, "
                      f"PxErr={repro_err[idx].item():.1f}, "
                      f"Pred={coords[idx].tolist()}")

        # =========================================================
        # 掩码逻辑 (Mask Logic)
        # =========================================================
        # 1. 基础无效条件 (ACE 原始逻辑)
        invalid_min_depth = (z_flat < depth_min)
        invalid_max_depth = (z_flat > depth_max)
        invalid_repro = (repro_err > float(getattr(loss_cfg, "repro_loss_hard_clamp", 100.0)))

        base_invalid_mask = invalid_min_depth | invalid_repro | invalid_max_depth

        # 2. 最终掩码 (剔除 nuclear 点)
        # Valid:  原来 Valid  且  非 Nuclear
        valid_mask = (~base_invalid_mask) & (~nuclear_mask)
        # Invalid: 原来 Invalid 且  非 Nuclear (保留这部分做 Proxy Loss)
        invalid_mask = base_invalid_mask & (~nuclear_mask)

        # =========================================================
        # Loss 计算
        # =========================================================
        loss_valid = torch.tensor(0.0, device=device)
        loss_invalid = torch.tensor(0.0, device=device)

        # Part 1: Valid Loss (使用 ReproLoss Tanh 抑制大梯度)
        if valid_mask.any():
            loss_valid = repro_loss.compute(repro_err[valid_mask], global_step)

        # Part 2: Invalid Loss (你要求的 Proxy Logic)
        if invalid_mask.any():
            if "intrinsics_inv" in batch:
                invK = batch["intrinsics_inv"].to(device)
            else:
                invK = torch.inverse(K)

            # 构造 Proxy Target: 沿着光线方向，强制拉到 depth_target
            uv1 = torch.cat([px, torch.ones_like(px[:, :1])], dim=1).unsqueeze(-1)
            Xc_tgt = float(getattr(loss_cfg, "depth_target", 10.0)) * (invK @ uv1)

            # 使用 Huber Loss 计算 3D 距离，比直接求和更稳健
            # 计算当前预测 Xc 与 目标 Xc_tgt 的距离
            dist_error = (Xc_tgt - Xc).abs().sum(dim=1).squeeze()  # L1 Dist [N]
            loss_invalid = dist_error[invalid_mask].sum()

        # 总 Loss：归一化 (分母为 Batch Size，或者非 Nuclear 的数量)
        # 这里使用 batch size (coords.shape[0]) 以保持梯度幅度的一致性，
        # 意味着如果大量点被熔断，总 Loss 会变小，这是合理的（不要更新错误的梯度）。
        total = (loss_valid + loss_invalid) / coords.shape[0]

        # ---------- scale regularization ----------
        sr = getattr(loss_cfg, "scale_reg", None)
        scale_reg_val = 0.0
        if sr and getattr(sr, "enabled", False):
            # [修改] 仅对 非 Nuclear 的点进行正则化计算
            safe_indices = ~nuclear_mask
            if safe_indices.any():
                z_flat_safe = z_flat[safe_indices]
                coords_unit_safe = coords_unit[safe_indices] if coords_unit is not None else None
                # 注意：target_world 和 c2w 也需要切片，但 scale_reg_add 内部可能有逻辑
                # 为了简单起见，这里传入全量，但在函数内只用 z_flat_safe 的 median
                # 更好的做法是：

                variant = getattr(sr, "variant", "prior")
                weight = float(getattr(sr, "weight", 1e-3))

                if variant == "match_to_gt" and ("target_world" in batch):
                    # 需要切片对应的 c2w 和 target_world
                    c2w_safe = c2w[safe_indices]
                    target_world_safe = batch["target_world"].to(device)[safe_indices]
                    total, scale_reg_val = _scale_reg_add(
                        total, z_flat_safe, variant="match_to_gt", weight=weight,
                        c2w=c2w_safe, target_world=target_world_safe
                    )
                elif variant == "unit" and (coords_unit_safe is not None):
                    # 解耦：对无尺度深度做 unit 正则更合理
                    # 重新计算 Zu (无尺度深度)
                    R_safe, t_safe = _invert_c2w_to_w2c(c2w[safe_indices])
                    Xw_u_safe = coords_unit_safe.unsqueeze(-1)
                    Zu_safe = (R_safe @ Xw_u_safe + t_safe)[:, 2, 0].clamp_min(depth_min)

                    total, scale_reg_val = _scale_reg_add(
                        total, Zu_safe, variant="unit", weight=weight
                    )
                else:
                    depth_prior = getattr(sr, "depth_prior", 1.0)
                    total, scale_reg_val = _scale_reg_add(
                        total, z_flat_safe, variant="prior", weight=weight, depth_prior=depth_prior
                    )
        metrics = {
            "err_mean_px": float(repro_err[~nuclear_mask].mean().detach().cpu()) if (~nuclear_mask).any() else 0.0,
            "loss_val": float(loss_valid.detach().cpu() / coords.shape[0]),
            "loss_inv": float(loss_invalid.detach().cpu() / coords.shape[0]),
            "nuclear_cnt": float(nuclear_mask.sum().cpu()),  # 监控熔断数量
            "loss": float(total.detach().cpu()),
            "mode": "reproj-safe",
        }
        if sr and getattr(sr, "enabled", False):
            metrics["scale_reg"] = scale_reg_val

        if "target_world" in batch:
            target_world_dbg = batch["target_world"].to(device)

            # 计算欧氏距离
            diff_3d = coords - target_world_dbg
            dist_3d = torch.norm(diff_3d, dim=1)  # [N]
            # 仅统计非熔断（Safe）的点
            safe_indices = ~nuclear_mask
            if safe_indices.any():
                d_safe = dist_3d[safe_indices]
                metrics["err_3d_mean"] = float(d_safe.mean().detach().cpu())
                metrics["err_3d_median"] = float(d_safe.median().detach().cpu())
                metrics["err_3d_min"] = float(d_safe.min().detach().cpu())
                metrics["err_3d_max"] = float(d_safe.max().detach().cpu())

                # 随机抽取一个点查看具体数值 (Pred vs GT)
                # 这能让你一眼看出是否存在 Grid Sample 错位或 Mean 丢失
                sample_idx = torch.nonzero(safe_indices, as_tuple=True)[0][0]
                metrics["dbg_pred"] = coords[sample_idx].detach().cpu().tolist()
                metrics["dbg_gt"] = target_world_dbg[sample_idx].detach().cpu().tolist()

                # 监控深度范围
                metrics["depth_mean"] = float(coords[sample_idx].norm().detach().cpu())  # 粗略估计
            else:
                # 如果全都被熔断了
                metrics["err_3d_mean"] = -1.0

        return total, metrics
# ---------------------------------------------------------------------------
# 训练入口
# ---------------------------------------------------------------------------
def run_training(cfg: DictConfig) -> Dict[str, str]:
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    # 检查是否启用混合精度 (建议在 config 中添加 use_half: True)
    use_half = getattr(cfg.training, "use_half", False)

    # 1. 加载模型
    model = init_model(cfg.model.model_str, cfg.model.model_config, torch_hub_force_reload=False)
    model.to(device).eval()
    if cfg.model.pretrained:
        print(f"Loading pretrained backbone: {cfg.model.pretrained}")
        ckpt = torch.load(cfg.model.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    # 2. Upsampler 设为 None
    upsampler = None

    # 计算总迭代次数用于 Loss 调度
    # 注意：Buffer 是固定大小，bufferloader 会遍历整个 buffer
    # ACE logic: iterations = epochs * (buffer_size // batch_size)
    total_buffer_samples = int(getattr(cfg.training, "buffer_size", getattr(cfg.training, "buffer_capacity", 0)))
    steps_per_epoch = total_buffer_samples // cfg.training.batch_size
    total_iterations = cfg.training.epochs * steps_per_epoch

    repro_loss = ReproLoss(
        total_iterations=total_iterations,
        soft_clamp=cfg.loss.repro_loss_soft_clamp,
        soft_clamp_min=cfg.loss.repro_loss_soft_clamp_min,
        type=cfg.loss.repro_loss_type,
        circle_schedule=(cfg.loss.repro_loss_schedule == 'circle')
    )

    # 3. 加载 Memory (即使不用也保持逻辑)
    memory_feats, memory_token = load_memory_features(cfg.fusion.stored_feature_file, device)

    # 4. 创建 Dataset
    if isinstance(cfg.dataset.train_dataset, str):
        dataset = eval(cfg.dataset.train_dataset)
    dataloader = ForcedRandomDataLoader(dataset=dataset, batch_size=1)

    if hasattr(dataset, "mean_cam_center"):
        scene_mean = dataset.mean_cam_center.to(device)
        print(f"Initializing Regression Head with Scene Mean: {scene_mean.tolist()}")
    else:
        print("Warning: Dataset does not have mean_cam_center. Using zero mean.")
        scene_mean = torch.zeros(3, device=device)

    # 5. 构建 Buffer
    buffer, in_channels = _collect_buffer(cfg, model, upsampler, dataloader, memory_feats, memory_token, device)

    bufferloader = DataLoader(
        BufferDataset(buffer),
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle,
        num_workers=cfg.training.num_workers,
        drop_last=True  # 建议丢弃最后不足一个 batch 的数据以保持 steps_per_epoch 稳定
    )

    # 6. 初始化 Head
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

    # ACE 使用 learning_rate_min 作为初始 LR (但在 OneCycleLR 中这通常被忽略，因为由 scheduler 接管)
    # 我们这里直接用 AdamW，lr 参数会被 Scheduler 覆盖
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    # max_lr: 对应 ACE 中的 learning_rate_max，这里我们假设 cfg.training.lr 就是 max_lr
    # pct_start: ACE 默认没有显式设置，OneCycleLR 默认为 0.3。ACE 实际上是自己算的 linear schedule，
    # 但 PyTorch 的 OneCycleLR 效果通常更好或相当。可以保持默认。
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.training.lr,
        epochs=cfg.training.epochs,
        steps_per_epoch=len(bufferloader),  # 确保准确
        cycle_momentum=False  # ACE 设置为 False
    )
    scaler = GradScaler(enabled=use_half)

    output_dir = Path(cfg.training.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 7. 训练循环
    global_step = 0
    training_start = time.time()

    print(f"Start training: Epochs={cfg.training.epochs}, Batch={cfg.training.batch_size}, "
          f"Use Half={use_half}, Steps/Epoch={len(bufferloader)}")
    debug_log_path = output_dir / "training_debug.txt"
    # 清空旧日志
    if debug_log_path.exists():
        with open(debug_log_path, "w") as f:
            f.write("Step\tLoss\tPxErr\t3D_Mean\t3D_Med\tSample_Pred\tSample_GT\n")
    warmup_steps = total_iterations * 0.05  # 或者 total_steps * 0.05
    for epoch in range(cfg.training.epochs):
        head.train()
        # 记录每个 epoch 的开始时间，或者不需要
        # epoch_start_time = time.time()
        for batch in bufferloader:
            # 数据搬运
            features = batch["features"].to(device)
            scale = batch["scale_token"].to(device)

            with torch.autocast(device_type=cfg.training.device, dtype=torch.float16, enabled=use_half):
                preds = head(features, scale)
                if global_step < warmup_steps:
                    cfg.loss.mode = "xyz"  # 强制直接回归坐标
                    cfg.loss.scale_reg.enabled = False  # Warmup 阶段通常不需要 scale reg
                else:
                    cfg.loss.mode = "reproj"  # 之后切换回重投影
                    cfg.loss.scale_reg.enabled = True
                loss, metrics = _loss_fn(preds, batch, repro_loss, global_step, cfg.loss)

            optimizer.zero_grad(set_to_none=True)
            # Scaled Backward
            scaler.scale(loss).backward()
            # Unscale 之后才能进行梯度裁剪!
            scaler.unscale_(optimizer)
            # Scaler Step & Update
            scaler.step(optimizer)
            scaler.update()
            # Scheduler Step (每个 iteration 都更新)
            # 注意：如果 scaler 跳过了这一步（因为 inf/nan），scheduler 通常也应该跳过，
            # 但 standard implementation 通常直接 step，ACE 代码中有个检查 old_optimizer_step 的逻辑。
            # 这里我们简化处理，直接 step，因为 OneCycleLR 对跳过几步不敏感。
            scheduler.step()
            global_step += 1

            if global_step % cfg.training.log_interval == 0:
                time_since_start = time.time() - training_start
                current_lr = scheduler.get_last_lr()[0]
                # --- 控制台打印 (精简) ---
                log_msg = (f"Iter: {global_step:6d} | Loss: {metrics['loss']:.4f} | "
                           f"PxErr: {metrics.get('err_mean_px', 0):.1f} | "
                           f"3DMed: {metrics.get('err_3d_median', -1):.3f}m")  # 实时看3D中位数误差
                print(log_msg)

                # --- 文件日志 (详细) ---
                # 将详细数据追加到文件，方便事后分析
                with open(debug_log_path, "a") as f:
                    pred_str = "[" + ",".join([f"{x:.2f}" for x in metrics.get('dbg_pred', [])]) + "]"
                    gt_str = "[" + ",".join([f"{x:.2f}" for x in metrics.get('dbg_gt', [])]) + "]"

                    f.write(f"Step {global_step}:\n")
                    f.write(f"  Loss: {metrics['loss']:.6f}\n")
                    f.write(
                        f"  3D Error (m): Mean={metrics.get('err_3d_mean', -1):.3f}, Median={metrics.get('err_3d_median', -1):.3f}, Min={metrics.get('err_3d_min', -1):.3f}\n")
                    f.write(f"  Sample Val  : Pred={pred_str}  vs  GT={gt_str}\n")
                    f.write(f"  Nuclear Cnt : {metrics.get('nuclear_cnt', 0)}\n")
                    f.write("-" * 40 + "\n")

    # 8. 保存
    try:
        task_name = HydraConfig.get().runtime.choices.get("model/task", "default")
    except Exception:
        task_name = "unknown"

    ckpt_name = (
        f"ace-downsample-debug_task-{task_name}_head-{cfg.model.head_mode}_loss-{cfg.loss.mode}_"
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