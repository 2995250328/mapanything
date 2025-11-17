"""ACE 风格的回归头训练入口。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import math
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mapanything.datasets import SevenScenesWAI
from mapanything.datasets.base.base_dataset import ForcedRandomDataLoader
from mapanything.models import init_model
from mapanything.tasks.ace import ACEHead_Pointwise_Decoupled, load_memory_features, ReproLoss
from mapanything.utils.geometry import quaternion_to_rotation_matrix

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class BufferTensors:
    features: torch.Tensor
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
            fused_feature: torch.Tensor,  # [1,C,H,W] 或 [C,H,W]；已上采样到原图分辨率
            target_world: torch.Tensor,  # [H,W,3] / [3,H,W] / [1,3,H,W] 世界坐标
            valid_mask: torch.Tensor,  # [H,W] / [1,1,H,W] / [1,H,W] 有效像素
            intrinsics: torch.Tensor,  # [3,3]
            c2w: torch.Tensor,  # [4,4] 或 [3,4]
            max_per_view: int,
    ) -> int:
        """从特征图随机采样 max_per_view 个有效像素，写入缓冲区（不超过剩余容量）。"""
        if self.is_full:
            return 0

        # -------- 形状归一 --------
        # features -> [C,H,W]
        if fused_feature.dim() == 4:
            b, c, h, w = fused_feature.shape
            assert b == 1, f"add_view expects single-view tensor, got batch={b}"
            feat_CHW = fused_feature[0]
        elif fused_feature.dim() == 3:
            c, h, w = fused_feature.shape
            feat_CHW = fused_feature
        else:
            raise ValueError(f"Unexpected fused_feature shape: {tuple(fused_feature.shape)}")

        # target_world -> [H,W,3]
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

        # valid_mask -> [H,W] (bool)
        if valid_mask is None:
            vm_HW = torch.ones((h, w), dtype=torch.bool, device=feat_CHW.device)
        else:
            if valid_mask.dim() == 4:  # [1,1,H,W]
                vm_HW = valid_mask[0, 0].to(torch.bool)
            elif valid_mask.dim() == 3:  # [1,H,W] 或 [C,H,W](C==1)
                vm_HW = valid_mask[0].to(torch.bool)
            elif valid_mask.dim() == 2:  # [H,W]
                vm_HW = valid_mask.to(torch.bool)
            else:
                raise ValueError(f"Unexpected valid_mask shape: {tuple(valid_mask.shape)}")

        # 尺寸一致性检查
        assert tw_HWC.shape[0] == h and tw_HWC.shape[1] == w, \
            f"Size mismatch: features ({h},{w}) vs target_world {tuple(tw_HWC.shape[:2])}"
        assert vm_HW.shape[0] == h and vm_HW.shape[1] == w, \
            f"Size mismatch: features ({h},{w}) vs valid_mask {tuple(vm_HW.shape)}"

        # -------- 展平并构建像素坐标 --------
        # features_flat: [N,C]，targets_flat: [N,3]，pixels_flat: [N,2] (u,v)
        feats_flat = feat_CHW.view(c, -1).t()  # [N,C]
        targets_flat = tw_HWC.view(-1, 3)  # [N,3]
        mask_flat = vm_HW.view(-1)  # [N]

        if mask_flat.any().item() is False:
            return 0

        # 像素网格 (u,v)
        ys, xs = torch.meshgrid(
            torch.arange(h, device=feat_CHW.device),
            torch.arange(w, device=feat_CHW.device),
            indexing="ij"
        )
        pixels_flat = torch.stack([xs, ys], dim=-1).view(-1, 2)  # [N,2]; 注意顺序(u,x列),(v,y行)

        # -------- 采样（仅有效区域，且不超过剩余容量）--------
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

        # 视角常量（对该视角的每个样本相同），依靠广播赋值
        self.storage.intrinsics[slot] = intrinsics.to(self.device)
        self.storage.c2w[slot] = c2w.to(self.device)

        self.storage.valid_mask[slot] = True

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
            "target_world": self.buffer.storage.target_world[idx],
            "pixels": self.buffer.storage.pixels[idx],
            "intrinsics": self.buffer.storage.intrinsics[idx],
            "c2w": self.buffer.storage.c2w[idx],
        }

# ---------------------------------------------------------------------------
# 基础几何工具
# ---------------------------------------------------------------------------
def fmap_to_image_pixel_centers(
    orig_hw,     # (H, W)
    feat_hw,     # (Hf, Wf)
    device=None,
    dtype=torch.float32,
):
    """
    返回 coords: [Hf, Wf, 2]，其中 coords[..., 0]=u (列/x), coords[..., 1]=v (行/y)
    采用像素中心系：u = (xf + 0.5) * Sx - 0.5, v = (yf + 0.5) * Sy - 0.5
    适用于常规 SAME padding 下采样（无特殊对齐/裁剪）。
    """
    H, W   = int(orig_hw[0]), int(orig_hw[1])
    Hf, Wf = int(feat_hw[0]), int(feat_hw[1])
    assert H > 0 and W > 0 and Hf > 0 and Wf > 0, "sizes must be positive"

    Sx = W / float(Wf)
    Sy = H / float(Hf)

    yf = torch.arange(Hf, dtype=dtype, device=device)  # 行
    xf = torch.arange(Wf, dtype=dtype, device=device)  # 列
    YF, XF = torch.meshgrid(yf, xf, indexing="ij")     # [Hf,Wf]

    U = (XF + 0.5) * Sx - 0.5   # 原图的列/x
    V = (YF + 0.5) * Sy - 0.5   # 原图的行/y
    coords = torch.stack([U, V], dim=-1)               # [Hf,Wf,2] -> (u,v)
    return coords

def _pixel_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(height, dtype=torch.float32, device=device) + 0.5
    xs = torch.arange(width, dtype=torch.float32, device=device) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=0)

def _project_points(
    points_world: torch.Tensor, intrinsics: torch.Tensor, c2w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """将世界坐标投影到像素平面，返回像素坐标与有效性掩码。"""

    if points_world.dim() == 2:
        points_world = points_world.unsqueeze(-1)

    R = c2w[:, :3, :3]
    t = c2w[:, :3, 3:4]
    cam = torch.bmm(R.transpose(1, 2), points_world - t)

    z = cam[:, 2:3]
    valid = z > 1e-6
    pixels = torch.bmm(intrinsics, cam)
    pixels = pixels[:, :2] / z.clamp(min=1e-6)
    return pixels.transpose(1, 2), valid.squeeze(1)

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

def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    return F.smooth_l1_loss(pred[mask], target[mask])

def _resolve_intrinsics(view: Dict[str, Any], device: torch.device) -> torch.Tensor:
    if "camera_intrinsics" in view:
        intr = torch.as_tensor(view["camera_intrinsics"], device=device, dtype=torch.float32)
    elif "intrinsics" in view:
        intr = torch.as_tensor(view["intrinsics"], device=device, dtype=torch.float32)
    else:
        raise KeyError("Camera intrinsics not available in view dictionary.")
    if intr.dim() == 2:
        intr = intr.unsqueeze(0)
    return intr

def _resolve_pose(view: Dict[str, Any], device: torch.device) -> torch.Tensor:
    if "camera_pose" in view:
        pose = torch.as_tensor(view["camera_pose"], device=device, dtype=torch.float32)
    elif "camera_pose_quats" in view and "camera_pose_trans" in view:
        quats = torch.as_tensor(view["camera_pose_quats"], device=device, dtype=torch.float32)
        trans = torch.as_tensor(view["camera_pose_trans"], device=device, dtype=torch.float32)
        rot = quaternion_to_rotation_matrix(quats).reshape(-1, 3, 3)
        pose = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
        pose[:, :3, :3] = rot
        pose[:, :3, 3] = trans.view(-1, 3)
    else:
        raise KeyError("Camera pose not available in view dictionary.")
    if pose.dim() == 2:
        pose = pose.unsqueeze(0)
    return pose

def _prepare_targets(view: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if "pts3d" not in view:
        raise KeyError("View is missing dense world coordinates (pts3d).")
    pts3d = torch.as_tensor(view["pts3d"], device=device, dtype=torch.float32)
    if pts3d.dim() == 3:
        pts3d = pts3d.permute(2, 0, 1).unsqueeze(0)
    valid_mask = torch.ones_like(pts3d[:, :1])
    if "valid_mask" in view:
        mask = torch.as_tensor(view["valid_mask"], device=device, dtype=torch.float32)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        valid_mask = mask
    return pts3d, valid_mask

# ---------------------------------------------------------------------------
# 缓冲区构建与损失
# ---------------------------------------------------------------------------
def _collect_buffer(
    cfg: DictConfig,
    model,
    upsampler,
    dataset,
    memory_feats,
    memory_token,
    device: torch.device,
) -> Tuple[FeatureReplayBuffer, int]:
    buffer: FeatureReplayBuffer | None = None

    for batch_id, views in tqdm(enumerate(dataset), desc="Creating Training Buffer"):
        batch = views
        # 移除不必要的键
        for view in batch:
            if "idx" in view:
                view["idx"] = view["idx"][2:]
        # 转到 GPU
        ignore_keys = {
            "dataset", "label", "instance", "idx",
            "true_shape", "rng", "data_norm_type",
        }
        for view in batch:
            for name in view.keys():
                if name in ignore_keys:
                    continue
                view[name] = view[name].to(device, non_blocking=True)

        with torch.no_grad():
            fused_feature, _ = model.forward_with_memory_dense_feature(
                query_view=batch,
                device=str(device),
                memory_feats=memory_feats,
                additional_tokens=memory_token,
                memory_keep_ratio=cfg.fusion.memory_keep_ratio,
                memory_efficient_inference=cfg.training.memory_efficient_inference,
            )
            fused_feature = upsampler(batch[0]["img"],fused_feature)

        target_world, valid_mask = _prepare_targets(view, device)
        intrinsics = _resolve_intrinsics(view, device).to(torch.float32)
        c2w = _resolve_pose(view, device).to(torch.float32)

        if buffer is None:
            buffer = FeatureReplayBuffer(cfg.training.buffer_size, fused_feature.shape[1], device=torch.device("cpu"))

        added = buffer.add_view(
            fused_feature.squeeze(0).detach(),
            target_world.squeeze(0).detach(),
            valid_mask.squeeze(0).detach(),
            intrinsics.squeeze(0).detach(),
            c2w.squeeze(0).detach(),
            cfg.training.samples_per_view,
        )

        if buffer.is_full():
            break

    if buffer is None or buffer.size == 0:
        raise RuntimeError("No training samples were collected. Check dataset or configuration.")
    return buffer, buffer.storage.features.shape[1]

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
    preds: torch.Tensor,           # [N,4]  => [X,Y,Z, raw]
    batch: DictConfig | dict,
    repro_loss,
    global_step,
    loss_cfg: DictConfig,
):
    """
    Heteroscedastic loss over sparse buffer samples (+ optional scale regularization).

    Required in batch:
      - intrinsics: [N,3,3]
      - c2w: [N,3,4] or [N,4,4]
      - pixels: [N,2]       (if mode='reproj')
      - target_world: [N,3] (if mode='xyz' or scale_reg.match_to_gt)

    loss_cfg (keys used here):
      - mode: 'reproj' | 'xyz'                      (default 'reproj')
      - conf_mode: 'log_sigma' | 'confidence'       (default 'log_sigma')
      - sigma_min: 1e-4, sigma_max: 10.0
      - depth_min: 1e-4, depth_max: 50.0
      - repro_loss_hard_clamp: 100.0
      - depth_target: 2.0
      - eps: 1e-8
      - scale_reg:
          enabled: bool
          variant: 'match_to_gt' | 'unit' | 'prior'
          weight: float
          depth_prior: float  # for 'prior'
    """
    device     = preds.device
    mode       = getattr(loss_cfg, "mode", "reproj")
    conf_mode  = getattr(loss_cfg, "conf_mode", "log_sigma")
    sigma_min  = float(getattr(loss_cfg, "sigma_min", 1e-4))
    sigma_max  = float(getattr(loss_cfg, "sigma_max", 10.0))
    depth_min  = float(getattr(loss_cfg, "depth_min", 1e-4))
    depth_max  = float(getattr(loss_cfg, "depth_max", 50.0))
    eps        = float(getattr(loss_cfg, "eps", 1e-8))
    sqrt2      = math.sqrt(2.0)

    coords = preds[:, :3]  # [N,3]
    raw    = preds[:, 3]   # [N]

    # -------- map raw -> sigma (>0) --------
    if conf_mode == "confidence":
        p = torch.sigmoid(raw)  # [N]
        sigma = (1.0 - p) * sigma_max + p * sigma_min
    else:  # 'log_sigma'
        sigma = F.softplus(raw) + eps
        sigma = torch.clamp(sigma, min=sigma_min, max=sigma_max)
    log_sigma = torch.log(sigma)  # [N]

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

    def _scale_reg_add(total_loss, metrics, z_pred_flat, *, variant: str, weight: float,
                       c2w=None, target_world=None, depth_prior=None):
        """
        轻微尺度正则；返回(updated_total, reg_value)
        - z_pred_flat: [N] 预测相机系深度（已 clamp）
        - variant:
            - 'match_to_gt': 需要 target_world, c2w
            - 'unit':        中位数 -> 1
            - 'prior':       中位数 -> depth_prior
        """
        if weight <= 0:
            return total_loss, 0.0

        # 预测深度中位数
        med_pred = torch.median(z_pred_flat)

        if variant == "match_to_gt":
            if target_world is None or c2w is None:
                return total_loss, 0.0
            # 计算 GT 深度中位数
            R, t = _invert_c2w_to_w2c(c2w.to(z_pred_flat.device))
            Xw_gt = target_world.to(z_pred_flat.device).unsqueeze(-1)  # [N,3,1]
            Zg = (R @ Xw_gt + t)[:, 2, 0].clamp_min(depth_min)
            med_gt = torch.median(Zg)
            reg = weight * torch.abs(torch.log((med_pred + eps) / (med_gt + eps)))

        elif variant == "unit":
            # 让 log(median) ~ 0  <=> median ~ 1
            reg = weight * torch.abs(torch.log(med_pred + eps))

        elif variant == "prior":
            if depth_prior is None:
                return total_loss, 0.0
            reg = weight * torch.abs(torch.log((med_pred + eps) / (float(depth_prior) + eps)))

        else:
            return total_loss, 0.0

        total_loss = total_loss + reg
        metrics["scale_reg"] = float(reg.detach().cpu())
        return total_loss, float(reg.detach().cpu())

    # =============== main branches ===============
    if mode == "xyz":
        target_world = batch["target_world"].to(device)  # [N,3]
        diff = coords - target_world                     # [N,3]
        err = torch.norm(diff, dim=1, p=2)               # [N]
        loss_vec = log_sigma + sqrt2 * (err / sigma)
        total = loss_vec.mean()

        # ---------- scale regularization (optional) ----------
        sr = getattr(loss_cfg, "scale_reg", None)
        scale_reg_val = 0.0
        if sr and getattr(sr, "enabled", False) and "c2w" in batch:
            # 计算预测相机深度（用于尺度统计）
            c2w = batch["c2w"].to(device)
            R, t = _invert_c2w_to_w2c(c2w)
            Xw = coords.unsqueeze(-1)                      # [N,3,1]
            Zp = (R @ Xw + t)[:, 2, 0].clamp_min(depth_min)  # [N]
            total, scale_reg_val = _scale_reg_add(
                total, {},
                Zp,
                variant=getattr(sr, "variant", "match_to_gt"),
                weight=float(getattr(sr, "weight", 1e-3)),
                c2w=c2w,
                target_world=target_world,
                depth_prior=getattr(sr, "depth_prior", None),
            )

        metrics = {
            "err_mean_m": float(err.mean().detach().cpu()),
            "sigma_mean": float(sigma.mean().detach().cpu()),
            "loss":       float(total.detach().cpu()),
            "mode":       "xyz-sparse",
        }
        if sr and getattr(sr, "enabled", False):
            metrics["scale_reg"] = scale_reg_val
        return total, metrics

    else:  # 'reproj'
        K   = batch["intrinsics"].to(device)  # [N,3,3]
        c2w = batch["c2w"].to(device)         # [N,3,4] or [N,4,4]
        px  = batch["pixels"].to(device)      # [N,2]

        # world -> cam
        R, t = _invert_c2w_to_w2c(c2w)        # [N,3,3], [N,3,1]
        Xw   = coords.unsqueeze(-1)           # [N,3,1]
        Xc   = R @ Xw + t                     # [N,3,1]

        z         = Xc[:, 2:3, :]                    # [N,1,1]
        z_clamped = z.clamp_min(depth_min)
        z_flat    = z_clamped[:, 0, 0]               # [N] for reg

        # 像素投影（ACE）
        uvh = K @ Xc                                  # [N,3,1]
        uv  = (uvh[:, :2, :] / z_clamped).squeeze(-1) # [N,2]

        # L1 重投影误差（ACE）
        repro_diff = (uv - px)                        # [N,2]
        repro_err  = repro_diff.abs().sum(dim=1)      # [N] (L1)

        # 有效/无效掩码（ACE）
        invalid_min_depth = (z.squeeze(-1).squeeze(-1) < depth_min)
        invalid_repro     = (repro_err > float(getattr(loss_cfg, "repro_loss_hard_clamp", 100.0)))
        invalid_max_depth = (z.squeeze(-1).squeeze(-1) > depth_max)
        invalid_mask = invalid_min_depth | invalid_repro | invalid_max_depth
        valid_mask   = ~invalid_mask

        # 有效：ACE ReproLoss
        if valid_mask.any():
            loss_valid = repro_loss.compute(repro_err[valid_mask], global_step)
        else:
            loss_valid = coords.sum() * 0.0

        # 无效：proxy 到相机坐标（ACE）
        if invalid_mask.any():
            if "intrinsics_inv" in batch:
                invK = batch["intrinsics_inv"].to(device)
            else:
                invK = torch.inverse(K)

            uv1    = torch.cat([px, torch.ones_like(px[:, :1])], dim=1).unsqueeze(-1)  # [N,3,1]
            Xc_tgt = float(getattr(loss_cfg, "depth_target", 2.0)) * (invK @ uv1)      # [N,3,1]
            loss_invalid = (Xc_tgt - Xc).abs()[invalid_mask].sum()
        else:
            loss_invalid = coords.sum() * 0.0

        total = (loss_valid + loss_invalid) / coords.shape[0]

        # ---------- scale regularization (optional) ----------
        sr = getattr(loss_cfg, "scale_reg", None)
        scale_reg_val = 0.0
        if sr and getattr(sr, "enabled", False):
            variant = getattr(sr, "variant", "prior")
            weight  = float(getattr(sr, "weight", 1e-3))
            if variant == "match_to_gt" and ("target_world" in batch):
                # 有 GT 时也可做 match_to_gt
                target_world = batch["target_world"].to(device)
                total, scale_reg_val = _scale_reg_add(
                    total, {},
                    z_flat,
                    variant="match_to_gt",
                    weight=weight,
                    c2w=c2w,
                    target_world=target_world,
                )
            elif variant in ("prior", "unit"):
                # reproj 分支通常不是“无尺度”，更合理的是 'prior'
                depth_prior = getattr(sr, "depth_prior", 1.0)
                use_variant = "prior" if variant != "match_to_gt" else "prior"
                total, scale_reg_val = _scale_reg_add(
                    total, {},
                    z_flat,
                    variant=use_variant,
                    weight=weight,
                    depth_prior=depth_prior,
                )

        metrics = {
            "err_mean_px": float(repro_err.mean().detach().cpu()),
            "sigma_mean":  float(sigma.mean().detach().cpu()),
            "loss":        float(total.detach().cpu()),
            "mode":        "reproj-sparse",
        }
        if sr and getattr(sr, "enabled", False):
            metrics["scale_reg"] = scale_reg_val
        return total, metrics

# ---------------------------------------------------------------------------
# 训练入口
# ---------------------------------------------------------------------------
def run_training(cfg: DictConfig) -> Dict[str, str]:

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    # 加载mapanything模型
    model = init_model(cfg.model.model_str, cfg.model.model_config, torch_hub_force_reload=False)
    model.to(device).eval()
    if cfg.model.pretrained:
        ckpt = torch.load(cfg.model.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    # 使用anyup进行特征图上采样
    upsampler = torch.hub.load('wimmerth/anyup', 'anyup')  # 如需可：, trust_repo=True
    upsampler.eval()  # 推理模式
    # 总迭代次数
    iterations = cfg.training.epochs * (cfg.training.buffer_size // cfg.training.batch_size )
    # 构建重投影损失计算类
    repro_loss = ReproLoss(
        total_iterations=iterations,
        soft_clamp=cfg.loss.repro_loss_soft_clamp,
        soft_clamp_min=cfg.loss.repro_loss_soft_clamp_min,
        type=cfg.loss.repro_loss_type,
        circle_schedule=(cfg.loss.repro_loss_schedule == 'circle')
    )
    # 从保存的中间特征文件中读取
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
    # 创建 training buffer
    buffer, in_channels = _collect_buffer(cfg, model,upsampler ,dataloader, memory_feats, memory_token, device)
    bufferloader = DataLoader(
        BufferDataset(buffer),
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle,
        num_workers=cfg.training.num_workers,
    )

    head = ACEHead_Pointwise_Decoupled(in_channels=in_channels, hidden_dim=cfg.head.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    output_dir = Path(cfg.training.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(cfg.training.epochs):
        head.train()
        for batch in bufferloader:
            features = batch["features"].to(device)
            preds = head(features)
            preds[:3] = preds[:3] *
            loss, metrics = _loss_fn(preds, batch,repro_loss,global_step, cfg.loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % cfg.training.log_interval == 0:
                print(json.dumps({"epoch": epoch, "step": global_step, "metrics": metrics}, ensure_ascii=False))

    ckpt_path = output_dir / "ace_regression_head.pt"
    torch.save({"state_dict": head.state_dict(), "in_channels": in_channels}, ckpt_path)
    return {"checkpoint": str(ckpt_path), "steps": str(global_step), "in_channels": str(in_channels)}

@hydra.main(version_base=None, config_path="../../../configs", config_name="train")
def main(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    info = run_training(cfg)
    print(json.dumps(info, indent=2, ensure_ascii=False))

if __name__ == "__main__":  # pragma: no cover
    main()
