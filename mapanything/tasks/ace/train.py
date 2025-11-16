"""ACE 风格的回归头训练入口。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from mapanything.models import init_model
from mapanything.tasks.aa_feature_fusion.common import instantiate_dataset
from mapanything.tasks.ace import ACERegressionHead, load_memory_features, move_view_to_device
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
    valid_mask: torch.Tensor


class FeatureReplayBuffer:
    """仿照 ACE，把随机采样的像素特征/位姿放入固定大小的缓冲区。"""

    def __init__(self, capacity: int, feat_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.size = 0
        self.storage = BufferTensors(
            features=torch.empty((capacity, feat_dim), device=device, dtype=torch.float32),
            target_world=torch.empty((capacity, 3), device=device, dtype=torch.float32),
            pixels=torch.empty((capacity, 2), device=device, dtype=torch.float32),
            intrinsics=torch.empty((capacity, 3, 3), device=device, dtype=torch.float32),
            c2w=torch.empty((capacity, 4, 4), device=device, dtype=torch.float32),
            valid_mask=torch.zeros((capacity,), device=device, dtype=torch.bool),
        )

    @property
    def is_full(self) -> bool:
        return self.size >= self.capacity

    def _remaining(self) -> int:
        return max(0, self.capacity - self.size)

    def add_view(
        self,
        fused_feature: torch.Tensor,
        target_world: torch.Tensor,
        valid_mask: torch.Tensor,
        intrinsics: torch.Tensor,
        c2w: torch.Tensor,
        max_per_view: int,
    ) -> int:
        """向缓冲区写入当前视角的随机像素样本。"""

        if self.is_full:
            return 0

        c, h, w = fused_feature.shape
        feats_flat = fused_feature.reshape(c, -1).transpose(0, 1)
        targets_flat = target_world.reshape(3, -1).transpose(0, 1)
        mask_flat = valid_mask.reshape(-1) > 0.5
        if mask_flat.sum() == 0:
            return 0

        pixel_grid = _pixel_grid(h, w, fused_feature.device)
        pixels_flat = pixel_grid.reshape(2, -1).transpose(0, 1)

        valid_idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
        perm = torch.randperm(valid_idx.numel(), device=fused_feature.device)
        chosen = valid_idx[perm[:max_per_view]]
        if chosen.numel() == 0:
            return 0

        chosen = chosen[: self._remaining()]
        slot = slice(self.size, self.size + chosen.numel())

        self.storage.features[slot] = feats_flat[chosen].to(self.device)
        self.storage.target_world[slot] = targets_flat[chosen].to(self.device)
        self.storage.pixels[slot] = pixels_flat[chosen].to(self.device)
        self.storage.intrinsics[slot] = intrinsics.to(self.device)
        self.storage.c2w[slot] = c2w.to(self.device)
        self.storage.valid_mask[slot] = True

        self.size += chosen.numel()
        return chosen.numel()


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
            "valid_mask": self.buffer.storage.valid_mask[idx],
        }


# ---------------------------------------------------------------------------
# 基础几何工具
# ---------------------------------------------------------------------------


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


def _prepare_targets(view: Dict[str, Any], feature_hw: Tuple[int, int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
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

    h, w = feature_hw
    pts3d = F.interpolate(pts3d, size=(h, w), mode="bilinear", align_corners=False)
    valid_mask = F.interpolate(valid_mask, size=(h, w), mode="nearest")
    return pts3d, valid_mask


# ---------------------------------------------------------------------------
# 缓冲区构建与损失
# ---------------------------------------------------------------------------


def _collect_buffer(
    cfg: DictConfig,
    model,
    dataset,
    memory_feats,
    memory_token,
    device: torch.device,
) -> Tuple[FeatureReplayBuffer, int]:
    samples_seen = 0
    buffer: FeatureReplayBuffer | None = None

    for idx in range(len(dataset)):
        if buffer is not None and buffer.is_full:
            break

        views = dataset[idx]
        if not views:
            continue
        view = views[0]
        prepared = move_view_to_device(view, device)

        with torch.no_grad():
            fused_feature, _ = model.forward_with_memory_dense_feature(
                query_view=[prepared],
                device=str(device),
                memory_feats=memory_feats,
                additional_tokens=memory_token,
                memory_keep_ratio=cfg.fusion.memory_keep_ratio,
                memory_efficient_inference=cfg.training.memory_efficient_inference,
            )

        feature_hw = fused_feature.shape[-2:]
        target_world, valid_mask = _prepare_targets(view, feature_hw, device)
        intrinsics = _resolve_intrinsics(view, device).to(torch.float32)
        c2w = _resolve_pose(view, device).to(torch.float32)

        if buffer is None:
            buffer = FeatureReplayBuffer(cfg.training.buffer_capacity, fused_feature.shape[1], device=torch.device("cpu"))

        added = buffer.add_view(
            fused_feature.squeeze(0).detach(),
            target_world.squeeze(0).detach(),
            valid_mask.squeeze(0).detach(),
            intrinsics.squeeze(0).detach(),
            c2w.squeeze(0).detach(),
            cfg.training.samples_per_view,
        )

        samples_seen += 1
        if samples_seen >= cfg.training.max_buffer_views:
            break

    if buffer is None or buffer.size == 0:
        raise RuntimeError("No training samples were collected. Check dataset or configuration.")
    return buffer, buffer.storage.features.shape[1]


def _loss_fn(preds: torch.Tensor, batch: Dict[str, torch.Tensor], loss_cfg: DictConfig):
    # 稠密图像推理（eval）使用 Bx4xHxW 形状
    if preds.dim() == 4:
        coords = preds[:, :3]
        conf_logits = preds[:, 3:4]
        confidence = torch.sigmoid(conf_logits)

        target_world = batch["target_world"].to(coords.device)
        intrinsics = batch["intrinsics"].to(coords.device)
        c2w = batch["c2w"].to(coords.device)
        valid_mask = batch["valid_mask"].to(coords.device)

        pred_pixels, pred_valid = _project_world_points(coords, intrinsics, c2w)
        target_pixels, target_valid = _project_world_points(target_world, intrinsics, c2w)
        combined_mask = valid_mask * pred_valid * target_valid

        reprojection = _masked_smooth_l1(pred_pixels, target_pixels, combined_mask)
        xyz_loss = _masked_smooth_l1(coords, target_world, valid_mask)
        conf_target = valid_mask.clamp(min=0.0, max=1.0)
        conf_loss = F.binary_cross_entropy(confidence, conf_target, reduction="mean")
    else:
        coords = preds[:, :3]
        conf_logits = preds[:, 3]
        confidence = torch.sigmoid(conf_logits)

        target_world = batch["target_world"].to(coords.device)
        intrinsics = batch["intrinsics"].to(coords.device)
        c2w = batch["c2w"].to(coords.device)
        pixels = batch["pixels"].to(coords.device)
        valid_mask = batch["valid_mask"].to(coords.device)

        coords_exp = coords.unsqueeze(-1)
        proj_pixels, proj_valid = _project_points(coords_exp, intrinsics, c2w)
        pixel_mask = (valid_mask & proj_valid.squeeze(1)).unsqueeze(-1).expand_as(proj_pixels.squeeze(1))
        xyz_mask = valid_mask.unsqueeze(-1).expand_as(coords)

        reprojection = _masked_smooth_l1(proj_pixels.squeeze(1), pixels, pixel_mask)
        xyz_loss = _masked_smooth_l1(coords, target_world, xyz_mask)
        conf_loss = F.binary_cross_entropy(confidence, valid_mask.float(), reduction="mean")

    total = (
        loss_cfg.reprojection_weight * reprojection
        + loss_cfg.confidence_weight * conf_loss
        + loss_cfg.xyz_weight * xyz_loss
    )

    metrics = {
        "reprojection": float(reprojection.detach().cpu()),
        "xyz": float(xyz_loss.detach().cpu()),
        "confidence": float(conf_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, metrics


# ---------------------------------------------------------------------------
# 训练入口
# ---------------------------------------------------------------------------


def run_training(cfg: DictConfig) -> Dict[str, str]:
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model = init_model(cfg.model.model_str, cfg.model.model_config, torch_hub_force_reload=False)
    model.to(device).eval()

    if cfg.model.pretrained:
        ckpt = torch.load(cfg.model.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    memory_feats, memory_token = load_memory_features(cfg.fusion.stored_feature_file, device)
    dataset = instantiate_dataset(cfg.dataset.dataset_str)

    buffer, in_channels = _collect_buffer(cfg, model, dataset, memory_feats, memory_token, device)
    dataloader = DataLoader(
        BufferDataset(buffer),
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle,
        num_workers=cfg.training.num_workers,
    )

    head = ACERegressionHead(in_channels=in_channels, hidden_dim=cfg.head.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    output_dir = Path(cfg.training.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(cfg.training.max_epochs):
        head.train()
        for batch in dataloader:
            features = batch["features"].to(device)
            preds = head(features)
            loss, metrics = _loss_fn(preds, batch, cfg.loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % cfg.training.log_interval == 0:
                print(json.dumps({"epoch": epoch, "step": global_step, "metrics": metrics}, ensure_ascii=False))

    ckpt_path = output_dir / "ace_regression_head.pt"
    torch.save({"state_dict": head.state_dict(), "in_channels": in_channels}, ckpt_path)
    return {"checkpoint": str(ckpt_path), "steps": str(global_step), "in_channels": str(in_channels)}


@hydra.main(version_base=None, config_path="../../configs/ace", config_name="train")
def main(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    info = run_training(cfg)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
