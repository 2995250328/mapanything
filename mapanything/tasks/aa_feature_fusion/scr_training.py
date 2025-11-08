"""Train a scene coordinate regression head using ACE buffers and AA fusion."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from mapanything.tasks.aa_feature_fusion.builder import build_pipeline_from_cfg
from mapanything.tasks.aa_feature_fusion.common import (
    extract_pointmap_like_outputs,
    instantiate_dataset,
    prepare_view,
)
from mapanything.tasks.aa_feature_fusion.pose_stats import (
    load_or_compute_mean_pose,
    world_from_mean_pose,
)
from mapanything.utils.geometry import quaternion_to_rotation_matrix


class ACEBufferDataset(Dataset):
    """Thin dataset wrapper around buffered ACE samples."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self._samples = samples

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # pragma: no cover - thin wrapper
        return self._samples[idx]


class ACETrainingBuffer:
    """FIFO buffer that stores ACE-style training tuples."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.samples: deque[Dict[str, Any]] = deque(maxlen=capacity)

    def add_sample(self, sample: Dict[str, Any]) -> None:
        self.samples.append(sample)

    def extend(self, samples: Iterable[Dict[str, Any]]) -> None:
        for sample in samples:
            self.add_sample(sample)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    @property
    def feature_channels(self) -> int:
        if not self.samples:
            raise RuntimeError("ACE buffer is empty; populate it before querying feature channels.")
        return self.samples[0]["features"].shape[0]

    def as_dataloader(self, batch_size: int, shuffle: bool) -> DataLoader:
        if not self.samples:
            raise RuntimeError("The ACE training buffer is empty; add samples before training.")
        dataset = ACEBufferDataset(list(self.samples))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _to_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().float()


def _pixel_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(height, dtype=torch.float32, device=device) + 0.5
    xs = torch.arange(width, dtype=torch.float32, device=device) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=0)


def _broadcast_rotation(rotation: torch.Tensor, batch: int) -> torch.Tensor:
    if rotation.dim() == 2:
        rotation = rotation.unsqueeze(0)
    if rotation.size(0) == 1:
        rotation = rotation.expand(batch, -1, -1)
    return rotation


def _broadcast_translation(translation: torch.Tensor, batch: int) -> torch.Tensor:
    if translation.dim() == 1:
        translation = translation.unsqueeze(0)
    if translation.size(0) == 1:
        translation = translation.expand(batch, -1)
    return translation


def _world_to_mean_frame(
    points_world: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> torch.Tensor:
    batch, _, height, width = points_world.shape
    rotation_batched = _broadcast_rotation(rotation, batch)
    translation_batched = _broadcast_translation(translation, batch).unsqueeze(-1)
    points_flat = points_world.view(batch, 3, -1) - translation_batched
    rotation_t = rotation_batched.transpose(1, 2)
    points_rel = torch.bmm(rotation_t, points_flat)
    return points_rel.view(batch, 3, height, width)


def _mean_frame_to_world(
    points_relative: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> torch.Tensor:
    batch, _, height, width = points_relative.shape
    rotation_batched = _broadcast_rotation(rotation, batch)
    translation_batched = _broadcast_translation(translation, batch).unsqueeze(-1)
    points_flat = points_relative.view(batch, 3, -1)
    world = torch.bmm(rotation_batched, points_flat) + translation_batched
    return world.view(batch, 3, height, width)


def _world_to_camera(points_world: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    if c2w.dim() == 2:
        c2w = c2w.unsqueeze(0)
    rotation = c2w[:, :3, :3]
    translation = c2w[:, :3, 3:4]
    rotation_t = rotation.transpose(1, 2)
    return torch.bmm(rotation_t, points_world - translation)


def _project_world_points(
    points_world: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    cam_points = _world_to_camera(points_flat, c2w)
    z = cam_points[:, 2:3, :]
    valid = z > 1e-6
    pixels = torch.bmm(intrinsics, cam_points)
    pixels = pixels[:, :2, :] / z.clamp(min=1e-6)
    return pixels.view(batch, 2, height, width), valid.view(batch, 1, height, width)


def _expand_mask(mask: torch.Tensor, channels: int) -> torch.Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.size(1) == 1 and channels > 1:
        mask = mask.expand(-1, channels, -1, -1)
    return mask


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    mask = _expand_mask(mask, pred.size(1))
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    return F.smooth_l1_loss(pred[mask], target[mask])


def _resolve_intrinsics(
    view: Dict[str, Any], prepared: Dict[str, Any], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if "camera_intrinsics" in view:
        return torch.as_tensor(view["camera_intrinsics"], device=device, dtype=dtype)
    if "intrinsics" in prepared:
        return prepared["intrinsics"].to(device=device, dtype=dtype).squeeze(0)
    raise KeyError("Camera intrinsics not available in view or prepared tensors.")


def _resolve_camera_pose(view: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if "camera_pose" in view:
        pose = torch.as_tensor(view["camera_pose"], device=device, dtype=dtype)
        if pose.dim() == 3:
            pose = pose.squeeze(0)
        return pose
    if "camera_pose_quats" in view and "camera_pose_trans" in view:
        quats = torch.as_tensor(view["camera_pose_quats"], device=device, dtype=dtype)
        trans = torch.as_tensor(view["camera_pose_trans"], device=device, dtype=dtype)
        rot = quaternion_to_rotation_matrix(quats).reshape(-1, 3, 3)
        if rot.shape[0] > 1:
            rot = rot[0]
        pose = torch.eye(4, device=device, dtype=dtype)
        pose[:3, :3] = rot
        pose[:3, 3] = trans.view(-1)[:3]
        return pose
    raise KeyError("Camera pose not available in view dictionary.")


def _build_valid_mask(reconstruction: Dict[str, Any], points_world: torch.Tensor) -> torch.Tensor:
    mask = reconstruction.get("mask")
    if isinstance(mask, torch.Tensor):
        mask_tensor = mask.to(dtype=torch.float32, device=points_world.device)
    elif mask is not None:
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=points_world.device)
    else:
        mask_tensor = torch.ones(
            points_world.shape[0], 1, points_world.shape[2], points_world.shape[3], device=points_world.device
        )

    if mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.unsqueeze(1)
    elif mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.view(1, 1, *mask_tensor.shape)

    finite_mask = torch.isfinite(points_world).all(dim=1, keepdim=True).float()
    mask_tensor = mask_tensor * finite_mask

    depth = reconstruction.get("depth_along_ray")
    if depth is not None:
        if not isinstance(depth, torch.Tensor):
            depth = torch.as_tensor(depth, device=points_world.device)
        depth_mask = (depth > 0).float()
        if depth_mask.dim() == 3:
            depth_mask = depth_mask.unsqueeze(1)
        elif depth_mask.dim() == 2:
            depth_mask = depth_mask.view(1, 1, *depth_mask.shape)
        mask_tensor = mask_tensor * depth_mask

    return mask_tensor


class SCRRegressionHead(nn.Module):
    """Lightweight convolutional regressor mapping dense features to XYZ maps."""

    def __init__(self, in_channels: int, hidden_dim: int, output_channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.net(x)


def _iter_training_indices(cfg: DictConfig, dataset_length: int) -> Iterable[int]:
    if cfg.training.sample_indices is not None:
        return cfg.training.sample_indices
    return range(min(cfg.training.num_samples, dataset_length))


def _collect_buffer(
    cfg: DictConfig,
    pipeline,
    device: torch.device,
    dataset,
) -> Tuple[ACETrainingBuffer, int]:
    buffer = ACETrainingBuffer(capacity=cfg.buffer.capacity)

    dataset_root = getattr(dataset, "ROOT", None)
    if dataset_root is None:
        raise AttributeError("Dataset instance must expose a ROOT attribute for mean pose computation.")
    dataset_root = Path(dataset_root)

    for idx in _iter_training_indices(cfg, len(dataset)):
        views = dataset[idx]
        if not views:
            continue
        single_view = views[0]
        scene_name = single_view.get("label") or single_view.get("scene_name") or str(idx)
        scene_root = dataset_root / scene_name
        mean_pose = load_or_compute_mean_pose(scene_root)
        mean_rotation, mean_translation = world_from_mean_pose(mean_pose)

        prepared = prepare_view(
            single_view,
            device=device,
            include_intrinsics=pipeline.include_intrinsics,
            include_depth=pipeline.include_depth,
            include_pose=pipeline.include_pose,
            include_scale=pipeline.include_scale,
        )

        with torch.no_grad():
            fused_tokens = pipeline.fuse(prepared)
            feature_map = pipeline.fusion_module.tokens_to_feature_map(fused_tokens)
            scale_token = pipeline.fusion_module.get_additional_token(
                device=device, dtype=feature_map.dtype
            )
            if scale_token is None:
                scale_token = torch.zeros(
                    feature_map.shape[0],
                    feature_map.shape[1],
                    1,
                    device=device,
                    dtype=feature_map.dtype,
                )
            dense_outputs, pose_outputs, scale_output = pipeline.base_model.downstream_head(
                dense_head_inputs=feature_map,
                scale_head_inputs=scale_token,
                img_shape=feature_map.shape[-2:],
                memory_efficient_inference=cfg.training.memory_efficient_inference,
            )

        reconstruction = extract_pointmap_like_outputs(
            pipeline, dense_outputs, pose_outputs, scale_output
        )

        features = dense_outputs.value
        points_world = reconstruction.get("pts3d")
        if points_world is None:
            continue

        mask_tensor = _build_valid_mask(reconstruction, points_world)

        intrinsics = _resolve_intrinsics(
            single_view,
            prepared,
            device=points_world.device,
            dtype=points_world.dtype,
        )
        camera_pose = _resolve_camera_pose(
            single_view,
            device=points_world.device,
            dtype=points_world.dtype,
        )

        mean_rotation = mean_rotation.to(device=points_world.device, dtype=points_world.dtype)
        mean_translation = mean_translation.to(device=points_world.device, dtype=points_world.dtype)
        target_relative = _world_to_mean_frame(points_world, mean_rotation, mean_translation)

        pixel_grid = _pixel_grid(
            height=features.shape[-2],
            width=features.shape[-1],
            device=points_world.device,
        )

        cpu_intrinsics = _to_cpu_float(intrinsics)
        cpu_pose = _to_cpu_float(camera_pose)
        cpu_rotation = _to_cpu_float(mean_rotation)
        cpu_translation = _to_cpu_float(mean_translation)
        cpu_quaternion = _to_cpu_float(mean_pose.quaternion)
        cpu_pixel_grid = _to_cpu_float(pixel_grid)

        for batch_idx in range(features.shape[0]):
            sample = {
                "features": _to_cpu_float(features[batch_idx]),
                "target_world": _to_cpu_float(points_world[batch_idx]),
                "target_relative": _to_cpu_float(target_relative[batch_idx]),
                "intrinsics": cpu_intrinsics,
                "c2w": cpu_pose,
                "valid_mask": _to_cpu_float(mask_tensor[batch_idx]),
                "pixel_grid": cpu_pixel_grid,
                "mean_rotation": cpu_rotation,
                "mean_translation": cpu_translation,
                "mean_quaternion": cpu_quaternion,
                "scene_name": scene_name,
            }
            buffer.add_sample(sample)

        if len(buffer) >= cfg.training.num_samples:
            break

    if not buffer:
        raise RuntimeError("Failed to populate ACE buffer; confirm dataset indices and configuration.")

    return buffer, buffer.feature_channels


def _ace_reprojection_loss(
    predictions_relative: torch.Tensor,
    batch: Dict[str, Any],
    loss_cfg: DictConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = predictions_relative.device

    target_relative = batch["target_relative"].to(device)
    mean_rotation = batch["mean_rotation"].to(device)
    mean_translation = batch["mean_translation"].to(device)

    predictions_world = _mean_frame_to_world(predictions_relative, mean_rotation, mean_translation)
    target_world = batch["target_world"].to(device)

    intrinsics = batch["intrinsics"].to(device)
    c2w = batch["c2w"].to(device)
    valid_mask = batch["valid_mask"].to(device)

    pred_pixels, pred_valid = _project_world_points(predictions_world, intrinsics, c2w)
    gt_pixels, gt_valid = _project_world_points(target_world, intrinsics, c2w)

    combined_mask = valid_mask * pred_valid * gt_valid

    reprojection_loss = _masked_smooth_l1(pred_pixels, gt_pixels, combined_mask)
    total_loss = loss_cfg.reprojection_weight * reprojection_loss
    metrics = {"reprojection": float(reprojection_loss.detach().cpu())}

    if loss_cfg.xyz_weight > 0:
        xyz_loss = _masked_smooth_l1(predictions_world, target_world, combined_mask)
        total_loss = total_loss + loss_cfg.xyz_weight * xyz_loss
        metrics["xyz"] = float(xyz_loss.detach().cpu())

    if loss_cfg.relative_weight > 0:
        relative_loss = F.smooth_l1_loss(predictions_relative, target_relative)
        total_loss = total_loss + loss_cfg.relative_weight * relative_loss
        metrics["relative"] = float(relative_loss.detach().cpu())

    metrics["total"] = float(total_loss.detach().cpu())
    return total_loss, metrics


def run_training(cfg: DictConfig) -> Dict[str, str]:
    if cfg.fusion.stored_feature_file is None:
        raise ValueError("Provide fusion.stored_feature_file to reuse AA memory during training.")

    requested_device = cfg.training.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU for SCR training.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    pipeline = build_pipeline_from_cfg(cfg)
    pipeline.base_model.to(device)
    pipeline.base_model.eval()
    pipeline.fusion_module.to(device)
    pipeline.fusion_module.eval()

    dataset = instantiate_dataset(cfg.dataset.dataset_str)

    buffer, in_channels = _collect_buffer(cfg, pipeline, device, dataset)

    dataloader = buffer.as_dataloader(
        batch_size=cfg.training.batch_size,
        shuffle=cfg.buffer.shuffle,
    )

    head = SCRRegressionHead(
        in_channels=in_channels,
        hidden_dim=cfg.scr_head.hidden_dim,
        output_channels=cfg.scr_head.output_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    output_dir = Path(cfg.training.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(cfg.training.max_epochs):
        head.train()
        for batch in dataloader:
            batch_features = batch["features"].to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = head(batch_features)
            loss, metrics = _ace_reprojection_loss(preds, batch, cfg.loss)
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % cfg.training.log_interval == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "loss": float(loss.detach().cpu()),
                            "buffer_size": len(buffer),
                            "metrics": metrics,
                        }
                    )
                )

    checkpoint_path = output_dir / "scr_head.pt"
    torch.save({"state_dict": head.state_dict()}, checkpoint_path)

    return {"checkpoint": str(checkpoint_path), "buffer_size": len(buffer)}


@hydra.main(
    version_base=None,
    config_path="../../../configs/tasks/aa_feature_fusion",
    config_name="scr_train",
)
def main(cfg: DictConfig) -> None:
    info = run_training(cfg)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

