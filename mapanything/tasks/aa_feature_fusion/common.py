"""Shared helpers for AA feature fusion tasks."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch

from mapanything import datasets as dataset_registry
from mapanything.tasks.aa_feature_fusion.pipeline import AAFeatureFusionPipeline
from mapanything.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
)


def instantiate_dataset(dataset_str: str):
    """Instantiate a dataset using the global registry side-effects."""

    if isinstance(dataset_str, str):
        return eval(dataset_str, vars(dataset_registry))
    raise TypeError(f"Expected dataset_str to be a string, got {type(dataset_str)!r}")


def to_tensor(value: Any, device: torch.device) -> torch.Tensor:
    """Convert common array-like inputs to tensors on the requested device."""

    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device)


def prepare_view(
    view: Dict[str, Any],
    *,
    device: torch.device,
    include_intrinsics: bool,
    include_depth: bool,
    include_pose: bool,
    include_scale: bool,
) -> Dict[str, Any]:
    """Convert a dataset view dictionary into the tensors required by the pipeline."""

    prepared: Dict[str, Any] = {
        "img": to_tensor(view["img"], device=device).unsqueeze(0),
        "data_norm_type": view["data_norm_type"],
    }

    if include_intrinsics:
        if "ray_directions_cam" in view:
            prepared["ray_directions_cam"] = to_tensor(
                view["ray_directions_cam"], device=device
            ).unsqueeze(0)
        elif "intrinsics" in view:
            prepared["intrinsics"] = to_tensor(view["intrinsics"], device=device).unsqueeze(0)
    if include_depth and "depth_along_ray" in view:
        prepared["depth_along_ray"] = to_tensor(view["depth_along_ray"], device=device).unsqueeze(0)
    if include_pose:
        if "camera_pose_quats" in view:
            prepared["camera_pose_quats"] = to_tensor(
                view["camera_pose_quats"], device=device
            ).unsqueeze(0)
        if "camera_pose_trans" in view:
            prepared["camera_pose_trans"] = to_tensor(
                view["camera_pose_trans"], device=device
            ).unsqueeze(0)
    if include_scale:
        if "is_metric_scale" in view:
            prepared["is_metric_scale"] = torch.as_tensor(
                [[bool(view["is_metric_scale"])]], device=device
            )
        if "depth_scale" in view:
            prepared["depth_scale"] = to_tensor(view["depth_scale"], device=device).unsqueeze(0)
        if "pose_scale" in view:
            prepared["pose_scale"] = to_tensor(view["pose_scale"], device=device).unsqueeze(0)
    return prepared


def extract_pointmap_like_outputs(
    pipeline: AAFeatureFusionPipeline,
    dense_outputs,
    pose_outputs,
    scale_output,
) -> Dict[str, torch.Tensor]:
    """Normalize downstream head outputs to a pointmap-style dictionary."""

    scene_rep_type = pipeline.base_model.scene_rep_type
    results: Dict[str, torch.Tensor] = {}

    scale_value = scale_output.value
    scale_hw = scale_value.view(scale_value.shape[0], scale_value.shape[1], 1, 1)

    if scene_rep_type.startswith("pointmap"):
        pts3d = dense_outputs.value * scale_hw
        results["pts3d"] = pts3d
        if hasattr(dense_outputs, "confidence"):
            results["confidence"] = dense_outputs.confidence
        if hasattr(dense_outputs, "mask"):
            results["mask"] = dense_outputs.mask
    elif scene_rep_type.startswith("raymap+depth"):
        raymap = dense_outputs.value
        ray_origins, ray_directions, depth_along_ray = raymap.split([3, 3, 1], dim=1)
        pts3d = (ray_origins + ray_directions * depth_along_ray) * scale_hw
        results.update(
            {
                "pts3d": pts3d,
                "ray_origins": ray_origins * scale_hw,
                "ray_directions": ray_directions,
                "depth_along_ray": depth_along_ray * scale_hw,
            }
        )
        if hasattr(dense_outputs, "confidence"):
            results["confidence"] = dense_outputs.confidence
        if hasattr(dense_outputs, "mask"):
            results["mask"] = dense_outputs.mask
    elif scene_rep_type.startswith("raydirs+depth+pose"):
        if pose_outputs is None:
            raise RuntimeError(
                "Pose outputs are required for raydirs+depth+pose scene representations."
            )
        ray_dirs, depth = dense_outputs.value.split([3, 1], dim=1)
        cam_trans, cam_quats = pose_outputs.value.split([3, 4], dim=1)
        pts3d = convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
            ray_dirs.permute(0, 2, 3, 1),
            depth.permute(0, 2, 3, 1),
            cam_trans,
            cam_quats,
        ).permute(0, 3, 1, 2)
        pts3d = pts3d * scale_hw
        results.update(
            {
                "pts3d": pts3d,
                "ray_directions": ray_dirs,
                "depth_along_ray": depth * scale_hw,
                "cam_trans": cam_trans * scale_value,
                "cam_quats": cam_quats,
            }
        )
    elif scene_rep_type.startswith("campointmap+pose"):
        if pose_outputs is None:
            raise RuntimeError(
                "Pose outputs are required for campointmap+pose scene representations."
            )
        pts3d_cam = dense_outputs.value
        depth = torch.norm(pts3d_cam, dim=1, keepdim=True)
        ray_dirs = pts3d_cam / depth.clamp(min=1e-6)
        cam_trans, cam_quats = pose_outputs.value.split([3, 4], dim=1)
        pts3d_world = convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
            ray_dirs.permute(0, 2, 3, 1),
            depth.permute(0, 2, 3, 1),
            cam_trans,
            cam_quats,
        ).permute(0, 3, 1, 2)
        results.update(
            {
                "pts3d": pts3d_world * scale_hw,
                "pts3d_cam": pts3d_cam,
                "ray_directions": ray_dirs,
                "depth_along_ray": depth * scale_hw,
                "cam_trans": cam_trans * scale_value,
                "cam_quats": cam_quats,
            }
        )
    else:
        results["dense_value"] = dense_outputs.value
    results["scale"] = scale_value
    return results


def detach_to_cpu(obj: Any) -> Any:
    """Recursively move tensors to CPU for serialization."""

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: detach_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [detach_to_cpu(value) for value in obj]
        return type(obj)(out)
    return obj

