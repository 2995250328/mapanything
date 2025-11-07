"""Run a small AA feature fusion reconstruction demo on 7Scenes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import hydra
import torch
from omegaconf import DictConfig

from mapanything import datasets as dataset_registry
from mapanything.tasks.aa_feature_fusion.builder import build_pipeline_from_cfg
from mapanything.tasks.aa_feature_fusion.pipeline import AAFeatureFusionPipeline
from mapanything.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
)


def _instantiate_dataset(dataset_str: str):
    """Instantiate a dataset using the registry side-effects from ``mapanything.datasets``."""

    if isinstance(dataset_str, str):
        return eval(dataset_str, vars(dataset_registry))
    raise TypeError(f"Expected dataset_str to be a string, got {type(dataset_str)!r}")


def _to_tensor(array: Any, device: torch.device) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.to(device)
    return torch.as_tensor(array, device=device)


def _prepare_view(
    view: Dict[str, Any],
    *,
    device: torch.device,
    include_intrinsics: bool,
    include_depth: bool,
    include_pose: bool,
    include_scale: bool,
) -> Dict[str, Any]:
    """Convert a dataset view dictionary into the tensors required by the model."""

    prepared: Dict[str, Any] = {
        "img": _to_tensor(view["img"], device=device).unsqueeze(0),
        "data_norm_type": [view["data_norm_type"]],
    }

    if include_intrinsics:
        if "ray_directions_cam" in view:
            prepared["ray_directions_cam"] = _to_tensor(
                view["ray_directions_cam"], device=device
            ).unsqueeze(0)
        elif "intrinsics" in view:
            prepared["intrinsics"] = _to_tensor(view["intrinsics"], device=device).unsqueeze(0)
    if include_depth and "depth_along_ray" in view:
        prepared["depth_along_ray"] = _to_tensor(view["depth_along_ray"], device=device).unsqueeze(0)
    if include_pose:
        if "camera_pose_quats" in view:
            prepared["camera_pose_quats"] = _to_tensor(
                view["camera_pose_quats"], device=device
            ).unsqueeze(0)
        if "camera_pose_trans" in view:
            prepared["camera_pose_trans"] = _to_tensor(
                view["camera_pose_trans"], device=device
            ).unsqueeze(0)
    if include_scale:
        if "is_metric_scale" in view:
            prepared["is_metric_scale"] = torch.as_tensor(
                [[bool(view["is_metric_scale"])]], device=device
            )
        if "depth_scale" in view:
            prepared["depth_scale"] = _to_tensor(view["depth_scale"], device=device).unsqueeze(0)
        if "pose_scale" in view:
            prepared["pose_scale"] = _to_tensor(view["pose_scale"], device=device).unsqueeze(0)
    return prepared


def _extract_pointmap_like_outputs(
    pipeline: AAFeatureFusionPipeline,
    dense_outputs,
    pose_outputs,
    scale_output,
) -> Dict[str, torch.Tensor]:
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
            raise RuntimeError("Pose outputs are required for raydirs+depth+pose scene representations.")
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
            raise RuntimeError("Pose outputs are required for campointmap+pose scene representations.")
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


def run_demo(cfg: DictConfig) -> Dict[str, Any]:
    pipeline = build_pipeline_from_cfg(cfg)
    device = torch.device(cfg.demo.device)
    pipeline.base_model.to(device)
    pipeline.base_model.eval()
    pipeline.fusion_module.to(device)
    pipeline.fusion_module.eval()

    if pipeline.base_model.pred_head_type != "linear":
        raise NotImplementedError(
            "The demo currently supports models with linear dense prediction heads."
        )

    dataset = _instantiate_dataset(cfg.dataset.dataset_str)
    output_dir = Path(cfg.demo.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    indices: Iterable[int]
    if cfg.demo.sample_indices is not None:
        indices = cfg.demo.sample_indices
    else:
        indices = range(len(dataset))

    saved_items: List[Dict[str, Any]] = []
    processed = 0

    for idx in indices:
        if processed >= cfg.demo.num_samples:
            break
        views = dataset[idx]
        if not views:
            continue
        single_view = views[0]

        prepared = _prepare_view(
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
                    feature_map.shape[0], feature_map.shape[1], 1, device=device, dtype=feature_map.dtype
                )
            dense_outputs, pose_outputs, scale_output = pipeline.base_model.downstream_head(
                dense_head_inputs=feature_map,
                scale_head_inputs=scale_token,
                img_shape=feature_map.shape[-2:],
                memory_efficient_inference=cfg.demo.memory_efficient_inference,
            )

        reconstruction = _extract_pointmap_like_outputs(
            pipeline, dense_outputs, pose_outputs, scale_output
        )

        sample_name = single_view.get("instance", f"sample_{idx:06d}")
        sample_name = str(sample_name).replace("/", "_")
        output_path = output_dir / f"{sample_name}.pt"

        record = {
            "dataset": single_view.get("dataset"),
            "label": single_view.get("label"),
            "instance": single_view.get("instance"),
            "scene_rep_type": pipeline.base_model.scene_rep_type,
            "rgb": single_view["img"].detach().cpu(),
            "intrinsics": torch.as_tensor(single_view["camera_intrinsics"]).cpu(),
            "camera_pose": torch.as_tensor(single_view["camera_pose"]).cpu(),
            "reconstruction": {k: v.detach().cpu() for k, v in reconstruction.items()},
        }
        torch.save(record, output_path)

        saved_items.append(
            {
                "index": int(idx),
                "output": str(output_path),
                "scene": record["label"],
                "instance": record["instance"],
            }
        )
        processed += 1

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(saved_items, indent=2, ensure_ascii=False))
    return {"saved": saved_items, "output_dir": str(output_dir)}


@hydra.main(
    version_base=None,
    config_path="../../../configs/tasks/aa_feature_fusion",
    config_name="demo",
)
def main(cfg: DictConfig) -> None:
    info = run_demo(cfg)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
