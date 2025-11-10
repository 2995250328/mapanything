"""Run multi-view MapAnything reconstruction directly from a dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import hydra
import torch
from omegaconf import DictConfig

from mapanything.models.mapanything.model import MapAnything
from mapanything.tasks.aa_feature_fusion.common import (
    detach_to_cpu,
    instantiate_dataset,
    to_tensor,
)
from mapanything.utils.geometry import (
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
)


def _prepare_model_view(view: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Convert a dataset view into the tensors expected by :class:`MapAnything`."""

    prepared: Dict[str, Any] = {
        "img": to_tensor(view["img"], device=device).unsqueeze(0),
        "data_norm_type": [view["data_norm_type"]],
    }

    if "ray_directions_cam" in view:
        prepared["ray_directions_cam"] = to_tensor(
            view["ray_directions_cam"], device=device
        ).unsqueeze(0)
    if "depth_along_ray" in view:
        prepared["depth_along_ray"] = to_tensor(
            view["depth_along_ray"], device=device
        ).unsqueeze(0)
    if "camera_pose_quats" in view:
        prepared["camera_pose_quats"] = to_tensor(
            view["camera_pose_quats"], device=device
        ).unsqueeze(0)
    if "camera_pose_trans" in view:
        prepared["camera_pose_trans"] = to_tensor(
            view["camera_pose_trans"], device=device
        ).unsqueeze(0)
    if "is_metric_scale" in view:
        prepared["is_metric_scale"] = torch.as_tensor(
            [[bool(view["is_metric_scale"])]], device=device
        )
    if "depth_scale" in view:
        prepared["depth_scale"] = to_tensor(view["depth_scale"], device=device).unsqueeze(0)
    if "pose_scale" in view:
        prepared["pose_scale"] = to_tensor(view["pose_scale"], device=device).unsqueeze(0)
    return prepared


def _compute_pointmap_targets(view: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """Derive world-space pointmaps from dataset depth and pose metadata."""

    ray_dirs = to_tensor(view["ray_directions_cam"], device=device)
    depth = to_tensor(view["depth_along_ray"], device=device)
    cam_quats = to_tensor(view["camera_pose_quats"], device=device)
    cam_trans = to_tensor(view["camera_pose_trans"], device=device)
    pts = convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
        ray_dirs.unsqueeze(0), depth.unsqueeze(0), cam_trans.unsqueeze(0), cam_quats.unsqueeze(0)
    )
    return pts.squeeze(0).permute(2, 0, 1)


def _iter_indices(cfg, dataset_length: int) -> Iterable[int]:
    indices: Iterable[int]
    if cfg.reconstruction.sample_indices is not None:
        indices = cfg.reconstruction.sample_indices
    else:
        end_index = min(cfg.reconstruction.max_index, dataset_length)
        indices = range(cfg.reconstruction.start_index, end_index)
    return indices


def run_dataset_reconstruction(cfg: DictConfig) -> Dict[str, Any]:
    device = torch.device(cfg.reconstruction.device)
    model = MapAnything(**cfg.model.model_config).to(device)
    model.eval()

    dataset = instantiate_dataset(cfg.dataset.dataset_str)

    output_dir = Path(cfg.reconstruction.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    saved: List[Dict[str, Any]] = []

    max_views = cfg.reconstruction.views_per_sample

    dataset_length = len(dataset)

    for idx in _iter_indices(cfg, dataset_length):
        if processed >= cfg.reconstruction.num_samples:
            break

        if idx >= dataset_length:
            break

        views = dataset[idx]
        if not views:
            continue

        selected_views = views[:max_views] if max_views is not None else views
        model_views = [_prepare_model_view(view, device=device) for view in selected_views]

        with torch.no_grad():
            outputs = model.forward(
                model_views,
                memory_efficient_inference=cfg.reconstruction.memory_efficient_inference,
            )

        per_view_records: List[Dict[str, Any]] = []
        for view, model_view, output in zip(selected_views, model_views, outputs):
            record: Dict[str, Any] = {
                "dataset": view.get("dataset"),
                "label": view.get("label"),
                "instance": view.get("instance"),
                "inputs": {
                    "img": detach_to_cpu(model_view["img"]),
                    "ray_directions_cam": detach_to_cpu(model_view.get("ray_directions_cam")),
                    "depth_along_ray": detach_to_cpu(model_view.get("depth_along_ray")),
                    "camera_pose_quats": detach_to_cpu(model_view.get("camera_pose_quats")),
                    "camera_pose_trans": detach_to_cpu(model_view.get("camera_pose_trans")),
                },
                "outputs": detach_to_cpu(output),
            }

            if cfg.reconstruction.save_pointmap_targets:
                pointmap = _compute_pointmap_targets(view, device=device)
                record["target_pointmap"] = detach_to_cpu(pointmap)

            per_view_records.append(record)

        sample_name = selected_views[0].get("label", f"sample_{idx:06d}")
        sample_root = output_dir / str(sample_name).replace("/", "_")
        sample_root.mkdir(parents=True, exist_ok=True)

        sample_file = sample_root / "reconstruction.pt"
        torch.save(per_view_records, sample_file)

        saved.append(
            {
                "index": int(idx),
                "output": str(sample_file),
                "num_views": len(per_view_records),
                "scene": per_view_records[0]["label"],
            }
        )

        processed += 1

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(saved, indent=2, ensure_ascii=False))
    return {"saved": saved, "output_dir": str(output_dir)}


@hydra.main(config_path="../../../configs/tasks/aa_feature_fusion/",
            config_name="demo")
def main(cfg: DictConfig) -> None:
    info = run_dataset_reconstruction(cfg)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

