# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
"""Convert 7Scenes data laid out like ``pgt_7scenes_*`` folders into WAI format."""

import logging
from pathlib import Path

import numpy as np
import torch
from argconf import argconf_parse
from mapanything.utils.wai.core import store_data
from natsort import natsorted
from PIL import Image
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from wai_processing.utils.wrapper import convert_scenes_wrapper

logger = logging.getLogger(__name__)

RGB_SUFFIX = ".color.png"
DEPTH_SUFFIX = ".depth.png"
POSE_SUFFIX = ".pose.txt"
CALIB_SUFFIX = ".calibration.txt"


def _discover_scene_keys(cfg) -> list[str]:
    original_root = Path(cfg.original_root)
    scene_keys: list[str] = []
    dataset_whitelist_cfg = cfg.get("dataset_whitelist")
    dataset_whitelist = (
        {entry.lower() for entry in dataset_whitelist_cfg}
        if dataset_whitelist_cfg
        else None
    )
    split_whitelist_cfg = cfg.get("split_whitelist")
    split_whitelist = (
        {entry.lower() for entry in split_whitelist_cfg}
        if split_whitelist_cfg
        else None
    )
    sequence_whitelist_cfg = cfg.get("sequence_whitelist")
    if sequence_whitelist_cfg:
        sequence_whitelist_full = {entry for entry in sequence_whitelist_cfg}
        sequence_whitelist_short = {
            entry.split("/")[-1] for entry in sequence_whitelist_cfg
        }
    else:
        sequence_whitelist_full = set()
        sequence_whitelist_short = set()

    for scene_dir in sorted(original_root.glob("pgt_7scenes_*")):
        if not scene_dir.is_dir():
            continue
        dataset_name = scene_dir.name.replace("pgt_7scenes_", "", 1)
        if dataset_whitelist and dataset_name.lower() not in dataset_whitelist:
            continue
        for split_dir in sorted(scene_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            split_name = split_dir.name
            if split_whitelist and split_name.lower() not in split_whitelist:
                continue
            rgb_dir = split_dir / "rgb"
            if not rgb_dir.exists():
                logger.warning("Skipping %s; missing rgb/ directory", split_dir)
                continue
            for image_file in natsorted(rgb_dir.glob(f"*{RGB_SUFFIX}")):
                prefix = image_file.name.replace(RGB_SUFFIX, "")
                seq_name = prefix.split("-frame")[0]
                scene_key = f"{dataset_name}/{split_name}/{seq_name}"
                if sequence_whitelist_full and (
                    scene_key not in sequence_whitelist_full
                    and seq_name not in sequence_whitelist_short
                ):
                    continue
                if scene_key not in scene_keys:
                    scene_keys.append(scene_key)
    return scene_keys


def _parse_calibration(calib_path: Path, cfg) -> tuple[float, float, float, float]:
    if not calib_path.exists():
        fx = fy = float(cfg.default_focal_length)
        cx = float(cfg.image_width) / 2.0
        cy = float(cfg.image_height) / 2.0
        return fx, fy, cx, cy

    with open(calib_path, "r", encoding="utf-8") as handle:
        values: list[float] = []
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.replace(",", " ").split()
            values.extend(float(part) for part in parts)
    if not values:
        fx = fy = float(cfg.default_focal_length)
        cx = float(cfg.image_width) / 2.0
        cy = float(cfg.image_height) / 2.0
        return fx, fy, cx, cy

    if len(values) == 1:
        fx = fy = values[0]
    else:
        fx, fy = values[0], values[1]
    if len(values) >= 4:
        cx, cy = values[2], values[3]
    else:
        cx = float(cfg.image_width) / 2.0
        cy = float(cfg.image_height) / 2.0
    return float(fx), float(fy), float(cx), float(cy)


def _load_depth(depth_path: Path, invalid_values: set[int]) -> np.ndarray:
    with Image.open(depth_path) as depth_pil:
        depth = np.array(depth_pil, dtype=np.uint16)
    mask = np.isin(depth, list(invalid_values))
    depth = depth.astype(np.float32)
    depth[mask] = 0.0
    depth *= 1.0 / 1000.0
    return depth


def process_seven_scenes_scene(cfg, scene_key: str):
    dataset_name, split, sequence = scene_key.split("/")
    source_root = Path(cfg.original_root) / f"pgt_7scenes_{dataset_name}" / split

    rgb_dir = source_root / "rgb"
    depth_dir = source_root / "depth"
    pose_dir = source_root / "poses"
    calib_dir = source_root / "calibration"

    if not rgb_dir.exists():
        raise FileNotFoundError(f"Missing rgb directory at {rgb_dir}")

    wai_scene_name = f"{dataset_name}_{split}_{sequence}"
    target_scene_root = Path(cfg.root) / wai_scene_name
    image_dir = target_scene_root / "images"
    depth_out_dir = target_scene_root / "depth"
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_out_dir.mkdir(parents=True, exist_ok=True)

    invalid_values = set(cfg.get("invalid_depth_values", [0, 65535]))

    image_files = natsorted(rgb_dir.glob(f"{sequence}-frame-*{RGB_SUFFIX}"))
    if not image_files:
        logger.warning("No frames found for %s", scene_key)
        return "finished", "empty_sequence"

    frames = []
    for image_path in image_files:
        base_name = image_path.name.replace(RGB_SUFFIX, "")
        depth_path = depth_dir / f"{base_name}{DEPTH_SUFFIX}"
        pose_path = pose_dir / f"{base_name}{POSE_SUFFIX}"
        calib_path = calib_dir / f"{base_name}{CALIB_SUFFIX}"

        if not depth_path.exists():
            logger.warning("Missing depth for %s, skipping frame", base_name)
            continue
        if not pose_path.exists():
            logger.warning("Missing pose for %s, skipping frame", base_name)
            continue

        target_image_path = image_dir / image_path.name
        if target_image_path.exists():
            target_image_path.unlink()
        target_image_path.symlink_to(image_path.resolve())

        depth = _load_depth(depth_path, invalid_values)
        rel_depth_path = Path("depth") / f"{base_name}.exr"
        store_data(target_scene_root / rel_depth_path, torch.from_numpy(depth), "depth")

        pose = np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4)
        fx, fy, cx, cy = _parse_calibration(calib_path, cfg)

        frame_entry = {
            "frame_name": base_name,
            "image": str(Path("images") / image_path.name),
            "file_path": str(Path("images") / image_path.name),
            "depth": str(rel_depth_path),
            "transform_matrix": pose.tolist(),
            "h": int(depth.shape[0]),
            "w": int(depth.shape[1]),
            "fl_x": float(fx),
            "fl_y": float(fy),
            "cx": float(cx),
            "cy": float(cy),
        }
        frames.append(frame_entry)

    if not frames:
        logger.warning("All frames skipped for %s", scene_key)
        return "finished", "no_valid_frames"

    scene_meta = {
        "scene_name": wai_scene_name,
        "dataset_name": cfg.dataset_name,
        "version": cfg.version,
        "shared_intrinsics": False,
        "camera_model": "PINHOLE",
        "camera_convention": "opencv",
        "scale_type": "metric",
        "scene_modalities": {},
        "frames": frames,
        "frame_modalities": {
            "image": {"frame_key": "image", "format": "image"},
            "depth": {"frame_key": "depth", "format": "depth"},
        },
    }
    store_data(target_scene_root / "scene_meta.json", scene_meta, "scene_meta")


def get_original_scene_names(cfg):
    return _discover_scene_keys(cfg)


if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/seven_scenes.yaml")
    Path(cfg.root).mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(process_seven_scenes_scene, cfg, get_original_scene_names)
