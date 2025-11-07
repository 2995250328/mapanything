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
from mapanything.utils.wai.core import load_data, store_data
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
            has_valid_sequence = False
            for image_file in natsorted(rgb_dir.glob(f"*{RGB_SUFFIX}")):
                prefix = image_file.name.replace(RGB_SUFFIX, "")
                seq_name = prefix.split("-frame")[0]
                if sequence_whitelist_full and (
                    f"{dataset_name}/{split_name}/{seq_name}" not in sequence_whitelist_full
                    and seq_name not in sequence_whitelist_short
                ):
                    continue
                has_valid_sequence = True
                break
            if has_valid_sequence:
                split_key = f"{dataset_name}_{split_name}"
                if split_key not in scene_keys:
                    scene_keys.append(split_key)
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


def _load_depth(
    depth_path: Path, invalid_values: set[int], device: torch.device
) -> np.ndarray:
    with Image.open(depth_path) as depth_pil:
        depth_np = np.array(depth_pil, dtype=np.uint16)

    depth = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)

    if invalid_values:
        invalid_mask = torch.zeros_like(depth, dtype=torch.bool)
        for value in invalid_values:
            invalid_mask |= depth == float(value)
        depth = depth.masked_fill(invalid_mask, 0.0)

    depth *= 1.0 / 1000.0
    return depth.cpu().numpy()


def _sequence_filters(cfg):
    sequence_whitelist_cfg = cfg.get("sequence_whitelist")
    if not sequence_whitelist_cfg:
        return None, None
    full_keys = {entry for entry in sequence_whitelist_cfg}
    short_keys = {entry.split("/")[-1] for entry in sequence_whitelist_cfg}
    return full_keys, short_keys


def process_seven_scenes_scene(cfg, scene_key: str):
    if "_" not in scene_key:
        raise ValueError(
            "Expected scene keys to be formatted as '<dataset>_<split>' but received "
            f"{scene_key!r}."
        )
    dataset_name, split = scene_key.split("_", 1)
    source_root = Path(cfg.original_root) / f"pgt_7scenes_{dataset_name}" / split

    rgb_dir = source_root / "rgb"
    depth_dir = source_root / "depth"
    pose_dir = source_root / "poses"
    calib_dir = source_root / "calibration"

    if not rgb_dir.exists():
        raise FileNotFoundError(f"Missing rgb directory at {rgb_dir}")

    target_scene_root = Path(cfg.root) / scene_key
    image_dir = target_scene_root / "images"
    depth_out_dir = target_scene_root / "depth"
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_out_dir.mkdir(parents=True, exist_ok=True)

    invalid_values = set(cfg.get("invalid_depth_values", [0, 65535]))
    requested_device = cfg.get("device", "cpu")
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "CUDA 不可用，转换流程将改为在 CPU 上执行 (请求的设备为 %s)",
            requested_device,
        )
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)
    sequence_whitelist_full, sequence_whitelist_short = _sequence_filters(cfg)

    image_files = natsorted(rgb_dir.glob(f"*{RGB_SUFFIX}"))
    if sequence_whitelist_full or sequence_whitelist_short:
        filtered_files = []
        for image_path in image_files:
            base_name = image_path.name.replace(RGB_SUFFIX, "")
            seq_name = base_name.split("-frame")[0]
            if sequence_whitelist_full and (
                f"{dataset_name}/{split}/{seq_name}" not in sequence_whitelist_full
                and seq_name not in sequence_whitelist_short
            ):
                continue
            filtered_files.append(image_path)
        image_files = filtered_files

    if not image_files:
        logger.warning("No frames found for %s", scene_key)
        return "finished", "empty_split"

    existing_meta_path = target_scene_root / "scene_meta.json"
    if existing_meta_path.exists():
        scene_meta = load_data(existing_meta_path, "scene_meta")
        frames = list(scene_meta.get("frames", []))
    else:
        frames = []

    existing_frame_names = {frame["frame_name"] for frame in frames}

    for image_path in image_files:
        base_name = image_path.name.replace(RGB_SUFFIX, "")
        if base_name in existing_frame_names:
            continue

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
        if not target_image_path.exists():
            target_image_path.symlink_to(image_path.resolve())

        depth = _load_depth(depth_path, invalid_values, device)
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
        existing_frame_names.add(base_name)

    if not frames:
        logger.warning("All frames skipped for %s", scene_key)
        return "finished", "no_valid_frames"

    frames = sorted(frames, key=lambda frame: frame["frame_name"])

    scene_meta = {
        "scene_name": scene_key,
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
