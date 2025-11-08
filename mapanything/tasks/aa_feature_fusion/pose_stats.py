"""Pose statistics utilities supporting ACE-style SCR training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch

from mapanything.utils.geometry import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from mapanything.utils.wai.core import load_data


@dataclass(frozen=True)
class MeanPose:
    """Container for the mean camera pose of a scene."""

    translation: torch.Tensor  # (3,)
    quaternion: torch.Tensor  # (4,) stored as xyzw

    @property
    def rotation(self) -> torch.Tensor:
        """Rotation matrix corresponding to the quaternion."""

        return quaternion_to_rotation_matrix(self.quaternion)

    def to_serializable(self) -> dict[str, list[float]]:
        """Convert the pose into a JSON/NumPy friendly representation."""

        return {
            "translation": self.translation.tolist(),
            "quaternion": self.quaternion.tolist(),
        }


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Ensure a quaternion is unit-norm with a positive scalar component."""

    quaternion = quaternion / np.linalg.norm(quaternion)
    # Enforce a consistent hemisphere to avoid averaging antipodal representations.
    if quaternion[3] < 0:
        quaternion = -quaternion
    return quaternion


def _average_quaternions(quaternions: Iterable[np.ndarray]) -> np.ndarray:
    """Average quaternions using eigen-decomposition of the scatter matrix."""

    accum = np.zeros((4, 4), dtype=np.float64)
    count = 0
    for quat_xyzw in quaternions:
        quat = _normalize_quaternion(np.asarray(quat_xyzw, dtype=np.float64))
        # Convert to wxyz ordering for the averaging method.
        quat_wxyz = np.concatenate([quat[3:], quat[:3]])
        accum += np.outer(quat_wxyz, quat_wxyz)
        count += 1

    if count == 0:
        raise ValueError("Cannot average zero quaternions.")

    eigvals, eigvecs = np.linalg.eigh(accum / count)
    avg_wxyz = eigvecs[:, np.argmax(eigvals)]
    if avg_wxyz[0] < 0:
        avg_wxyz = -avg_wxyz
    avg_xyzw = np.concatenate([avg_wxyz[1:], avg_wxyz[:1]])
    return avg_xyzw.astype(np.float32)


def _iter_pose_matrices(scene_meta: dict) -> Iterable[np.ndarray]:
    """Yield camera-to-world matrices from the scene metadata."""

    frames = scene_meta.get("frames", [])
    for frame in frames:
        transform = frame.get("transform_matrix")
        if transform is None:
            continue
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.size == 16:
            matrix = matrix.reshape(4, 4)
        elif matrix.shape != (4, 4):
            continue
        yield matrix


def compute_mean_pose(scene_meta: dict) -> MeanPose:
    """Compute the mean pose for a scene from its metadata."""

    translations: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []

    for matrix in _iter_pose_matrices(scene_meta):
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        translations.append(translation.astype(np.float64))
        quat = rotation_matrix_to_quaternion(torch.from_numpy(rotation).float())
        quaternions.append(quat.numpy())

    if not translations:
        raise ValueError("Scene metadata does not contain any transform matrices.")

    mean_translation = torch.from_numpy(np.mean(translations, axis=0).astype(np.float32))
    mean_quaternion = torch.from_numpy(_average_quaternions(quaternions))
    return MeanPose(translation=mean_translation, quaternion=mean_quaternion)


def _stats_file(scene_root: Path) -> Path:
    return scene_root / "statistics" / "mean_pose.json"


def load_or_compute_mean_pose(scene_root: Path) -> MeanPose:
    """Load the cached mean pose for a scene, computing it if necessary."""

    stats_path = _stats_file(scene_root)
    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        translation = torch.tensor(payload["translation"], dtype=torch.float32)
        quaternion = torch.tensor(payload["quaternion"], dtype=torch.float32)
        return MeanPose(translation=translation, quaternion=quaternion)

    scene_meta = load_data(scene_root / "scene_meta.json", "scene_meta")
    mean_pose = compute_mean_pose(scene_meta)

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(mean_pose.to_serializable(), handle, indent=2)

    return mean_pose


def world_from_mean_pose(mean_pose: MeanPose) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return rotation and translation tensors for broadcasting operations."""

    rotation = mean_pose.rotation.to(dtype=torch.float32)
    translation = mean_pose.translation.to(dtype=torch.float32)
    return rotation, translation

