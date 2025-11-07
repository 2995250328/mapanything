"""High-level orchestration utilities for alternating-attention feature fusion."""

from __future__ import annotations

from typing import Dict

import torch

from mapanything.models.mapanything.model import MapAnything

from .fusion import AAFeatureFusionModule, extract_single_view_tokens


class AAFeatureFusionPipeline:
    """Pipeline that combines stored AA memory with a new single-view input."""

    def __init__(
        self,
        *,
        base_model: MapAnything,
        fusion_module: AAFeatureFusionModule,
        include_intrinsics: bool = True,
        include_depth: bool = False,
        include_pose: bool = False,
        include_scale: bool = False,
    ) -> None:
        self.base_model = base_model.eval()
        self.fusion_module = fusion_module
        self.include_intrinsics = include_intrinsics
        self.include_depth = include_depth
        self.include_pose = include_pose
        self.include_scale = include_scale

    def _filter_view(self, view: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        allowed_keys = {"img", "data_norm_type"}
        if self.include_intrinsics:
            allowed_keys.add("ray_directions_cam")
        if self.include_depth:
            allowed_keys.add("depth_along_ray")
        if self.include_pose:
            allowed_keys.update({"camera_pose_quats", "camera_pose_trans"})
        if self.include_scale:
            allowed_keys.update({"is_metric_scale", "depth_scale", "pose_scale"})
        return {key: value for key, value in view.items() if key in allowed_keys}

    def encode_view(self, view: Dict[str, torch.Tensor]) -> torch.Tensor:
        filtered_view = self._filter_view(view)
        return extract_single_view_tokens(self.base_model, filtered_view)

    def fuse(self, view: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.encode_view(view)
            fused = self.fusion_module(tokens)
        return fused

    def __call__(self, view: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.fuse(view)
