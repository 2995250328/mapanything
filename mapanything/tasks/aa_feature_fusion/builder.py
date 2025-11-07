"""Helpers to construct AA feature fusion pipelines from Hydra configs."""

from __future__ import annotations

from omegaconf import DictConfig

from mapanything.models.mapanything.model import MapAnything

from .fusion import AAFeatureFusionModule
from .pipeline import AAFeatureFusionPipeline


def build_pipeline_from_cfg(cfg: DictConfig) -> AAFeatureFusionPipeline:
    """Instantiate the fusion pipeline described by the provided config."""

    base_model = MapAnything(**cfg.model.model_config)
    fusion_module = AAFeatureFusionModule(
        stored_feature_file=cfg.fusion.stored_feature_file,
        num_heads=cfg.fusion.num_heads,
        mlp_ratio=cfg.fusion.mlp_ratio,
        dropout=cfg.fusion.dropout,
        map_location=cfg.fusion.map_location,
    )
    pipeline = AAFeatureFusionPipeline(
        base_model=base_model,
        fusion_module=fusion_module,
        include_intrinsics=cfg.single_view.include_intrinsics,
        include_depth=cfg.single_view.include_depth,
        include_pose=cfg.single_view.include_pose,
        include_scale=cfg.single_view.include_scale,
    )
    return pipeline
