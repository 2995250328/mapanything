"""Helpers to construct AA feature fusion pipelines from Hydra configs."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf  # 确保导入 OmegaConf

from mapanything.models.mapanything.model import MapAnything

from .fusion import AAFeatureFusionModule
from .pipeline import AAFeatureFusionPipeline


def build_pipeline_from_cfg(cfg: DictConfig) -> AAFeatureFusionPipeline:
    """Instantiate the fusion pipeline described by the provided config."""

    # [关键修复]：不要转换为 dict，而是创建一个 "unlocked" 的配置副本。
    # 1. OmegaConf.create(cfg.model.model_config) 创建一个副本。
    # 2. OmegaConf.set_struct(..., False) 允许在 __init__ 中 'del' 键。
    # 这样可以避免 'ConfigTypeError'，同时也不会触发 'UnsupportedInterpolationType'
    model_config_unlocked = OmegaConf.create(cfg.model.model_config)
    OmegaConf.set_struct(model_config_unlocked, False)

    base_model = MapAnything(**model_config_unlocked)

    # 按照你的原始代码构建 fusion_module 和 pipeline
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