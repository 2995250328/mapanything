"""Skeleton training loop for alternating-attention feature fusion."""

from __future__ import annotations

import logging
from typing import Any

import hydra
from omegaconf import DictConfig

def _build_pipeline(cfg: DictConfig):
    from mapanything.models.mapanything.model import MapAnything
    from .fusion import AAFeatureFusionModule
    from .pipeline import AAFeatureFusionPipeline

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


def run_training(cfg: DictConfig) -> Any:
    log = logging.getLogger(__name__)
    pipeline = _build_pipeline(cfg)
    log.info("Initialized AA feature fusion pipeline for training.")
    log.info("Dataset config: %s", cfg.dataset)
    log.info("Training parameters: %s", cfg.training)
    log.info(
        "Stored AA memory loaded: %s",
        getattr(pipeline.fusion_module, "has_memory", False),
    )
    # Placeholder for the actual optimization loop.
    return {"status": "training_loop_not_implemented"}


@hydra.main(
    version_base=None,
    config_path="../../../configs/tasks/aa_feature_fusion",
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
