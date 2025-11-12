"""Helpers to construct AA feature fusion pipelines from Hydra configs."""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf  # 确保导入 OmegaConf

from mapanything.models.mapanything.model import MapAnything
from mapanything.models import init_model

from .fusion import AAFeatureFusionModule
from .pipeline import AAFeatureFusionPipeline


def build_pipeline_from_cfg(cfg: DictConfig) -> AAFeatureFusionPipeline:
    """Instantiate the fusion pipeline described by the provided config."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    # Load Model
    model = init_model(
        cfg.model.model_str, cfg.model.model_config, torch_hub_force_reload=False
    )
    model.to(device)  # Move model to device
    # Load pretrained model
    if cfg.model.pretrained:
        print("Loading pretrained: ", cfg.model.pretrained)
        ckpt = torch.load(
            cfg.model.pretrained, map_location=device, weights_only=False
        )
        print(model.load_state_dict(ckpt["model"], strict=False))
        del ckpt  # in case it occupies memory
    # 按照你的原始代码构建 fusion_module 和 pipeline
    fusion_module = AAFeatureFusionModule(
        stored_feature_file=cfg.fusion.stored_feature_file,
        num_heads=cfg.fusion.num_heads,
        mlp_ratio=cfg.fusion.mlp_ratio,
        dropout=cfg.fusion.dropout,
        map_location=cfg.fusion.map_location,
    )
    pipeline = AAFeatureFusionPipeline(
        base_model=model,
        fusion_module=fusion_module,
        include_intrinsics=cfg.single_view.include_intrinsics,
        include_depth=cfg.single_view.include_depth,
        include_pose=cfg.single_view.include_pose,
        include_scale=cfg.single_view.include_scale,
    )
    import sys
    sys.exit()

    return pipeline