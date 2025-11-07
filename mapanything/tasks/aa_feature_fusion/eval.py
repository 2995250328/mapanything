"""Skeleton evaluation loop for alternating-attention feature fusion."""

from __future__ import annotations

import logging
from typing import Any

import hydra
from omegaconf import DictConfig

from .builder import build_pipeline_from_cfg


def run_evaluation(cfg: DictConfig) -> Any:
    log = logging.getLogger(__name__)
    pipeline = build_pipeline_from_cfg(cfg)
    log.info("Initialized AA feature fusion pipeline for evaluation.")
    log.info("Dataset config: %s", cfg.dataset)
    log.info("Evaluation parameters: %s", cfg.evaluation)
    log.info(
        "Stored AA memory loaded: %s",
        getattr(pipeline.fusion_module, "has_memory", False),
    )
    # Placeholder for evaluation loop.
    return {"status": "evaluation_loop_not_implemented"}


@hydra.main(
    version_base=None,
    config_path="../../../configs/tasks/aa_feature_fusion",
    config_name="test",
)
def main(cfg: DictConfig) -> None:
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
