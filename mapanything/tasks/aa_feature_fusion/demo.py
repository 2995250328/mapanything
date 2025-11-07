"""Run a small AA feature fusion reconstruction demo on 7Scenes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import hydra
import torch
from omegaconf import DictConfig

from mapanything.tasks.aa_feature_fusion.builder import build_pipeline_from_cfg
from mapanything.tasks.aa_feature_fusion.common import (
    detach_to_cpu,
    extract_pointmap_like_outputs,
    instantiate_dataset,
    prepare_view,
)
from mapanything.tasks.aa_feature_fusion.pipeline import AAFeatureFusionPipeline


def run_demo(cfg: DictConfig) -> Dict[str, Any]:
    pipeline = build_pipeline_from_cfg(cfg)
    device = torch.device(cfg.demo.device)
    pipeline.base_model.to(device)
    pipeline.base_model.eval()
    pipeline.fusion_module.to(device)
    pipeline.fusion_module.eval()

    if pipeline.base_model.pred_head_type != "linear":
        raise NotImplementedError(
            "The demo currently supports models with linear dense prediction heads."
        )

    dataset = instantiate_dataset(cfg.dataset.dataset_str)
    output_dir = Path(cfg.demo.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    indices: Iterable[int]
    if cfg.demo.sample_indices is not None:
        indices = cfg.demo.sample_indices
    else:
        indices = range(len(dataset))

    saved_items: List[Dict[str, Any]] = []
    processed = 0

    for idx in indices:
        if processed >= cfg.demo.num_samples:
            break
        views = dataset[idx]
        if not views:
            continue
        single_view = views[0]

        prepared = prepare_view(
            single_view,
            device=device,
            include_intrinsics=pipeline.include_intrinsics,
            include_depth=pipeline.include_depth,
            include_pose=pipeline.include_pose,
            include_scale=pipeline.include_scale,
        )

        with torch.no_grad():
            fused_tokens = pipeline.fuse(prepared)
            feature_map = pipeline.fusion_module.tokens_to_feature_map(fused_tokens)
            scale_token = pipeline.fusion_module.get_additional_token(
                device=device, dtype=feature_map.dtype
            )
            if scale_token is None:
                scale_token = torch.zeros(
                    feature_map.shape[0], feature_map.shape[1], 1, device=device, dtype=feature_map.dtype
                )
            dense_outputs, pose_outputs, scale_output = pipeline.base_model.downstream_head(
                dense_head_inputs=feature_map,
                scale_head_inputs=scale_token,
                img_shape=feature_map.shape[-2:],
                memory_efficient_inference=cfg.demo.memory_efficient_inference,
            )

        reconstruction = extract_pointmap_like_outputs(
            pipeline, dense_outputs, pose_outputs, scale_output
        )

        sample_name = single_view.get("instance", f"sample_{idx:06d}")
        sample_name = str(sample_name).replace("/", "_")
        output_path = output_dir / f"{sample_name}.pt"

        record = {
            "dataset": single_view.get("dataset"),
            "label": single_view.get("label"),
            "instance": single_view.get("instance"),
            "scene_rep_type": pipeline.base_model.scene_rep_type,
            "rgb": single_view["img"].detach().cpu(),
            "intrinsics": torch.as_tensor(single_view["camera_intrinsics"]).cpu(),
            "camera_pose": torch.as_tensor(single_view["camera_pose"]).cpu(),
            "reconstruction": detach_to_cpu(reconstruction),
        }
        torch.save(record, output_path)

        saved_items.append(
            {
                "index": int(idx),
                "output": str(output_path),
                "scene": record["label"],
                "instance": record["instance"],
            }
        )
        processed += 1

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(saved_items, indent=2, ensure_ascii=False))
    return {"saved": saved_items, "output_dir": str(output_dir)}


@hydra.main(
    version_base=None,
    config_path="../../../configs/tasks/aa_feature_fusion",
    config_name="demo",
)
def main(cfg: DictConfig) -> None:
    info = run_demo(cfg)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
