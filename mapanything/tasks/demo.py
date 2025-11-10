import hydra
import torch
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import mapanything.models
from omegaconf import OmegaConf

def _resolve_special_float(x: str):
    # 兼容 inf / -inf / nan 以及普通数值写法（1e-4、0.001 等）
    sx = str(x).strip()
    if sx.lower() in ("inf", "+inf"):
        return float("inf")
    if sx.lower() in ("-inf",):
        return float("-inf")
    if sx.lower() in ("nan",):
        return float("nan")
    return float(sx)

# 只有未注册时才注册，避免重复注册异常
if OmegaConf._get_resolver("special_float") is None:
    OmegaConf.register_new_resolver("special_float", _resolve_special_float, replace=True)

from mapanything.tasks.aa_feature_fusion.builder import build_pipeline_from_cfg
from mapanything.tasks.aa_feature_fusion.common import (
    detach_to_cpu,
    extract_pointmap_like_outputs,
    instantiate_dataset,
    prepare_view,
)

def run_demo(cfg: DictConfig):
    print(f"[Demo] Feature File: {cfg.fusion.stored_feature_file}")

    # 2. 构建管道
    pipeline = build_pipeline_from_cfg(cfg)
    device = torch.device(cfg.demo.device)
    pipeline.base_model.to(device).eval()
    pipeline.fusion_module.to(device).eval()

    # 3. 加载数据
    dataset = instantiate_dataset(cfg)
    output_dir = Path(cfg.demo.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. 确定要处理的样本
    indices = cfg.demo.sample_indices if cfg.demo.sample_indices is not None else range(len(dataset))

    saved_items = []
    processed_count = 0

    # 5. 主循环
    for idx in tqdm(indices, desc="Running Reconstruction"):
        if processed_count >= cfg.demo.num_samples:
            break

        views = dataset[idx]
        if not views: continue
        single_view = views[0]

        # 准备数据
        prepared = prepare_view(
            single_view,
            device=device,
            include_intrinsics=pipeline.include_intrinsics,
            include_depth=pipeline.include_depth,
            include_pose=pipeline.include_pose,
            include_scale=pipeline.include_scale
        )

        # 推理
        with torch.no_grad():
            fused_tokens = pipeline.fuse(prepared)
            feature_map = pipeline.fusion_module.tokens_to_feature_map(fused_tokens)

            # 获取或创建 scale token
            scale_token = pipeline.fusion_module.get_additional_token(device=device, dtype=feature_map.dtype)
            if scale_token is None:
                scale_token = torch.zeros(feature_map.shape[0], feature_map.shape[1], 1, device=device, dtype=feature_map.dtype)

            dense, pose, scale = pipeline.base_model.downstream_head(
                dense_head_inputs=feature_map,
                scale_head_inputs=scale_token,
                img_shape=feature_map.shape[-2:],
                memory_efficient_inference=cfg.memory_efficient_inference,
            )

        # 保存结果
        reconstruction = extract_pointmap_like_outputs(pipeline, dense, pose, scale)
        sample_name = str(single_view.get("instance", f"sample_{idx:06d}")).replace("/", "_")
        output_path = output_dir / f"{sample_name}.pt"

        torch.save({
            "rgb": single_view["img"].detach().cpu(),
            "reconstruction": detach_to_cpu(reconstruction),
            "label": single_view.get("label"),
        }, output_path)

        saved_items.append({"index": int(idx), "output": str(output_path)})
        processed_count += 1

    # 6. 完成
    (output_dir / "summary.json").write_text(json.dumps(saved_items, indent=2))
    return {"output_dir": str(output_dir), "count": len(saved_items)}

@hydra.main(
    version_base=None, config_path="../../configs", config_name="memory"
)
def main(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    run_demo(cfg)

if __name__ == "__main__":
    main()