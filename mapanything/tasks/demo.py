import hydra
import torch
import json
from pathlib import Path
from typing import List, Dict, Any
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from mapanything.datasets import SevenScenesWAI
from mapanything.datasets.base.base_dataset import ForcedRandomDataLoader
# 模型初始化
from mapanything.models import init_model

# 复用你现有的通用工具
from mapanything.tasks.aa_feature_fusion.common import (
    detach_to_cpu,
    extract_pointmap_like_outputs,   # 仍可用于把 dense/pose/scale 打包成点图等结构；若 forward_with_memory 已返回最终结构，可不调用
    instantiate_dataset,
    prepare_view,
)

# 兼容 inf / nan 解析
def _resolve_special_float(x: str):
    sx = str(x).strip()
    if sx.lower() in ("inf", "+inf"): return float("inf")
    if sx.lower() in ("-inf",): return float("-inf")
    if sx.lower() in ("nan",): return float("nan")
    return float(sx)
if OmegaConf._get_resolver("special_float") is None:
    OmegaConf.register_new_resolver("special_float", _resolve_special_float, replace=True)

def load_memory_features(path: str, device: torch.device):
    """
    加载 Memory 文件，返回：
      - memory_feats: 所有 blocks（intermediate + final）的 multi-view features
        结构: List[List[Tensor(B,C,H,W)]]
      - memory_scale_token: 仅 final block 的 additional_token_features (B,C,1)
    """
    payload = torch.load(path, map_location=device)

    memory_feats = []            # List[List[Tensor]]
    memory_scale_token = None    # Tensor(B,C,1)
    # ---- (1) 读取 intermediate blocks 的 features ----
    interm_blocks = payload.get("intermediate", None)
    if interm_blocks is not None:
        for block in interm_blocks:
            feats = [t.to(device) for t in block["features"]]     # List[(B,C,H,W)]
            memory_feats.append(feats)
    # ---- (2) 加上 final block 的 features ----
    final_block = payload.get("final", None)
    if final_block is not None:
        final_feats = [t.to(device) for t in final_block["features"]]
        memory_feats.append(final_feats)   # ✅ 放在最后
    # ---- (3) scale token 只使用 final block 的 additional_token_features ----
    if final_block is not None and final_block["additional_token_features"] is not None:
        memory_scale_token = final_block["additional_token_features"].to(device)
    if memory_scale_token is None:
        raise ValueError("❌ Memory file has no final additional_token_features (scale token).")

    return memory_feats, memory_scale_token

# ============================================================
# 主流程：不再使用 pipeline/AAFeatureFusionModule
# 直接：init_model -> load pretrained -> dataset -> prepare_view ->
#       model.forward_with_memory(query, memory_features, ...)
# ============================================================
def run_demo(cfg: DictConfig):
    device = "cuda"
    # 1) 构建模型
    model = init_model(
        cfg.model.model_str,
        cfg.model.model_config,
        torch_hub_force_reload=False
    )
    model.to(device).eval()
    # 2) 加载预训练
    if cfg.model.pretrained:
        print("[Demo] Loading pretrained:", cfg.model.pretrained)
        ckpt = torch.load(cfg.model.pretrained, map_location=device, weights_only=False)
        print(model.load_state_dict(ckpt["model"], strict=False))
        del ckpt

    # 3) 读取 AA 中间特征（记忆库）
    print(f"[Demo] Loading AA memory from: {cfg.fusion.stored_feature_file}")
    memory_feats, memory_scale_token = load_memory_features(cfg.fusion.stored_feature_file, device)
    print(f"[Demo] Loaded memory blocks: {len(memory_feats)}")
    print(memory_scale_token.size())
    import sys
    sys.exit()
    # 4) 数据集
    # dataset = instantiate_dataset(cfg)
    dataset = SevenScenesWAI(
        num_views=cfg.dataset.num_views,
        split="test",
        covisibility_thres=0.025,
        ROOT="/mnt/storage/xwh/mapanything-dataset/wai_data/7scenes",
        dataset_metadata_dir="/mnt/storage/xwh/map-anything/mapanything_dataset_metadata",
        sample_specific_scene=True,
        specific_scene_name='chess_test',
        resolution=(518, 392),
        transform="imgnorm",
        data_norm_type="dinov2",
        seed=777
    )
    dataloader = ForcedRandomDataLoader(
        dataset=dataset,
        batch_size=1  # 你想要的 batch size
    )

    # 5) 输出目录
    output_dir = Path(cfg.demo.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_items: List[Dict[str, Any]] = []

    # 6) 推理循环（单视图查询）
    for batch_id, views in tqdm(enumerate(dataloader), desc="Running Memory-Augmented Reconstruction"):

        # views 是 batch 列表（因为 batch_size=1，所以取 views[0]）
        single_view = views[0]
        # 准备模型输入 (和你之前一样)
        prepared = prepare_view(
            single_view,
            device=device,
            include_intrinsics=cfg.single_view.include_intrinsics,
            include_depth=cfg.single_view.include_depth,
            include_pose=cfg.single_view.include_pose,
            include_scale=cfg.single_view.include_scale,
        )
        # --------------------------
        # 6. Memory-Augmented 推理部分
        # --------------------------
        with torch.no_grad():
            # (1) 提取查询视图特征
            query_features = model.extract_features(prepared)

            # (2) 载入 memory features（你之前保存的中间特征文件）
            memory_features = torch.load(cfg.fusion.stored_feature_file, map_location=device)

            with torch.no_grad():
                result = model.forward_with_memory(
                    views=prepared,
                    memory_feats=memory_feats,
                    memory_keep_ratio=cfg.fusion.memory_keep_ratio,
                    memory_efficient_inference=cfg.memory_efficient_inference,
                )

            reconstruction = result[0]

        save_path = output_dir / f"memory_sample_{batch_id:06d}.pt"
        torch.save({
            "rgb": single_view["img"].cpu(),
            "reconstruction": reconstruction,
        }, save_path)

        saved_items.append({"index": batch_id, "output": str(save_path)})

        # 达到目标样本数量自动停止
        if cfg.demo.num_samples > 0 and batch_id + 1 >= cfg.demo.num_samples:
            break

        # 8. 生成 summary
    (output_dir / "summary.json").write_text(json.dumps(saved_items, indent=2))
    return {"output_dir": str(output_dir), "count": len(saved_items)}


@hydra.main(version_base=None, config_path="../../configs", config_name="memory")
def main(cfg: DictConfig):
    # 保留 structured round-trip，与你现有用法一致
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    run_demo(cfg)


if __name__ == "__main__":
    main()
