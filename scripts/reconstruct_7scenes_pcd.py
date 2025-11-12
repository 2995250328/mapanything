#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MapAnything 7Scenes Reconstruction Script (using ForcedRandomDataLoader)
功能：
1. 使用自定义 ForcedRandomDataLoader 替代标准 Loader，强制从单场景中提取多样化数据。
2. 支持通过 --num_batches 参数控制循环次数（重建 1 次或多次用于测试）。
3. 执行推理、点云融合变换至 View0、体素下采样并保存 PLY。

Usage:
python scripts/reconstruct_7scenes_loader.py \
    --scene chess \
    --num_views 24 \
    --num_batches 1 \
    --ckpt /path/to/mapanything_model.pth \
    --data_root /path/to/wai_data/7scenes \
    --voxel_size 0.05
"""

import sys
import os
import argparse
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到 sys.path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapanything.datasets.wai.seven_scenes import SevenScenesWAI
from mapanything.models import init_model
from mapanything.utils.geometry import geotrf, inv

# --- 按照你的要求，从 base_dataset 导入自定义 Loader ---
# (请确保你已完成了步骤 1，将该类添加到了 mapanything/datasets/base/base_dataset.py)
try:
    from mapanything.datasets.base.base_dataset import ForcedRandomDataLoader
except ImportError:
    print("[ERROR] 无法导入 ForcedRandomDataLoader。请确保你已将其添加到 mapanything/datasets/base/base_dataset.py 中。")
    exit(1)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_and_clean_point_cloud(batch_data, preds, voxel_size=0.02):
    """融合多视角点云到 View 0 坐标系并下采样"""
    n_views = len(batch_data)
    # 获取 View 0 的 world-to-camera 变换矩阵 (B=1)
    pose0_c2w = batch_data[0]["camera_pose"][0]
    pose0_w2c = inv(pose0_c2w.unsqueeze(0))[0]

    all_points, all_colors = [], []
    for i in range(n_views):
        pts_world = preds[i]["pts3d"][0]  # (H, W, 3)
        img = batch_data[i]["img"][0].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)

        # 变换到 View 0
        pts_view0 = geotrf(pose0_w2c, pts_world)
        all_points.append(pts_view0.cpu().numpy().reshape(-1, 3))
        all_colors.append(img.reshape(-1, 3))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_points, axis=0).astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(all_colors, axis=0).astype(np.float64))

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd


def main(args):
    device = get_device()
    print(f"[INFO] Device: {device}")

    # 1. 初始化数据集
    # 关键：seed=None 确保每次调用 __getitem__ 都能拿到随机视图
    dataset = SevenScenesWAI(
        num_views=args.num_views,
        split="test",
        covisibility_thres=0.0,
        ROOT=args.data_root,
        sample_specific_scene=True,
        specific_scene_name=args.scene,
        resolution=(518, 392),
        transform="imgnorm",
        data_norm_type="dinov2",
        seed=None,
    )

    # 2. 使用自定义 Loader 替代标准 DataLoader
    loader = ForcedRandomDataLoader(
        dataset=dataset,
        batch_size=1,  # 推理时通常 BS=1
        num_batches=args.num_batches  # 决定了循环会运行多少次
    )
    print(f"[INFO] Loader prepared. Will yield {len(loader)} batches from scene '{args.scene}'.")

    # 3. 初始化模型
    model = init_model(model_str="mapanything", model_config="images_only", torch_hub_force_reload=False)
    state_dict = torch.load(args.ckpt, map_location='cpu')
    if "model" in state_dict: state_dict = state_dict["model"]
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)
    model.to(device).eval()

    # 4. 主循环 (现在看起来和标准 PyTorch 循环一样了)
    os.makedirs(args.out_dir, exist_ok=True)

    for i, batch_data in enumerate(tqdm(loader, desc="Inference Loop")):
        # --- 数据移动到 GPU ---
        ignore_keys = set(["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng", "data_norm_type"])
        views_on_device = []
        for view_data in batch_data:
            view_device = {}
            for k, v in view_data.items():
                if k in ignore_keys:
                    view_device[k] = v
                elif isinstance(v, torch.Tensor):
                    view_device[k] = v.to(device, non_blocking=True)
                else:
                    view_device[k] = v
            views_on_device.append(view_device)

        # --- 推理与重建 ---
        with torch.no_grad():
            preds = model(views_on_device, memory_efficient_inference=True)

        pcd = build_and_clean_point_cloud(views_on_device, preds, voxel_size=args.voxel_size)

        save_path = os.path.join(args.out_dir, f"{args.scene}_batch{i:03d}.ply")
        o3d.io.write_point_cloud(save_path, pcd)
        # print(f"Saved batch {i} to {save_path}")

    print(f"[SUCCESS] All {args.num_batches} batches processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="chess")
    parser.add_argument("--num_views", type=int, default=24)
    parser.add_argument("--num_batches", type=int, default=1,
                        help="Number of times to repeat extraction (useful for future training loops)")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./output_loader_recon")
    parser.add_argument("--voxel_size", type=float, default=0.05)
    args = parser.parse_args()
    main(args)