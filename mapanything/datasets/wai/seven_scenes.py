# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""7Scenes Dataset using WAI format data."""

import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from mapanything.datasets.base.base_dataset import BaseDataset, ForcedRandomDataLoader
from mapanything.utils.debugprinter import DebugPrinter
from mapanything.utils.wai.core import load_data, load_frame
printer = DebugPrinter()


class SevenScenesWAI(BaseDataset):
    """
    7Scenes dataset standard WAI implementation, following ETH3DWAI structure.
    Scenes are explicitly split by train/test suffixes (e.g., 'chess_train', 'chess_test') in their folder names.
    """

    def __init__(
            self,
            *args,
            ROOT,
            dataset_metadata_dir,
            split="test",  # 7Scenes通常需要显式指定split
            overfit_num_sets=None,
            sample_specific_scene: bool = False,
            specific_scene_name: str = None,
            sequential_view_mode: bool = False,  # [修复] 显式添加参数，将其从 kwargs 中剥离
            **kwargs,
    ):
        """
        Initialize the dataset attributes.
        Args:
            ROOT: Root directory of the dataset.
            dataset_metadata_dir: Path to the dataset metadata directory.
            split: "train" or "test", determines which subset of scenes to load.
            overfit_num_sets: If None, use all sets. Else, truncate to this number of sets.
            sample_specific_scene: Whether to sample a specific scene.
            specific_scene_name: Name of the specific scene (e.g., "chess_test").
            sequential_view_mode: If True, iterates over all views sequentially instead of sampling scenes.
        """
        # Initialize the dataset attributes
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.dataset_metadata_dir = dataset_metadata_dir
        self.split = split
        self.overfit_num_sets = overfit_num_sets
        self.sample_specific_scene = sample_specific_scene
        self.specific_scene_name = specific_scene_name
        self.sequential_view_mode = sequential_view_mode

        self._load_data()

        # 7Scenes provides metric-scale poses (approximate GT)
        self.is_metric_scale = True
        self.is_synthetic = False

    def _load_data(self):
        "Load the precomputed dataset metadata"
        # Load the dataset metadata corresponding to the split
        # Expects files like: seven_scenes_scene_list_test.npy containing ['chess_test', 'fire_test', ...]
        split_metadata_path = os.path.join(
            self.dataset_metadata_dir,
            self.split,
            f"seven_scenes_scene_list_{self.split}.npy",
        )

        # Relaxed loading: if metadata file is missing, scan the ROOT directory for matching suffixes
        if not os.path.exists(split_metadata_path):
            # Fallback: scan directory if explicit list file is missing
            all_folders = sorted([d for d in os.listdir(self.ROOT) if os.path.isdir(os.path.join(self.ROOT, d))])
            # Filter scenes that match the requested split suffix (e.g., endswith "_test")
            self.scenes = [s for s in all_folders if s.endswith(f"_{self.split}")]
        else:
            split_scene_list = np.load(split_metadata_path, allow_pickle=True)
            self.scenes = list(split_scene_list)

        # Handle specific scene sampling
        if self.sample_specific_scene:
            if self.specific_scene_name not in self.scenes:
                # Optional: Try appending split suffix if user forgot it (e.g. passed "chess" instead of "chess_test")
                potential_name = f"{self.specific_scene_name}_{self.split}"
                if potential_name in self.scenes:
                    self.specific_scene_name = potential_name
            self.scenes = [self.specific_scene_name]

        if self.overfit_num_sets is not None:
            self.scenes = self.scenes[:self.overfit_num_sets]

        if self.sequential_view_mode:
            # Mode: Sequential View Traversal
            # Expand the scenes into a flat list of (scene_name, frame_name)
            self.flat_view_list = []
            self.scene_meta_cache = {}

            print(f"Sequential Mode: Indexing frames from {len(self.scenes)} scenes...")
            for scene_name in self.scenes:
                try:
                    scene_root = os.path.join(self.ROOT, scene_name)
                    # We load meta here to build the full index
                    scene_meta = load_data(
                        os.path.join(scene_root, "scene_meta.json"), "scene_meta"
                    )

                    # Ensure frame_names exists
                    if "frame_names" not in scene_meta:
                        scene_meta["frame_names"] = {f["frame_name"]: i for i, f in enumerate(scene_meta["frames"])}

                    # Sort frames to ensure deterministic order
                    sorted_frames = sorted(list(scene_meta["frame_names"].keys()))

                    for frame_name in sorted_frames:
                        self.flat_view_list.append((scene_name, frame_name))

                    # Cache meta to avoid re-loading in _get_views
                    self.scene_meta_cache[scene_name] = scene_meta

                except Exception as e:
                    print(f"Warning: Failed to load metadata for scene {scene_name}: {e}")

            # Hijack num_of_scenes to represent total views, so __len__ returns total views
            self.num_of_scenes = len(self.flat_view_list)
            print(f"Sequential Mode: Found {self.num_of_scenes} total views.")
        else:
            # Mode: Standard Scene Sampling
            self.num_of_scenes = len(self.scenes)

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        if self.sequential_view_mode:
            # --- Sequential Mode Logic ---
            # sampled_idx corresponds to the index in self.flat_view_list
            scene_name, view_file_name = self.flat_view_list[sampled_idx]

            # Retrieve cached meta
            scene_meta = self.scene_meta_cache[scene_name]
            scene_root = os.path.join(self.ROOT, scene_name)

            # Load the specific frame
            view_data = load_frame(
                scene_root,
                view_file_name,
                modalities=["image", "depth"],
                scene_meta=scene_meta,
            )

            # Process Data (same as below)
            image = view_data["image"].permute(1, 2, 0).numpy()
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

            depthmap = view_data["depth"].numpy().astype(np.float32)
            intrinsics = view_data["intrinsics"].numpy().astype(np.float32)
            c2w_pose = view_data["extrinsics"].numpy().astype(np.float32)

            # Resize
            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image=image,
                resolution=resolution,
                depthmap=depthmap,
                intrinsics=intrinsics,
                additional_quantities=None,
            )

            # Return as a list containing a single view
            # Ignoring num_views_to_sample (or assuming it is 1) for sequential scan
            return [
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=c2w_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    dataset="7Scenes",
                    label=scene_name,
                    instance=os.path.join("images", str(view_file_name)),
                )
            ]

        else:
            # --- Original Random Scene Logic ---
            # Get the scene name of the sampled index (e.g., "chess_test")
            scene_index = sampled_idx
            scene_name = self.scenes[scene_index]

            # Get the metadata corresponding to the scene
            scene_root = os.path.join(self.ROOT, scene_name)
            scene_meta = load_data(
                os.path.join(scene_root, "scene_meta.json"), "scene_meta"
            )

            # Ensure frame_names exists (compatibility for older WAI versions)
            if "frame_names" not in scene_meta:
                scene_meta["frame_names"] = {f["frame_name"]: i for i, f in enumerate(scene_meta["frames"])}

            scene_file_names = list(scene_meta["frame_names"].keys())
            num_views_in_scene = len(scene_file_names)

            # Load the scene pairwise covisibility mmap
            # Strictly following ETH3D's 'v0' structure, with a safe fallback if missing
            covisibility_version_key = "v0"
            covisibility_map_dir = os.path.join(
                scene_root, "covisibility", covisibility_version_key
            )
            pairwise_covisibility = None
            if os.path.exists(covisibility_map_dir):
                try:
                    covisibility_map_name = next(
                        f for f in os.listdir(covisibility_map_dir) if f.endswith(".npy")
                    )
                    covisibility_map_path = os.path.join(covisibility_map_dir, covisibility_map_name)
                    pairwise_covisibility = load_data(covisibility_map_path, "mmap")
                except StopIteration:
                    pass  # Directory exists but empty

            # Fallback to fully connected if no covisibility map is found
            if pairwise_covisibility is None:
                pairwise_covisibility = np.ones((num_views_in_scene, num_views_in_scene), dtype=np.float32)

            # Get the indices of the N views in the scene
            view_indices = self._sample_view_indices(
                num_views_to_sample, num_views_in_scene, pairwise_covisibility
            )

            # Get the views corresponding to the selected view indices
            views = []
            for view_index in view_indices:
                # Load the data corresponding to the view
                view_file_name = scene_file_names[view_index]
                view_data = load_frame(
                    scene_root,
                    view_file_name,
                    modalities=["image", "depth"],
                    scene_meta=scene_meta,
                )

                # Convert necessary data to numpy
                image = view_data["image"].permute(1, 2, 0).numpy()
                # Using clip is safer for 7Scenes data before casting to uint8
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

                depthmap = view_data["depth"].numpy().astype(np.float32)
                intrinsics = view_data["intrinsics"].numpy().astype(np.float32)
                c2w_pose = view_data["extrinsics"].numpy().astype(np.float32)

                # Resize the data to match the desired resolution
                image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    image=image,
                    resolution=resolution,
                    depthmap=depthmap,
                    intrinsics=intrinsics,
                    additional_quantities=None,
                )

                # Append the view dictionary to the list of views
                views.append(
                    dict(
                        img=image,
                        depthmap=depthmap,
                        camera_pose=c2w_pose,  # cam2world
                        camera_intrinsics=intrinsics,
                        dataset="7Scenes",
                        label=scene_name,
                        instance=os.path.join("images", str(view_file_name)),
                    )
                )

            return views


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-rd", "--root_dir", default="/data/xwh/mapanything-dataset/wai_data/7scenes", type=str)
    parser.add_argument(
        "-dmd",
        "--dataset_metadata_dir",
        default="",
        type=str,
        help="Optional metadata directory; omit when scenes are already split by folder name.",
    )
    parser.add_argument("-s", "--split", default="test", type=str)
    parser.add_argument(
        "-nv",
        "--num_of_views",
        default=1,
        type=int,
    )
    # 添加 sequential_mode 参数
    parser.add_argument("--sequential_mode", default=True, help="Enable sequential traversal of all frames.")

    return parser

def check_depth_and_pts3d_with_image(
    depthmap,
    pts3d_world_map,
    intrinsics,
    c2w,
    num_samples=2048,
    label="",
):
    """
    depthmap:      (H, W) numpy 数组，深度 Z（在相机坐标系下）
    pts3d_world_map: (H, W, 3) numpy 数组，每个像素的世界坐标
    intrinsics:    (3, 3) 相机内参 K（与 depth / image 分辨率匹配）
    c2w:           (4, 4) 或 (3, 4) 相机位姿，cam2world
    num_samples:   随机采样的像素数量
    label:         打印标记用
    """

    depthmap = depthmap.astype(np.float64)
    pts3d_world_map = pts3d_world_map.astype(np.float64)
    K = intrinsics.astype(np.float64)

    H, W = depthmap.shape

    # 1) 处理 c2w 形状，变成 4x4
    if c2w.shape == (3, 4):
        c2w_4 = np.eye(4, dtype=np.float64)
        c2w_4[:3, :4] = c2w
        c2w = c2w_4
    elif c2w.shape == (4, 4):
        c2w = c2w.astype(np.float64)
    else:
        raise ValueError(f"[{label}] Unsupported c2w shape: {c2w.shape}")

    # world -> cam
    w2c = np.linalg.inv(c2w)

    # 2) 采样有效像素：depth > 0 且 pts3d 没有 NaN
    valid_mask = (depthmap > 0) & np.isfinite(pts3d_world_map[..., 0]) & np.isfinite(pts3d_world_map[..., 1]) & np.isfinite(pts3d_world_map[..., 2])
    ys, xs = np.where(valid_mask)
    n_valid = len(xs)
    if n_valid == 0:
        print(f"[{label}] 没有有效像素（depth>0 且 pts3d 有效），跳过检查。")
        return

    n_sample = min(num_samples, n_valid)
    idx = np.random.choice(n_valid, size=n_sample, replace=False)

    u_int = xs[idx]  # 列坐标 x
    v_int = ys[idx]  # 行坐标 y

    # 3) 从 pts3d_world_map 取世界坐标
    pts_world = pts3d_world_map[v_int, u_int, :]  # (N,3)

    # 4) world -> cam
    ones = np.ones((n_sample, 1), dtype=np.float64)
    pts_world_h = np.concatenate([pts_world, ones], axis=-1)  # (N,4)
    pts_cam_h = (w2c @ pts_world_h.T).T  # (N,4)
    Xc = pts_cam_h[:, 0]
    Yc = pts_cam_h[:, 1]
    Zc = pts_cam_h[:, 2]

    # 5) 深度一致性：Zc vs depthmap
    depth_from_pts3d = Zc
    depth_sampled = depthmap[v_int, u_int]
    depth_err = depth_from_pts3d - depth_sampled

    # 6) 像素投影一致性：Xc,Yc,Zc -> (u_proj, v_proj)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    Z_safe = Zc + 1e-8
    u_proj = fx * (Xc / Z_safe) + cx
    v_proj = fy * (Yc / Z_safe) + cy

    u = u_int.astype(np.float64)
    v = v_int.astype(np.float64)

    du = u_proj - u
    dv = v_proj - v
    pix_err = np.sqrt(du ** 2 + dv ** 2)

    print(f"[{label}] depth & pts3d_world 与 image 对齐检查:")
    print(f"    有效像素总数: {n_valid}, 抽样: {n_sample}")
    print(f"    像素误差 mean   = {pix_err.mean():.4f}")
    print(f"    像素误差 median = {np.median(pix_err):.4f}")
    print(f"    像素误差 90%%   = {np.quantile(pix_err, 0.9):.4f}")
    print(f"    像素误差 max    = {pix_err.max():.4f}")
    print(f"    深度误差 mean   = {depth_err.mean():.4f}")
    print(f"    深度误差 median = {np.median(depth_err):.4f}")
    print(f"    深度误差 90%%   = {np.quantile(depth_err, 0.9):.4f}")
    print(f"    深度误差 max    = {np.max(np.abs(depth_err)):.4f}")

if __name__ == "__main__":
    import numpy as np
    from tqdm import tqdm
    import cv2
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader
    from PIL import Image

    from mapanything.utils.wai.core import load_frame

    parser = get_parser()
    args = parser.parse_args()

    # ============ 配置 ============
    BATCH_SIZE = 1
    SAVE_DIR = Path("/home/xwh/project/tmp/7scenes_size_debug")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    NUM_SAMPLES = 2  # 只看前几个 batch，方便调试

    # ============ 初始化数据集 ============
    dataset = SevenScenesWAI(
        num_views=args.num_of_views,
        split=args.split,
        covisibility_thres=0.025,
        ROOT=args.root_dir,
        dataset_metadata_dir=args.dataset_metadata_dir,
        sample_specific_scene=True,
        specific_scene_name="chess_test",
        resolution=(518, 392),
        transform="imgnorm",
        data_norm_type="dinov2",
        sequential_view_mode=True,  # 顺序模式
    )

    print(f"Sequential Mode: Indexing frames from {len(dataset.scenes)} scenes...")
    print(f"Sequential Mode: Found {len(dataset.flat_view_list)} total views.")
    print(f"Dataset initialized. Length (num views): {len(dataset)}")
    print(f"Scenes: {dataset.scenes}")
    print(f"Sequential mode: {dataset.sequential_view_mode}")

    num_to_process = min(NUM_SAMPLES, len(dataset))
    print(f"Will process first {num_to_process} views (out of {len(dataset)}).")
    print(f"Images will be saved to: {SAVE_DIR}")

    # ============ 构建 DataLoader ============
    collate_fn = getattr(dataset, "collate_fn", None)
    if collate_fn is not None:
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

    # 一些小工具函数
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def get_depthmap(sample):
        """
        sample['depthmap']: (1, H, W, 1) -> (H, W)
        """
        d = _to_numpy(sample["depthmap"])
        # d: (1, H, W, 1)
        return d[0, :, :, 0]

    def get_pts3d_world(sample):
        """
        sample['pts3d']: (1, H, W, 3) -> (H, W, 3)
        """
        pw = _to_numpy(sample["pts3d"])  # (1, H, W, 3)
        return pw[0]

    def get_intrinsics(sample):
        """
        sample['camera_intrinsics']: (1, 3, 3) -> (3, 3)
        """
        K = _to_numpy(sample["camera_intrinsics"])
        return K[0]

    def get_pose_c2w(sample):
        """
        sample['camera_pose']: (1, 4, 4) -> (4, 4)
        """
        c2w = _to_numpy(sample["camera_pose"])
        return c2w[0]

    def check_pts3d_world_vs_depth_and_image(sample, label=""):
        """
        使用世界坐标系下的 pts3d 验证：
          - world -> cam -> 像素 是否和像素网格对齐
          - cam.z 是否和 depthmap 对齐
        """
        depth = get_depthmap(sample)          # (H, W)
        pts_world = get_pts3d_world(sample)   # (H, W, 3)
        K = get_intrinsics(sample)            # (3, 3)
        c2w = get_pose_c2w(sample)            # (4, 4)

        H, W = depth.shape

        # world -> cam
        w2c = np.linalg.inv(c2w)

        # 只用有深度的像素
        valid = depth > 0
        ys, xs = np.where(valid)
        n_valid = len(xs)
        if n_valid == 0:
            print(f"[{label}] 无有效深度像素，跳过。")
            return

        n_sample = min(2048, n_valid)
        idxs = np.random.choice(n_valid, size=n_sample, replace=False)
        v = ys[idxs]
        u = xs[idxs]

        depth_s = depth[v, u]            # (N,)
        pts_w = pts_world[v, u, :]      # (N, 3)

        # 世界坐标 -> 齐次 -> 相机坐标
        ones = np.ones((n_sample, 1), dtype=np.float64)
        pts_w_h = np.concatenate([pts_w.astype(np.float64), ones], axis=-1)  # (N, 4)
        pts_c_h = (w2c @ pts_w_h.T).T                                        # (N, 4)
        Xc = pts_c_h[:, 0]
        Yc = pts_c_h[:, 1]
        Zc = pts_c_h[:, 2]

        # 深度一致性：Zc vs depthmap
        depth_err = Zc - depth_s

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        Z_safe = Zc + 1e-8

        u_proj = fx * Xc / Z_safe + cx
        v_proj = fy * Yc / Z_safe + cy

        u_f = u.astype(np.float64)
        v_f = v.astype(np.float64)

        du = u_proj - u_f
        dv = v_proj - v_f
        pix_err = np.sqrt(du ** 2 + dv ** 2)

        print(f"[{label}] 世界坐标 pts3d 与 depth & image 对齐检查:")
        print(f"    像素误差 mean   = {pix_err.mean():.4f}")
        print(f"    像素误差 median = {np.median(pix_err):.4f}")
        print(f"    像素误差 90%   = {np.quantile(pix_err, 0.9):.4f}")
        print(f"    像素误差 max    = {pix_err.max():.4f}")
        print(f"    深度误差 mean   = {depth_err.mean():.4f}")
        print(f"    深度误差 median = {np.median(depth_err):.4f}")
        print(f"    深度误差 90%   = {np.quantile(depth_err, 0.9):.4f}")
        print(f"    深度误差 max    = {np.max(np.abs(depth_err)):.4f}")

    # ============ 主循环：用 DataLoader 读取 ============
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Loading Data")):
        if batch_idx >= num_to_process:
            break

        # 你的 debug 已经证明：batch_data 是 list(len=1)
        sample = batch_data[0]
        print("[DEBUG] batch_data[0] keys:", list(sample.keys()))

        # 1) dataloader 输出图像 (normalized)
        img_tensor = sample["img"]  # (1, 3, H, W) = (1,3,392,518)
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError(f"sample['img'] is not Tensor, got {type(img_tensor)}")

        if img_tensor.ndim == 4:
            img_tensor = img_tensor[0]  # -> (3, H, W)

        C, H1, W1 = img_tensor.shape
        print(f"[{batch_idx:04d}] dataloader 图像尺寸: C={C}, H={H1}, W={W1}")

        # 反归一化（简单 min-max）后保存可视化图
        img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
        img_min, img_max = float(img_np.min()), float(img_np.max())
        if img_max > img_min:
            img_np = (img_np - img_min) / (img_max - img_min)
        else:
            img_np = np.zeros_like(img_np)
        img_uint8 = (img_np * 255.0).clip(0, 255).astype(np.uint8)

        scene_name, frame_name = dataset.flat_view_list[batch_idx]
        resized_path = SAVE_DIR / f"{scene_name}_{frame_name}_from_dataloader.jpg"
        Image.fromarray(img_uint8).save(resized_path)

        # 2) 用 load_frame 读取原始 480x640 图像
        scene_root = os.path.join(dataset.ROOT, scene_name)
        scene_meta = dataset.scene_meta_cache[scene_name]
        raw_view = load_frame(
            scene_root,
            frame_name,
            modalities=["image"],
            scene_meta=scene_meta,
        )
        img_orig = raw_view["image"].permute(1, 2, 0).numpy()
        img_orig = np.clip(img_orig * 255.0, 0, 255).astype(np.uint8)
        H0, W0 = img_orig.shape[:2]
        print(f"    原始图像大小: H={H0}, W={W0}")

        orig_path = SAVE_DIR / f"{scene_name}_{frame_name}_orig.jpg"
        Image.fromarray(img_orig).save(orig_path)

        # 3) 世界坐标几何一致性检查
        check_pts3d_world_vs_depth_and_image(
            sample, label=f"{scene_name}_{frame_name}_world"
        )

        print(f"    原始图保存:   {orig_path}")
        print(f"    Dataloader 图保存: {resized_path}")
        print("-" * 60)

    print("Finished sampling via DataLoader.")



