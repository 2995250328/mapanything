# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""7Scenes Dataset using WAI format data."""

import os
import torch
import numpy as np
from PIL import Image

from mapanything.datasets.base.base_dataset import BaseDataset, ForcedRandomDataLoader
from mapanything.utils.debugprinter import DebugPrinter
from mapanything.utils.wai.core import load_data, load_frame

printer = DebugPrinter()

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""7Scenes Dataset using WAI format data."""

import os
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
            split="test",  # 7Scenes usually requires explicit split
            overfit_num_sets=None,
            sample_specific_scene: bool = False,
            specific_scene_name: str = None,
            sequential_view_mode: bool = False,  # Explicitly add parameter to strip from kwargs
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

        # 1. Load scene list and metadata structure
        self._load_data()

        # 7Scenes provides metric-scale poses (approximate GT)
        self.is_metric_scale = True
        self.is_synthetic = False

        # 2. Compute mean camera center (Calculated over all frames in the selected scenes)
        # [New Feature] Added to match CamLocDataset behavior
        self.mean_cam_center = self._compute_mean_camera_center()
        # print(f"Computed mean camera center for {self.scenes}: {self.mean_cam_center}")

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

        # Pre-cache metadata for sequential mode or just to speed up things
        self.scene_meta_cache = {}

        if self.sequential_view_mode:
            # Mode: Sequential View Traversal
            # Expand the scenes into a flat list of (scene_name, frame_name)
            self.flat_view_list = []

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

    def _compute_mean_camera_center(self):
        """
        Computes the mean translation (camera center) of all frames in the dataset.
        Iterates through the loaded scene metadata to extract poses from 'transform_matrix'.
        Returns:
            torch.Tensor: (3,) mean translation vector.
        """
        print(f"Computing mean camera center for {len(self.scenes)} scenes: {self.scenes}...")

        total_translation = np.zeros(3, dtype=np.float64)
        count = 0

        # 遍历当前 split 下的所有场景
        for scene_name in self.scenes:
            # 1. 获取 scene_meta
            # 如果在 _load_data 中已经缓存了 meta (Sequential 模式)，直接使用
            if hasattr(self, "scene_meta_cache") and scene_name in self.scene_meta_cache:
                scene_meta = self.scene_meta_cache[scene_name]
            else:
                # 否则从磁盘加载
                try:
                    scene_root = os.path.join(self.ROOT, scene_name)
                    scene_meta_path = os.path.join(scene_root, "scene_meta.json")
                    if not os.path.exists(scene_meta_path):
                        print(f"[Warning] Metadata not found: {scene_meta_path}")
                        continue
                    scene_meta = load_data(scene_meta_path, "scene_meta")

                    # 顺便缓存下来，避免后续 _get_views 再次加载
                    if not hasattr(self, "scene_meta_cache"):
                        self.scene_meta_cache = {}
                    self.scene_meta_cache[scene_name] = scene_meta
                except Exception as e:
                    print(f"[Error] Failed to load metadata for {scene_name}: {e}")
                    continue

            # 2. 遍历该场景下的所有帧
            frames = scene_meta.get("frames", [])
            for frame in frames:
                # 获取位姿矩阵
                # 根据提供的 json 示例，key 为 "transform_matrix"
                pose_matrix = frame.get("transform_matrix")

                # 兼容性处理：如果没找到 transform_matrix，尝试找 extrinsics (标准 WAI 格式)
                if pose_matrix is None:
                    pose_matrix = frame.get("extrinsics")

                if pose_matrix is not None:
                    try:
                        c2w = np.array(pose_matrix, dtype=np.float64)
                        # 确保是 4x4 矩阵
                        if c2w.shape == (4, 4):
                            # 累加平移部分 (前三行的最后一列)
                            total_translation += c2w[:3, 3]
                            count += 1
                    except Exception:
                        pass

        if count == 0:
            print("Warning: Failed to compute mean camera center (no valid poses found). Defaulting to [0, 0, 0].")
            return torch.zeros(3, dtype=torch.float32)

        # 计算均值
        mean_center = total_translation / count
        mean_tensor = torch.from_numpy(mean_center).float()
        print(f"Computed Mean Camera Center (from {count} frames): {mean_tensor.tolist()}")

        return mean_tensor

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        if self.sequential_view_mode:
            # --- Sequential Mode Logic ---
            # sampled_idx corresponds to the index in self.flat_view_list (0 to total_views-1)
            scene_name, target_view_file_name = self.flat_view_list[sampled_idx]

            # Retrieve cached meta
            scene_meta = self.scene_meta_cache[scene_name]
            scene_root = os.path.join(self.ROOT, scene_name)

            # 1. Identify which frames to load
            sorted_frame_names = sorted(list(scene_meta["frame_names"].keys()))
            frame_name_to_idx = {name: i for i, name in enumerate(sorted_frame_names)}
            target_local_idx = frame_name_to_idx[target_view_file_name]

            # Initialize list with the target frame
            selected_local_indices = [target_local_idx]

            if num_views_to_sample > 1:
                # Try to load covisibility
                covisibility_version_key = "v0"
                covisibility_map_dir = os.path.join(scene_root, "covisibility", covisibility_version_key)
                pairwise_covisibility = None

                if os.path.exists(covisibility_map_dir):
                    try:
                        covisibility_map_name = next(f for f in os.listdir(covisibility_map_dir) if f.endswith(".npy"))
                        covisibility_map_path = os.path.join(covisibility_map_dir, covisibility_map_name)
                        pairwise_covisibility = load_data(covisibility_map_path, "mmap")
                    except Exception:
                        pass

                if pairwise_covisibility is not None and target_local_idx < pairwise_covisibility.shape[0]:
                    # Strategy: Best Covisibility
                    covis_row = pairwise_covisibility[target_local_idx]
                    # Get indices sorted by covisibility descending
                    sorted_neighbors = np.argsort(covis_row)[::-1]

                    count_added = 0
                    for nb_idx in sorted_neighbors:
                        if nb_idx != target_local_idx:
                            selected_local_indices.append(nb_idx)
                            count_added += 1
                        if count_added >= (num_views_to_sample - 1):
                            break
                else:
                    # Strategy: Temporal Fallback (Next frames)
                    for i in range(1, num_views_to_sample):
                        next_idx = (target_local_idx + i) % len(sorted_frame_names)
                        selected_local_indices.append(next_idx)

                # Pad if still not enough (e.g. scene smaller than N)
                while len(selected_local_indices) < num_views_to_sample:
                    selected_local_indices.append(selected_local_indices[0])

            # 2. Load the views
            views = []
            for local_idx in selected_local_indices:
                view_file_name = sorted_frame_names[local_idx]
                view_data = load_frame(
                    scene_root,
                    view_file_name,
                    modalities=["image", "depth"],
                    scene_meta=scene_meta,
                )

                # Process Data
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

        else:
            # --- Original Random Scene Logic ---
            # Get the scene name of the sampled index (e.g., "chess_test")
            scene_index = sampled_idx
            scene_name = self.scenes[scene_index]

            # Get the metadata corresponding to the scene
            scene_root = os.path.join(self.ROOT, scene_name)

            # Use cached meta if possible (though random mode logic typically reloads)
            if hasattr(self, 'scene_meta_cache') and scene_name in self.scene_meta_cache:
                scene_meta = self.scene_meta_cache[scene_name]
            else:
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
        default=4,
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
    Helper function for debug checks (from original snippet).
    """
    depthmap = depthmap.astype(np.float64)
    pts3d_world_map = pts3d_world_map.astype(np.float64)
    K = intrinsics.astype(np.float64)

    H, W = depthmap.shape

    if c2w.shape == (3, 4):
        c2w_4 = np.eye(4, dtype=np.float64)
        c2w_4[:3, :4] = c2w
        c2w = c2w_4
    elif c2w.shape == (4, 4):
        c2w = c2w.astype(np.float64)
    else:
        raise ValueError(f"[{label}] Unsupported c2w shape: {c2w.shape}")

    w2c = np.linalg.inv(c2w)

    valid_mask = (depthmap > 0) & np.isfinite(pts3d_world_map[..., 0]) & np.isfinite(
        pts3d_world_map[..., 1]) & np.isfinite(pts3d_world_map[..., 2])
    ys, xs = np.where(valid_mask)
    n_valid = len(xs)
    if n_valid == 0:
        print(f"[{label}] No valid pixels (depth>0 and pts3d valid), skipping check.")
        return

    n_sample = min(num_samples, n_valid)
    idx = np.random.choice(n_valid, size=n_sample, replace=False)

    u_int = xs[idx]
    v_int = ys[idx]

    pts_world = pts3d_world_map[v_int, u_int, :]
    ones = np.ones((n_sample, 1), dtype=np.float64)
    pts_world_h = np.concatenate([pts_world, ones], axis=-1)
    pts_cam_h = (w2c @ pts_world_h.T).T
    Xc = pts_cam_h[:, 0]
    Yc = pts_cam_h[:, 1]
    Zc = pts_cam_h[:, 2]

    depth_from_pts3d = Zc
    depth_sampled = depthmap[v_int, u_int]
    depth_err = depth_from_pts3d - depth_sampled

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

    print(f"[{label}] depth & pts3d_world vs image alignment check:")
    print(f"    Valid pixels: {n_valid}, Sampled: {n_sample}")
    print(f"    Pixel error mean   = {pix_err.mean():.4f}")
    print(f"    Depth error mean   = {depth_err.mean():.4f}")


if __name__ == "__main__":
    import numpy as np
    from pathlib import Path
    import torch
    from torch.utils.data import DataLoader
    from PIL import Image
    import os

    parser = get_parser()
    args = parser.parse_args()

    # ================== 通用配置 ==================
    BASE_SAVE_DIR = Path("/home/xwh/project/tmp")
    SEQ_SAVE_DIR = BASE_SAVE_DIR / "7scenes_seq_debug"
    RAND_SAVE_DIR = BASE_SAVE_DIR / "7scenes_random_debug"
    SEQ_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    RAND_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    NUM_SEQ_TO_SAVE = 10   # 顺序模式：最多保存多少张（均匀抽样）
    NUM_RAND_TO_SAVE = 10  # 随机模式：保存多少张

    def tensor_to_img_uint8(img_tensor: torch.Tensor) -> np.ndarray:
        """
        通用 Tensor -> uint8 RGB，可处理:
          - (B, V, C, H, W)
          - (B, C, H, W)
          - (C, H, W)
        自动去掉前面的 batch/视图维度。
        """
        while img_tensor.ndim > 3:
            img_tensor = img_tensor[0]

        assert img_tensor.ndim == 3, f"Expect img tensor with 3 dims, got {img_tensor.shape}"
        img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()

        img_min = float(img_np.min())
        img_max = float(img_np.max())
        if img_max > img_min:
            img_np = (img_np - img_min) / (img_max - img_min)
        else:
            img_np = np.zeros_like(img_np)

        img_uint8 = (img_np * 255.0).clip(0, 255).astype(np.uint8)
        return img_uint8

    def extract_first_sample(item_or_batch):
        """
        统一从 dataset item 或 DataLoader batch 中取一个 sample(dict):

          - 如果是 dict（已经是单个 sample），直接返回
          - 如果是 list/tuple（多视图或 batch 列表），取第 0 个
        不再对每个字段做 v[0] 这种操作，避免误删维度。
        """
        if isinstance(item_or_batch, dict):
            return item_or_batch
        elif isinstance(item_or_batch, (list, tuple)):
            if len(item_or_batch) == 0:
                raise ValueError("Empty list/tuple passed to extract_first_sample")
            return item_or_batch[0]
        else:
            raise TypeError(f"Unsupported type for extract_first_sample: {type(item_or_batch)}")

    def normalize_to_str(x, fallback: str) -> str:
        """
        把 str / list[str] / tuple[str] / None 统一成 str。
        """
        if isinstance(x, (list, tuple)):
            if len(x) > 0:
                return str(x[0])
            else:
                return fallback
        elif x is None:
            return fallback
        else:
            return str(x)

    # # =========================================================
    # # [Test 1A] Sequential 模式：直接 dataset[idx]，均匀抽取远间隔视图
    # # =========================================================
    # print("========== [Test 1A] Sequential mode via direct dataset indexing ==========")
    # dataset_seq = SevenScenesWAI(
    #     num_views=args.num_of_views,
    #     split=args.split,
    #     covisibility_thres=0.025,
    #     ROOT=args.root_dir,
    #     dataset_metadata_dir=args.dataset_metadata_dir,
    #     sample_specific_scene=True,
    #     specific_scene_name="chess_test",
    #     resolution=(518, 392),
    #     transform="imgnorm",
    #     data_norm_type="dinov2",
    #     sequential_view_mode=True,   # ★ 开启顺序模式 ★
    # )
    #
    # print(f"[Seq-ds] scenes: {dataset_seq.scenes}")
    # total_views = len(dataset_seq)
    # print(f"[Seq-ds] total views (len(dataset)): {total_views}")
    #
    # # 在 [0, total_views-1] 范围内均匀抽 NUM_SEQ_TO_SAVE 个 index
    # num_to_sample = min(NUM_SEQ_TO_SAVE, total_views)
    # if num_to_sample <= 0:
    #     raise RuntimeError("No views in dataset to sample.")
    #
    # stride = max(1, total_views // num_to_sample)
    # indices_to_sample = []
    # for i in range(num_to_sample):
    #     idx = i * stride
    #     if idx >= total_views:
    #         idx = total_views - 1
    #     indices_to_sample.append(idx)
    #
    # print(f"[Seq] indices_to_sample (far apart): {indices_to_sample}")
    #
    # # 直接 dataset[idx]
    # for idx in indices_to_sample:
    #     raw_item = dataset_seq[idx]           # 可能是 [view_dict] 或 dict
    #     sample = extract_first_sample(raw_item)
    #
    #     scene_raw = sample.get("label", "unknown_scene")
    #     scene = normalize_to_str(scene_raw, "unknown_scene")
    #     instance_raw = sample.get("instance", f"idx_{idx}")
    #     instance = normalize_to_str(instance_raw, f"idx_{idx}")
    #     frame_tag = os.path.basename(instance)
    #
    #     img_tensor = sample["img"]
    #     img_uint8 = tensor_to_img_uint8(img_tensor)
    #     mean_val = float(img_tensor.mean().item())
    #
    #     save_path = SEQ_SAVE_DIR / f"ds_{idx:04d}_{scene}_{frame_tag}_seq.jpg"
    #     Image.fromarray(img_uint8).save(save_path)
    #
    #     print(
    #         f"[Seq-ds] idx={idx:04d}, scene={scene}, instance={instance}, "
    #         f"mean={mean_val:.6f}, saved={save_path}"
    #     )
    #
    # print(f"[Seq-ds] Done. Saved {len(indices_to_sample)} images to {SEQ_SAVE_DIR}")
    #
    # # =========================================================
    # # [Test 1B] Sequential 模式：DataLoader(shuffle=False)，同样抽这些 index
    # # =========================================================
    # print("========== [Test 1B] Sequential mode via DataLoader ==========")
    #
    # collate_fn_seq = getattr(dataset_seq, "collate_fn", None)
    # if collate_fn_seq is not None:
    #     loader_seq = DataLoader(
    #         dataset_seq,
    #         batch_size=1,
    #         shuffle=False,           # 顺序遍历
    #         num_workers=0,
    #         collate_fn=collate_fn_seq,
    #     )
    # else:
    #     loader_seq = DataLoader(
    #         dataset_seq,
    #         batch_size=1,
    #         shuffle=False,
    #         num_workers=0,
    #     )
    #
    # indices_to_sample_set = set(indices_to_sample)
    # for idx, batch in enumerate(loader_seq):
    #     if idx not in indices_to_sample_set:
    #         continue
    #
    #     sample = extract_first_sample(batch)  # 兼容 dict / [dict]
    #
    #     scene_raw = sample.get("label", "unknown_scene")
    #     scene = normalize_to_str(scene_raw, "unknown_scene")
    #     instance_raw = sample.get("instance", f"idx_{idx}")
    #     instance = normalize_to_str(instance_raw, f"idx_{idx}")
    #     frame_tag = os.path.basename(instance)
    #
    #     img_tensor = sample["img"]
    #     img_uint8 = tensor_to_img_uint8(img_tensor)
    #     mean_val = float(img_tensor.mean().item())
    #
    #     save_path = SEQ_SAVE_DIR / f"dl_{idx:04d}_{scene}_{frame_tag}_seq.jpg"
    #     Image.fromarray(img_uint8).save(save_path)
    #
    #     print(
    #         f"[Seq-dl] idx={idx:04d}, scene={scene}, instance={instance}, "
    #         f"mean={mean_val:.6f}, saved={save_path}"
    #     )
    #
    # print(f"[Seq-dl] Done. Saved {len(indices_to_sample)} images to {SEQ_SAVE_DIR}")

    # =========================================================
    # [Test 2] 非 Sequential 模式：ForcedRandomDataLoader 随机采样
    # =========================================================
    print("========== [Test 2] Random (non-sequential) mode ==========")
    dataset_rand = SevenScenesWAI(
        num_views=args.num_of_views,
        split=args.split,
        covisibility_thres=0.0025,
        ROOT=args.root_dir,
        dataset_metadata_dir=args.dataset_metadata_dir,
        sample_specific_scene=True,
        specific_scene_name="chess_test",
        resolution=(518, 392),
        transform="imgnorm",
        data_norm_type="dinov2",
        sequential_view_mode=False,  # ★ 关闭顺序模式，使用原始随机采样逻辑 ★
    )

    print(f"[Rand] scenes: {dataset_rand.scenes}")
    print(f"[Rand] len(dataset): {len(dataset_rand)}")

    # 用 ForcedRandomDataLoader 验证随机性
    loader_rand = ForcedRandomDataLoader(dataset=dataset_rand, batch_size=1)

    seen_pairs = set()
    means_rand = []
    for idx, batch in enumerate(loader_rand):
        if idx >= NUM_RAND_TO_SAVE:
            break

        sample = extract_first_sample(batch)

        scene_raw = sample.get("label", "unknown_scene")
        scene = normalize_to_str(scene_raw, "unknown_scene")
        instance_raw = sample.get("instance", f"idx_{idx}")
        instance = normalize_to_str(instance_raw, f"idx_{idx}")
        frame_tag = os.path.basename(instance)
        seen_pairs.add((scene, frame_tag))

        img_tensor = sample["img"]
        img_uint8 = tensor_to_img_uint8(img_tensor)
        mean_val = float(img_tensor.mean().item())
        means_rand.append(mean_val)

        save_path = RAND_SAVE_DIR / f"{idx:04d}_{scene}_{frame_tag}_rand.jpg"
        Image.fromarray(img_uint8).save(save_path)

        print(
            f"[Rand] iter={idx:04d}, scene={scene}, instance={instance}, "
            f"mean={mean_val:.6f}, saved={save_path}"
        )

    print(f"[Rand] Unique (scene, frame) pairs among {len(means_rand)} samples: {len(seen_pairs)}")
    print(f"[Rand] Per-image mean values: {means_rand}")
    print(f"[Rand] Images saved to {RAND_SAVE_DIR}")
    print("All tests finished.")
