# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""7Scenes Dataset using WAI format data."""

import os
import torch
import numpy as np

from mapanything.datasets.base.base_dataset import BaseDataset, ForcedRandomDataLoader
from mapanything.utils.wai.core import load_data, load_frame


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
            sequential_view_mode: bool = False,  # 新增参数：是否开启顺序视图遍历模式
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
    parser.add_argument("-rd", "--root_dir", default="/mnt/storage/xwh/mapanything-dataset/wai_data/7scenes", type=str)
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


if __name__ == "__main__":
    import numpy as np
    from tqdm import tqdm
    import cv2
    from pathlib import Path
    import matplotlib.pyplot as plt

    from mapanything.datasets.base.base_dataset import view_name
    from mapanything.utils.image import rgb
    from mapanything.utils.viz import script_add_rerun_args

    parser = get_parser()
    script_add_rerun_args(parser)
    args = parser.parse_args()

    # =========================================
    # 配置
    # =========================================
    BATCH_SIZE = 1
    # 如果开启顺序模式，我们通常不希望截断，或者设置为非常大的数
    if args.sequential_mode:
        print("Sequential mode enabled. Will iterate through dataset linearly.")
        # 注意：ForcedRandomDataLoader 默认是随机的，如果是顺序模式，建议使用标准的 DataLoader 或
        # 确保 ForcedRandomDataLoader 的实现不shuffle，或者我们在这里手动处理
        # 但由于 SevenScenesWAI 继承自 BaseDataset，这里的 num_batches 是基于 len(dataset) 的
        pass

    SAVE_DIR = Path("/home/xwh/project/tmp/")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================
    # 1. 初始化数据集
    # 关键：确保 seed=None 或者不传，以允许随机性
    # =========================================
    try:
        dataset = SevenScenesWAI(
            num_views=args.num_of_views,
            split=args.split,
            covisibility_thres=0.025,
            ROOT=args.root_dir,
            dataset_metadata_dir=args.dataset_metadata_dir,
            sample_specific_scene=True,
            specific_scene_name='chess_test',
            resolution=(518, 392),
            transform="imgnorm",
            data_norm_type="dinov2",
            sequential_view_mode=args.sequential_mode,  # 传入参数
            # seed=777,  <--- 务必注释掉或设为 None
        )
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        exit()

    print(f"Dataset initialized. Length: {len(dataset)}")

    # 如果是顺序模式，我们想要遍历整个数据集
    num_batches_to_run = len(dataset) // BATCH_SIZE if args.sequential_mode else 1

    print(f"Starting extraction of {num_batches_to_run} batches...")

    # 注意：ForcedRandomDataLoader 设计初衷是随机采样。
    # 如果要严格顺序，最好用 pytorch 标准 DataLoader (shuffle=False)。
    # 但为了兼容现有代码结构，如果 BaseDataset 实现了 __getitem__ 为根据 index 访问，
    # 我们可以用 torch.utils.data.DataLoader

    if args.sequential_mode:
        from torch.utils.data import DataLoader

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None)
    else:
        dataloader = ForcedRandomDataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            num_batches=num_batches_to_run
        )

    # =========================================
    # 2. 循环构建 Batch
    # =========================================
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Loading Data")):
        if batch_idx >= num_batches_to_run:
            break

        # =========================================
        # 3. 保存逻辑
        # =========================================

        # 简单的 collate 处理 (如果使用标准 DataLoader 且没自定义 collate，batch_data 结构可能不同)
        # 这里假设 batch_data 结构兼容

        # 兼容性处理：如果是列表（BaseDataset通常返回list of views），解包
        if isinstance(batch_data, list):
            # 如果 batch_size=1，batch_data 可能就是 [view1_dict, view2_dict...] 的 batched 版本
            # 通常 BaseDataset 的 collate 会把 list of dicts 变成 dict of batched tensors
            pass

        current_batch_size = BATCH_SIZE  # 简化假设
        # 实际使用中需根据 batch_data 真实结构解析

        # (此处省略具体的保存代码，保持原样或根据需要调整)
        pass

    print("Finished sampling.")