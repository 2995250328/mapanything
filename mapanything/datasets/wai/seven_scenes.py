# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""7Scenes Dataset using WAI format data."""

import os
from pathlib import Path
from typing import Optional
import open3d as o3d
import numpy as np
import torch

from mapanything.datasets.base.base_dataset import BaseDataset,ForcedRandomDataLoader
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
        """
        # Initialize the dataset attributes
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.dataset_metadata_dir = dataset_metadata_dir
        self.split = split
        self.overfit_num_sets = overfit_num_sets
        self.sample_specific_scene = sample_specific_scene
        self.specific_scene_name = specific_scene_name
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

        self.num_of_scenes = len(self.scenes)

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
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
    parser.add_argument("-s", "--split", default="train", type=str)
    parser.add_argument(
        "-nv",
        "--num_of_views",
        default=1,
        type=int,
    )

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
    MAX_BATCHES = 1  # 你想要提取的批次数量
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
            # seed=777,  <--- 务必注释掉或设为 None
        )
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        exit()

    print(f"Dataset initialized. Length: {len(dataset)}")
    print(f"Starting manual extraction of {MAX_BATCHES} diverse batches...")

    dataloader = ForcedRandomDataLoader(
        dataset=dataset,
        batch_size=1,  # 你想要的 batch size
        num_batches=1  # 你想要遍历多少个 batch 后结束
    )

    # =========================================
    # 2. 手动循环构建 Batch
    # =========================================
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Loading Data")):

   # =========================================
        # 3. 保存逻辑 (与之前相同，略有调整以适应 batch_data 结构)
        # =========================================
        # 注意：default_collate 打包后，batch_data 依然是一个 list (长度为 num_views)，
        # 其中的每个元素是一个 dict，dict 里的值现在有了 (B, ...) 的维度。

        num_views_in_batch = len(batch_data)
        # 获取当前 batch 的实际大小 (应该等于 BATCH_SIZE)
        current_batch_size = batch_data[0]['img'].shape[0]

        for sample_i in range(current_batch_size):
            global_sample_idx = batch_idx * BATCH_SIZE + sample_i
            sample_dir = SAVE_DIR / f"batch{batch_idx:03d}_sample{sample_i:03d}_idx{global_sample_idx:06d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # 从 batch 中解压出单个样本的数据用于保存
            views = []
            for view_idx in range(num_views_in_batch):
                view_sample = {}
                for key, value in batch_data[view_idx].items():
                    if isinstance(value, torch.Tensor):
                        view_sample[key] = value[sample_i].cpu().numpy()
                    elif isinstance(value, (list, tuple)):
                        view_sample[key] = value[sample_i]
                    else:
                        view_sample[key] = value
                views.append(view_sample)

            # --- 执行你的保存和可视化代码 ---
            # save_views_data(views, sample_dir)
            # 这里简写，请替换为你完整的保存 RGB/Depth/Pose 的循环代码
            for v_idx, view in enumerate(views):
                # 示例：仅保存 RGB 验证差异性
                if 'img' in view:
                    # 假设 rgb() 是你的反归一化函数
                    img_np = rgb(view['img'], norm_type=view.get('data_norm_type', 'dinov2'))
                    cv2.imwrite(str(sample_dir / f"view{v_idx:02d}_rgb.png"),
                                cv2.cvtColor((img_np * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    print("Finished manual diverse sampling.")