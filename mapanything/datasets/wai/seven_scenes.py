# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""7Scenes Dataset using WAI format data."""

import os
from pathlib import Path
from typing import Optional

import numpy as np

from mapanything.datasets.base.base_dataset import BaseDataset
from mapanything.utils.wai.core import load_data, load_frame


class SevenScenesWAI(BaseDataset):
    """7Scenes dataset captured with RGB-D sensors and converted to WAI format."""

    def __init__(
        self,
        *args,
        ROOT,
        dataset_metadata_dir=None,
        split=None,
        overfit_num_sets=None,
        sample_specific_scene: bool = False,
        specific_scene_name: str | None = None,
        **kwargs,
    ):
        """Initialize the dataset attributes."""
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.dataset_metadata_dir = dataset_metadata_dir
        self.split = split or "train"
        self.overfit_num_sets = overfit_num_sets
        self.sample_specific_scene = sample_specific_scene
        self.specific_scene_name = specific_scene_name
        self._load_data()

        # 7Scenes provides metric-scale poses derived from pseudo ground truth
        self.is_metric_scale = True
        self.is_synthetic = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _metadata_path(self) -> Optional[Path]:
        if not self.dataset_metadata_dir or str(self.dataset_metadata_dir).lower() == "none":
            return None
        return Path(self.dataset_metadata_dir) / self.split / f"seven_scenes_scene_list_{self.split}.npy"

    def _infer_scene_split(self, scene_name: str) -> str | None:
        parts = scene_name.split("_")
        if len(parts) >= 2:
            return parts[1].lower()
        return None

    def _scan_split_directory(self, split: str) -> list[str]:
        """Discover scenes on disk when metadata is missing."""
        root = Path(self.ROOT)
        if not root.exists():
            return []

        split_key = split.lower()
        fallback_splits = [split_key]
        # Commonly we do not ship a dedicated validation split; fall back to test scenes.
        if split_key == "val":
            fallback_splits.append("test")

        scene_names: list[str] = []
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            scene_split = self._infer_scene_split(entry.name)
            if scene_split in fallback_splits:
                scene_names.append(entry.name)
        return scene_names

    def _maybe_overfit(self, scenes: list[str]) -> list[str]:
        if self.overfit_num_sets is not None:
            return scenes[: self.overfit_num_sets]
        return scenes

    def _ensure_frame_names(self, scene_meta: dict) -> dict:
        if "frame_names" not in scene_meta:
            frame_names = {
                frame["frame_name"]: idx
                for idx, frame in enumerate(scene_meta.get("frames", []))
            }
            scene_meta["frame_names"] = frame_names
        return scene_meta

    def _load_pairwise_covisibility(self, scene_root: Path, num_views: int) -> np.ndarray:
        covisibility_root = scene_root / "covisibility"
        if covisibility_root.exists():
            for version_dir in sorted(covisibility_root.iterdir()):
                if not version_dir.is_dir():
                    continue
                npy_files = sorted(version_dir.glob("*.npy"))
                if not npy_files:
                    continue
                try:
                    return load_data(npy_files[0], "mmap")
                except FileNotFoundError:
                    continue
        # Fallback: fully connected graph ensures deterministic sampling when covisibility is unused.
        return np.ones((num_views, num_views), dtype=np.float32)

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def _load_data(self):
        metadata_path = self._metadata_path()
        if metadata_path is not None and metadata_path.exists():
            split_scene_list = list(np.load(metadata_path, allow_pickle=True))
        else:
            split_scene_list = self._scan_split_directory(self.split)

        if self.sample_specific_scene and self.specific_scene_name is not None:
            self.scenes = [self.specific_scene_name]
        else:
            self.scenes = self._maybe_overfit(split_scene_list)
        self.num_of_scenes = len(self.scenes)

    def _get_views(self, sampled_idx, num_views_to_sample, resolution):
        scene_name = self.scenes[sampled_idx]
        scene_root = Path(self.ROOT) / scene_name
        scene_meta = load_data(scene_root / "scene_meta.json", "scene_meta")
        scene_meta = self._ensure_frame_names(scene_meta)

        scene_file_names = list(scene_meta["frame_names"].keys())
        num_views_in_scene = len(scene_file_names)

        pairwise_covisibility = self._load_pairwise_covisibility(scene_root, num_views_in_scene)
        view_indices = self._sample_view_indices(
            num_views_to_sample,
            num_views_in_scene,
            pairwise_covisibility,
        )

        views = []
        for view_index in view_indices:
            view_key = scene_file_names[view_index]
            view_data = load_frame(
                scene_root,
                view_key,
                modalities=["image", "depth"],
                scene_meta=scene_meta,
            )

            image = view_data["image"].permute(1, 2, 0).numpy()
            image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
            depthmap = view_data["depth"].numpy().astype(np.float32)
            depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
            intrinsics = view_data["intrinsics"].numpy().astype(np.float32)
            camera_pose = view_data["extrinsics"].numpy().astype(np.float32)

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
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="7Scenes",
                    label=scene_name,
                    instance=os.path.join("images", str(view_key)),
                )
            )

        return views


def get_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-rd", "--root_dir", default="/fsx/xrtech/data/seven_scenes_wai", type=str)
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
        default=2,
        type=int,
    )
    parser.add_argument("--viz", action="store_true")

    return parser


if __name__ == "__main__":
    import rerun as rr
    from tqdm import tqdm

    from mapanything.datasets.base.base_dataset import view_name
    from mapanything.utils.image import rgb
    from mapanything.utils.viz import script_add_rerun_args

    parser = get_parser()
    script_add_rerun_args(parser)
    args = parser.parse_args()

    dataset = SevenScenesWAI(
        num_views=args.num_of_views,
        split=args.split,
        covisibility_thres=0.0,
        ROOT=args.root_dir,
        dataset_metadata_dir=args.dataset_metadata_dir,
        resolution=(518, 392),
        transform="imgnorm",
        data_norm_type="dinov2",
        seed=777,
    )
    print(dataset.get_stats())

    if args.viz:
        rr.script_setup(args, "SevenScenes_Dataloader")
        rr.set_time("stable_time", sequence=0)
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

        for idx in tqdm(range(len(dataset))):
            views = dataset[idx]
            for view in views:
                rr.log(view_name(view), rgb(view["img"]))
