# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""7Scenes Dataset using WAI format data."""

import os
import torch
import numpy as np

from mapanything.datasets.base.base_dataset import BaseDataset,ForcedRandomDataLoader
from mapanything.utils.wai.core import load_data, load_frame


import os
import numpy as np

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
        split_metadata_path = os.path.join(
            self.dataset_metadata_dir,
            self.split,
            f"seven_scenes_scene_list_{self.split}.npy",
        )

        if not os.path.exists(split_metadata_path):
            all_folders = sorted([d for d in os.listdir(self.ROOT) if os.path.isdir(os.path.join(self.ROOT, d))])
            self.scenes = [s for s in all_folders if s.endswith(f"_{self.split}")]
        else:
            split_scene_list = np.load(split_metadata_path, allow_pickle=True)
            self.scenes = list(split_scene_list)

        if self.sample_specific_scene:
            if self.specific_scene_name not in self.scenes:
                potential_name = f"{self.specific_scene_name}_{self.split}"
                if potential_name in self.scenes:
                    self.specific_scene_name = potential_name
            self.scenes = [self.specific_scene_name]

        if self.overfit_num_sets is not None:
            self.scenes = self.scenes[:self.overfit_num_sets]

        self.num_of_scenes = len(self.scenes)

    @staticmethod
    def _depth_to_world_points(depthmap: np.ndarray,
                               intrinsics: np.ndarray,
                               c2w_pose: np.ndarray) -> np.ndarray:
        """
        将深度图用内参反投影到相机坐标，再用 cam2world 外参变换到世界坐标。
        对无效深度（<=0 或非有限）返回 NaN。

        Args:
            depthmap: (H, W) float32
            intrinsics: (3, 3) float32, 对应已经 resize/crop 后的图像
            c2w_pose: (4, 4) float32, 相机到世界的变换

        Returns:
            pts3d_world: (H, W, 3) float32，世界坐标；无效深度处为 NaN
        """
        H, W = depthmap.shape
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

        # 像素网格 (u 对应列，v 对应行)
        u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                           np.arange(H, dtype=np.float32))

        z = depthmap.astype(np.float32)
        valid = np.isfinite(z) & (z > 0.0)

        # 相机坐标系下三维点
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        xyz_cam = np.stack([x, y, z], axis=-1)  # (H, W, 3)

        # 变换到世界坐标：X_w = R * X_c + t
        R = c2w_pose[:3, :3].astype(np.float32)
        t = c2w_pose[:3, 3].astype(np.float32)

        # (H, W, 3) @ (3, 3)^T -> (H, W, 3)，再加 t
        pts3d_world = xyz_cam @ R.T + t

        # 无效位置写 NaN
        if not valid.all():
            pts3d_world[~valid] = np.nan

        return pts3d_world.astype(np.float32)

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

        if pairwise_covisibility is None:
            pairwise_covisibility = np.ones((num_views_in_scene, num_views_in_scene), dtype=np.float32)

        # Get the indices of the N views in the scene
        view_indices = self._sample_view_indices(
            num_views_to_sample, num_views_in_scene, pairwise_covisibility
        )

        views = []
        for view_index in view_indices:
            view_file_name = scene_file_names[view_index]
            view_data = load_frame(
                scene_root,
                view_file_name,
                modalities=["image", "depth"],
                scene_meta=scene_meta,
            )

            # HWC uint8 image
            image = view_data["image"].permute(1, 2, 0).numpy()
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

            depthmap = view_data["depth"].numpy().astype(np.float32)     # (H, W)
            intrinsics = view_data["intrinsics"].numpy().astype(np.float32)  # (3, 3)
            c2w_pose = view_data["extrinsics"].numpy().astype(np.float32)    # (4, 4)

            # Resize/crop to target resolution (intrinsics updated inside)
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
                    camera_pose=c2w_pose,          # cam2world
                    camera_intrinsics=intrinsics,
                    dataset="7Scenes",
                    label=scene_name,
                    instance=os.path.join("images", str(view_file_name)),
                    # pts3d=pts3d,                   # <<< 新增键值对
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
        default=2,
        type=int,
    )

    return parser

if __name__ == "__main__":
    import numpy as np
    from tqdm import tqdm
    import cv2, json
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from mapanything.utils.viz import script_add_rerun_args
    from mapanything.utils.image import rgb

    # ========= 形状规格化工具 =========
    def to_hw(depth):
        """返回 (H, W) float32；把 (1,H,W)/(H,W,1) 等 squeeze 成 2D"""
        arr = np.asarray(depth)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            # 常见情况：C×H×W，把 C==1 的 squeeze
            if arr.ndim == 3 and 1 in arr.shape:
                arr = np.squeeze(arr)
        assert arr.ndim == 2, f"depth must be HxW, got shape {arr.shape}"
        return arr.astype(np.float32)

    def as33(K):
        """返回 (3,3) float32"""
        K = np.asarray(K).squeeze()
        assert K.shape[-2:] == (3,3), f"K must be 3x3, got {K.shape}"
        return K.astype(np.float32)

    def as44(T):
        """返回 (4,4) float32"""
        T = np.asarray(T).squeeze()
        assert T.shape[-2:] == (4,4), f"pose must be 4x4, got {T.shape}"
        return T.astype(np.float32)

    def img_to_hwc01(img, norm_type="dinov2"):
        """
        把数据里的 img 还原为 H×W×3、[0,1] 用于可视化。
        你原来用的 rgb() 会做反归一化，这里统一转成 HWC、[0,1]。
        """
        img_np = rgb(img, norm_type=norm_type)  # 返回一般是 H×W×3 或 C×H×W
        arr = np.asarray(img_np)
        if arr.ndim == 3 and arr.shape[0] in (1,3) and arr.shape[-1] not in (1,3):
            # 形如 C×H×W -> H×W×C
            arr = np.moveaxis(arr, 0, -1)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        # 夹紧到 [0,1]
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
        return arr

    # ========= 几何函数 =========
    def invert_se3(c2w: np.ndarray) -> np.ndarray:
        R, t = c2w[:3, :3], c2w[:3, 3]
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R.T
        w2c[:3, 3] = -R.T @ t
        return w2c

    def depth_to_world_points(depthmap: np.ndarray, K: np.ndarray, c2w_pose: np.ndarray) -> np.ndarray:
        """depth: (H,W); K:(3,3); c2w:(4,4) -> pts3d:(H,W,3)；无效处 NaN"""
        H, W = depthmap.shape
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        z = depthmap.astype(np.float32)
        valid = np.isfinite(z) & (z > 0.0)

        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        xyz_cam = np.stack([x, y, z], axis=-1)

        R = c2w_pose[:3, :3].astype(np.float32)
        t = c2w_pose[:3, 3].astype(np.float32)
        pts3d_world = xyz_cam @ R.T + t
        pts3d_world[~valid] = np.nan
        return pts3d_world.astype(np.float32)

    def single_view_diagnostics(view: dict):
        """
        要求 view 里：
          depthmap: H×W
          camera_intrinsics: 3×3
          camera_pose: 4×4
          pts3d: H×W×3
        """
        K   = as33(view["camera_intrinsics"])
        c2w = as44(view["camera_pose"])
        w2c = invert_se3(c2w)

        depth = to_hw(view["depthmap"])
        pts3d = np.asarray(view["pts3d"]).astype(np.float32)
        assert pts3d.ndim == 3 and pts3d.shape[2] == 3, f"pts3d must be HxWx3, got {pts3d.shape}"
        H, W = depth.shape

        valid = np.isfinite(depth) & (depth > 0) & np.isfinite(pts3d).all(axis=-1)
        if valid.sum() == 0:
            raise RuntimeError("有效深度点为 0。")

        Xw = pts3d[valid]
        Xw_h = np.concatenate([Xw, np.ones((Xw.shape[0], 1), np.float32)], axis=1)
        Xc = (w2c @ Xw_h.T).T[:, :3]
        zc = Xc[:, 2]

        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        up = fx * (Xc[:, 0] / zc) + cx
        vp = fy * (Xc[:, 1] / zc) + cy

        U, V = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        du = up - U[valid]; dv = vp - V[valid]
        dpx = np.sqrt(du**2 + dv**2)

        dz = np.abs(zc - depth[valid])
        med_depth = float(np.median(depth[valid]))
        rel_dz_mean = float((dz / max(med_depth, 1e-6)).mean())

        nan_in_pts = np.isnan(pts3d).any(axis=-1)
        invalid_depth = ~np.isfinite(depth) | (depth <= 0)
        nan_invalid_agree_ratio = float((nan_in_pts == invalid_depth).mean())

        report = {
            "valid_ratio": float(valid.mean()),
            "front_ratio": float((zc > 0).mean()),
            "reproj_err_px_mean": float(dpx.mean()),
            "reproj_err_px_p95": float(np.percentile(dpx, 95)),
            "depth_abs_err_mean": float(dz.mean()),
            "depth_abs_err_p95": float(np.percentile(dz, 95)),
            "depth_rel_err_mean": rel_dz_mean,
            "median_depth": med_depth,
            "nan_invalid_agree_ratio": nan_invalid_agree_ratio,
        }
        return report, dpx, up, vp, valid

    def pair_view_reprojection_stats(view_src: dict, view_tgt: dict) -> dict:
        """把 src 的 pts3d 投到 tgt，返回 inside/infront 比例等。"""
        Kt   = as33(view_tgt["camera_intrinsics"])
        c2w_t = as44(view_tgt["camera_pose"])
        w2c_t = invert_se3(c2w_t)

        depth_t = to_hw(view_tgt["depthmap"])
        Ht, Wt = depth_t.shape

        pts3d_src = np.asarray(view_src["pts3d"]).astype(np.float32)
        valid_src = np.isfinite(pts3d_src).all(axis=-1)
        if valid_src.sum() == 0:
            return {"num_src_valid": 0}

        Xw = pts3d_src[valid_src]
        Xw_h = np.concatenate([Xw, np.ones((Xw.shape[0], 1), np.float32)], axis=1)
        Xt = (w2c_t @ Xw_h.T).T[:, :3]
        zt = Xt[:, 2]

        fx, fy, cx, cy = float(Kt[0, 0]), float(Kt[1, 1]), float(Kt[0, 2]), float(Kt[1, 2])
        ut = fx * (Xt[:, 0] / zt) + cx
        vt = fy * (Xt[:, 1] / zt) + cy

        infront = zt > 0
        inside = (ut >= 0) & (ut < Wt) & (vt >= 0) & (vt < Ht) & infront

        margin_u = np.minimum(ut, Wt - 1 - ut)
        margin_v = np.minimum(vt, Ht - 1 - vt)
        margin = np.minimum(margin_u, margin_v)
        margin_inside = margin[inside]

        return {
            "num_src_valid": int(valid_src.sum()),
            "infront_ratio": float(infront.mean()),
            "inside_ratio": float(inside.mean()),
            "margin_px_p50": float(np.percentile(margin_inside, 50)) if margin_inside.size else 0.0,
            "margin_px_p10": float(np.percentile(margin_inside, 10)) if margin_inside.size else 0.0,
        }

    def save_overlay(img01, up, vp, valid_mask, out_path: Path, stride=8):
        H, W = valid_mask.shape
        valid_idx_flat = np.flatnonzero(valid_mask.ravel())
        if valid_idx_flat.size == 0:
            return
        sel_flat = valid_idx_flat[::max(1, stride)]
        # 建立 valid_flat -> 顺序索引 的映射（因为 up/vp 只对应 valid 的顺序）
        order = np.empty(valid_mask.size, dtype=np.int64); order.fill(-1)
        order[valid_idx_flat] = np.arange(valid_idx_flat.size)
        sel_order = order[sel_flat]

        plt.figure(figsize=(6, 6))
        plt.imshow((img01 * 255).astype(np.uint8))
        plt.scatter(up[sel_order], vp[sel_order], s=2, alpha=0.7)
        plt.title("Reprojected pixels overlay")
        plt.axis("off"); plt.tight_layout()
        plt.savefig(out_path, dpi=150); plt.close()

    def save_hist(dpx, out_path: Path):
        plt.figure()
        plt.hist(dpx, bins=60)
        plt.xlabel("Reprojection error (px)"); plt.ylabel("Count")
        plt.title("Reprojection error histogram")
        plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

    # ========= 你的原始参数与数据加载 =========
    parser = get_parser()
    script_add_rerun_args(parser)
    args = parser.parse_args()

    BATCH_SIZE = 1
    MAX_BATCHES = 1
    SAVE_DIR = Path("/home/xwh/project/tmp/"); SAVE_DIR.mkdir(parents=True, exist_ok=True)

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
    )

    print(f"Dataset initialized. Length: {len(dataset)}")
    print(f"Starting manual extraction of {MAX_BATCHES} diverse batches...")

    dataloader = ForcedRandomDataLoader(
        dataset=dataset,
        batch_size=1,
        num_batches=MAX_BATCHES
    )

    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Loading Data")):
        num_views_in_batch = len(batch_data)
        current_batch_size = batch_data[0]['img'].shape[0]

        for sample_i in range(current_batch_size):
            sample_dir = SAVE_DIR / f"batch{batch_idx:03d}_sample{sample_i:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # 解包单样本
            raw_views = []
            for view_idx in range(num_views_in_batch):
                d = {}
                for k, v in batch_data[view_idx].items():
                    if isinstance(v, torch.Tensor):
                        d[k] = v[sample_i].detach().cpu().numpy()
                    elif isinstance(v, (list, tuple)):
                        d[k] = v[sample_i]
                    else:
                        d[k] = v
                raw_views.append(d)
            # 规格化 + 点云计算 + 诊断
            views = []
            for v_idx, v in enumerate(raw_views):
                img01 = img_to_hwc01(v['img'], norm_type=v.get('data_norm_type', 'dinov2'))
                depth = to_hw(v['depthmap'])
                K     = as33(v['camera_intrinsics'])
                c2w   = as44(v['camera_pose'])
                pts3d = v['pts3d']
                print(pts3d.shape)

                view_local = dict(v)
                view_local['img01'] = img01     # 仅可视化用
                view_local['depthmap'] = depth  # 2D
                view_local['camera_intrinsics'] = K
                view_local['camera_pose'] = c2w
                view_local['pts3d'] = pts3d

                # 保存 RGB 便于肉眼看
                cv2.imwrite(str(sample_dir / f"view{v_idx:02d}_rgb.png"),
                            cv2.cvtColor((img01 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                try:
                    report, dpx, up, vp, valid = single_view_diagnostics(view_local)
                except Exception as e:
                    print(f"[view {v_idx}] diagnostics failed: {e}")
                    continue

                with open(sample_dir / f"view{v_idx:02d}_diagnostics.json", "w") as f:
                    json.dump(report, f, indent=2)

                save_overlay(img01, up, vp, valid, sample_dir / f"view{v_idx:02d}_reproj_overlay.png", stride=8)
                save_hist(dpx, sample_dir / f"view{v_idx:02d}_reproj_hist.png")

                print(f"[batch{batch_idx} view{v_idx}] "
                      f"front={report['front_ratio']:.3f} "
                      f"reproj_mean={report['reproj_err_px_mean']:.3f} "
                      f"p95={report['reproj_err_px_p95']:.3f} "
                      f"valid={report['valid_ratio']:.3f}")

                views.append(view_local)

            # 跨视角：需要至少两帧
            if len(views) >= 2:
                for a in range(len(views)):
                    for b in range(len(views)):
                        if a == b: continue
                        cross = pair_view_reprojection_stats(views[a], views[b])
                        with open(sample_dir / f"cross_{a}_to_{b}.json", "w") as f:
                            json.dump(cross, f, indent=2)
                        print(f"[cross {a}->{b}] inside={cross.get('inside_ratio',0):.3f} "
                              f"infront={cross.get('infront_ratio',0):.3f} "
                              f"N={cross.get('num_src_valid',0)}")

    print("Finished manual diverse sampling & diagnostics.")

