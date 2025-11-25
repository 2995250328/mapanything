import warnings

import hydra
import torch
import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm
import imageio
import numpy as np
import open3d as o3d
import torch

from mapanything.utils.geometry import (
    geotrf,
    inv,
    normalize_multiple_pointclouds,
    quaternion_to_rotation_matrix,
    transform_pose_using_quats_and_trans_2_to_1,
)
from mapanything.utils.metrics import (
    calculate_auc_np,
    evaluate_ate,
    l2_distance_of_unit_ray_directions_to_angular_error,
    m_rel_ae,
    se3_to_relative_pose_error,
    thresh_inliers,
)
from mapanything.tasks.ace import load_memory_features
from mapanything.datasets import SevenScenesWAI
from mapanything.datasets.base.base_dataset import ForcedRandomDataLoader
from mapanything.models import init_model
from mapanything.utils.debugprinter import DebugPrinter
import torch.nn.functional as F

printer = DebugPrinter()

def debug_check_layernorm_like_seq(seq: torch.Tensor, name: str = "seq"):
    """
    seq: (B, L, C) 或 (L, C)
    按 IFR/LayerNorm 的使用方式检查每个 token 的 channel 统计量：
      - 对每个 token（长度 C 向量）算 mean/std
      - 再看 mean/std 在所有 token 之间的分布是否很“集中”
      - 若非常集中 -> 很像 LN 输出；若变化很大 -> 更像 raw 特征
    """
    if seq.ndim == 2:
        seq = seq.unsqueeze(0)   # (L,C) -> (1,L,C)
    assert seq.ndim == 3, f"{name} must be (B,L,C) or (L,C), got {seq.shape}"

    B, L, C = seq.shape
    tokens = seq.reshape(-1, C)  # (B*L, C)

    mean_c = tokens.mean(dim=1)            # 每个 token 的 channel 均值
    std_c  = tokens.std(dim=1, unbiased=False)  # 每个 token 的 channel std

    mean_mean = mean_c.mean().item()
    mean_std  = std_c.mean().item()

    print(f"[{name}] token-wise channel mean stats:")
    print(f"  mean(mean_c) = {mean_mean:.6f}")
    print(f"  min(mean_c)  = {mean_c.min().item():.6f}")
    print(f"  max(mean_c)  = {mean_c.max().item():.6f}")
    print(f"  max |mean_c - mean(mean_c)| = {(mean_c - mean_mean).abs().max().item():.6f}")

    print(f"[{name}] token-wise channel std stats:")
    print(f"  mean(std_c) = {mean_std:.6f}")
    print(f"  min(std_c)  = {std_c.min().item():.6f}")
    print(f"  max(std_c)  = {std_c.max().item():.6f}")
    print(f"  max |std_c - mean(std_c)| = {(std_c - mean_std).abs().max().item():.6f}")
def debug_check_memory_layer_seq(memory_feats, layer_idx: int, name_prefix: str = "mem_layer"):
    """
    memory_feats: list[len = 1 + depth]
      memory_feats[layer_idx]: list[len = V]
        memory_feats[layer_idx][v]: (B, C, H, W)

    将某一层的所有 view stack 成 (B, V*H*W, C)，
    再用 sequence 版 LN-like 检查。
    """
    feats_per_view = memory_feats[layer_idx]
    assert isinstance(feats_per_view, (list, tuple)) and len(feats_per_view) > 0

    B, C, H, W = feats_per_view[0].shape
    V = len(feats_per_view)

    # (B, V, C, H, W)
    stacked = torch.stack(feats_per_view, dim=1)
    # -> (B, V, H, W, C)
    stacked = stacked.permute(0, 1, 3, 4, 2).contiguous()
    # -> (B, V*H*W, C)
    seq = stacked.reshape(B, V * H * W, C)

    debug_check_layernorm_like_seq(seq, f"{name_prefix}{layer_idx}_allviews")

def get_all_info_for_metric_computation(batch, preds, norm_mode="avg_dis"):
    """
    提取并归一化用于指标计算的所有信息。
    所有量都将归一化到 view0 的相机坐标系下。
    """
    n_views = len(batch)
    batch_size = batch[0]["camera_pose"].shape[0]

    # --- 1. 处理 Ground Truth (转换到 View 0 坐标系) ---
    in_camera0 = inv(batch[0]["camera_pose"])
    no_norm_gt_pts = []
    no_norm_gt_pts3d_cam = []
    no_norm_gt_pose_trans = []
    valid_masks = []
    gt_ray_directions = []
    gt_pose_quats = []

    # --- 2. 处理 Predictions (转换到 View 0 坐标系) ---
    pred_camera0 = torch.eye(4, device=preds[0]["cam_quats"].device).unsqueeze(0).repeat(batch_size, 1, 1)
    pred_camera0_rot = quaternion_to_rotation_matrix(preds[0]["cam_quats"].clone())
    pred_camera0[..., :3, :3] = pred_camera0_rot
    pred_camera0[..., :3, 3] = preds[0]["cam_trans"].clone()
    pred_in_camera0 = inv(pred_camera0)

    no_norm_pr_pts = []
    no_norm_pr_pts3d_cam = []
    no_norm_pr_pose_trans = []
    pr_ray_directions = []
    pr_pose_quats = []
    metric_pr_pts_to_compute_scale = []

    # ---------------- GT & Pred 收集 ----------------
    for i in range(n_views):
        # --- GT 点云 / mask / ray ---
        no_norm_gt_pts.append(geotrf(in_camera0, batch[i]["pts3d"]))   # [B,H,W,3] in view0 cam
        valid_masks.append(batch[i]["valid_mask"].clone())
        gt_ray_directions.append(batch[i]["ray_directions_cam"])
        no_norm_gt_pts3d_cam.append(batch[i]["pts3d_cam"])

        if i == 0:
            # view0 在自身坐标系中位姿为单位
            gt_pose_quats.append(
                torch.tensor([0, 0, 0, 1],
                             dtype=gt_ray_directions[0].dtype,
                             device=gt_ray_directions[0].device
                             ).unsqueeze(0).repeat(batch_size, 1)
            )
            no_norm_gt_pose_trans.append(
                torch.tensor([0, 0, 0],
                             dtype=gt_ray_directions[0].dtype,
                             device=gt_ray_directions[0].device
                             ).unsqueeze(0).repeat(batch_size, 1)
            )
        else:
            gt_q_in_v0, gt_t_in_v0 = transform_pose_using_quats_and_trans_2_to_1(
                batch[0]["camera_pose_quats"], batch[0]["camera_pose_trans"],
                batch[i]["camera_pose_quats"], batch[i]["camera_pose_trans"]
            )
            gt_pose_quats.append(gt_q_in_v0)
            no_norm_gt_pose_trans.append(gt_t_in_v0)

        # --- Pred 点云 / pose -> view0 ---
        pr_q_in_v0, pr_t_in_v0 = transform_pose_using_quats_and_trans_2_to_1(
            preds[0]["cam_quats"], preds[0]["cam_trans"],
            preds[i]["cam_quats"], preds[i]["cam_trans"]
        )
        pr_pts3d_in_v0 = geotrf(pred_in_camera0, preds[i]["pts3d"])

        # metric_scaling_factor 处理
        scale = preds[i].get("metric_scaling_factor", torch.ones(batch_size, device=preds[i]["pts3d"].device))
        if scale.ndim == 1:
            scale = scale.unsqueeze(-1).unsqueeze(-1)   # [B,1,1]
        elif scale.ndim == 2:
            scale = scale.unsqueeze(-1)                 # [B,1,1]

        curr_view_no_norm_pr_pts = pr_pts3d_in_v0 / scale
        curr_view_no_norm_pr_pts3d_cam = preds[i]["pts3d_cam"] / scale
        curr_view_no_norm_pr_pose_trans = pr_t_in_v0 / scale.squeeze(-1).squeeze(-1)

        no_norm_pr_pts.append(curr_view_no_norm_pr_pts)
        no_norm_pr_pts3d_cam.append(curr_view_no_norm_pr_pts3d_cam)
        no_norm_pr_pose_trans.append(curr_view_no_norm_pr_pose_trans)
        pr_ray_directions.append(preds[i]["ray_directions"])
        pr_pose_quats.append(pr_q_in_v0)

        metric_pr_pts_to_compute_scale.append(curr_view_no_norm_pr_pts.detach() * scale)

    # ---------------- 归一化 (Scale Alignment) ----------------
    # 注意：normalize_multiple_pointclouds 在不同版本下可能：
    #   - 只返回 normalized_pts
    #   - 返回 (normalized_pts, factor)
    #   - 返回 (normalized_pts, factor, 其他信息…)
    # 这里统一做兼容：只取前两个，其余丢弃；若没有 factor，则 factor 设为 None。

    pr_res = normalize_multiple_pointclouds(no_norm_pr_pts, valid_masks, norm_mode, ret_factor=True)
    if isinstance(pr_res, tuple):
        pr_pts_norm = pr_res[0]
        pr_norm_factor = pr_res[1] if len(pr_res) > 1 else None
    else:
        pr_pts_norm = pr_res
        pr_norm_factor = None

    gt_res = normalize_multiple_pointclouds(no_norm_gt_pts, valid_masks, norm_mode, ret_factor=True)
    if isinstance(gt_res, tuple):
        gt_pts_norm = gt_res[0]
        gt_norm_factor = gt_res[1] if len(gt_res) > 1 else None
    else:
        gt_pts_norm = gt_res
        gt_norm_factor = None

    # ---------------- 根据是否有 factor 选择是否再 / factor ----------------
    gt_pts, gt_pts3d_cam, gt_pose_trans = [], [], []
    pr_pts, pr_pts3d_cam, pr_pose_trans = [], [], []

    for i in range(n_views):
        # GT
        gt_pts.append(gt_pts_norm[i].cpu())

        if gt_norm_factor is not None:
            # [B,1,1,1] 广播到 [B,H,W,3]
            gt_pts3d_cam.append((no_norm_gt_pts3d_cam[i] / gt_norm_factor).cpu())
            # trans 用标量 scale：gt_norm_factor[:,0,0,0] -> [B]
            gt_scale_1d = gt_norm_factor[:, 0, 0, 0]
            gt_pose_trans.append((no_norm_gt_pose_trans[i] / gt_scale_1d.unsqueeze(-1)).cpu())
        else:
            gt_pts3d_cam.append(no_norm_gt_pts3d_cam[i].cpu())
            gt_pose_trans.append(no_norm_gt_pose_trans[i].cpu())

        # Pred
        pr_pts.append(pr_pts_norm[i].cpu())

        if pr_norm_factor is not None:
            pr_pts3d_cam.append((no_norm_pr_pts3d_cam[i] / pr_norm_factor).cpu())
            pr_scale_1d = pr_norm_factor[:, 0, 0, 0]
            pr_pose_trans.append((no_norm_pr_pose_trans[i] / pr_scale_1d.unsqueeze(-1)).cpu())
        else:
            pr_pts3d_cam.append(no_norm_pr_pts3d_cam[i].cpu())
            pr_pose_trans.append(no_norm_pr_pose_trans[i].cpu())

    # ---------------- Metric Scale Error ----------------
    gt_metric_norm, pr_metric_norm = None, None
    metric_scale_mask = batch[0].get("is_metric_scale", torch.zeros(batch_size, dtype=torch.bool))

    if gt_norm_factor is not None and metric_scale_mask is not None:
        valid_gt_norm = gt_norm_factor[:, 0, 0, 0] > 1e-8
        valid_metric_mask = metric_scale_mask & valid_gt_norm

        if valid_metric_mask.any():
            metric_res = normalize_multiple_pointclouds(
                metric_pr_pts_to_compute_scale, valid_masks, norm_mode, ret_factor=True
            )
            if isinstance(metric_res, tuple):
                # 只取 factor
                pr_metric_factor = metric_res[1] if len(metric_res) > 1 else None
            else:
                pr_metric_factor = None

            if pr_metric_factor is not None:
                gt_metric_norm = gt_norm_factor[valid_metric_mask].cpu()
                pr_metric_norm = pr_metric_factor[valid_metric_mask].cpu()

    # ---------------- Pose 矩阵 (4x4) & Z-depth ----------------
    gt_poses, pr_poses = [], []
    gt_z_depths, pr_z_depths = [], []

    for i in range(n_views):
        # GT Pose
        gt_p = torch.eye(4).repeat(batch_size, 1, 1)
        gt_p[..., :3, :3] = quaternion_to_rotation_matrix(gt_pose_quats[i].cpu())
        gt_p[..., :3, 3] = gt_pose_trans[i]
        gt_poses.append(gt_p)

        # Pred Pose
        pr_p = torch.eye(4).repeat(batch_size, 1, 1)
        pr_p[..., :3, :3] = quaternion_to_rotation_matrix(pr_pose_quats[i].cpu())
        pr_p[..., :3, 3] = pr_pose_trans[i]
        pr_poses.append(pr_p)

        # Z-depth
        gt_z_depths.append(gt_pts3d_cam[i][..., 2:])
        pr_z_depths.append(pr_pts3d_cam[i][..., 2:])

        # mask 转 CPU
        valid_masks[i] = valid_masks[i].cpu()

    return {
        "ray_directions": gt_ray_directions,
        "z_depths": gt_z_depths,
        "poses": gt_poses,
        "pts3d": gt_pts,
        "metric_scale": gt_metric_norm,
    }, {
        "ray_directions": pr_ray_directions,
        "z_depths": pr_z_depths,
        "poses": pr_poses,
        "pts3d": pr_pts,
        "metric_scale": pr_metric_norm,
    }, valid_masks

@torch.no_grad()
def kmeans_merge(x: torch.Tensor, k: int, iters: int = 6, metric: str = "cosine"):
    assert x.dim() == 2
    n, c = x.shape
    k = min(k, n)
    if k == n:
        return x.clone(), torch.ones(k, 1, device=x.device, dtype=x.dtype)

    if metric == "cosine":
        x_norm = F.normalize(x, dim=-1)
    else:
        x_norm = x

    idx = torch.randperm(n, device=x.device)[:k]
    centers = x_norm[idx].clone()
    true_centers = x[idx].clone()

    for _ in range(iters):
        if metric == "cosine":
            sim = torch.matmul(x_norm, centers.T)
            labels = sim.argmax(dim=1)
        else:
            dist = torch.cdist(x_norm, centers, p=2)
            labels = dist.argmin(dim=1)

        counts = torch.bincount(labels, minlength=k).clamp_(min=1).to(x.dtype)
        sums = torch.zeros(k, c, device=x.device, dtype=x.dtype)
        sums.index_add_(0, labels, x)
        new_centers = sums / counts.unsqueeze(1)

        shift = (new_centers - true_centers).pow(2).mean()
        true_centers = new_centers
        if shift < 1e-6:
            break

        centers = F.normalize(true_centers, dim=-1) if metric == "cosine" else true_centers

    assign_weights = torch.sqrt(counts).unsqueeze(1)
    return true_centers, assign_weights


class BlockwiseAggregator(nn.Module):
    def __init__(
        self,
        num_blocks: int = 24,
        tokens_per_block: Union[int, List[int]] = 256,
        kmeans_iters: int = 6,
        metric: str = "cosine",
        pre_cap: Optional[int] = 8192,
        weight_scale: bool = True,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        if isinstance(tokens_per_block, int):
            self.tokens_per_block = [tokens_per_block] * num_blocks
        else:
            assert len(tokens_per_block) == num_blocks
            self.tokens_per_block = tokens_per_block
        self.kmeans_iters = kmeans_iters
        self.metric = metric
        self.pre_cap = pre_cap
        self.weight_scale = weight_scale

    @staticmethod
    def _to_tokens(feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 2:
            return feat.contiguous()
        elif feat.dim() == 3:
            C, H, W = feat.shape
            return feat.permute(1, 2, 0).reshape(H * W, C).contiguous()
        elif feat.dim() == 4:
            B, C, H, W = feat.shape
            return feat.permute(0, 2, 3, 1).reshape(B * H * W, C).contiguous()
        else:
            raise ValueError(f"Unsupported feat dim={feat.dim()}, expected 2/3/4.")

    @torch.no_grad()
    def forward(self, memory_feats: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        assert len(memory_feats) == self.num_blocks, \
            f"期望 {self.num_blocks} 个块，收到 {len(memory_feats)}"

        memory_tokens_per_block: List[torch.Tensor] = []

        for b in range(self.num_blocks):
            views = memory_feats[b]
            assert len(views) > 0, f"block {b} 为空"
            first_valid = None
            for t in views:
                if t is not None:
                    first_valid = t
                    break
            if first_valid is None:
                raise ValueError(f"block {b} 全为 None")

            dev = first_valid.device
            dtype = first_valid.dtype

            tokens_b = []
            for v, feat in enumerate(views):
                if feat is None:
                    continue
                if feat.device != dev:
                    feat = feat.to(dev, non_blocking=True)
                tb = self._to_tokens(feat)
                assert tb.dim() == 2, f"block {b}, view {v} 转换失败，得到维度 {tb.dim()}"
                tokens_b.append(tb)

            if len(tokens_b) == 0:
                memory_tokens_per_block.append(
                    torch.empty(
                        0,
                        first_valid.shape[-3] if first_valid.dim() >= 3 else first_valid.shape[-1],
                        device=dev,
                        dtype=dtype,
                    )
                )
                continue

            x = torch.cat(tokens_b, dim=0)
            if self.pre_cap is not None and x.shape[0] > self.pre_cap:
                idx = torch.randperm(x.shape[0], device=dev)[: self.pre_cap]
                x = x.index_select(0, idx)

            Kb = int(self.tokens_per_block[b])
            centers, weights = kmeans_merge(x, k=Kb, iters=self.kmeans_iters, metric=self.metric)
            if self.weight_scale:
                centers = centers * weights

            memory_tokens_per_block.append(centers.to(device=dev, dtype=dtype))

        return memory_tokens_per_block


@torch.no_grad()
def run_demo(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = not cfg.disable_cudnn_benchmark

    # AMP 类型
    if cfg.amp:
        if cfg.amp_dtype == "fp16":
            amp_dtype = torch.float16
        elif cfg.amp_dtype == "bf16":
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                warnings.warn("bf16 is not supported on this device. Using fp16 instead.")
                amp_dtype = torch.float16
        elif cfg.amp_dtype == "fp32":
            amp_dtype = torch.float32
    else:
        amp_dtype = torch.float32

    # 1. 初始化模型
    model = init_model(cfg.model.model_str, cfg.model.model_config, torch_hub_force_reload=False)
    model.to(device).eval()

    # 2. 加载预训练权重
    if cfg.model.pretrained:
        print("[Demo] Loading pretrained:", cfg.model.pretrained)
        ckpt = torch.load(cfg.model.pretrained, map_location=device, weights_only=False)
        print(model.load_state_dict(ckpt["model"], strict=False))
        del ckpt

    # 3. 加载记忆特征 (Memory)
    print(f"[Demo] Loading AA memory from: {cfg.fusion.stored_feature_file}")
    memory_feats, memory_scale_token = load_memory_features(cfg.fusion.stored_feature_file, device)
    printer.print(memory_feats,'memory_feats')

    # 4. 初始化数据集（这里示例 num_views=4，可按需在 cfg 里配置）
    print(f"[Demo] Initializing Dataset with num_views=4 ...")
    dataset = SevenScenesWAI(
        num_views=1,
        split="train",
        covisibility_thres=0,
        ROOT="/data/xwh/mapanything-dataset/wai_data/7scenes",
        dataset_metadata_dir="",
        sample_specific_scene=True,
        specific_scene_name="chess_train",
        resolution=(518, 392),
        transform="imgnorm",
        data_norm_type="dinov2",
        sequential_view_mode=False,
        seed=777
    )
    data_loader = ForcedRandomDataLoader(dataset=dataset, batch_size=1)

    output_dir = Path(cfg.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5. 初始化指标容器
    metrics_accumulator = {
        "pointmaps_abs_rel": [],
        "pointmaps_inlier_thres_103": [],
        "z_depth_abs_rel": [],
        "z_depth_inlier_thres_103": [],
        "ray_dirs_err_deg": [],
        "pose_ate_rmse": [],
        "pose_auc_5": [],
        "metric_scale_abs_rel": [],
    }
    saved_items = []
    target_num_views: Optional[int] = None

    print("[Demo] Running evaluation loop...")

    # ===================== 多视图主循环（改为通用 n_views） =====================
    for batch_id, batch in enumerate(data_loader):
        n_views = len(batch)
        if target_num_views is None:
            target_num_views = n_views

        # 清理 idx
        for view in batch:
            if "idx" in view:
                view["idx"] = view["idx"][2:]

        # 数据集名称 / 场景名称，用于 save_filename
        ds_name = batch[0].get("dataset", ["unknown"])[0]
        if isinstance(ds_name, torch.Tensor):
            ds_name = ds_name.item() if ds_name.ndim == 0 else str(ds_name)
        scene_name = batch[0].get("label", ["unknown"])[0]
        print(f"\n[Debug] Processing Batch: Dataset={ds_name}, Scene={scene_name}, Views={n_views}")
        for i, view in enumerate(batch):
            # 获取 instance 字段，它通常包含文件名或相对路径
            file_path = view.get("instance", ["unknown"])

            # DataLoader 可能会把字符串打包成列表 (例如 ['path/to/img.jpg'])，这里解包取第一个
            if isinstance(file_path, (list, tuple)):
                file_path = file_path[0]

            print(f"  View {i}: {file_path}")
        if isinstance(scene_name, torch.Tensor):
            scene_name = scene_name.item() if scene_name.ndim == 0 else str(scene_name)

        current_save_filename = f"{ds_name}_{scene_name}_{n_views}v_intermediates.pt"
        current_save_filename = current_save_filename.replace("/", "_").replace(" ", "")

        # 上 GPU
        ignore_keys = {
            "dataset",
            "label",
            "instance",
            "idx",
            "true_shape",
            "rng",
            "data_norm_type",
        }
        for view in batch:
            for name in list(view.keys()):
                if name in ignore_keys:
                    continue
                view[name] = view[name].to(device, non_blocking=True)

        # 模型推理（支持多视图输入）
        with torch.autocast("cuda", enabled=bool(cfg.amp), dtype=amp_dtype):
            # preds = model(batch, cfg.memory_efficient_inference, save_filename=current_save_filename)
            preds = model.forward_with_memory(
            query_view=batch,
            device=device,
            memory_feats=memory_feats,
            additional_tokens=memory_scale_token,
            memory_efficient_inference=cfg.memory_efficient_inference,
            )
            printer.print(preds, '')  # 如需 debug 可打开

        # GT / Pred 信息整理（内部已经支持多视图）
        gt_info, pr_info, valid_masks = get_all_info_for_metric_computation(batch, preds)

        # ----- 逐 batch 元素、逐视图累积指标 -----
        batch_size = batch[0]["img"].shape[0]

        for batch_idx in range(batch_size):
            pointmaps_abs_rel_across_views = []
            pointmaps_inlier_thres_103_across_views = []
            z_depth_abs_rel_across_views = []
            z_depth_inlier_thres_103_across_views = []
            ray_dirs_err_deg_across_views = []

            gt_poses_curr_set = []
            pr_poses_curr_set = []

            for view_idx in range(n_views):
                valid_mask_curr_view = valid_masks[view_idx][batch_idx].numpy()

                # A. 3D 点云误差
                pointmaps_abs_rel_curr_view = m_rel_ae(
                    gt=gt_info["pts3d"][view_idx][batch_idx].numpy(),
                    pred=pr_info["pts3d"][view_idx][batch_idx].numpy(),
                    mask=valid_mask_curr_view,
                )
                pointmaps_inlier_thres_103_curr_view = thresh_inliers(
                    gt=gt_info["pts3d"][view_idx][batch_idx].numpy(),
                    pred=pr_info["pts3d"][view_idx][batch_idx].numpy(),
                    mask=valid_mask_curr_view,
                    thresh=1.03,
                )

                # B. 深度误差
                z_depth_abs_rel_curr_view = m_rel_ae(
                    gt=gt_info["z_depths"][view_idx][batch_idx].numpy(),
                    pred=pr_info["z_depths"][view_idx][batch_idx].numpy(),
                    mask=valid_mask_curr_view,
                )
                z_depth_inlier_thres_103_curr_view = thresh_inliers(
                    gt=gt_info["z_depths"][view_idx][batch_idx].numpy(),
                    pred=pr_info["z_depths"][view_idx][batch_idx].numpy(),
                    mask=valid_mask_curr_view,
                    thresh=1.03,
                )

                pointmaps_abs_rel_across_views.append(pointmaps_abs_rel_curr_view)
                pointmaps_inlier_thres_103_across_views.append(pointmaps_inlier_thres_103_curr_view)
                z_depth_abs_rel_across_views.append(z_depth_abs_rel_curr_view)
                z_depth_inlier_thres_103_across_views.append(z_depth_inlier_thres_103_curr_view)

                # C. 光线方向误差
                ray_dirs_l2 = torch.norm(
                    gt_info["ray_directions"][view_idx][batch_idx]
                    - pr_info["ray_directions"][view_idx][batch_idx],
                    dim=-1,
                )
                ray_dirs_err_deg_curr_view = l2_distance_of_unit_ray_directions_to_angular_error(ray_dirs_l2)
                ray_dirs_err_deg_curr_view = torch.mean(ray_dirs_err_deg_curr_view)
                ray_dirs_err_deg_across_views.append(ray_dirs_err_deg_curr_view.cpu().numpy())

                # 轨迹
                gt_poses_curr_set.append(gt_info["poses"][view_idx][batch_idx])
                pr_poses_curr_set.append(pr_info["poses"][view_idx][batch_idx])

            # 各视图平均
            pointmaps_abs_rel_curr_set = np.mean(pointmaps_abs_rel_across_views)
            pointmaps_inlier_thres_103_curr_set = np.mean(pointmaps_inlier_thres_103_across_views)
            z_depth_abs_rel_curr_set = np.mean(z_depth_abs_rel_across_views)
            z_depth_inlier_thres_103_curr_set = np.mean(z_depth_inlier_thres_103_across_views)
            ray_dirs_err_deg_curr_set = np.mean(ray_dirs_err_deg_across_views)

            # D. 位姿误差
            pose_ate_curr_set = evaluate_ate(gt_traj=gt_poses_curr_set, est_traj=pr_poses_curr_set)

            gt_poses_stack = torch.stack(gt_poses_curr_set)
            pr_poses_stack = torch.stack(pr_poses_curr_set)
            rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
                pred_se3=pr_poses_stack,
                gt_se3=gt_poses_stack,
                num_frames=pr_poses_stack.shape[0],
            )

            rError = rel_rangle_deg.cpu().numpy()
            tError = rel_tangle_deg.cpu().numpy()
            pose_auc_5_curr_set, _ = calculate_auc_np(rError, tError, max_threshold=5)
            pose_auc_5_curr_set = pose_auc_5_curr_set * 100.0

            # E. 尺度误差（如果有）
            if gt_info["metric_scale"] is not None and pr_info["metric_scale"] is not None:
                gt_metric_scale_curr_set = gt_info["metric_scale"][batch_idx].numpy()
                pr_metric_scale_curr_set = pr_info["metric_scale"][batch_idx].numpy()
                metric_scale_err_curr_set = pr_metric_scale_curr_set - gt_metric_scale_curr_set
                metric_scale_abs_rel_curr_set = (
                    np.abs(metric_scale_err_curr_set) / gt_metric_scale_curr_set
                )
                metrics_accumulator["metric_scale_abs_rel"].append(metric_scale_abs_rel_curr_set.item())

            # 全局累积
            metrics_accumulator["pointmaps_abs_rel"].append(pointmaps_abs_rel_curr_set.item())
            metrics_accumulator["pointmaps_inlier_thres_103"].append(pointmaps_inlier_thres_103_curr_set.item())
            metrics_accumulator["z_depth_abs_rel"].append(z_depth_abs_rel_curr_set.item())
            metrics_accumulator["z_depth_inlier_thres_103"].append(z_depth_inlier_thres_103_curr_set.item())
            metrics_accumulator["ray_dirs_err_deg"].append(ray_dirs_err_deg_curr_set.item())
            metrics_accumulator["pose_ate_rmse"].append(pose_ate_curr_set.item())
            metrics_accumulator["pose_auc_5"].append(pose_auc_5_curr_set.item())

        # ---- 可视化保存：只对前 3 个 batch 的 view0 做 PNG + PLY ----
        if batch_id < 3:
            n_views = len(batch)
            batch_img_paths = []

            all_points = []
            all_colors = []

            for v in range(n_views):
                # 1) 当前视图的重建点云和 RGB
                reconstruction_v = preds[v]["pts3d"]  # [B, H, W, 3]
                rgb_tensor_v = batch[v]["img"][0]  # [C, H, W]
                rgb_v = rgb_tensor_v.permute(1, 2, 0).cpu()
                if rgb_v.dtype in (torch.float32, torch.float64):
                    rgb_v = (rgb_v.clamp(0, 1) * 255).byte()

                # 2) 保存当前视图 PNG
                png_path_v = output_dir / f"sample_{batch_id:04d}_view{v}.png"
                imageio.imwrite(png_path_v, rgb_v.numpy())
                batch_img_paths.append(str(png_path_v))

                # 3) 展平当前视图点云 + 颜色，并做 NaN / Inf 掩码
                recon_np_v = (
                    reconstruction_v.detach()
                    .cpu()
                    .numpy()[0]
                    .reshape(-1, 3)
                    .astype(np.float64)
                )
                mask_v = np.isfinite(recon_np_v).all(axis=1)
                recon_np_v = recon_np_v[mask_v]

                rgb_np_v = (
                        rgb_v.detach()
                        .cpu()
                        .numpy()
                        .reshape(-1, 3)
                        .astype(np.float64) / 255.0
                )
                rgb_np_v = rgb_np_v[mask_v]

                all_points.append(recon_np_v)
                all_colors.append(rgb_np_v)

            # 4) 合并所有视图的点云 + 颜色，并保存为一个 PLY
            if len(all_points) > 0:
                merged_points = np.concatenate(all_points, axis=0)
                merged_colors = np.concatenate(all_colors, axis=0)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(merged_points)
                if merged_colors.shape[0] == merged_points.shape[0]:
                    pcd.colors = o3d.utility.Vector3dVector(merged_colors)

                ply_path = output_dir / f"sample_{batch_id:04d}_merged.ply"
                o3d.io.write_point_cloud(str(ply_path), pcd)

                saved_items.append(
                    {
                        "index": batch_id,
                        "images": batch_img_paths,  # 所有视图 PNG 路径
                        "ply_merged": str(ply_path),  # 合并点云路径
                    }
                )
        else:
            # 已经处理完前三个 batch，提前结束整个 demo
            print("已保存 3 个 batch 的所有视图和合并点云，提前结束 demo。")
            break

    # 7. 打印并保存最终平均指标
    print("\n" + "=" * 50)
    print(f"  Quantitative Results (Num Views: {target_num_views})")
    print("=" * 50)

    final_results = {}
    for key, values in metrics_accumulator.items():
        if len(values) > 0:
            mean_val = np.nanmean(values)
            final_results[key] = mean_val
            print(f"{key:<35}: {mean_val:.4f}")
        else:
            print(f"{key:<35}: N/A")

    json_path = output_dir / "metrics_summary.json"
    with open(json_path, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"\n[Demo] Metrics saved to {json_path}")
    return {"output_dir": str(output_dir), "metrics": final_results}


@hydra.main(version_base=None, config_path="../../configs", config_name="memory")
def main(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    run_demo(cfg)


if __name__ == "__main__":
    main()
