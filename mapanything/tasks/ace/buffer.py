from __future__ import annotations
import torch
import numpy as np
import imageio
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig, open_dict
import torch.nn.functional as F

from mapanything.tasks.ace.geometry import (
    _prepare_targets, _resolve_intrinsics, _resolve_pose
)


@dataclass
class BufferTensors:
    features: torch.Tensor
    scale_token: torch.Tensor
    target_world: torch.Tensor
    pixels: torch.Tensor
    intrinsics: torch.Tensor
    c2w: torch.Tensor


class FeatureReplayBuffer:
    """仿照 ACE，把随机采样的像素特征/位姿放入固定大小的缓冲区。"""

    def __init__(self, buffer_size: int, feat_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.size = 0
        self.storage = BufferTensors(
            features=torch.empty((buffer_size, feat_dim), device=device, dtype=torch.float32),
            scale_token=torch.empty((buffer_size, feat_dim), device=device, dtype=torch.float32),
            target_world=torch.empty((buffer_size, 3), device=device, dtype=torch.float32),
            pixels=torch.empty((buffer_size, 2), device=device, dtype=torch.float32),
            intrinsics=torch.empty((buffer_size, 3, 3), device=device, dtype=torch.float32),
            c2w=torch.empty((buffer_size, 4, 4), device=device, dtype=torch.float32),
        )

    @property
    def is_full(self) -> bool:
        return self.size >= self.buffer_size

    def _remaining(self) -> int:
        return max(0, self.buffer_size - self.size)

    def add_view(
            self,
            fused_feature: torch.Tensor,  # [1,C,H,W] 或 [C,H,W]
            scale_token: torch.Tensor,
            target_world: torch.Tensor,  # [H,W,3]
            valid_mask: torch.Tensor,  # [H,W]
            intrinsics: torch.Tensor,  # [3,3]
            c2w: torch.Tensor,  # [4,4]
            max_per_view: int,
            pixel_grid: Optional[torch.Tensor] = None,  # [H, W, 2]
            importance_map: Optional[torch.Tensor] = None,  # [H, W] 置信度图
            sampling_mode: str = "topk"  # 'topk' 或 'weighted'
    ) -> int:
        """从特征图采样像素，支持基于置信度的采样。"""
        if self.is_full:
            return 0

        # -------- 形状归一：features -> [C,H,W] --------
        if fused_feature.dim() == 4:
            b, c, h, w = fused_feature.shape
            assert b == 1
            feat_CHW = fused_feature[0]
        else:
            c, h, w = fused_feature.shape
            feat_CHW = fused_feature

        # -------- 形状归一：scale_token --------
        st = scale_token.view(-1)
        scale_vec = st.to(feat_CHW.device).contiguous()

        # -------- target_world -> [H,W,3] --------
        if target_world.dim() == 4:
            tw_HWC = target_world[0].permute(1, 2, 0).contiguous()
        elif target_world.dim() == 3:
            if target_world.shape[0] == 3:
                tw_HWC = target_world.permute(1, 2, 0).contiguous()
            else:
                tw_HWC = target_world.contiguous()
        else:
            tw_HWC = target_world  # Assume correct if matched

        # -------- valid_mask -> [H,W] --------
        if valid_mask is None:
            vm_HW = torch.ones((h, w), dtype=torch.bool, device=feat_CHW.device)
        else:
            if valid_mask.dim() == 4:
                vm_HW = valid_mask[0, 0].bool()
            elif valid_mask.dim() == 3:
                vm_HW = valid_mask[0].bool()
            else:
                vm_HW = valid_mask.bool()

        # -------- importance_map -> Flatten --------
        importance_flat = None
        if importance_map is not None:
            # 简单的resize以防万一（例如特征图有padding差异）
            if importance_map.shape != (h, w):
                importance_map = importance_map.view(1, 1, *importance_map.shape)
                importance_map = F.interpolate(importance_map, size=(h, w), mode='bilinear', align_corners=False)
                importance_map = importance_map.view(h, w)
            importance_flat = importance_map.view(-1)

        # -------- 展平并构建像素坐标 --------
        feats_flat = feat_CHW.view(c, -1).t()  # [N,C]
        targets_flat = tw_HWC.view(-1, 3)  # [N,3]
        mask_flat = vm_HW.view(-1)  # [N]
        if not mask_flat.any().item():
            return 0

        if pixel_grid is not None:
            pixels_flat = pixel_grid.to(feat_CHW.device).view(-1, 2)
        else:
            ys, xs = torch.meshgrid(
                torch.arange(h, device=feat_CHW.device),
                torch.arange(w, device=feat_CHW.device),
                indexing="ij"
            )
            pixels_flat = torch.stack([xs, ys], dim=-1).view(-1, 2)

        # -------- 采样逻辑 (Top-K / Weighted / Random) --------
        valid_idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            return 0

        remaining = self._remaining()
        num_to_take = min(max_per_view, valid_idx.numel(), remaining)
        if num_to_take <= 0:
            return 0

        if importance_flat is not None:
            # 取出有效像素对应的分数
            scores = importance_flat[valid_idx]

            if sampling_mode == "topk":
                # 策略: 选取分数最高的 K 个点
                if num_to_take >= scores.numel():
                    chosen = valid_idx
                else:
                    _, topk_indices = torch.topk(scores, k=num_to_take)
                    chosen = valid_idx[topk_indices]
            elif sampling_mode == "weighted":
                # 策略: 按分数加权随机采样
                weights = scores + 1e-6
                sampled_indices = torch.multinomial(weights, num_to_take, replacement=False)
                chosen = valid_idx[sampled_indices]
            else:
                # Fallback to random
                perm = torch.randperm(valid_idx.numel(), device=feat_CHW.device)
                chosen = valid_idx[perm[:num_to_take]]
        else:
            # 原始随机策略
            perm = torch.randperm(valid_idx.numel(), device=feat_CHW.device)
            chosen = valid_idx[perm[:num_to_take]]

        # -------- 写入缓冲区 --------
        slot = slice(self.size, self.size + chosen.numel())
        self.storage.features[slot] = feats_flat[chosen].to(self.device)
        self.storage.target_world[slot] = targets_flat[chosen].to(self.device)
        self.storage.pixels[slot] = pixels_flat[chosen].to(self.device)
        self.storage.intrinsics[slot] = intrinsics.to(self.device)
        self.storage.c2w[slot] = c2w.to(self.device)
        self.storage.scale_token[slot] = scale_vec.unsqueeze(0).expand(chosen.numel(), -1).to(self.device)
        self.size += chosen.numel()

        return int(chosen.numel())


class BufferDataset(Dataset):
    def __init__(self, buffer: FeatureReplayBuffer):
        self.buffer = buffer

    def __len__(self) -> int:
        return self.buffer.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.buffer.storage.features[idx],
            "scale_token": self.buffer.storage.scale_token[idx],
            "target_world": self.buffer.storage.target_world[idx],
            "pixels": self.buffer.storage.pixels[idx],
            "intrinsics": self.buffer.storage.intrinsics[idx],
            "c2w": self.buffer.storage.c2w[idx],
        }


def _collect_buffer(
        cfg: DictConfig,
        model,
        upsampler,
        dataset,
        memory_feats,
        memory_token,
        device: torch.device,
        capacity_override: int | None = None,
        confidence_predictor: torch.nn.Module | None = None,
        confidence_hook: Any | None = None,
) -> Tuple[FeatureReplayBuffer, int]:
    buffer: FeatureReplayBuffer | None = None

    # 1. 确定 Buffer 容量 (优先使用 override 值)
    if capacity_override is not None:
        capacity = int(capacity_override)
    else:
        capacity = int(getattr(cfg.training, "buffer_size", getattr(cfg.training, "buffer_capacity", 0)))

    if capacity <= 0:
        raise ValueError("Buffer capacity must be positive.")

    # 2. 进度条与 Debug 路径
    pbar = tqdm(total=capacity, desc="Filling buffer", unit="sample", leave=False)
    debug_dir = Path("debug_vis")  # 可选：用于保存 Debug 图像
    debug_dir.mkdir(exist_ok=True)

    try:
        for batch_id, views in enumerate(dataset):

            # [Debug] 打印前 10 个 Batch 的图片名称 (用于验证数据加载)
            if batch_id < 10:
                print(f"\n[Buffer Collection] Batch {batch_id}:")
                for v_idx, v in enumerate(views):
                    # 尝试获取文件名，不同数据集 key 可能不同，做兼容处理
                    img_name = v.get('image_path',
                                     v.get('file_path', v.get('name', f"Index {v.get('idx', 'Unknown')}")))
                    # 如果是完整路径，只取最后一部分保持简洁
                    if isinstance(img_name, (str, Path)):
                        img_name = Path(img_name).name
                    print(f"  - View {v_idx}: {img_name}")

            batch = views
            for view in batch:
                if "idx" in view: view["idx"] = view["idx"][2:]

            ignore_keys = {"dataset", "label", "instance", "idx", "true_shape", "rng", "data_norm_type"}
            for view in batch:
                for name in list(view.keys()):
                    if name in ignore_keys: continue
                    view[name] = view[name].to(device, non_blocking=True)

            with torch.no_grad():
                # 3. 特征提取 (触发 Hook)
                fused_feature, fused_token = model.forward_dense_feats(
                    batch,
                    cfg.training.memory_efficient_inference
                )
                # fused_feature: [1, C, H, W]
                feat_down = fused_feature

                # 4. [置信度推理] (含 Hook 处理与退化机制)
                importance_map = None
                if confidence_predictor is not None:
                    try:
                        # 情况 A: 有 Hook 数据 (CoMe 标准逻辑)
                        if confidence_hook is not None and confidence_hook.features is not None:
                            raw_feat = confidence_hook.features  # [B, N, C_in]

                            # 解析形状: 将 Sequence 还原为 Spatial
                            B_raw, N_raw, C_raw = raw_feat.shape
                            H_f, W_f = fused_feature.shape[-2:]  # 参考最终特征图的 H, W
                            num_spatial = H_f * W_f

                            # 简单的 Token 截取策略 (适配 DINOv2 可能存在的 CLS/Register tokens)
                            if N_raw == num_spatial:
                                spatial_tokens = raw_feat
                            elif N_raw == num_spatial + 1:
                                spatial_tokens = raw_feat[:, 1:, :]  # Skip CLS
                            elif N_raw == num_spatial + 5:
                                spatial_tokens = raw_feat[:, 5:, :]  # Skip CLS + 4 Reg
                            else:
                                # 形状不匹配时的保底: 取最后 N 个
                                spatial_tokens = raw_feat[:, -num_spatial:, :]

                            # Reshape: [B, H*W, C] -> [B, C, H, W]
                            spatial_tokens = spatial_tokens.reshape(B_raw, H_f, W_f, C_raw).permute(0, 3, 1, 2)

                            # 推理: [B, 1, H, W]
                            conf_out = confidence_predictor(spatial_tokens.to(dtype=torch.float32))
                            importance_map = conf_out[0, 0]  # [H, W]

                        # 情况 B: 没有 Hook 但有 Predictor (尝试直接预测最终特征)
                        # 注意: 除非 Predictor 是针对最终层训练的，否则效果可能不佳，这里作为一种兼容性保留
                        elif importance_map is None:
                            conf_out = confidence_predictor(feat_down)
                            importance_map = conf_out[0, 0]

                    except Exception as e:
                        # 仅在第一个 Batch 打印警告，避免刷屏
                        if batch_id == 0:
                            print(f"[Warning] Confidence prediction failed: {e}. Fallback to Random Sampling.")
                        importance_map = None
                    finally:
                        # [重要] 必须清理 Hook，否则显存会爆炸
                        if confidence_hook is not None:
                            confidence_hook.clear()

            view = batch[0]
            target_world_full, valid_mask_full = _prepare_targets(view, device)

            # 5. 坐标网格生成 (对应 feat_down 的分辨率)
            H0, W0 = target_world_full.shape[-2:]
            Hf, Wf = feat_down.shape[-2:]

            ys, xs = torch.meshgrid(
                torch.arange(Hf, device=device),
                torch.arange(Wf, device=device),
                indexing='ij'
            )
            # 计算每个特征点对应的原图中心坐标
            u_c = (xs + 0.5) * (W0 / Wf)
            v_c = (ys + 0.5) * (H0 / Hf)

            # 构建归一化 Grid 用于 Grid Sample
            grid_x = (u_c / (W0 - 1)) * 2 - 1
            grid_y = (v_c / (H0 - 1)) * 2 - 1
            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, Hf, Wf, 2]

            # 6. 下采样 GT 和 Mask
            target_world_down = F.grid_sample(target_world_full, grid, mode='nearest', align_corners=True)
            target_world_for_buffer = target_world_down.squeeze(0).permute(1, 2, 0)

            valid_mask_down = F.grid_sample(valid_mask_full.float(), grid, mode='nearest', align_corners=True)
            valid_mask_for_buffer = valid_mask_down.squeeze(0).squeeze(0).bool()

            pixel_grid_for_buffer = torch.stack([u_c, v_c], dim=-1)

            intrinsics = _resolve_intrinsics(view, device).to(torch.float32)
            c2w = _resolve_pose(view, device).to(torch.float32)

            if buffer is None:
                in_channels = int(feat_down.shape[1])
                buffer = FeatureReplayBuffer(capacity, in_channels, device=torch.device("cpu"))

            # 7. 读取采样模式配置
            sampling_mode = getattr(cfg.training, "sampling_mode", "topk")

            # 8. 写入 Buffer (核心调用)
            # add_view 内部逻辑: 如果 importance_map 为 None，自动退化为 torch.randperm
            added = buffer.add_view(
                feat_down.squeeze(0).detach(),
                fused_token.squeeze(0).detach(),
                target_world_for_buffer.detach(),
                valid_mask_for_buffer.detach(),
                intrinsics.squeeze(0).detach(),
                c2w.squeeze(0).detach(),
                cfg.training.samples_per_view,
                pixel_grid=pixel_grid_for_buffer.detach(),
                importance_map=importance_map.detach() if importance_map is not None else None,
                sampling_mode=sampling_mode
            )

            if added > 0:
                pbar.update(added)
            if buffer.is_full:
                break

        if buffer is None or buffer.size == 0:
            raise RuntimeError("Buffer empty. No valid samples collected.")
        return buffer, buffer.storage.features.shape[1]

    finally:
        pbar.close()