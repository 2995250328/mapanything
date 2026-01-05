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
import matplotlib.pyplot as plt

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
        cfg: DictConfig, model, upsampler, dataset, memory_feats, memory_token, device: torch.device,
        capacity_override: int | None = None, confidence_predictor: torch.nn.Module | None = None, confidence_hook: Any | None = None,
) -> Tuple[FeatureReplayBuffer, int]:
    buffer: FeatureReplayBuffer | None = None
    capacity = int(capacity_override) if capacity_override is not None else int(getattr(cfg.training, "buffer_size", 0))
    if capacity <= 0: raise ValueError("Buffer capacity must be positive.")

    pbar = tqdm(total=capacity, desc="Filling buffer", unit="sample", leave=False)
    debug_vis_dir = Path("debug_vis_confidence"); debug_vis_dir.mkdir(exist_ok=True, parents=True)

    try:
        for batch_id, views in enumerate(dataset):
            batch = views
            for view in batch:
                if "idx" in view: view["idx"] = view["idx"][2:]

            ignore_keys = {"dataset", "label", "instance", "idx", "true_shape", "rng", "data_norm_type"}
            for view in batch:
                for name in list(view.keys()):
                    if name in ignore_keys: continue
                    view[name] = view[name].to(device, non_blocking=True)

            with torch.no_grad():
                fused_feature, fused_token = model.forward_dense_feats(batch, cfg.training.memory_efficient_inference)
                feat_down = fused_feature
                importance_map = None

                if confidence_predictor is not None:
                    try:
                        if confidence_hook is not None and confidence_hook.features is not None:
                            raw_feat = confidence_hook.features
                            if batch_id == 0:  # 只在第一个 Batch 打印，防止刷屏
                                B, N, C = raw_feat.shape
                                Hf, Wf = feat_down.shape[-2:] # MapAnything 输出的特征图高度和宽度
                                expected_spatial_tokens = Hf * Wf

                                print("\n" + "="*50)
                                print(f" [DEBUG] Hook Feature Diagnosis")
                                print(f" > Hook Output Shape (B, N, C): {raw_feat.shape}")
                                print(f" > MapAnything Feat Shape (H, W): ({Hf}, {Wf})")
                                print(f" > Expected Spatial Tokens (H*W): {expected_spatial_tokens}")
                                print(f" > Difference (N - H*W): {N - expected_spatial_tokens}")

                                diff = N - expected_spatial_tokens
                                if diff == 0:
                                    print(" > Type: Pure Spatial (No CLS, No Registers)")
                                elif diff == 1:
                                    print(" > Type: Standard ViT (1 CLS + Spatial)")
                                elif diff == 5:
                                    print(" > Type: DINOv2 with Registers (1 CLS + 4 Registers + Spatial)")
                                else:
                                    print(f" > Type: Unknown structure! Diff={diff}")
                                print("="*50 + "\n")
                                import matplotlib.pyplot as plt

                                # 尝试按照不同假设进行 Reshape 和可视化
                                # 假设 1: 只有 CLS (跳过 1 个)
                                try:
                                    feat_cls = raw_feat[:, 1:, :].reshape(B, Hf, Wf, C).permute(0, 3, 1, 2)
                                    # 计算通道平均热力图
                                    heatmap_cls = feat_cls[0].mean(dim=0).float().cpu().numpy()

                                    plt.figure(figsize=(10, 5))
                                    plt.subplot(1, 2, 1)
                                    plt.title("Assumption: Skip 1 (CLS only)")
                                    plt.imshow(heatmap_cls, cmap='viridis')
                                    plt.colorbar()
                                except: pass

                                # 假设 2: DINOv2 (跳过 5 个)
                                try:
                                    if N > 5:
                                        feat_reg = raw_feat[:, 5:, :].reshape(B, Hf, Wf, C).permute(0, 3, 1, 2)
                                        heatmap_reg = feat_reg[0].mean(dim=0).float().cpu().numpy()

                                        plt.subplot(1, 2, 2)
                                        plt.title("Assumption: Skip 5 (CLS+4 Reg)")
                                        plt.imshow(heatmap_reg, cmap='viridis')
                                        plt.colorbar()
                                except: pass

                                plt.savefig("debug_hook_feature_alignment.png")
                                print(" > Saved debug visualization to debug_hook_feature_alignment.png")
                            B_raw, N_raw, C_raw = raw_feat.shape
                            H_f, W_f = fused_feature.shape[-2:]
                            target_len = H_f * W_f
                            diff = N_raw - target_len
                            if diff == 0:
                                # 情况: 纯空间特征
                                spatial_tokens = raw_feat
                            elif diff == 1:
                                # 情况: 1 CLS
                                spatial_tokens = raw_feat[:, 1:, :]
                            elif diff == 5:
                                # 情况: 1 CLS + 4 Registers (DINOv2 default)
                                spatial_tokens = raw_feat[:, 5:, :]
                            elif diff == 9:
                                # 情况: 有些变体有 8 个 Registers
                                spatial_tokens = raw_feat[:, 9:, :]
                            else:
                                # 情况: 未知结构，尝试取最后 N 个 (保底策略)
                                # 警告：这可能还是会导致错位，如果 Register 在最后面的话
                                if batch_id == 0:
                                    print(f"[Warning] Unknown token structure! N={N_raw}, HW={target_len}, diff={diff}. Taking last {target_len}.")
                                spatial_tokens = raw_feat[:, -target_len:, :]

                            # Reshape [B, HW, C] -> [B, H, W, C] -> [B, C, H, W]
                            # 这一步如果不报错，且上面的切片是对的，那么空间位置就是对齐的
                            spatial_tokens = spatial_tokens.reshape(B_raw, H_f, W_f, C_raw).permute(0, 3, 1, 2)
                            importance_map = confidence_predictor(spatial_tokens.to(dtype=torch.float32))[0, 0]
                        elif importance_map is None:
                            importance_map = confidence_predictor(feat_down)[0, 0]
                    except Exception as e:
                        if batch_id == 0: tqdm.write(f"[Warning] Confidence failed: {e}. Fallback to Random.")
                        importance_map = None
                    finally:
                        if confidence_hook is not None: confidence_hook.clear()

            # --- 可视化修复版 (叠加热力图 + 分辨率处理) ---
            if batch_id < 5 and importance_map is not None:
                try:
                    # 1. 准备原图
                    img_t = batch[0]["img"].detach().cpu()
                    if img_t.dim() == 4: img_t = img_t.squeeze(0)
                    H_img, W_img = img_t.shape[-2:] # 获取原图尺寸
                    img_np = img_t.permute(1, 2, 0).numpy()
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)

                    # 2. 准备原始低分辨率热力图 (用于对比展示)
                    conf_np_raw = importance_map.detach().cpu().numpy()

                    # 3. 准备高分辨率叠加热力图 (核心逻辑)
                    # 将 [H_feat, W_feat] 扩展为 [1, 1, H_feat, W_feat] 以进行插值
                    heatmap_t = importance_map.detach().unsqueeze(0).unsqueeze(0)
                    # 双线性插值上采样到原图尺寸
                    heatmap_resized_t = F.interpolate(heatmap_t, size=(H_img, W_img), mode='bilinear', align_corners=False)
                    conf_np_resized = heatmap_resized_t.squeeze().cpu().numpy() # [H_img, W_img]
                    # 归一化到 0-1 以便更好地映射颜色
                    conf_np_resized = (conf_np_resized - conf_np_resized.min()) / (conf_np_resized.max() - conf_np_resized.min() + 1e-6)

                    # 4. 绘图 (3个子图：原图 | 原始热力图 | 叠加图)
                    plt.figure(figsize=(15, 5))

                    plt.subplot(1, 3, 1); plt.imshow(img_np); plt.title(f"Image ({W_img}x{H_img})"); plt.axis('off')

                    plt.subplot(1, 3, 2); plt.imshow(conf_np_raw, cmap='jet'); plt.colorbar()
                    plt.title(f"Raw Confidence ({conf_np_raw.shape[1]}x{conf_np_raw.shape[0]})"); plt.axis('off')

                    plt.subplot(1, 3, 3)
                    plt.imshow(img_np) # 先画背景原图
                    plt.imshow(conf_np_resized, cmap='jet', alpha=0.5) # 再叠加半透明热力图 (alpha=0.5)
                    plt.title("Overlay (Superimposed)"); plt.axis('off')

                    # 文件名处理
                    raw_name = batch[0].get('instance', str(batch_id))
                    if isinstance(raw_name, list): raw_name = raw_name[0]
                    if hasattr(raw_name, 'name'): name_str = raw_name.name
                    elif hasattr(raw_name, 'item'): name_str = str(raw_name.item())
                    else: name_str = str(raw_name)
                    safe_name = name_str.replace("/", "_").replace("\\", "_").replace("[", "").replace("]", "").replace("'", "")

                    save_path = debug_vis_dir / f"batch_{batch_id}_{safe_name}_overlay.png"
                    plt.tight_layout(); plt.savefig(save_path); plt.close()
                    tqdm.write(f"[Debug] Saved overlay vis: {save_path}")
                except Exception as e: tqdm.write(f"[Warn] Vis failed: {e}")
            # ---------------------------

            view = batch[0]
            target_world_full, valid_mask_full = _prepare_targets(view, device)
            H0, W0 = target_world_full.shape[-2:]
            Hf, Wf = feat_down.shape[-2:]

            ys, xs = torch.meshgrid(torch.arange(Hf, device=device), torch.arange(Wf, device=device), indexing='ij')
            u_c = (xs + 0.5) * (W0 / Wf); v_c = (ys + 0.5) * (H0 / Hf)
            grid = torch.stack([(u_c / (W0 - 1)) * 2 - 1, (v_c / (H0 - 1)) * 2 - 1], dim=-1).unsqueeze(0)

            target_world_down = F.grid_sample(target_world_full, grid, mode='nearest', align_corners=True)
            valid_mask_down = F.grid_sample(valid_mask_full.float(), grid, mode='nearest', align_corners=True)
            pixel_grid = torch.stack([u_c, v_c], dim=-1)
            intrinsics = _resolve_intrinsics(view, device).to(torch.float32)
            c2w = _resolve_pose(view, device).to(torch.float32)

            if buffer is None:
                buffer = FeatureReplayBuffer(capacity, int(feat_down.shape[1]), device=torch.device("cpu"))

            added = buffer.add_view(
                feat_down.squeeze(0).detach(), fused_token.squeeze(0).detach(),
                target_world_down.squeeze(0).permute(1, 2, 0).detach(),
                valid_mask_down.squeeze(0).squeeze(0).bool(),
                intrinsics.squeeze(0).detach(), c2w.squeeze(0).detach(),
                cfg.training.samples_per_view, pixel_grid=pixel_grid.detach(),
                importance_map=importance_map.detach() if importance_map is not None else None,
                sampling_mode=getattr(cfg.training, "sampling_mode", "topk")
            )
            if added > 0: pbar.update(added)
            if buffer.is_full: break

        if buffer is None or buffer.size == 0: raise RuntimeError("Buffer empty.")
        return buffer, buffer.storage.features.shape[1]
    finally:
        pbar.close()