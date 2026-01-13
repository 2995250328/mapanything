import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from typing import Dict, Tuple

from mapanything.tasks.ace.geometry import _invert_c2w_to_w2c

def weighted_tanh(repro_errs, weight):
    return weight * torch.tanh(repro_errs / weight).sum()


class ReproLoss:
    """
    Compute per-pixel reprojection loss using different configurable approaches.

    - tanh:     tanh loss with a constant scale factor given by the `soft_clamp` parameter (when a pixel's reprojection
                error is equal to `soft_clamp`, its loss is equal to `soft_clamp * tanh(1)`).
    - dyntanh:  Used in the paper, similar to the tanh loss above, but the scaling factor decreases during the course of
                the training from `soft_clamp` to `soft_clamp_min`. The decrease is linear, unless `circle_schedule`
                is True (default), in which case it applies a circular scheduling. See paper for details.
    - l1:       Standard L1 loss, computed only on those pixels having an error lower than `soft_clamp`
    - l1+sqrt:  L1 loss for pixels with reprojection error smaller than `soft_clamp` and
                `sqrt(soft_clamp * reprojection_error)` for pixels with a higher error.
    - l1+logl1: Similar to the above, but using log L1 for pixels with high reprojection error.
    """

    def __init__(self,
                 total_iterations,
                 soft_clamp,
                 soft_clamp_min,
                 type='dyntanh',
                 circle_schedule=True):

        self.total_iterations = total_iterations
        self.soft_clamp = soft_clamp
        self.soft_clamp_min = soft_clamp_min
        self.type = type
        self.circle_schedule = circle_schedule

    def compute(self, repro_errs_b1N, iteration):
        if repro_errs_b1N.nelement() == 0:
            return 0

        if self.type == "tanh":
            return weighted_tanh(repro_errs_b1N, self.soft_clamp)

        elif self.type == "dyntanh":
            # Compute the progress over the training process.
            schedule_weight = iteration / self.total_iterations

            if self.circle_schedule:
                # Optionally scale it using the circular schedule.
                schedule_weight = 1 - np.sqrt(1 - schedule_weight ** 2)

            # Compute the weight to use in the tanh loss.
            loss_weight = (1 - schedule_weight) * self.soft_clamp + self.soft_clamp_min

            # Compute actual loss.
            return weighted_tanh(repro_errs_b1N, loss_weight)

        elif self.type == "l1":
            # L1 loss on all pixels with small-enough error.
            softclamp_mask_b1 = repro_errs_b1N > self.soft_clamp
            return repro_errs_b1N[~softclamp_mask_b1].sum()

        elif self.type == "l1+sqrt":
            # L1 loss on pixels with small errors and sqrt for the others.
            softclamp_mask_b1 = repro_errs_b1N > self.soft_clamp
            loss_l1 = repro_errs_b1N[~softclamp_mask_b1].sum()
            loss_sqrt = torch.sqrt(self.soft_clamp * repro_errs_b1N[softclamp_mask_b1]).sum()

            return loss_l1 + loss_sqrt

        else:
            # l1+logl1: same as above, but use log(L1) for pixels with a larger error.
            softclamp_mask_b1 = repro_errs_b1N > self.soft_clamp
            loss_l1 = repro_errs_b1N[~softclamp_mask_b1].sum()
            loss_logl1 = torch.log(1 + (self.soft_clamp * repro_errs_b1N[softclamp_mask_b1])).sum()

            return loss_l1 + loss_logl1

def _compute_scale_reg(z_flat, target_world, c2w, loss_cfg, device):
    """计算尺度正则化 Loss"""
    sr = getattr(loss_cfg, "scale_reg", None)
    if not (sr and getattr(sr, "enabled", False)):
        return torch.tensor(0.0, device=device), 0.0

    variant = getattr(sr, "variant", "prior")
    weight = float(getattr(sr, "weight", 1e-3))
    depth_min = float(getattr(loss_cfg, "depth_min", 0.1))
    eps = float(getattr(loss_cfg, "eps", 1e-8))

    if weight <= 0 or z_flat.numel() == 0:
        return torch.tensor(0.0, device=device), 0.0

    med_pred = torch.median(z_flat)
    reg = torch.tensor(0.0, device=device)

    if variant == "match_to_gt":
        if target_world is not None and c2w is not None:
            R, t = _invert_c2w_to_w2c(c2w)
            Xw_gt = target_world.unsqueeze(-1)
            Zg = (R @ Xw_gt + t)[:, 2, 0].clamp_min(depth_min)
            med_gt = torch.median(Zg)
            reg = weight * torch.abs(torch.log((med_pred + eps) / (med_gt + eps)))
    elif variant == "unit":
        reg = weight * torch.abs(torch.log(med_pred + eps))
    elif variant == "prior":
        depth_prior = getattr(sr, "depth_prior", 1.0)
        reg = weight * torch.abs(torch.log((med_pred + eps) / (float(depth_prior) + eps)))

    return reg, float(reg.detach().cpu())

def save_batch_error_histograms(step, dist_3d_np, reproj_err_np, save_dir):
    """绘制并保存当前 Batch 的误差分布直方图"""
    save_path = save_dir / f"step_{step:06d}_dist.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if dist_3d_np is not None and len(dist_3d_np) > 0:
        limit_3d = np.percentile(dist_3d_np, 99) if len(dist_3d_np) > 100 else dist_3d_np.max()
        axes[0].hist(dist_3d_np, bins=50, range=(0, limit_3d), color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_title(f'3D Error Dist (m)\nMedian: {np.median(dist_3d_np):.3f}m')
        axes[0].set_xlabel('Error (m)')
        axes[0].set_ylabel('Count')
    else:
        axes[0].text(0.5, 0.5, 'No 3D Data', ha='center')

    if reproj_err_np is not None and len(reproj_err_np) > 0:
        limit_px = np.percentile(reproj_err_np, 99) if len(reproj_err_np) > 100 else reproj_err_np.max()
        axes[1].hist(reproj_err_np, bins=50, range=(0, limit_px), color='salmon', edgecolor='black', alpha=0.7)
        axes[1].set_title(f'Reproj Error Dist (px)\nMedian: {np.median(reproj_err_np):.2f}px')
        axes[1].set_xlabel('Error (px)')
    else:
        axes[1].text(0.5, 0.5, 'No Reproj Data', ha='center')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def _loss_fn(
        preds,
        batch: Dict[str, torch.Tensor],
        repro_loss,
        global_step,
        loss_cfg: DictConfig,
):
    device = preds.device
    mode = getattr(loss_cfg, "mode", "reproj")
    conf_mode = getattr(loss_cfg, "conf_mode", "log_sigma")

    sigma_min = float(getattr(loss_cfg, "sigma_min", 1e-4))
    sigma_max = float(getattr(loss_cfg, "sigma_max", 10.0))
    depth_min = float(getattr(loss_cfg, "depth_min", 0.1))
    depth_max = float(getattr(loss_cfg, "depth_max", 50.0))
    eps = float(getattr(loss_cfg, "eps", 1e-8))

    SANITY_PIXEL_ERR = 50000.0
    SANITY_COORD_VAL = 10000.0

    scale = None
    if isinstance(preds, (tuple, list)):
        preds, scale = preds
    elif isinstance(preds, dict):
        scale = preds.get("scale", None)
        preds = preds["preds"]

    if preds.dim() == 4:
        preds = preds.permute(0, 2, 3, 1).reshape(-1, preds.shape[1])

    if torch.isnan(preds).any() or torch.isinf(preds).any():
        print(f"[CRITICAL] Iter {global_step}: Input preds contain NaN/Inf! Returning zero loss.")
        return preds.sum() * 0.0, {"loss": 0.0, "mode": "nan_skipped"}

    coords_pred_in = preds[:, :3]
    raw = preds[:, 3]

    if conf_mode == "confidence":
        p = torch.sigmoid(raw)
        sigma = (1.0 - p) * sigma_max + p * sigma_min
    else:
        sigma = F.softplus(raw) + eps
        sigma = torch.clamp(sigma, min=sigma_min, max=sigma_max)

    if scale is not None:
        coords = coords_pred_in * scale.to(device).view(-1).unsqueeze(-1)
    else:
        coords = coords_pred_in

    target_world = batch.get("target_world", None)
    if target_world is not None:
        target_world = target_world.to(device)
        diff_3d = coords - target_world
        dist_3d = torch.norm(diff_3d, dim=1, p=2)
    else:
        dist_3d = None

    nuclear_mask = torch.zeros(coords.shape[0], dtype=torch.bool, device=device)
    repro_err = None
    metrics = {}

    if mode == "xyz":
        if target_world is None:
            raise ValueError("XYZ mode requires 'target_world' in batch.")
        loss_huber = F.huber_loss(coords, target_world, delta=1.0, reduction='none').sum(dim=1)
        total = loss_huber.mean() * 100.0
        metrics["loss"] = float(total.detach().cpu())
        metrics["mode"] = "xyz-huber"
        repro_err = torch.zeros_like(dist_3d)
    else:
        K = batch["intrinsics"].to(device)
        c2w = batch["c2w"].to(device)
        px = batch["pixels"].to(device)

        R, t = _invert_c2w_to_w2c(c2w)
        Xw = coords.unsqueeze(-1)
        Xc = R @ Xw + t
        z = Xc[:, 2:3, :]
        z_safe = z.clamp(min=1e-3)
        z_flat = z[:, 0, 0]

        uvh = K @ Xc
        uv = (uvh[:, :2, :] / z_safe).squeeze(-1)
        repro_err = (uv - px).abs().sum(dim=1)

        nuclear_mask = (repro_err > SANITY_PIXEL_ERR) | \
                       (torch.abs(coords).max(dim=1)[0] > SANITY_COORD_VAL) | \
                       (torch.isnan(repro_err)) | \
                       (torch.isinf(repro_err))

        if nuclear_mask.any():
            metrics["nuclear_cnt"] = float(nuclear_mask.sum().cpu())

        invalid_min_depth = (z_flat < depth_min)
        invalid_max_depth = (z_flat > depth_max)
        repro_clamp = float(getattr(loss_cfg, "repro_loss_hard_clamp", 100.0))
        invalid_repro = (repro_err > repro_clamp)

        base_invalid_mask = invalid_min_depth | invalid_repro | invalid_max_depth
        valid_mask = (~base_invalid_mask) & (~nuclear_mask)
        invalid_mask = base_invalid_mask & (~nuclear_mask)

        loss_valid = torch.tensor(0.0, device=device)
        loss_invalid = torch.tensor(0.0, device=device)

        if valid_mask.any():
            loss_valid = repro_loss.compute(repro_err[valid_mask], global_step)

        if invalid_mask.any():
            invK = batch.get("intrinsics_inv", torch.inverse(K)).to(device)
            uv1 = torch.cat([px, torch.ones_like(px[:, :1])], dim=1).unsqueeze(-1)
            Xc_tgt = float(getattr(loss_cfg, "depth_target", 10.0)) * (invK @ uv1)
            dist_error = (Xc_tgt - Xc).abs().sum(dim=1).squeeze()
            loss_invalid = dist_error[invalid_mask].sum()

        total = (loss_valid + loss_invalid) / coords.shape[0]

        scale_reg_val = 0.0
        safe_indices = ~nuclear_mask
        if safe_indices.any():
            c2w_safe = c2w[safe_indices] if target_world is not None else None
            reg_loss, scale_reg_val = _compute_scale_reg(
                z_flat[safe_indices],
                target_world[safe_indices] if target_world is not None else None,
                c2w_safe, loss_cfg, device
            )
            total = total + reg_loss

        metrics["loss"] = float(total.detach().cpu())
        metrics["mode"] = "reproj-safe"
        metrics["scale_reg"] = scale_reg_val

    safe_indices = ~nuclear_mask
    if safe_indices.any():
        if dist_3d is not None:
            d_safe = dist_3d[safe_indices]
            metrics["err_3d_mean"] = float(d_safe.mean().detach().cpu())
            metrics["err_3d_median"] = float(d_safe.median().detach().cpu())
            metrics["err_3d_min"] = float(d_safe.min().detach().cpu())
            metrics["err_3d_max"] = float(d_safe.max().detach().cpu())
            metrics["raw_dist_3d"] = d_safe.detach().cpu().numpy()
            sample_idx = torch.nonzero(safe_indices, as_tuple=True)[0][0]
            metrics["dbg_pred"] = coords[sample_idx].detach().cpu().tolist()
            metrics["dbg_gt"] = target_world[sample_idx].detach().cpu().tolist()
        else:
            metrics["err_3d_mean"] = -1.0
            metrics["raw_dist_3d"] = np.array([])

        if repro_err is not None:
            r_safe = repro_err[safe_indices]
            metrics["err_mean_px"] = float(r_safe.mean().detach().cpu())
            metrics["raw_reproj_err"] = r_safe.detach().cpu().numpy()
        else:
            metrics["err_mean_px"] = 0.0
            metrics["raw_reproj_err"] = np.array([])
    else:
        metrics["err_3d_mean"] = -1.0
        metrics["nuclear_cnt"] = float(nuclear_mask.sum().cpu())

    return total, metrics