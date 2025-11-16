"""ACE-style regression head training using stored AA memory features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from mapanything.models import init_model
from mapanything.tasks.aa_feature_fusion.common import instantiate_dataset
from mapanything.tasks.ace import ACERegressionHead, load_memory_features, move_view_to_device
from mapanything.utils.geometry import quaternion_to_rotation_matrix


class FeatureBufferDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # pragma: no cover - thin wrapper
        return self.samples[idx]


@dataclass
class TrainingSample:
    features: torch.Tensor
    target_world: torch.Tensor
    valid_mask: torch.Tensor
    intrinsics: torch.Tensor
    c2w: torch.Tensor
    pixel_grid: torch.Tensor
    scene_name: str


def _pixel_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(height, dtype=torch.float32, device=device) + 0.5
    xs = torch.arange(width, dtype=torch.float32, device=device) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=0)


def _project_world_points(
    points_world: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, _, height, width = points_world.shape
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if intrinsics.size(0) == 1 and batch > 1:
        intrinsics = intrinsics.expand(batch, -1, -1)
    if c2w.dim() == 2:
        c2w = c2w.unsqueeze(0)
    if c2w.size(0) == 1 and batch > 1:
        c2w = c2w.expand(batch, -1, -1)

    points_flat = points_world.view(batch, 3, -1)
    rotation = c2w[:, :3, :3]
    translation = c2w[:, :3, 3:4]
    cam_points = torch.bmm(rotation.transpose(1, 2), points_flat - translation)

    z = cam_points[:, 2:3, :]
    valid = z > 1e-6
    pixels = torch.bmm(intrinsics, cam_points)
    pixels = pixels[:, :2, :] / z.clamp(min=1e-6)
    return pixels.view(batch, 2, height, width), valid.view(batch, 1, height, width)


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    return F.smooth_l1_loss(pred[mask.expand_as(pred)], target[mask.expand_as(target)])


def _resolve_intrinsics(view: Dict[str, Any], device: torch.device) -> torch.Tensor:
    if "camera_intrinsics" in view:
        intr = torch.as_tensor(view["camera_intrinsics"], device=device, dtype=torch.float32)
    elif "intrinsics" in view:
        intr = torch.as_tensor(view["intrinsics"], device=device, dtype=torch.float32)
    else:
        raise KeyError("Camera intrinsics not available in view dictionary.")
    if intr.dim() == 2:
        intr = intr.unsqueeze(0)
    return intr


def _resolve_pose(view: Dict[str, Any], device: torch.device) -> torch.Tensor:
    if "camera_pose" in view:
        pose = torch.as_tensor(view["camera_pose"], device=device, dtype=torch.float32)
    elif "camera_pose_quats" in view and "camera_pose_trans" in view:
        quats = torch.as_tensor(view["camera_pose_quats"], device=device, dtype=torch.float32)
        trans = torch.as_tensor(view["camera_pose_trans"], device=device, dtype=torch.float32)
        rot = quaternion_to_rotation_matrix(quats).reshape(-1, 3, 3)
        pose = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
        pose[:, :3, :3] = rot
        pose[:, :3, 3] = trans.view(-1, 3)
    else:
        raise KeyError("Camera pose not available in view dictionary.")
    if pose.dim() == 2:
        pose = pose.unsqueeze(0)
    return pose


def _prepare_targets(view: Dict[str, Any], feature_hw: Tuple[int, int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if "pts3d" not in view:
        raise KeyError("View is missing dense world coordinates (pts3d).")
    pts3d = torch.as_tensor(view["pts3d"], device=device, dtype=torch.float32)
    if pts3d.dim() == 3:
        pts3d = pts3d.permute(2, 0, 1).unsqueeze(0)
    valid_mask = torch.ones_like(pts3d[:, :1])
    if "valid_mask" in view:
        mask = torch.as_tensor(view["valid_mask"], device=device, dtype=torch.float32)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        valid_mask = mask

    h, w = feature_hw
    pts3d = F.interpolate(pts3d, size=(h, w), mode="bilinear", align_corners=False)
    valid_mask = F.interpolate(valid_mask, size=(h, w), mode="nearest")
    return pts3d, valid_mask


def _collect_buffer(
    cfg: DictConfig,
    model,
    dataset,
    memory_feats,
    memory_token,
    device: torch.device,
):
    samples: List[Dict[str, Any]] = []
    for idx in range(min(cfg.training.num_samples, len(dataset))):
        views = dataset[idx]
        if not views:
            continue
        view = views[0]
        scene_name = view.get("label") or view.get("scene_name") or str(idx)
        prepared = move_view_to_device(view, device)

        with torch.no_grad():
            dense_feature, _ = model.forward_with_memory_dense_feature(
                query_view=[prepared],
                device=str(device),
                memory_feats=memory_feats,
                additional_tokens=memory_token,
                memory_keep_ratio=cfg.fusion.memory_keep_ratio,
                memory_efficient_inference=cfg.training.memory_efficient_inference,
            )

        feature_hw = dense_feature.shape[-2:]
        target_world, valid_mask = _prepare_targets(view, feature_hw, device)
        intrinsics = _resolve_intrinsics(view, device).to(torch.float32)
        c2w = _resolve_pose(view, device).to(torch.float32)
        pixel_grid = _pixel_grid(feature_hw[0], feature_hw[1], device)

        feature_cpu = dense_feature.detach().cpu()
        if feature_cpu.shape[0] == 1:
            feature_cpu = feature_cpu.squeeze(0)

        target_world = target_world.detach().cpu()
        valid_mask = valid_mask.detach().cpu()
        if target_world.shape[0] == 1:
            target_world = target_world.squeeze(0)
        if valid_mask.shape[0] == 1:
            valid_mask = valid_mask.squeeze(0)

        intrinsics_cpu = intrinsics.detach().cpu()
        c2w_cpu = c2w.detach().cpu()
        if intrinsics_cpu.shape[0] == 1:
            intrinsics_cpu = intrinsics_cpu.squeeze(0)
        if c2w_cpu.shape[0] == 1:
            c2w_cpu = c2w_cpu.squeeze(0)

        samples.append(
            TrainingSample(
                features=feature_cpu,
                target_world=target_world,
                valid_mask=valid_mask,
                intrinsics=intrinsics_cpu,
                c2w=c2w_cpu,
                pixel_grid=pixel_grid.detach().cpu(),
                scene_name=scene_name,
            ).__dict__
        )

        if len(samples) >= cfg.training.buffer_capacity:
            break

    if not samples:
        raise RuntimeError("No training samples were collected. Check dataset or configuration.")
    return FeatureBufferDataset(samples), samples[0]["features"].shape[0]


def _loss_fn(preds: torch.Tensor, batch: Dict[str, torch.Tensor], loss_cfg: DictConfig):
    coords = preds[:, :3]
    conf_logits = preds[:, 3:4]
    confidence = torch.sigmoid(conf_logits)

    target_world = batch["target_world"].to(coords.device)
    intrinsics = batch["intrinsics"].to(coords.device)
    c2w = batch["c2w"].to(coords.device)
    valid_mask = batch["valid_mask"].to(coords.device)

    pred_pixels, pred_valid = _project_world_points(coords, intrinsics, c2w)
    target_pixels, target_valid = _project_world_points(target_world, intrinsics, c2w)
    combined_mask = valid_mask * pred_valid * target_valid

    reprojection = _masked_smooth_l1(pred_pixels, target_pixels, combined_mask)
    xyz_loss = _masked_smooth_l1(coords, target_world, valid_mask)
    conf_target = valid_mask.clamp(min=0.0, max=1.0)
    conf_loss = F.binary_cross_entropy(confidence, conf_target, reduction="mean")

    total = (
        loss_cfg.reprojection_weight * reprojection
        + loss_cfg.confidence_weight * conf_loss
        + loss_cfg.xyz_weight * xyz_loss
    )

    metrics = {
        "reprojection": float(reprojection.detach().cpu()),
        "xyz": float(xyz_loss.detach().cpu()),
        "confidence": float(conf_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, metrics


def run_training(cfg: DictConfig) -> Dict[str, str]:
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model = init_model(cfg.model.model_str, cfg.model.model_config, torch_hub_force_reload=False)
    model.to(device).eval()

    if cfg.model.pretrained:
        ckpt = torch.load(cfg.model.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    memory_feats, memory_token = load_memory_features(cfg.fusion.stored_feature_file, device)
    dataset = instantiate_dataset(cfg.dataset.dataset_str)

    buffer_ds, in_channels = _collect_buffer(cfg, model, dataset, memory_feats, memory_token, device)
    dataloader = DataLoader(
        buffer_ds,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle,
        num_workers=cfg.training.num_workers,
    )

    head = ACERegressionHead(in_channels=in_channels, hidden_dim=cfg.head.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    output_dir = Path(cfg.training.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(cfg.training.max_epochs):
        head.train()
        for batch in dataloader:
            features = batch["features"].to(device)
            preds = head(features)
            loss, metrics = _loss_fn(preds, batch, cfg.loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % cfg.training.log_interval == 0:
                print(json.dumps({"epoch": epoch, "step": global_step, "metrics": metrics}))

    ckpt_path = output_dir / "ace_regression_head.pt"
    torch.save({"state_dict": head.state_dict(), "in_channels": in_channels}, ckpt_path)
    return {"checkpoint": str(ckpt_path), "steps": str(global_step), "in_channels": str(in_channels)}


@hydra.main(version_base=None, config_path="../../configs/ace", config_name="train")
def main(cfg: DictConfig):
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    info = run_training(cfg)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
