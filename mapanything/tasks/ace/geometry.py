import torch
from typing import Dict, Any, Tuple
from mapanything.utils.geometry import quaternion_to_rotation_matrix


def _resolve_intrinsics(view: Dict[str, Any], device: torch.device) -> torch.Tensor:
    if "camera_intrinsics" in view:
        intr = torch.as_tensor(view["camera_intrinsics"], device=device, dtype=torch.float32)
    elif "intrinsics" in view:
        intr = torch.as_tensor(view["intrinsics"], device=device, dtype=torch.float32)
    else:
        raise KeyError("Camera intrinsics not available in view dictionary.")

    # 统一到 [B, 3, 3]
    if intr.dim() == 2:  # [3, 3]
        intr = intr.unsqueeze(0)
    return intr


def _resolve_pose(view: Dict[str, Any], device: torch.device) -> torch.Tensor:
    if "camera_pose" in view:
        pose = torch.as_tensor(view["camera_pose"], device=device, dtype=torch.float32)
    elif "camera_pose_quats" in view and "camera_pose_trans" in view:
        # 合成 pose
        quats = torch.as_tensor(view["camera_pose_quats"], device=device, dtype=torch.float32)
        trans = torch.as_tensor(view["camera_pose_trans"], device=device, dtype=torch.float32)

        rot = quaternion_to_rotation_matrix(quats)  # [B, 3, 3] or [3, 3]
        if rot.dim() == 2: rot = rot.unsqueeze(0)
        if trans.dim() == 1: trans = trans.unsqueeze(0)

        B = rot.shape[0]
        pose = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)
        pose[:, :3, :3] = rot
        pose[:, :3, 3] = trans.view(B, 3)
    else:
        raise KeyError("Camera pose not available in view dictionary.")

    # 统一到 [B, 4, 4]
    if pose.dim() == 2:
        pose = pose.unsqueeze(0)
    return pose


def _prepare_targets(view: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and preprocess ground truth 3D points and valid mask.
    Returns:
        pts3d: [B, 3, H, W]
        valid_mask: [B, 1, H, W]
    """
    if "pts3d" not in view:
        raise KeyError("View is missing dense world coordinates (pts3d).")

    pts3d = torch.as_tensor(view["pts3d"], device=device, dtype=torch.float32)

    # 统一维度到 [B, 3, H, W]
    if pts3d.dim() == 3 and pts3d.shape[-1] == 3:
        pts3d = pts3d.permute(2, 0, 1).unsqueeze(0)
    elif pts3d.dim() == 4 and pts3d.shape[-1] == 3:
        pts3d = pts3d.permute(0, 3, 1, 2)
    elif pts3d.dim() == 4 and pts3d.shape[1] == 3:
        pass
    else:
        raise ValueError(f"Unsupported pts3d shape: {pts3d.shape}")

    # 处理 Mask
    if "valid_mask" in view:
        mask = torch.as_tensor(view["valid_mask"], device=device, dtype=torch.float32)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() == 4 and mask.shape[-1] == 1:
            mask = mask.permute(0, 3, 1, 2)
        valid_mask = mask
    else:
        valid_mask = torch.ones((pts3d.shape[0], 1, pts3d.shape[2], pts3d.shape[3]),
                                device=device, dtype=torch.float32)

    return pts3d, valid_mask


def _invert_c2w_to_w2c(c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    统一的位姿求逆函数。
    Returns:
        R: [..., 3, 3]
        t: [..., 3, 1]
    """
    if c2w.shape[-2:] == (4, 4):
        w2c = torch.inverse(c2w)
        R = w2c[..., :3, :3]
        t = w2c[..., :3, 3:4]
    elif c2w.shape[-2:] == (3, 4):
        R_c2w = c2w[..., :3, :3]
        t_c2w = c2w[..., :3, 3:4]
        R = R_c2w.transpose(-1, -2)
        t = -R @ t_c2w
    else:
        raise ValueError(f"Unexpected c2w shape: {tuple(c2w.shape)}")

    return R, t


def _project_world_points(
        points_world: torch.Tensor,
        intrinsics: torch.Tensor,
        c2w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """稠密投影版本，兼容 Bx3xHxW 的输入。"""
    batch, _, height, width = points_world.shape
    if intrinsics.dim() == 2: intrinsics = intrinsics.unsqueeze(0)
    if intrinsics.size(0) == 1 and batch > 1: intrinsics = intrinsics.expand(batch, -1, -1)
    if c2w.dim() == 2: c2w = c2w.unsqueeze(0)
    if c2w.size(0) == 1 and batch > 1: c2w = c2w.expand(batch, -1, -1)

    points_flat = points_world.view(batch, 3, -1)
    rotation = c2w[:, :3, :3]
    translation = c2w[:, :3, 3:4]
    cam_points = torch.bmm(rotation.transpose(1, 2), points_flat - translation)

    z = cam_points[:, 2:3, :]
    valid = z > 1e-6
    pixels = torch.bmm(intrinsics, cam_points)
    pixels = pixels[:, :2, :] / z.clamp(min=1e-6)
    return pixels.view(batch, 2, height, width), valid.view(batch, 1, height, width)


def _project_world_points_sparse(
        Xw_N3: torch.Tensor,
        K_N33: torch.Tensor,
        c2w_N: torch.Tensor,
        depth_min: float = 1e-4
) -> torch.Tensor:
    """
    Xw_N3 -> pixels_N2  (batched, no mask; z clamped to depth_min)
    """
    R, t = _invert_c2w_to_w2c(c2w_N)
    Xw = Xw_N3.unsqueeze(-1)
    Xc = R @ Xw + t
    Xc_z = Xc[:, 2:3, :]
    Xc_z = Xc_z.clamp(min=depth_min)
    uvd = K_N33 @ Xc
    uv = uvd[:, :2, :] / Xc_z
    return uv.squeeze(-1)