"""Utility to visualize AA fusion demo outputs on WSL or local machines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _load_sample(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _export_ply(points: np.ndarray, colors: Optional[np.ndarray], path: Path) -> None:
    mask = np.isfinite(points).all(axis=-1)
    pts = points[mask].reshape(-1, 3)
    if colors is not None:
        cols = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
        cols = cols[mask].reshape(-1, 3)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(pts)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if colors is not None:
        header.extend(
            [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        )
    header.append("end_header")

    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(header))
        fh.write("\n")
        if colors is None:
            for x, y, z in pts:
                fh.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        else:
            for (x, y, z), (r, g, b) in zip(pts, cols, strict=True):
                fh.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def visualize(sample_path: Path, save_path: Optional[Path], export_pts: Optional[Path], show: bool) -> None:
    payload = _load_sample(sample_path)

    rgb_tensor = payload.get("rgb")
    rgb = None
    if rgb_tensor is not None:
        rgb = np.clip(_tensor_to_numpy(rgb_tensor).transpose(1, 2, 0), 0.0, 1.0)

    reconstruction = payload.get("reconstruction", {})
    pts_tensor = reconstruction.get("pts3d")
    depth_map = None
    if pts_tensor is not None:
        pts = _tensor_to_numpy(pts_tensor.squeeze(0)).transpose(1, 2, 0)
        depth_map = pts[..., 2]
        if export_pts is not None:
            colors = rgb if rgb is not None else None
            _export_ply(pts, colors, export_pts)
    else:
        dense = reconstruction.get("dense_value")
        if dense is not None:
            depth_map = _tensor_to_numpy(dense.squeeze(0))[0]

    if save_path is None:
        save_path = sample_path.with_suffix(".png")

    if depth_map is None and rgb is None:
        raise RuntimeError("No visualization tensors found in the sample payload.")

    fig, axes = plt.subplots(1, 2 if (rgb is not None and depth_map is not None) else 1, figsize=(10, 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    axis_idx = 0
    if rgb is not None:
        axes[axis_idx].imshow(rgb)
        axes[axis_idx].set_title("RGB")
        axes[axis_idx].axis("off")
        axis_idx += 1
    if depth_map is not None:
        axes[axis_idx].imshow(depth_map, cmap="turbo")
        axes[axis_idx].set_title("Depth / Z")
        axes[axis_idx].axis("off")

    plt.tight_layout()
    if show:
        plt.show()
    else:
        fig.savefig(save_path, dpi=200)
        print(f"Saved visualization to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path, help="Path to a saved demo sample (.pt file)")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save the preview PNG")
    parser.add_argument("--export-pts", type=Path, default=None, help="Optional PLY export for the reconstructed points")
    parser.add_argument("--show", action="store_true", help="Display the figure instead of saving it")
    args = parser.parse_args()

    visualize(args.sample, args.save, args.export_pts, args.show)


if __name__ == "__main__":
    main()
