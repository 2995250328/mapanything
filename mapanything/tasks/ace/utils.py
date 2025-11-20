from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch


def load_memory_features(path: str, device: torch.device):
    """Load AA memory tensors from disk and move them to ``device``.

    返回:
      - memory_feats: List[List[Tensor(B,C,H,W)]]
      - memory_scale_token: Tensor(B,C,1)
    """

    payload = torch.load(path, map_location=device)

    memory_feats: List[List[torch.Tensor]] = []
    memory_scale_token = None

    interm_blocks = payload.get("intermediate")
    if interm_blocks is not None:
        for block in interm_blocks:
            feats = [t.to(device) for t in block["features"]]
            memory_feats.append(feats)

    final_block = payload.get("final")
    if final_block is not None and final_block.get("additional_token_features") is not None:
        memory_scale_token = final_block["additional_token_features"].to(device)

    if memory_scale_token is None:
        raise ValueError("Memory file has no final additional_token_features (scale token).")

    return memory_feats, memory_scale_token


def move_view_to_device(view: Dict[str, Any], device: torch.device, *, non_blocking: bool = True):
    """Move tensor-like view entries to the requested device.

    Non tensor/metadata entries are kept untouched. This mirrors the filtering
    strategy used in :mod:`map-anything.tasks.demo` to keep the calling contract
    aligned with existing inference code.
    """

    ignore_keys = {
        "dataset",
        "label",
        "instance",
        "idx",
        "true_shape",
        "rng",
        "data_norm_type",
        "scene_name",
    }

    moved: Dict[str, Any] = {}
    for name, value in view.items():
        if name in ignore_keys:
            moved[name] = value
            continue

        if torch.is_tensor(value):
            moved[name] = value.to(device, non_blocking=non_blocking)
        else:
            try:
                moved[name] = torch.as_tensor(value, device=device)
            except Exception:
                moved[name] = value
    return moved


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunked(iterable: Iterable[Any], n: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch
