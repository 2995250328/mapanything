from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


class ACERegressionHead(nn.Module):
    """Light-weight regression head that predicts XYZ + confidence."""

    def __init__(self, in_channels: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 4, kernel_size=1),  # 3D coords + confidence logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.net(x)


def load_regression_head(head: ACERegressionHead, checkpoint: Path | str) -> Dict[str, torch.Tensor]:
    """Load only the regression head parameters.

    This helper allows users to先加载完整的 mapanything 主干（通常较大），再加载轻量
    的回归头参数，避免在磁盘上重复存储完整模型。返回值是 ``state_dict`` 的加载
    结果，可用于调试。
    """

    payload = torch.load(checkpoint, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    return head.load_state_dict(state_dict, strict=False)
