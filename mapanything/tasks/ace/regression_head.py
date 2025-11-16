from __future__ import annotations

import torch.nn.functional as F
from __future__ import annotations
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Any, Iterable, Optional, Tuple
import torch
import torch.nn as nn

class PointwiseBlock(nn.Module):
    """纯 1x1 conv 残差块（逐点 MLP），不做空间聚合。"""
    def __init__(self, C):
        super().__init__()
        self.c0 = nn.Conv2d(C, C, 1, bias=False)
        self.c1 = nn.Conv2d(C, C, 1, bias=False)
        self.c2 = nn.Conv2d(C, C, 1, bias=False)

    def forward(self, x):
        y = F.gelu(self.c0(x))
        y = F.gelu(self.c1(y))
        y = self.c2(y)
        return x + y

class FiLM1x1(nn.Module):
    """token -> per-channel γ/β（逐点缩放与偏移），不引入空间耦合。"""
    def __init__(self, token_dim, channels, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 2*channels)
        )
        self.channels = channels

    def forward(self, x, token):
        # x: [B,C,H,W], token: [B,T] 或 [N,T]（N 当成 B, H=W=1 也可以）
        gb = self.mlp(token)                  # [B,2C]
        gamma, beta = gb.chunk(2, dim=1)      # [B,C],[B,C]
        gamma = gamma.view(-1, self.channels, 1, 1)
        beta  = beta.view(-1, self.channels, 1, 1)
        return x * (1 + gamma) + beta

class ACEHead_Pointwise_FiLM(nn.Module):
    """
    大容量逐点回归头（仅 1x1 卷积），将 scale_token 通过 FiLM 融入；输出带尺度 XYZ+conf。
    适于：batch 内样本互相独立的场景（缓冲区混合不同图像）。
    """
    def __init__(self, in_channels: int, hidden_dim: int = 512,
                 depth: int = 8, token_dim: int | None = None,
                 out_channels: int = 4, add_mean: bool = False, mean: torch.Tensor | None = None):
        super().__init__()
        self.add_mean = add_mean
        if add_mean:
            assert mean is not None
            self.register_buffer("mean", mean.clone().detach().view(1, 3, 1, 1))

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([PointwiseBlock(hidden_dim) for _ in range(depth)])
        self.films  = nn.ModuleList([FiLM1x1(token_dim, hidden_dim) if token_dim is not None else None
                                     for _ in range(depth)])
        self.head   = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False), nn.GELU(),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=True)  # 3D+conf_logit
        )

    def forward(self, x: torch.Tensor, scale_token: torch.Tensor | None = None):
        # 允许 x 为 [N,C]：自动 reshape 为 [N,C,1,1] 逐点处理
        squeeze_hw = False
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
            squeeze_hw = True

        h = self.stem(x)
        for blk, film in zip(self.blocks, self.films):
            h = blk(h)
            if film is not None and scale_token is not None:
                # 若传入的是 [N,T]，这里把 N 当成 batch 维度即可
                h = film(h, scale_token)

        out = self.head(h)  # [B,4,H,W]
        if self.add_mean:
            out[:, :3] = out[:, :3] + self.mean  # 与原 ACE 相同：回归相对 mean，再加回 mean

        if squeeze_hw:
            out = out.flatten(1)  # [N,4]
        return out

class ACEHead_Pointwise_Decoupled(nn.Module):
    """
    逐点 1x1 回归头：输出无尺度 XYZ_unit + conf_logit，不做任何空间聚合。
    """
    def __init__(self, in_channels: int, hidden_dim: int = 512, depth: int = 8, out_channels: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([PointwiseBlock(hidden_dim) for _ in range(depth)])
        self.head   = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False), nn.GELU(),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=True)  # 3D_unit + conf_logit
        )

    def forward(self, x: torch.Tensor):
        squeeze_hw = False
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
            squeeze_hw = True

        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        out = self.head(h)  # [B,4,H,W] or [N,4,1,1]
        if squeeze_hw:
            out = out.flatten(1)  # [N,4]
        return out

def load_regression_head(
    head: nn.Module,
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    strict: bool = False,
    allowed_prefixes: Iterable[str] = ("", "module.", "head.", "reg_head.", "ace_head."),
    rename_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    仅加载回归头参数：
    - 自动剥离常见前缀（module./head./reg_head./ace_head.）
    - 仅保留与 head.state_dict() 键名匹配且 shape 一致的权重
    - 跳过 shape 不一致项，避免误加载
    - 返回加载报告（loaded/missing/unexpected/skipped_shape）

    Args:
        head: 回归头模块（如 ACEHead_Pointwise_FiLM / ACEHead_Pointwise_Decoupled）
        checkpoint: ckpt 路径；支持 {'state_dict': ...} 或 直接 state_dict
        device: 加载映射设备（建议 'cpu'）
        strict: 传给 load_state_dict 的 strict 标志
        allowed_prefixes: 允许剥离的前缀集合
        rename_map: 旧->新 键名映射（可用于兼容老版本命名）

    Returns:
        report: {
            'loaded': List[str],
            'missing': List[str],
            'unexpected': List[str],
            'skipped_shape': List[Tuple[name, ckpt_shape, head_shape]],
            'total_in_ckpt': int,
            'used_from_ckpt': int
        }
    """
    payload = torch.load(str(checkpoint), map_location=device)
    sd = payload.get("state_dict", payload)

    tgt_sd = head.state_dict()
    filtered = OrderedDict()
    skipped_shape: list[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []

    def strip_prefix(k: str) -> str:
        for p in allowed_prefixes:
            if p and k.startswith(p):
                return k[len(p):]
        return k  # 包含空前缀情况

    for k, v in sd.items():
        name = strip_prefix(k)
        if rename_map and name in rename_map:
            name = rename_map[name]
        if name in tgt_sd:
            if tgt_sd[name].shape == v.shape:
                filtered[name] = v
            else:
                skipped_shape.append((name, tuple(v.shape), tuple(tgt_sd[name].shape)))

    incompat = head.load_state_dict(filtered, strict=strict)

    report = {
        "loaded": sorted(list(filtered.keys())),
        "missing": sorted(list(incompat.missing_keys)),
        "unexpected": sorted(list(incompat.unexpected_keys)),
        "skipped_shape": skipped_shape,
        "total_in_ckpt": len(sd),
        "used_from_ckpt": len(filtered),
    }
    return report

"""
head_ckpt = getattr(cfg.model, "head_checkpoint", None)
if head_ckpt:
    report = load_regression_head(
        self.reg_head,
        head_ckpt,
        device="cpu",        # 建议先在 CPU 过一遍
        strict=False,        # 不要求严格，允许部分缺失
        allowed_prefixes=("", "module.", "head.", "reg_head.", "ace_head."),
        rename_map=None,     # 如需兼容旧命名，传映射表
    )
    _logger.info(
        "Loaded regression head from %s | used=%d, total_in_ckpt=%d, missing=%d, unexpected=%d, skipped_shape=%d",
        head_ckpt, report["used_from_ckpt"], report["total_in_ckpt"],
        len(report["missing"]), len(report["unexpected"]), len(report["skipped_shape"])
    )
    if report["skipped_shape"]:
        _logger.debug("Skipped (shape mismatch): %s", report["skipped_shape"][:10])

report = load_regression_head(
    self.reg_head,
    cfg.model.head_checkpoint,
    strict=cfg.model.head_strict,
    allowed_prefixes=tuple(cfg.model.head_prefixes),
)
"""
