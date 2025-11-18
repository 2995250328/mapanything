from __future__ import annotations
import torch.nn.functional as F
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

class ScaleMLP(nn.Module):
    """仅用 token 预测正数尺度 s（逐样本/逐视角独立）。"""
    def __init__(self, token_dim: int, hidden: int = 128,
                 s_min: float = 1e-3, s_max: float = 1e3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1)
        )
        self.s_min = s_min
        self.s_max = s_max

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        # raw -> softplus 保证 >0，再 clamp 到合理范围
        raw = self.mlp(token).squeeze(-1)           # [N] 或 [B]
        s = F.softplus(raw) + 1e-8
        s = torch.clamp(s, min=self.s_min, max=self.s_max)
        return s  # [N] 或 [B]

class ACEHead_Pointwise_Decoupled_WithScale(nn.Module):
    """
    逐点 1x1 回归头：输出无尺度 XYZ_unit 与尺度 s。
    - 特征 -> 1x1 MLP -> XYZ_unit（不聚合邻域）
    - scale_token -> MLP -> s（正数）
    - 不做空间卷积，适合来自不同图像/不同区域的混合 batch

    forward 返回:
      - coords_unit: [B,3,H,W] 或 [N,3]  （输入为 [N,C] 时返回 [N,3]）
      - scale:       [B] 或 [N]          （与 batch 维对应）
    """
    def __init__(self,
                 in_channels: int,
                 token_dim: int,
                 hidden_dim: int = 512,
                 depth: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([PointwiseBlock(hidden_dim) for _ in range(depth)])
        self.head_coords = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False), nn.GELU(),
            nn.Conv2d(hidden_dim, 4, 1, bias=True)  # 仅输出 XYZ_unit
        )
        self.scale_head = ScaleMLP(token_dim=token_dim)

    def forward(self, x: torch.Tensor, scale_token: torch.Tensor):
        """
        x: [B,C,H,W] 或 [N,C]（逐点特征）
        scale_token: [B,T] 或 [N,T]（与 batch 维一一对应）
        """
        squeeze_hw = False
        if x.dim() == 2:
            # 稀疏样本：按 [N,C] 处理（不聚合），内部 reshape 成 [N,C,1,1]
            x = x.unsqueeze(-1).unsqueeze(-1)
            squeeze_hw = True

        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        out = self.head_coords(h)  # [B,3,H,W] 或 [N,3,1,1]

        # 尺度仅由 token 决定，与空间无关
        s = self.scale_head(scale_token)   # [B] 或 [N]

        if squeeze_hw:
            out = out.squeeze(-1).squeeze(-1)  # [N,3]

        return out, s


def load_regression_head(
        head: nn.Module,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        strict: bool = False,
        # 这里的 prefixes 支持递归剥离，顺序很重要，长的放前面更安全
        allowed_prefixes: Iterable[str] = ("module.", "ace_head.", "reg_head.", "head."),
        rename_map: Optional[Dict[str, str]] = None,
        verbose: bool = False
) -> Dict[str, Any]:
    """
    通用回归头加载器。支持自动前缀剥离、键名重映射和形状安全检查。
    """

    # 1. 加载 Checkpoint
    if isinstance(checkpoint, (str, Path)):
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint}")
        payload = torch.load(str(checkpoint), map_location=device)
    else:
        payload = checkpoint  # 允许直接传入 dict

    source_sd = payload.get("state_dict", payload)
    target_sd = head.state_dict()

    loaded_sd = OrderedDict()
    report = {
        "loaded": [],
        "missing": [],  # 目标有，但在 ckpt 没找到
        "unexpected": [],  # ckpt 有，但在目标不需要
        "skipped_shape": [],  # 名字对上了，但形状不对
        "remapped": [],  # 发生了重命名
        "total_in_ckpt": len(source_sd),
    }

    # --- 内部辅助函数：尝试剥离前缀 ---
    def try_match_key(src_k: str, target_keys: set) -> str | None:
        """
        尝试通过剥离前缀或重命名来匹配 target_keys 中的键。
        返回匹配到的 target_key，如果没有匹配则返回 None。
        """
        # 1. 直接匹配
        if src_k in target_keys:
            return src_k

        # 2. 应用 rename_map (在剥离前缀之前尝试，处理如 'head.' -> 'head_coords.' 的情况)
        current_k = src_k
        if rename_map:
            for old_pattern, new_pattern in rename_map.items():
                if old_pattern in current_k:
                    # 简单的字符串替换，通常用于前缀替换
                    candidate = current_k.replace(old_pattern, new_pattern)
                    if candidate in target_keys:
                        report["remapped"].append(f"{src_k} -> {candidate}")
                        return candidate
                    # 更新 current_k 以便后续剥离前缀逻辑基于新名字继续尝试
                    # current_k = candidate

        # 3. 递归/贪婪剥离前缀
        # 逻辑：比如 key 是 "module.ace_head.stem.0.weight"
        # 我们尝试剥离 "module." -> "ace_head.stem.0.weight" -> 检查是否存在
        # 再剥离 "ace_head." -> "stem.0.weight" -> 检查是否存在

        potential_key = src_k
        found = False

        # 为了防止死循环，设置最大剥离深度
        max_depth = 5
        for _ in range(max_depth):
            matched_prefix = None
            for p in allowed_prefixes:
                if potential_key.startswith(p):
                    potential_key = potential_key[len(p):]
                    matched_prefix = p
                    break  # 找到一个前缀就剥离，然后重新检查是否存在

            if potential_key in target_keys:
                return potential_key

            # 如果还要应用 rename_map (在剥离后)
            if rename_map:
                for old_p, new_p in rename_map.items():
                    if old_p in potential_key:
                        cand = potential_key.replace(old_p, new_p)
                        if cand in target_keys:
                            report["remapped"].append(f"{src_k} -> {cand}")
                            return cand

            # 如果这一轮没有剥离任何前缀，说明已经到底了，无法匹配
            if matched_prefix is None:
                break

        return None

    # --- 主循环：遍历 Source ---
    target_keys_set = set(target_sd.keys())
    used_source_keys = set()

    for src_k, src_v in source_sd.items():
        tgt_k = try_match_key(src_k, target_keys_set)

        if tgt_k:
            # 名字匹配上了，检查形状
            tgt_v = target_sd[tgt_k]
            if src_v.shape != tgt_v.shape:
                report["skipped_shape"].append((tgt_k, tuple(src_v.shape), tuple(tgt_v.shape)))
            else:
                loaded_sd[tgt_k] = src_v
                report["loaded"].append(tgt_k)
                used_source_keys.add(src_k)  # 记录原始 key 被使用了
        else:
            pass  # 这个 key 在 target 中完全没用到，属于 unexpected

    # --- 填充结果 ---
    incompat = head.load_state_dict(loaded_sd, strict=strict)

    # 修正 missing_keys：load_state_dict 返回的是相对于 loaded_sd 的缺失
    # 但我们更关心相对于 target_sd 到底缺了啥
    # 这里 incompat.missing_keys 实际上就是 report['missing']
    report["missing"] = sorted(list(incompat.missing_keys))

    # unexpected keys 计算：ckpt 中没被用到的 key
    report["unexpected"] = sorted(list(set(source_sd.keys()) - used_source_keys))

    report["used_from_ckpt"] = len(loaded_sd)

    # --- 简单的打印/日志 ---
    if verbose:
        print(f"[LoadReport] Loaded: {len(report['loaded'])}/{len(target_sd)} keys.")
        if report["remapped"]:
            print(f"[LoadReport] Remapped {len(report['remapped'])} keys (e.g., {report['remapped'][0]}).")
        if report["skipped_shape"]:
            print(f"[LoadReport] Shape Mismatch: {len(report['skipped_shape'])} keys.")
        if report["missing"]:
            print(f"[LoadReport] Missing: {len(report['missing'])} keys (e.g., {report['missing'][0]}).")

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
