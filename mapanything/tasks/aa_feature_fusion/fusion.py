"""Modules for fusing stored alternating-attention features with single-view inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import torch
import torch.nn as nn

from mapanything.models.mapanything.model import MapAnything
from uniception.models.info_sharing.base import MultiViewTransformerOutput


def _flatten_view_features(view_features: torch.Tensor) -> torch.Tensor:
    """Convert per-view feature maps ``(B, C, H, W)`` into token sequences ``(B, S, C)``."""

    if view_features.dim() != 4:
        raise ValueError(
            "Expected a 4D tensor shaped as (batch, channels, height, width) "
            f"but received {tuple(view_features.shape)}"
        )
    batch, channels, height, width = view_features.shape
    sequence = height * width
    tokens = view_features.reshape(batch, channels, sequence).transpose(1, 2).contiguous()
    return tokens


def extract_single_view_tokens(
    model: MapAnything,
    view: dict,
    use_autocast: bool = False,
) -> torch.Tensor:
    """Run the MapAnything encoders on a single view and return flattened tokens."""

    encoded = model._encode_n_views([view])
    # Align with ``forward`` by defaulting to full precision during fusion.
    with torch.autocast("cuda", enabled=use_autocast and torch.cuda.is_available()):
        fused = model._encode_and_fuse_optional_geometric_inputs([view], encoded)
    per_view = fused[0]
    return _flatten_view_features(per_view)


@dataclass(frozen=True)
class StoredAAFeatureSequence:
    """Container holding the memory tokens reconstructed from stored AA blocks."""

    memory_tokens: Sequence[torch.Tensor]
    embed_dim: int

    @classmethod
    def from_outputs(
        cls,
        blocks: Iterable[MultiViewTransformerOutput],
    ) -> "StoredAAFeatureSequence":
        token_groups: List[torch.Tensor] = []
        for block in blocks:
            per_view_tokens = [_flatten_view_features(feature) for feature in block.features]
            memory = torch.cat(per_view_tokens, dim=1)
            token_groups.append(memory.detach())
        if not token_groups:
            raise ValueError("No alternating-attention blocks were provided for memory construction.")
        embed_dim = token_groups[0].shape[-1]
        return cls(memory_tokens=token_groups, embed_dim=embed_dim)

    @classmethod
    def from_file(
        cls,
        path: Optional[Path],
        *,
        map_location: Optional[torch.device | str] = None,
    ) -> Optional["StoredAAFeatureSequence"]:
        if path is None:
            return None
        payload = MapAnything.load_info_sharing_features_from_file(
            path, map_location=map_location, as_outputs=True
        )
        blocks: List[MultiViewTransformerOutput] = []
        intermediate = payload.get("intermediate")
        if intermediate:
            blocks.extend(intermediate)
        final = payload.get("final")
        if final is not None:
            blocks.append(final)
        return cls.from_outputs(blocks)

    @property
    def num_blocks(self) -> int:
        return len(self.memory_tokens)

    def iter_tokens(
        self,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Iterator[torch.Tensor]:
        for tokens in self.memory_tokens:
            if device is not None or dtype is not None:
                yield tokens.to(device=device, dtype=dtype)
            else:
                yield tokens


class AlternatingAttentionMemoryBlock(nn.Module):
    """Single block that mirrors AA frame/global attention using stored memory tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.frame_norm1 = nn.LayerNorm(embed_dim)
        self.frame_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.frame_dropout = nn.Dropout(dropout)
        self.frame_norm2 = nn.LayerNorm(embed_dim)
        self.frame_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.global_norm1 = nn.LayerNorm(embed_dim)
        self.global_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.global_dropout = nn.Dropout(dropout)
        self.global_norm2 = nn.LayerNorm(embed_dim)
        self.global_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query_tokens: torch.Tensor, memory_tokens: torch.Tensor) -> torch.Tensor:
        # Frame attention: operate on the single-view tokens only.
        residual = query_tokens
        frame_input = self.frame_norm1(query_tokens)
        attn_output, _ = self.frame_attention(frame_input, frame_input, frame_input, need_weights=False)
        query_tokens = residual + self.frame_dropout(attn_output)
        residual = query_tokens
        mlp_input = self.frame_norm2(query_tokens)
        query_tokens = residual + self.frame_mlp(mlp_input)

        # Global attention: reuse stored AA block features as keys/values.
        residual = query_tokens
        global_input = self.global_norm1(query_tokens)
        attn_output, _ = self.global_attention(
            global_input, memory_tokens, memory_tokens, need_weights=False
        )
        query_tokens = residual + self.global_dropout(attn_output)
        residual = query_tokens
        mlp_input = self.global_norm2(query_tokens)
        query_tokens = residual + self.global_mlp(mlp_input)
        return query_tokens


class AAFeatureFusionModule(nn.Module):
    """Fuse single-view tokens with stored alternating-attention memory."""

    def __init__(
        self,
        *,
        stored_feature_file: Optional[str | Path] = None,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        map_location: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()
        self.stored_feature_file = (
            Path(stored_feature_file).expanduser() if stored_feature_file is not None else None
        )
        self.map_location = map_location
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        self._memory = StoredAAFeatureSequence.from_file(
            self.stored_feature_file, map_location=map_location
        )
        if self._memory is None:
            self.blocks = nn.ModuleList()
        else:
            embed_dim = self._memory.embed_dim
            self.blocks = nn.ModuleList(
                [
                    AlternatingAttentionMemoryBlock(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                    for _ in range(self._memory.num_blocks)
                ]
            )

    @property
    def has_memory(self) -> bool:
        return self._memory is not None and self._memory.num_blocks > 0

    def forward(self, query_tokens: torch.Tensor) -> torch.Tensor:
        if not self.has_memory:
            raise RuntimeError(
                "AAFeatureFusionModule was constructed without stored alternating-attention memory."
            )

        fused = query_tokens
        for block, memory_tokens in zip(
            self.blocks,
            self._memory.iter_tokens(device=query_tokens.device, dtype=query_tokens.dtype),
        ):
            fused = block(fused, memory_tokens)
        return fused

    def reload(self, path: str | Path, *, map_location: Optional[torch.device | str] = None) -> None:
        """Replace the stored memory with a new alternating-attention capture."""

        memory = StoredAAFeatureSequence.from_file(Path(path), map_location=map_location)
        if memory is None:
            raise ValueError("The provided alternating-attention file did not contain any feature blocks.")
        if memory.embed_dim != (self.blocks[0].frame_attention.embed_dim if self.blocks else memory.embed_dim):
            self.blocks = nn.ModuleList(
                [
                    AlternatingAttentionMemoryBlock(
                        embed_dim=memory.embed_dim,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        dropout=self.dropout,
                    )
                    for _ in range(memory.num_blocks)
                ]
            )
        elif len(self.blocks) != memory.num_blocks:
            self.blocks = nn.ModuleList(
                [
                    AlternatingAttentionMemoryBlock(
                        embed_dim=memory.embed_dim,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        dropout=self.dropout,
                    )
                    for _ in range(memory.num_blocks)
                ]
            )
        self._memory = memory
        self.stored_feature_file = Path(path).expanduser()
