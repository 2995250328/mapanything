# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mapanything.models.external.vggt.layers import PatchEmbed
from mapanything.models.external.vggt.layers.block import Block
from mapanything.models.external.vggt.layers.rope import (
    PositionGetter,
    RotaryPositionEmbedding2D,
)
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
        store_intermediate_features (bool): Whether to retain intermediate tokens from each alternating-attention
            block. When True, call :meth:`get_intermediate_features` after ``forward`` to retrieve them. (default: False)
        intermediate_storage_device (str): Optional device string (e.g., "cpu") where the stored intermediate
            features should be moved. If ``None`` the tensors are cloned on their current device. (default: None)
        intermediate_storage_path (str or Path): Optional directory where intermediate tensors should be serialized
            to disk. When provided, stored metadata will reference files on disk instead of in-memory tensors.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        store_intermediate_features: bool = False,
        intermediate_storage_device: Optional[Union[str, torch.device]] = None,
        intermediate_storage_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__()

        self.__build_patch_embed__(
            patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim
        )

        # Initialize rotary position embedding if frequency > 0
        self.rope = (
            RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        )
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size
        self.store_intermediate_features = store_intermediate_features
        self.intermediate_storage_device = intermediate_storage_device
        self.intermediate_storage_path = (
            Path(intermediate_storage_path).expanduser()
            if intermediate_storage_path is not None
            else None
        )
        self._storage_run_uuid: Optional[str] = None
        self._stored_intermediate_features = None

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(
                f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})"
            )

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(
            torch.randn(1, 2, num_register_tokens, embed_dim)
        )

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 3, 1, 1),
                persistent=False,
            )

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=embed_dim,
            )
        else:
            ### From original VGGT codebase: Doesn't load pre-trained DINOv2 weights
            # vit_models = {
            #     "dinov2_vitl14_reg": vit_large,
            #     "dinov2_vitb14_reg": vit_base,
            #     "dinov2_vits14_reg": vit_small,
            #     "dinov2_vitg2_reg": vit_giant2,
            # }

            # self.patch_embed = vit_models[patch_embed](
            #     img_size=img_size,
            #     patch_size=patch_size,
            #     num_register_tokens=num_register_tokens,
            #     interpolate_antialias=interpolate_antialias,
            #     interpolate_offset=interpolate_offset,
            #     block_chunks=block_chunks,
            #     init_values=init_values,
            # )

            ### Use pre-trained DINOv2 with gradient checkpointing
            self.patch_embed = torch.hub.load("facebookresearch/dinov2", patch_embed)
            for i in range(len(self.patch_embed.blocks)):
                self.patch_embed.blocks[i] = (
                    self.wrap_module_with_gradient_checkpointing(
                        self.patch_embed.blocks[i]
                    )
                )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    ### Gradient Checkpointing Wrapper from UniCeption:
    def wrap_module_with_gradient_checkpointing(self, module: nn.Module):
        """
        Wrapper for Gradient Checkpointing
        References: https://github.com/microsoft/MoGe
        """

        class _CheckpointingWrapper(module.__class__):
            _restore_cls = module.__class__

            def forward(self, *args, **kwargs):
                return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)

        module.__class__ = _CheckpointingWrapper
        return module

    def forward(
        self,
        images: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed.forward_features(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=images.device
            )

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = (
                torch.zeros(B * S, self.patch_start_idx, 2)
                .to(images.device)
                .to(pos.dtype)
            )
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []
        stored_frame_intermediates = [] if self.store_intermediate_features else None
        stored_global_intermediates = [] if self.store_intermediate_features else None
        stored_concat_intermediates = [] if self.store_intermediate_features else None
        storage_run_dir = (
            self._prepare_storage_run_directory()
            if self.store_intermediate_features
            and self.intermediate_storage_path is not None
            else None
        )
        self._stored_intermediate_features = None

        block_counter = 0

        for _ in range(self.aa_block_num):
            frame_intermediates = None
            global_intermediates = None

            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = (
                        self._process_frame_attention(
                            tokens, B, S, P, C, frame_idx, pos=pos
                        )
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = (
                        self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos
                        )
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            if frame_intermediates is None or global_intermediates is None:
                raise ValueError(
                    "aa_order must include both 'frame' and 'global' attention blocks"
                )

            if len(frame_intermediates) != len(global_intermediates):
                raise RuntimeError(
                    "Mismatched number of frame and global intermediates after attention"
                )

            for frame_feat_tokens, global_feat_tokens in zip(
                frame_intermediates, global_intermediates
            ):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat(
                    [frame_feat_tokens, global_feat_tokens], dim=-1
                )
                output_list.append(concat_inter)

                if self.store_intermediate_features:
                    target_device = self.intermediate_storage_device
                    if storage_run_dir is not None:
                        frame_meta = self._store_tensor_to_disk(
                            frame_feat_tokens,
                            storage_run_dir,
                            block_counter,
                            "frame",
                        )
                        global_meta = self._store_tensor_to_disk(
                            global_feat_tokens,
                            storage_run_dir,
                            block_counter,
                            "global",
                        )
                        concat_meta = self._store_tensor_to_disk(
                            concat_inter,
                            storage_run_dir,
                            block_counter,
                            "concatenated",
                        )
                    else:
                        frame_feat = frame_feat_tokens.detach().clone()
                        global_feat = global_feat_tokens.detach().clone()
                        concat_feat = concat_inter.detach().clone()

                        if target_device is not None:
                            frame_feat = frame_feat.to(target_device)
                            global_feat = global_feat.to(target_device)
                            concat_feat = concat_feat.to(target_device)

                        frame_meta = frame_feat
                        global_meta = global_feat
                        concat_meta = concat_feat

                    stored_frame_intermediates.append(frame_meta)
                    stored_global_intermediates.append(global_meta)
                    stored_concat_intermediates.append(concat_meta)
                    block_counter += 1

        if self.store_intermediate_features:
            self._stored_intermediate_features = {
                "frame": stored_frame_intermediates,
                "global": stored_global_intermediates,
                "concatenated": stored_concat_intermediates,
            }
            if storage_run_dir is not None:
                self._stored_intermediate_features["storage_directory"] = str(
                    storage_run_dir
                )

        return output_list, self.patch_start_idx

    def get_intermediate_features(self):
        """Return the intermediate features captured during the last forward pass.

        When ``intermediate_storage_path`` is set the returned dictionary contains
        file metadata (including ``storage_directory`` and per-block paths). If the
        path is ``None`` the lists contain in-memory tensors.
        """

        return self._stored_intermediate_features

    def _prepare_storage_run_directory(self) -> Path:
        base_dir = self.intermediate_storage_path
        assert base_dir is not None
        base_dir.mkdir(parents=True, exist_ok=True)
        run_dir = base_dir / (
            datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            + f"_{uuid.uuid4().hex[:8]}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        self._storage_run_uuid = run_dir.name
        return run_dir

    def _store_tensor_to_disk(
        self,
        tensor: torch.Tensor,
        run_dir: Path,
        block_index: int,
        kind: str,
    ) -> dict:
        block_dir = run_dir / f"block_{block_index:03d}"
        block_dir.mkdir(parents=True, exist_ok=True)
        file_path = block_dir / f"{kind}.pt"
        tensor_to_save = tensor.detach().cpu()
        torch.save(tensor_to_save, file_path)
        return {
            "path": str(file_path),
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "block_index": block_index,
            "kind": kind,
        }

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
