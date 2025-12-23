"""
CoMe Confidence Predictor Definition.
Self-contained implementation of GlobalAttentionConfidence matching the provided checkpoint structure.
"""
import torch
import torch.nn as nn
import json
import os
from typing import Tuple, Any, List, Optional, Sequence

# -----------------------------------------------------------------------
# 1. Local Implementations of Dependencies (StreamBlock, Attention, MLP)
#    We implement these locally to avoid dependencies on 'Network.stream_vggt'.
# -----------------------------------------------------------------------

class Mlp(nn.Module):
    """ Standard MLP for Transformer Block """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """ Standard Multi-Head Self-Attention """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class StreamBlock(nn.Module):
    """
    Implementation of the StreamBlock structure matching the checkpoint keys.
    Structure: Norm1 -> Attn -> Norm2 -> MLP
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        # Pre-Norm architecture matching standard ViT/CoMe
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# -----------------------------------------------------------------------
# 2. GlobalAttentionConfidence (The Main Model)
# -----------------------------------------------------------------------

class GlobalAttentionConfidence(nn.Module):
    def __init__(self, hidden_dims: Sequence[int], activation: str):
        super().__init__()

        # Ensure we have at least 3 dims: [In, Hidden, Out]
        # e.g., [1024, 384, 1]

        # 1. Input Projection: Linear
        self.proj1 = nn.Linear(in_features=hidden_dims[0], out_features=hidden_dims[1])

        # 2. Transformer Block
        # Checkpoint structure implies: num_heads=1, mlp_ratio=2 based on your snippet
        self.block = StreamBlock(dim=hidden_dims[1], num_heads=1, mlp_ratio=2)

        # 3. Output Projection: Conv2d -> GELU -> Conv2d
        # Matches: proj2.0.weight, proj2.2.weight
        self.proj2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dims[1], out_channels=hidden_dims[1], kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(in_channels=hidden_dims[1], out_channels=hidden_dims[2], kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        )
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Modified forward to handle standard image tensors [B, C, H, W] or sequence [B, N, C].
        """
        B, C, H, W = 0, 0, 0, 0
        is_spatial = False

        # Handle Input Shape: expect [B, C, H, W]
        if x.dim() == 4:
            is_spatial = True
            B, C, H, W = x.shape
            # [B, C, H, W] -> [B, H*W, C] (Channel Last for Linear/Block)
            x = x.flatten(2).transpose(1, 2)
        else:
            # Assume [B, N, C], try to infer H, W from N if square
            B, N, C = x.shape
            S = int(N**0.5)
            if S*S == N:
                H, W = S, S
            else:
                # If N != H*W (e.g. register tokens), we might need to handle it.
                # For now assume N is spatial area.
                H, W = S, S

                # 1. Proj1 (Linear)
        x = self.proj1(x) # [B, N, hidden]

        # 2. Block (Transformer)
        x = self.block(x) # [B, N, hidden]

        # 3. Reshape for Proj2 (Conv2d requires [B, C, H, W])
        if H > 0 and W > 0:
            x = x.transpose(1, 2).reshape(B, -1, H, W)
            x = self.proj2(x) # [B, out_dim, H, W]

            # 4. Activation
            if self.activation == "expp1":
                return torch.exp(x) + 1.
            elif self.activation == "none":
                return x
            else:
                raise ValueError(f"Unrecognized activation: {self.activation}")
        else:
            raise ValueError("Could not infer spatial dimensions for Conv2d head.")

# -----------------------------------------------------------------------
# 3. Helper Classes & Loader
# -----------------------------------------------------------------------

class FeatureHook:
    def __init__(self):
        self.features = None

    def __call__(self, module, inputs, output):
        # output: [B, N, C]
        self.features = output

    def clear(self):
        self.features = None

def load_confidence_predictor_and_hook(
        checkpoint_path: str,
        config_path: str,
        device: torch.device,
        model_backbone: nn.Module,
        probe_loc_override: Optional[List[Any]] = None
) -> Tuple[nn.Module, FeatureHook]:
    """
    加载 GlobalAttentionConfidence 并注册 Hook。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # 这里的 config["method_dim"] 应该是一个列表，例如 [1024, 384, 1]
    hidden_dims = config.get("method_dim", [1024, 384, 1])
    activation = config.get("method_act", "expp1")

    print(f"[ACE] Building GlobalAttentionConfidence with dims={hidden_dims}, act={activation}")

    # 实例化模型
    predictor = GlobalAttentionConfidence(hidden_dims=hidden_dims, activation=activation)

    if os.path.exists(checkpoint_path):
        print(f"[ACE] Loading weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # [关键] 移除可能存在的前缀 (如 "module.")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # 加载权重 (strict=True 以确保结构完全匹配)
        try:
            predictor.load_state_dict(new_state_dict, strict=True)
        except RuntimeError as e:
            print(f"[ERROR] Weight mismatch! Please check config vs checkpoint.")
            print(f"Expected keys examples: proj1.weight, block.norm1.weight, proj2.0.weight")
            raise e
    else:
        print(f"[Warning] Checkpoint not found at {checkpoint_path}, using random weights!")

    predictor.to(device).eval()

    # Hook 注册逻辑
    if probe_loc_override is not None:
        probe_loc = probe_loc_override
    else:
        probe_loc = config.get("probe_loc", None)

    if probe_loc is None:
        raise ValueError("Config missing 'probe_loc'")

    _, layer_idx = probe_loc
    print(f"[ACE] Hooking into Layer {layer_idx} for confidence prediction.")

    hook = FeatureHook()
    if hasattr(model_backbone, "blocks"):
        if layer_idx >= len(model_backbone.blocks):
            raise ValueError(f"Layer index {layer_idx} out of range.")
        target_layer = model_backbone.blocks[layer_idx]
        target_layer.register_forward_hook(hook)
    else:
        raise AttributeError(f"Backbone does not have 'blocks' attribute.")

    return predictor, hook