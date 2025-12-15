import torch
import torch.nn as nn
import json
import os
from typing import Tuple


class Conv2DConfidence(nn.Module):
    """(保持不变) 轻量级置信度预测网络"""

    def __init__(self, hidden_dims, activation="expp1"):
        super().__init__()
        layers = []
        for in_feature, out_feature in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.extend([
                nn.Conv2d(in_channels=in_feature, out_channels=out_feature, kernel_size=3, stride=1, padding=1),
                nn.GELU()
            ])
        self.module = nn.Sequential(*layers)
        self.activation = activation

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        result = self.module(features)
        if self.activation == "expp1":
            return torch.exp(result) + 1.
        elif self.activation == "none":
            return result
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


class FeatureHook:
    """用于捕获中间层输出的 Hook"""

    def __init__(self):
        self.features = None

    def __call__(self, module, inputs, output):
        # output 通常是 [B, N, C] (ViT Block 输出)
        # 我们只保存数据，后续再处理形状
        self.features = output

    def clear(self):
        self.features = None


def load_confidence_predictor_and_hook(
        checkpoint_path: str,
        config_path: str,
        device: torch.device,
        model_backbone: nn.Module  # 传入 MapAnything 的 encoder.model
) -> Tuple[nn.Module, FeatureHook]:
    """
    加载预测器，并自动将 Hook 注册到正确的层上。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # 1. 加载预测器
    predictor = Conv2DConfidence(config["method_dim"], config["method_act"])
    if os.path.exists(checkpoint_path):
        print(f"[ACE] Loading Confidence Predictor from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        predictor.load_state_dict(state_dict)
    else:
        print(f"[Warning] Checkpoint not found at {checkpoint_path}, using random weights!")

    predictor.to(device).eval()

    # 2. 解析 Probe Location 并注册 Hook
    # CoMe config 格式通常为: "probe_loc": ["DINO", 3]
    probe_loc = config.get("probe_loc", None)
    if probe_loc is None:
        raise ValueError("Config missing 'probe_loc'")

    target_type, layer_idx = probe_loc
    print(f"[ACE] Hooking into {target_type} Layer {layer_idx} for confidence prediction.")

    hook = FeatureHook()

    # 假设 backbone 是标准的 ViT (如 DINOv2)，具有 blocks 列表
    # MapAnything.encoder.model -> ViT
    if hasattr(model_backbone, "blocks"):
        target_layer = model_backbone.blocks[layer_idx]
        target_layer.register_forward_hook(hook)
    else:
        raise AttributeError(f"Backbone does not have 'blocks' attribute. Check model structure.")

    return predictor, hook