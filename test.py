import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def verify_patch_centers():
    # 1. 模拟尺寸
    H0, W0 = 392, 518  # 原图
    Hf, Wf = 28, 37  # 特征图

    print(f"Original Image: {H0}x{W0}")
    print(f"Feature Map:    {Hf}x{Wf}")

    # 2. 计算逻辑 (与 _collect_buffer 保持一致)
    i = torch.arange(Hf)
    j = torch.arange(Wf)
    yy, xx = torch.meshgrid(i, j, indexing="ij")

    patch_h = H0 / Hf  # 14.0
    patch_w = W0 / Wf  # 14.0
    print(f"Patch Size:     {patch_h}x{patch_w}")

    # 计算中心
    u_c = (xx + 0.5) * patch_w
    v_c = (yy + 0.5) * patch_h

    # 3. 可视化
    # 创建一个空白图，画出 Patch 网格
    img = np.zeros((H0, W0, 3), dtype=np.uint8) + 255  # 白底

    plt.figure(figsize=(10, 8))
    plt.imshow(img)

    # 画网格线 (Patch 边界)
    for h_idx in range(Hf + 1):
        y = h_idx * patch_h
        plt.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)
    for w_idx in range(Wf + 1):
        x = w_idx * patch_w
        plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)

    # 画计算出的 Patch 中心点 (红点)
    # 展平坐标
    u_flat = u_c.flatten().numpy()
    v_flat = v_c.flatten().numpy()
    plt.scatter(u_flat, v_flat, c='red', s=2, label='Feature Centers')

    # 验证几个具体点
    print("\nVerification Samples:")
    print(f"Feature(0,0) -> Image({u_c[0, 0].item():.1f}, {v_c[0, 0].item():.1f}) | Expect: ({0.5 * 14}, {0.5 * 14})")
    print(f"Feature(0,1) -> Image({u_c[0, 1].item():.1f}, {v_c[0, 1].item():.1f}) | Expect: ({1.5 * 14}, {0.5 * 14})")

    plt.title(f"Alignment Verification: {Hf}x{Wf} Features on {H0}x{W0} Image")
    plt.legend()
    plt.tight_layout()
    plt.savefig("verify_alignment.png", dpi=150)
    print("\nVisualization saved to 'verify_alignment.png'. Please check if red dots are centered in grid cells.")


if __name__ == "__main__":
    verify_patch_centers()