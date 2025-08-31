import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def plot_heads_grid(attn: torch.Tensor, selected: List[Dict], meta: Dict, save_path: str, show_plot: bool) -> None:
    """Save a figure: original image + top-K attention maps.

    attn: [L, H, 1, V]
    """
    P = int(meta.get("patch_size"))
    W, H_img = meta["image_size"]
    n = len(selected)
    cols = n + 1
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))

    # Original image
    try:
        img = Image.open(meta["image_file"]).convert("RGB")
        axes[0].imshow(img)
        axes[0].set_title("Image")
        axes[0].axis("off")
    except Exception as e:
        axes[0].text(0.5, 0.5, f"Image load error\n{e}", ha='center', va='center')
        axes[0].axis("off")

    # Attention maps
    for i, hinfo in enumerate(selected):
        l, h = hinfo["layer"], hinfo["head"]
        a2d = attn[l, h, 0].reshape(P, P).detach().cpu().numpy()
        im = axes[i + 1].imshow(a2d, cmap="viridis")
        axes[i + 1].set_title(f"L{l}-H{h}\nSE={hinfo['spatial_entropy']:.2f}")
        axes[i + 1].axis("off")
        plt.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

