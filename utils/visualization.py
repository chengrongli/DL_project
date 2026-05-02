"""
Visualization utilities for LPC sprite diffusion.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torchvision.utils as vutils
from PIL import Image
import numpy as np


def tensor_to_pil(t: torch.Tensor, upscale: int = 4) -> Image.Image:
    """
    Convert a (C, H, W) float tensor in [−1, 1] to a PIL Image.

    Args:
        t:        Tensor in [−1, 1] with C ∈ {1, 3, 4}.
        upscale:  Nearest-neighbour upscale factor for easier viewing of
                  64×64 pixel art (default ×4 → 256×256).
    """
    t = t.detach().cpu().clamp(-1, 1)
    t = (t + 1.0) / 2.0  # → [0, 1]
    # Convert to uint8 numpy
    arr = (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    img = Image.fromarray(arr)
    if upscale > 1:
        w, h = img.size
        img = img.resize((w * upscale, h * upscale), Image.NEAREST)
    return img


def save_sample_grid(
    samples: torch.Tensor,
    path: str,
    nrow: int = 4,
    padding: int = 2,
    upscale: int = 4,
) -> None:
    """
    Save a grid of sample tensors to an image file.

    Args:
        samples: (N, C, H, W) tensor in [−1, 1].
        path:    Output file path.
        nrow:    Number of images per row.
        padding: Padding between images in the grid.
        upscale: Nearest-neighbour upscale applied to each tile before saving.
    """
    samples = samples.detach().cpu().clamp(-1, 1)
    samples = (samples + 1.0) / 2.0  # → [0, 1]

    # Upscale each tile
    if upscale > 1:
        N, C, H, W = samples.shape
        upscaled = torch.nn.functional.interpolate(
            samples, size=(H * upscale, W * upscale), mode="nearest"
        )
    else:
        upscaled = samples

    grid = vutils.make_grid(upscaled, nrow=nrow, padding=padding, normalize=False)
    arr = (grid.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def side_by_side(
    img_a: Image.Image,
    img_b: Image.Image,
    gap: int = 4,
    bg_color: tuple = (50, 50, 50),
) -> Image.Image:
    """
    Concatenate two PIL images side by side with a narrow gap.

    Args:
        img_a:     Left image.
        img_b:     Right image.
        gap:       Pixel gap between images.
        bg_color:  Background fill color for the gap.

    Returns:
        Combined PIL image.
    """
    h = max(img_a.height, img_b.height)
    w = img_a.width + gap + img_b.width
    out = Image.new("RGB", (w, h), bg_color)
    out.paste(img_a.convert("RGB"), (0, 0))
    out.paste(img_b.convert("RGB"), (img_a.width + gap, 0))
    return out


def make_comparison_grid(
    conditions: torch.Tensor,
    predictions: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    nrow: int = 4,
    upscale: int = 4,
) -> Image.Image:
    """
    Build a comparison grid: condition | prediction [| target].

    Args:
        conditions:  (N, 3, H, W) conditioning images.
        predictions: (N, 3, H, W) generated images.
        targets:     (N, 3, H, W) ground-truth images (optional).
        nrow:        Images per row.
        upscale:     Upscale factor.

    Returns:
        PIL image.
    """
    import tempfile

    rows = [conditions, predictions]
    if targets is not None:
        rows.append(targets)

    combined = torch.cat(rows, dim=0)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        save_sample_grid(combined, tmp, nrow=nrow, upscale=upscale)
        img = Image.open(tmp).copy()
    finally:
        import os
        os.unlink(tmp)
    return img
