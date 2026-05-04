"""
Data augmentation utilities for LPC sprite pairs.

All augmentations are applied consistently to both front and back images
of a pair so that they remain aligned.
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image



# ---------------------------------------------------------------------------
# Paired transforms (applied identically to front and back)
# ---------------------------------------------------------------------------

def random_horizontal_flip(
    front: Image.Image,
    back: Image.Image,
    p: float = 0.5,
) -> Tuple[Image.Image, Image.Image]:
    """
    Flip both images horizontally with probability p.

    Note: For LPC sprites the left/right views are distinct, so horizontal
    flipping is only valid as augmentation (not as a semantic flip).
    """
    if random.random() < p:
        front = TF.hflip(front)
        back = TF.hflip(back)
    return front, back


def random_color_jitter(
    front: Image.Image,
    back: Image.Image,
    brightness: float = 0.1,
    contrast: float = 0.1,
    saturation: float = 0.1,
    hue: float = 0.05,
) -> Tuple[Image.Image, Image.Image]:
    """
    Apply identical random color jitter to both images.

    The same random parameters are sampled once and applied to both so
    that palette consistency is preserved.
    """
    brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
    contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
    saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
    hue_factor = random.uniform(-hue, hue)

    def _jitter(img: Image.Image) -> Image.Image:
        img = TF.adjust_brightness(img, brightness_factor)
        img = TF.adjust_contrast(img, contrast_factor)
        img = TF.adjust_saturation(img, saturation_factor)
        img = TF.adjust_hue(img, hue_factor)
        return img

    return _jitter(front), _jitter(back)


def random_palette_shift(
    front: Image.Image,
    back: Image.Image,
    p: float = 0.5,
    max_h: float = 0.08,
    max_s: float = 0.2,
    max_v: float = 0.2,
) -> Tuple[Image.Image, Image.Image]:
    """Apply a consistent HSV shift to both images with probability ``p``.

    The hue shift is sampled uniformly from ``[-max_h, max_h]`` (in normalized
    HSV space, where 1.0 corresponds to 360 degrees). Saturation and value are
    scaled by factors sampled from ``[1-max_s, 1+max_s]`` and
    ``[1-max_v, 1+max_v]`` respectively. Alpha channels are preserved.
    """
    if random.random() >= p:
        return front, back

    hue_shift = random.uniform(-max_h, max_h)
    sat_scale = random.uniform(1.0 - max_s, 1.0 + max_s)
    val_scale = random.uniform(1.0 - max_v, 1.0 + max_v)

    def _shift(img: Image.Image) -> Image.Image:
        rgba = img.convert("RGBA")
        alpha = rgba.split()[-1]
        hsv = rgba.convert("RGB").convert("HSV")
        h, s, v = (np.array(ch, dtype=np.float32) for ch in hsv.split())
        h = (h + hue_shift * 255.0) % 255.0
        s = np.clip(s * sat_scale, 0.0, 255.0)
        v = np.clip(v * val_scale, 0.0, 255.0)
        hsv_shifted = Image.merge(
            "HSV",
            (
                Image.fromarray(h.astype(np.uint8), mode="L"),
                Image.fromarray(s.astype(np.uint8), mode="L"),
                Image.fromarray(v.astype(np.uint8), mode="L"),
            ),
        )
        rgb_shifted = hsv_shifted.convert("RGB")
        rgba_shifted = Image.merge("RGBA", (*rgb_shifted.split(), alpha))
        return rgba_shifted

    return _shift(front), _shift(back)


def random_occlusion(
    image: Image.Image,
    max_fraction: float = 0.2,
    p: float = 0.3,
) -> Image.Image:
    """
    Randomly occlude a rectangular patch of an image with zeros.

    Used to make front-to-back reconstruction robust to missing regions.
    Applied only to the front image (Task 2 robustness augmentation).
    """
    if random.random() >= p:
        return image

    arr = np.array(image).copy()
    h, w = arr.shape[:2]
    ph = random.randint(1, max(1, int(h * max_fraction)))
    pw = random.randint(1, max(1, int(w * max_fraction)))
    y0 = random.randint(0, h - ph)
    x0 = random.randint(0, w - pw)
    arr[y0 : y0 + ph, x0 : x0 + pw] = 0
    return Image.fromarray(arr)


def to_tensor_pair(
    front: Image.Image,
    back: Image.Image,
    size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resize to `size`×`size`, convert to float tensors in [−1, 1].

    Returns:
        front_tensor: (C, H, W) float32 in [−1, 1]
        back_tensor:  (C, H, W) float32 in [−1, 1]
    """
    front = front.convert("RGB").resize((size, size), Image.NEAREST)
    back = back.convert("RGB").resize((size, size), Image.NEAREST)
    ft = TF.to_tensor(front) * 2.0 - 1.0
    bt = TF.to_tensor(back) * 2.0 - 1.0
    return ft, bt
