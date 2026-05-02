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

def random_horizontal_flip(front: Image.Image, back: Image.Image,
                             p: float = 0.5) -> Tuple[Image.Image, Image.Image]:
    """
    Flip both images horizontally with probability p.

    Note: For LPC sprites the left/right views are distinct, so horizontal
    flipping is only valid as augmentation (not as a semantic flip).
    """
    if random.random() < p:
        front = TF.hflip(front)
        back = TF.hflip(back)
    return front, back


def random_color_jitter(front: Image.Image, back: Image.Image,
                         brightness: float = 0.1,
                         contrast: float = 0.1,
                         saturation: float = 0.1,
                         hue: float = 0.05) -> Tuple[Image.Image, Image.Image]:
    """
    Apply identical random color jitter to both images.

    The same random parameters are sampled once and applied to both so
    that palette consistency is preserved.
    """
    # Sample parameters once
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


def random_occlusion(image: Image.Image,
                      max_fraction: float = 0.2,
                      p: float = 0.3) -> Image.Image:
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
    arr[y0:y0 + ph, x0:x0 + pw] = 0
    return Image.fromarray(arr)


def to_tensor_pair(front: Image.Image,
                    back: Image.Image,
                    size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
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
