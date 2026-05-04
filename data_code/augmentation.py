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
    """对 front/back 同时施加相同的随机色调扰动。

    保留 alpha 通道：调色只作用在 RGB 上，免得 TF 函数对 RGBA 的不定行为
    或 alpha 被调成非二值。
    """
    brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
    contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
    saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
    hue_factor = random.uniform(-hue, hue)

    def _jitter(img: Image.Image) -> Image.Image:
        has_alpha = (img.mode == "RGBA")
        if has_alpha:
            alpha = img.getchannel("A")
            rgb = img.convert("RGB")
        else:
            rgb = img
        rgb = TF.adjust_brightness(rgb, brightness_factor)
        rgb = TF.adjust_contrast(rgb, contrast_factor)
        rgb = TF.adjust_saturation(rgb, saturation_factor)
        rgb = TF.adjust_hue(rgb, hue_factor)
        if has_alpha:
            return Image.merge("RGBA", (*rgb.split(), alpha))
        return rgb

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


def _rgba_to_rgba_tensor(img: Image.Image, size: int) -> torch.Tensor:
    """把 PIL RGBA 图转为 (4, H, W) 张量，RGB 在 [−1, 1]、A 在 [−1, 1]。

    关键修复：透明像素的 RGB 会被强制为 0（归一化后为 -1，即黑色）。
    LPC/一般的 PNG 透明区域常有 “垃圾 RGB”（如 (255,0,0,0)），直接拿来
    训练会导致背景被模型记下为红色/垃圾色。
    """
    rgba = img.convert("RGBA").resize((size, size), Image.NEAREST)
    # 把 alpha=0 的 RGB 强制清洗为 0
    np_rgba = np.array(rgba, dtype=np.uint8)  # (H, W, 4)
    a = np_rgba[..., 3:4]  # (H, W, 1)
    rgb = np_rgba[..., :3]
    mask = (a > 0).astype(np.uint8)  # 1 if visible, 0 otherwise
    rgb = rgb * mask  # zero out invisible pixels' RGB
    cleaned = np.concatenate([rgb, a], axis=-1)  # (H, W, 4)
    t = torch.from_numpy(cleaned).permute(2, 0, 1).float() / 255.0  # (4, H, W) in [0,1]
    t = t * 2.0 - 1.0  # 统一到 [-1, 1]，包括 alpha（这样全模型 I/O 一致）
    return t


def to_tensor_pair(
    front: Image.Image,
    back: Image.Image,
    size: int = 64,
    mode: str = "hstack_rgba",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 front/back 图对转为张量。

    默认模式 ``hstack_rgba``：
        - 每张图转成 (4, H, W)，RGB、A 都在 [-1, 1]。
        - 透明像素的 RGB 被清理为 0（归一化后 -1），避免 PNG 垃圾值座数据。

    Returns:
        front_tensor: (4, H, W) float32 in [-1, 1]，第 4 通道为 alpha
        back_tensor:  (4, H, W) float32 in [-1, 1]
        front_alpha:  (1, H, W) float32 in [0, 1]，二值化的前景 mask
        back_alpha:   (1, H, W) float32 in [0, 1]
    """
    if mode != "hstack_rgba":
        raise ValueError(f"Unsupported mode: {mode}")

    ft = _rgba_to_rgba_tensor(front, size)  # (4, H, W)
    bt = _rgba_to_rgba_tensor(back, size)

    # 二值化 alpha mask，从 [-1,1] 还原然后阀值
    fa = ((ft[3:4] + 1.0) / 2.0 > 0.5).float()  # (1, H, W)
    ba = ((bt[3:4] + 1.0) / 2.0 > 0.5).float()
    return ft, bt, fa, ba
