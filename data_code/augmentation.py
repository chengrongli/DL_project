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
    brightness: float = 0.25,
    contrast: float = 0.25,
    saturation: float = 0.3,
    hue: float = 0.12,
    strength: float = 1.0,
) -> Tuple[Image.Image, Image.Image]:
    """对 front/back 同时施加相同的随机色调扰动。

    保留 alpha 通道：调色只作用在 RGB 上，免得 TF 函数对 RGBA 的不定行为
    或 alpha 被调成非二值。
    """
    # Allow scaling the jitter strength (useful to switch between mild/strong)
    brightness_factor = random.uniform(
        max(0, 1 - brightness * strength), 1 + brightness * strength
    )
    contrast_factor = random.uniform(
        max(0, 1 - contrast * strength), 1 + contrast * strength
    )
    saturation_factor = random.uniform(
        max(0, 1 - saturation * strength), 1 + saturation * strength
    )
    hue_factor = random.uniform(-hue * strength, hue * strength)

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
    max_fraction: float = 0.35,
    p: float = 0.5,
    n_patches_max: int = 3,
    intensity: float = 0.5,
    fill: str = "zero",
) -> Image.Image:
    """
    Randomly occlude one or more rectangular patches of an image with zeros.

    Used to make front-to-back reconstruction robust to missing regions and
    to break the "copy-front-shape" shortcut that hurts OOD generalization.
    Applied only to the front image (Task 2 robustness augmentation).
    """
    if random.random() >= p:
        return image

    arr = np.array(image).copy()
    h, w = arr.shape[:2]
    # intensity controls size and fill type: higher intensity -> larger patches
    scale = 0.5 + float(np.clip(intensity, 0.0, 1.0)) * 1.5
    n_patches = random.randint(1, max(1, int(n_patches_max * scale)))
    for _ in range(n_patches):
        ph = random.randint(1, max(1, int(h * max_fraction * scale)))
        pw = random.randint(1, max(1, int(w * max_fraction * scale)))
        y0 = random.randint(0, max(0, h - ph))
        x0 = random.randint(0, max(0, w - pw))

        if fill == "zero":
            arr[y0 : y0 + ph, x0 : x0 + pw] = 0
        elif fill == "noise":
            noise = (np.random.randn(ph, pw, arr.shape[2]) * 255.0 * intensity).astype(
                np.int32
            )
            patch = arr[y0 : y0 + ph, x0 : x0 + pw].astype(np.int32) + noise
            arr[y0 : y0 + ph, x0 : x0 + pw] = np.clip(patch, 0, 255).astype(np.uint8)
        elif fill == "color":
            color = np.array(
                [
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ],
                dtype=np.uint8,
            )
            if arr.shape[2] == 4:
                alpha_val = int(255 * (1.0 - intensity))
                color = np.concatenate([color, np.array([alpha_val], dtype=np.uint8)])
            arr[y0 : y0 + ph, x0 : x0 + pw] = color
        else:
            # unknown fill mode -> fall back to zero
            arr[y0 : y0 + ph, x0 : x0 + pw] = 0

    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Front-only augmentations (used to break the copy-shortcut in Task 2)
# ---------------------------------------------------------------------------

def front_only_geometric_jitter(
    image: Image.Image,
    p: float = 0.7,
    max_translate: int = 2,
    max_rotate_deg: float = 6.0,
    max_scale: float = 0.06,
) -> Image.Image:
    """Apply a small geometric perturbation **only** to the front image.

    Goal: prevent the model from learning a per-pixel "copy front, paint back"
    shortcut. The back stays untouched, so the model has to align via
    semantics rather than pixel coordinates.

    Magnitudes are intentionally small (~2 px / ~6°) so the front still
    semantically matches the back; we are jittering, not breaking, alignment.
    """
    if random.random() >= p:
        return image

    angle = random.uniform(-max_rotate_deg, max_rotate_deg)
    tx = random.randint(-max_translate, max_translate)
    ty = random.randint(-max_translate, max_translate)
    scale = 1.0 + random.uniform(-max_scale, max_scale)

    return TF.affine(
        image,
        angle=angle,
        translate=(tx, ty),
        scale=scale,
        shear=(0.0, 0.0),
        interpolation=TF.InterpolationMode.NEAREST,
        fill=0,
    )


def front_only_palette_perturb(
    image: Image.Image,
    p: float = 0.5,
    grayscale_p: float = 0.15,
    posterize_p: float = 0.2,
    posterize_bits: int = 4,
    hue_shift: float = 0.2,
) -> Image.Image:
    """Aggressively perturb the front's palette while keeping alpha intact.

    Forces the model to treat the front condition as a *shape/identity cue*
    rather than a literal color source. Without this, the network learns to
    just copy the LPC palette and breaks on any OOD recoloring (real photos,
    other pixel-art styles, etc.).
    """
    if random.random() >= p:
        return image

    has_alpha = (image.mode == "RGBA")
    if has_alpha:
        alpha = image.getchannel("A")
        rgb = image.convert("RGB")
    else:
        rgb = image

    # Strong hue shift
    if hue_shift > 0:
        rgb = TF.adjust_hue(rgb, random.uniform(-hue_shift, hue_shift))

    # Occasionally collapse to grayscale
    if random.random() < grayscale_p:
        rgb = TF.rgb_to_grayscale(rgb, num_output_channels=3)

    # Occasionally posterize (cartoonify the palette further)
    if random.random() < posterize_p:
        rgb = TF.posterize(rgb, bits=max(1, posterize_bits))

    if has_alpha:
        return Image.merge("RGBA", (*rgb.split(), alpha))
    return rgb


def front_only_background_noise_tensor(
    cond_rgb: torch.Tensor,
    cond_alpha: torch.Tensor,
    p: float = 0.4,
    noise_strength: float = 0.6,
) -> torch.Tensor:
    """Replace the transparent background of the front with random color/noise.

    Operates **at the tensor stage** (RGB in [-1, 1], alpha in [0, 1]) so that
    it survives the alpha-premultiply step inside ``_rgba_to_rgba_tensor``.

    The model must not assume "background = black/transparent" — that's a cue
    that disappears on OOD inputs (e.g. photos with cluttered backgrounds).
    Only the alpha=0 region is overwritten; the foreground sprite is preserved.

    Args:
        cond_rgb:   (3, H, W) front RGB in [-1, 1] (already premultiplied).
        cond_alpha: (1, H, W) binary front alpha mask in [0, 1].

    Returns:
        (3, H, W) tensor in [-1, 1] with the background region replaced.
    """
    if random.random() >= p:
        return cond_rgb

    _, h, w = cond_rgb.shape
    bg_type = random.choice(["solid", "noise", "gradient"])

    if bg_type == "solid":
        color = torch.empty(3).uniform_(-1.0, 1.0)
        bg = color.view(3, 1, 1).expand(3, h, w).clone()
    elif bg_type == "noise":
        bg = torch.empty(3, h, w).uniform_(-1.0, 1.0)
        base = torch.empty(3).uniform_(-1.0, 1.0).view(3, 1, 1)
        bg = bg * noise_strength + base * (1.0 - noise_strength)
    else:  # gradient
        c0 = torch.empty(3).uniform_(-1.0, 1.0).view(3, 1, 1)
        c1 = torch.empty(3).uniform_(-1.0, 1.0).view(3, 1, 1)
        ramp = torch.linspace(0.0, 1.0, w).view(1, 1, w)
        bg = c0 * (1.0 - ramp) + c1 * ramp
        bg = bg.expand(3, h, w).clone()

    fg_mask = cond_alpha  # (1, H, W) in [0, 1]
    composed = cond_rgb * fg_mask + bg * (1.0 - fg_mask)
    return composed.clamp(-1.0, 1.0)


def _rgba_to_rgba_tensor(img: Image.Image, size: int) -> torch.Tensor:
    """把 PIL RGBA 图转为 (4, H, W) 张量，RGB/A 都映射到 [-1, 1]。

    关键修复（稳定训练）：
    1) 透明像素 RGB 清零，去掉 PNG 中常见的隐藏垃圾色；
    2) 对半透明像素执行 alpha 预乘（premultiply），避免边缘处 RGB 与 A
       不一致带来的高频噪声，减轻训练中背景发散/崩溃风险。
    """
    rgba = img.convert("RGBA").resize((size, size), Image.NEAREST)

    np_rgba = np.array(rgba, dtype=np.float32)  # (H, W, 4)
    rgb = np_rgba[..., :3]
    alpha = np_rgba[..., 3:4] / 255.0  # [0,1]

    # 透明区域强制黑底；半透明区域做预乘，去掉“看不见但颜色很亮”的噪声来源。
    rgb = rgb * alpha

    cleaned = np.concatenate([rgb, alpha * 255.0], axis=-1).clip(0.0, 255.0).astype(np.uint8)
    t = torch.from_numpy(cleaned).permute(2, 0, 1).float() / 255.0  # (4, H, W) in [0,1]
    t = t * 2.0 - 1.0
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
        - 透明像素 RGB 会被清理（并对半透明边缘做 alpha 预乘），
          避免 PNG 背景隐藏色造成训练不稳定。

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
