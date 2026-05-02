"""
Evaluation metrics for generated sprite quality.

Metrics implemented:
  - MSE / PSNR        – pixel-level fidelity.
  - SSIM              – structural similarity.
  - Color histogram distance – palette consistency between front and back.
  - FID (Fréchet Inception Distance) – distribution-level quality (optional).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _to_float(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is float32 in [0, 1].

    If the minimum value is clearly below -0.5 (i.e. in [−1, 1] diffusion
    space), rescale to [0, 1].  Small negative values caused by floating-point
    arithmetic on already-[0,1] tensors are handled by the final clamp.
    """
    x = x.float()
    if x.min() < -0.5:
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# MSE / PSNR
# ---------------------------------------------------------------------------

def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean squared error between two image tensors."""
    p, t = _to_float(pred), _to_float(target)
    return F.mse_loss(p, t).item()


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak signal-to-noise ratio (dB)."""
    err = mse(pred, target)
    if err == 0.0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / err)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.outer(g)


_SSIM_KERNEL: Optional[torch.Tensor] = None
_SSIM_KERNEL_SIZE = 11


def ssim(pred: torch.Tensor, target: torch.Tensor,
         window_size: int = 11, data_range: float = 1.0) -> float:
    """
    Structural Similarity Index (SSIM).

    Args:
        pred, target: (B, C, H, W) or (C, H, W) tensors.
        data_range:   Value range of the images (1.0 for [0,1]).

    Returns:
        Mean SSIM scalar.
    """
    p = _to_float(pred)
    t = _to_float(target)
    if p.ndim == 3:
        p = p.unsqueeze(0)
        t = t.unsqueeze(0)

    C = p.shape[1]
    device = p.device

    # Build per-channel Gaussian kernel
    kernel = _gaussian_kernel(window_size).to(device)
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)

    pad = window_size // 2

    def conv(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu_p = conv(p)
    mu_t = conv(t)
    mu_p2 = mu_p ** 2
    mu_t2 = mu_t ** 2
    mu_pt = mu_p * mu_t

    sigma_p2 = conv(p * p) - mu_p2
    sigma_t2 = conv(t * t) - mu_t2
    sigma_pt = conv(p * t) - mu_pt

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = (
        (2 * mu_pt + C1) * (2 * sigma_pt + C2)
        / ((mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2))
    )
    return ssim_map.mean().item()


# ---------------------------------------------------------------------------
# Color histogram distance
# ---------------------------------------------------------------------------

def color_histogram(img: torch.Tensor, bins: int = 32) -> np.ndarray:
    """
    Compute a 3-channel color histogram of an image.

    Args:
        img:  (C, H, W) or (B, C, H, W) tensor in [0, 1] or [−1, 1].
        bins: Number of histogram bins per channel.

    Returns:
        Normalised histogram as (3*bins,) numpy array.
    """
    img = _to_float(img)
    if img.ndim == 4:
        img = img[0]  # use first image if batched
    arr = img.cpu().numpy()  # (3, H, W)
    histograms = []
    for c in range(arr.shape[0]):
        h, _ = np.histogram(arr[c].ravel(), bins=bins, range=(0.0, 1.0))
        histograms.append(h.astype(np.float32))
    hist = np.concatenate(histograms)
    norm = hist.sum()
    return hist / norm if norm > 0 else hist


def histogram_distance(img_a: torch.Tensor, img_b: torch.Tensor,
                        bins: int = 32) -> float:
    """
    L1 distance between color histograms of two images.
    Lower → more similar palettes.
    """
    h_a = color_histogram(img_a, bins)
    h_b = color_histogram(img_b, bins)
    return float(np.abs(h_a - h_b).sum())


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_pair_batch(
    fronts: torch.Tensor,
    pred_backs: torch.Tensor,
    gt_backs: torch.Tensor,
) -> dict:
    """
    Evaluate a batch of front/back pairs.

    Args:
        fronts:     (B, 3, H, W) front images.
        pred_backs: (B, 3, H, W) predicted back images.
        gt_backs:   (B, 3, H, W) ground-truth back images.

    Returns:
        Dict with mean MSE, PSNR, SSIM, hist_dist_front_back,
        hist_dist_pred_gt.
    """
    B = fronts.shape[0]
    mse_vals, psnr_vals, ssim_vals = [], [], []
    hist_fb, hist_pg = [], []

    for i in range(B):
        mse_vals.append(mse(pred_backs[i], gt_backs[i]))
        psnr_vals.append(psnr(pred_backs[i], gt_backs[i]))
        ssim_vals.append(ssim(pred_backs[i], gt_backs[i]))
        # Palette consistency: predicted back vs front should be similar
        hist_fb.append(histogram_distance(pred_backs[i], fronts[i]))
        hist_pg.append(histogram_distance(pred_backs[i], gt_backs[i]))

    return {
        "mse":                float(np.mean(mse_vals)),
        "psnr":               float(np.mean(psnr_vals)),
        "ssim":               float(np.mean(ssim_vals)),
        "hist_pred_front":    float(np.mean(hist_fb)),
        "hist_pred_gt":       float(np.mean(hist_pg)),
    }
