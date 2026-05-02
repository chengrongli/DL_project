"""
Tests for evaluation metrics.
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.metrics import mse, psnr, ssim, color_histogram, histogram_distance, evaluate_pair_batch


def test_mse_identical():
    x = torch.rand(3, 16, 16)
    assert mse(x, x) < 1e-6


def test_mse_different():
    x = torch.zeros(3, 16, 16)
    y = torch.ones(3, 16, 16)
    assert mse(x, y) > 0


def test_psnr_identical():
    x = torch.rand(3, 16, 16)
    assert psnr(x, x) == float("inf")


def test_psnr_upper_bound():
    torch.manual_seed(0)
    x = torch.rand(3, 16, 16)
    # Add a tiny deterministic perturbation that stays in [0, 1]
    noise = 0.005 * torch.ones_like(x)
    y = (x + noise).clamp(0.0, 1.0)
    assert psnr(x, y) > 30  # high PSNR for tiny perturbation


def test_ssim_identical():
    x = torch.rand(1, 3, 32, 32)
    score = ssim(x, x)
    assert score > 0.99


def test_ssim_different():
    x = torch.rand(1, 3, 32, 32)
    y = torch.rand(1, 3, 32, 32)
    score = ssim(x, y)
    assert 0.0 <= score <= 1.0


def test_color_histogram_shape():
    img = torch.rand(3, 64, 64)
    hist = color_histogram(img, bins=32)
    assert hist.shape == (3 * 32,)


def test_color_histogram_normalized():
    img = torch.rand(3, 64, 64)
    hist = color_histogram(img, bins=32)
    import numpy as np
    assert abs(hist.sum() - 1.0) < 1e-5


def test_histogram_distance_identical():
    img = torch.rand(3, 64, 64)
    d = histogram_distance(img, img)
    assert d < 1e-5


def test_histogram_distance_range():
    a = torch.zeros(3, 64, 64)
    b = torch.ones(3, 64, 64)
    d = histogram_distance(a, b)
    assert d >= 0.0


def test_evaluate_pair_batch():
    fronts = torch.rand(4, 3, 32, 32)
    pred_backs = torch.rand(4, 3, 32, 32)
    gt_backs = torch.rand(4, 3, 32, 32)
    results = evaluate_pair_batch(fronts, pred_backs, gt_backs)
    assert "mse" in results
    assert "psnr" in results
    assert "ssim" in results
    assert "hist_pred_front" in results
    assert "hist_pred_gt" in results
    assert results["mse"] >= 0
