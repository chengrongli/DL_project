"""
Tests for the GaussianDiffusion wrapper (forward noising, DDPM/DDIM sampling).
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.unet import UNet
from models.diffusion import GaussianDiffusion, linear_beta_schedule, cosine_beta_schedule


# ---------------------------------------------------------------------------
# Beta schedules
# ---------------------------------------------------------------------------

def test_linear_schedule_range():
    betas = linear_beta_schedule(100)
    assert betas.shape == (100,)
    assert float(betas[0]) < float(betas[-1])
    assert (betas > 0).all() and (betas < 1).all()


def test_cosine_schedule_range():
    betas = cosine_beta_schedule(100)
    assert betas.shape == (100,)
    assert (betas > 0).all() and (betas < 1).all()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_diffusion(in_ch=6, out_ch=6, timesteps=10):
    unet = UNet(
        in_channels=in_ch,
        out_channels=out_ch,
        model_channels=8,
        channel_mults=(1, 2),
        n_res_blocks=1,
        attn_resolutions=(),
        time_emb_dim=32,
        image_size=8,
    )
    return GaussianDiffusion(unet, timesteps=timesteps, schedule="linear")


# ---------------------------------------------------------------------------
# Forward process
# ---------------------------------------------------------------------------

def test_q_sample_shape():
    diff = _small_diffusion()
    x0 = torch.randn(2, 6, 8, 8)
    t = torch.randint(0, 10, (2,))
    xt = diff.q_sample(x0, t)
    assert xt.shape == x0.shape


def test_q_sample_t0_close_to_x0():
    """At t=0, x_t should be very close to x_0 (sqrt(alpha_bar_0) ≈ 1)."""
    diff = _small_diffusion(timesteps=1000)
    diff2 = GaussianDiffusion(diff.model, timesteps=1000, schedule="cosine")
    x0 = torch.randn(2, 6, 8, 8)
    t = torch.zeros(2, dtype=torch.long)
    noise = torch.zeros_like(x0)   # fix noise to 0 to isolate schedule
    xt = diff2.q_sample(x0, t, noise=noise)
    # sqrt(alpha_bar_0) should be very close to 1 for cosine schedule
    assert (xt - x0).abs().mean().item() < 0.5


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------

def test_p_losses_scalar():
    diff = _small_diffusion()
    x0 = torch.randn(2, 6, 8, 8)
    t = torch.randint(0, 10, (2,))
    loss = diff.p_losses(x0, t)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_p_losses_with_cond_image():
    diff = _small_diffusion(in_ch=6, out_ch=3)
    target = torch.randn(2, 3, 8, 8)
    cond = torch.randn(2, 3, 8, 8)
    t = torch.randint(0, 10, (2,))
    loss = diff.p_losses(target, t, cond_image=cond)
    assert not torch.isnan(loss)


# ---------------------------------------------------------------------------
# DDPM sampling
# ---------------------------------------------------------------------------

def test_ddpm_sample_shape():
    diff = _small_diffusion(timesteps=5)
    shape = (2, 6, 8, 8)
    out = diff.ddpm_sample(shape, device=torch.device("cpu"))
    assert out.shape == shape


# ---------------------------------------------------------------------------
# DDIM sampling
# ---------------------------------------------------------------------------

def test_ddim_sample_shape():
    diff = _small_diffusion(timesteps=10)
    shape = (2, 6, 8, 8)
    out = diff.ddim_sample(shape, device=torch.device("cpu"), ddim_steps=5)
    assert out.shape == shape


def test_ddim_sample_with_cond():
    diff = _small_diffusion(in_ch=6, out_ch=3, timesteps=10)
    cond = torch.randn(2, 3, 8, 8)
    shape = (2, 3, 8, 8)
    out = diff.ddim_sample(
        shape, device=torch.device("cpu"), ddim_steps=5, cond_image=cond
    )
    assert out.shape == shape


def test_ddim_deterministic():
    """With eta=0, two runs with the same model weights should match."""
    diff = _small_diffusion(timesteps=10)
    shape = (1, 6, 8, 8)
    torch.manual_seed(42)
    a = diff.ddim_sample(shape, torch.device("cpu"), ddim_steps=5, eta=0.0)
    torch.manual_seed(42)
    b = diff.ddim_sample(shape, torch.device("cpu"), ddim_steps=5, eta=0.0)
    assert torch.allclose(a, b), "DDIM with eta=0 should be deterministic"
