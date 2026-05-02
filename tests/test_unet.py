"""
Tests for the U-Net model architecture.
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.unet import UNet, SinusoidalPositionEmbedding, ResBlock, SelfAttention


# ---------------------------------------------------------------------------
# SinusoidalPositionEmbedding
# ---------------------------------------------------------------------------

def test_sinusoidal_embedding_shape():
    emb = SinusoidalPositionEmbedding(dim=64)
    t = torch.arange(4)
    out = emb(t)
    assert out.shape == (4, 64), f"Unexpected shape: {out.shape}"


def test_sinusoidal_embedding_dtype():
    emb = SinusoidalPositionEmbedding(dim=32)
    t = torch.tensor([0, 100, 999])
    out = emb(t)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------

def test_resblock_same_channels():
    block = ResBlock(in_channels=32, out_channels=32, time_emb_dim=64)
    x = torch.randn(2, 32, 16, 16)
    t = torch.randn(2, 64)
    out = block(x, t)
    assert out.shape == x.shape


def test_resblock_different_channels():
    block = ResBlock(in_channels=16, out_channels=64, time_emb_dim=64)
    x = torch.randn(2, 16, 16, 16)
    t = torch.randn(2, 64)
    out = block(x, t)
    assert out.shape == (2, 64, 16, 16)


# ---------------------------------------------------------------------------
# SelfAttention
# ---------------------------------------------------------------------------

def test_selfattention_shape():
    attn = SelfAttention(channels=32, num_heads=4)
    x = torch.randn(2, 32, 8, 8)
    out = attn(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# UNet – Task 1 (6→6)
# ---------------------------------------------------------------------------

def test_unet_task1_forward():
    """Task 1: joint front+back generation (6-channel in/out)."""
    model = UNet(
        in_channels=6,
        out_channels=6,
        model_channels=16,       # tiny for test speed
        channel_mults=(1, 2),
        n_res_blocks=1,
        attn_resolutions=(8,),
        time_emb_dim=64,
        cond_emb_dim=0,
        image_size=16,
    )
    x = torch.randn(2, 6, 16, 16)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    assert out.shape == (2, 6, 16, 16), f"Unexpected shape: {out.shape}"


def test_unet_task2_forward():
    """Task 2: image-conditioned front→back (6-channel in, 3-channel out)."""
    model = UNet(
        in_channels=6,
        out_channels=3,
        model_channels=16,
        channel_mults=(1, 2),
        n_res_blocks=1,
        attn_resolutions=(8,),
        time_emb_dim=64,
        cond_emb_dim=0,
        image_size=16,
    )
    x = torch.randn(2, 6, 16, 16)  # noisy_back(3) + front_cond(3)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    assert out.shape == (2, 3, 16, 16)


def test_unet_with_cond_emb():
    """U-Net with attribute/class embedding conditioning."""
    model = UNet(
        in_channels=6,
        out_channels=6,
        model_channels=16,
        channel_mults=(1, 2),
        n_res_blocks=1,
        attn_resolutions=(),
        time_emb_dim=64,
        cond_emb_dim=32,
        image_size=16,
    )
    x = torch.randn(2, 6, 16, 16)
    t = torch.randint(0, 1000, (2,))
    cond = torch.randn(2, 32)
    out = model(x, t, cond_emb=cond)
    assert out.shape == (2, 6, 16, 16)


def test_unet_gradient_flow():
    """Ensure gradients flow back through the model."""
    model = UNet(
        in_channels=6,
        out_channels=6,
        model_channels=8,
        channel_mults=(1, 2),
        n_res_blocks=1,
        attn_resolutions=(),
        time_emb_dim=32,
        image_size=8,
    )
    x = torch.randn(1, 6, 8, 8, requires_grad=True)
    t = torch.tensor([5])
    out = model(x, t)
    loss = out.mean()
    loss.backward()
    assert x.grad is not None
