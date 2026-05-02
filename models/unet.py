"""
Lightweight U-Net backbone for pixel-art sprite diffusion.

Architecture:
  - Sinusoidal timestep embedding + optional attribute/condition embedding.
  - Encoder: [Conv → ResBlock × n_res] × n_levels with stride-2 downsampling.
  - Middle: ResBlock → Self-Attention → ResBlock.
  - Decoder: Upsample → ResBlock × n_res with skip connections.
  - Head: GroupNorm → SiLU → Conv 3×3.

All spatial resolutions are powers of two (64 → 32 → 16 → 8 by default).
The model supports:
  - Unconditional generation (no extra conditioning).
  - Classifier-free guidance via a null embedding dropped during training.
  - Image conditioning (Task 2): the condition image is channel-concatenated
    with the noisy input before the first conv.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with up to 32 groups."""
    groups = min(32, channels)
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbedding(nn.Module):
    """Maps a scalar timestep t to a (dim,) embedding vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer or float timestep tensor.
        Returns:
            emb: (B, dim) float32 tensor.
        """
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)  # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=1)   # (B, dim)
        return emb


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Residual block with time (and optional context) conditioning.

    time_emb_dim → learned scale+shift via linear projection.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 time_emb_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = _norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2),  # scale + shift
        )

        self.norm2 = _norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        # Time conditioning: scale + shift
        ts = self.time_proj(t_emb)            # (B, 2*C)
        scale, shift = ts.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))

        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Self-attention block
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Multi-head self-attention for spatial feature maps."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        assert channels % num_heads == 0
        self.norm = _norm(channels)
        self.heads = num_heads
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # (B, heads, H*W, head_dim)
        q = rearrange(q, "b (heads d) h w -> b heads (h w) d", heads=self.heads)
        k = rearrange(k, "b (heads d) h w -> b heads (h w) d", heads=self.heads)
        v = rearrange(v, "b (heads d) h w -> b heads (h w) d", heads=self.heads)

        scale = q.shape[-1] ** -0.5
        attn = torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1)
        out = attn @ v

        out = rearrange(out, "b heads (h w) d -> b (heads d) h w", h=H, w=W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# Encoder & Decoder levels
# ---------------------------------------------------------------------------

class DownBlock(nn.Module):
    """Encoder level: n_res ResBlocks then stride-2 conv for downsampling."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int,
                 n_res: int = 2, use_attn: bool = False,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attns = nn.ModuleList()
        ch = in_ch
        for _ in range(n_res):
            self.resnets.append(ResBlock(ch, out_ch, time_dim, dropout))
            self.attns.append(SelfAttention(out_ch) if use_attn else nn.Identity())
            ch = out_ch
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []
        for res, attn in zip(self.resnets, self.attns):
            x = res(x, t_emb)
            x = attn(x)
            skips.append(x)
        x = self.downsample(x)
        return x, skips


class UpBlock(nn.Module):
    """
    Decoder level: bilinear upsample then n_res ResBlocks with skip connections.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 time_dim: int, n_res: int = 2, use_attn: bool = False,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.resnets = nn.ModuleList()
        self.attns = nn.ModuleList()
        ch = in_ch
        for i in range(n_res):
            extra = skip_ch if i == 0 else 0
            self.resnets.append(ResBlock(ch + extra, out_ch, time_dim, dropout))
            self.attns.append(SelfAttention(out_ch) if use_attn else nn.Identity())
            ch = out_ch

    def forward(
        self, x: torch.Tensor, skips: List[torch.Tensor], t_emb: torch.Tensor
    ) -> torch.Tensor:
        x = self.upsample(x)
        for i, (res, attn) in enumerate(zip(self.resnets, self.attns)):
            if i == 0 and skips:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
            x = res(x, t_emb)
            x = attn(x)
        return x


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Conditional diffusion U-Net for pixel-art sprite generation.

    Args:
        in_channels:     Input channels to the model.
                         - Task 1 (joint): 6  (concat front+back, 3+3 RGB)
                         - Task 2 (cond):  6  (noisy target 3 + cond front 3)
                         Set via config.
        out_channels:    Channels of the predicted noise.
                         - Task 1: 6  (front+back noise)
                         - Task 2: 3  (back noise only)
        model_channels:  Base channel width.
        channel_mults:   Channel multiplier at each resolution level.
        n_res_blocks:    Number of residual blocks per level.
        attn_resolutions: Set of spatial resolutions where attention is applied.
        time_emb_dim:    Dimension of the sinusoidal timestep embedding.
        cond_emb_dim:    Dimension of optional attribute/class embedding.
                         Set to 0 to disable.
        dropout:         Dropout probability inside ResBlocks.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 6,
        model_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        n_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (8,),
        time_emb_dim: int = 256,
        cond_emb_dim: int = 0,
        dropout: float = 0.1,
        image_size: int = 64,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size

        # ------------------------------------------------------------------
        # Timestep embedding MLP
        # ------------------------------------------------------------------
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Optional attribute/class embedding (classifier-free guidance)
        total_emb_dim = time_emb_dim
        if cond_emb_dim > 0:
            self.cond_proj = nn.Linear(cond_emb_dim, time_emb_dim)
            total_emb_dim = time_emb_dim  # added, not concatenated
        else:
            self.cond_proj = None

        # ------------------------------------------------------------------
        # Build encoder
        # ------------------------------------------------------------------
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        ch = model_channels
        current_res = image_size
        skip_channels: List[int] = []

        for mult in channel_mults:
            out_ch = model_channels * mult
            use_attn = current_res in attn_resolutions
            block = DownBlock(ch, out_ch, total_emb_dim,
                              n_res=n_res_blocks, use_attn=use_attn,
                              dropout=dropout)
            self.down_blocks.append(block)
            # Each DownBlock produces n_res skip tensors
            for _ in range(n_res_blocks):
                skip_channels.append(out_ch)
            ch = out_ch
            current_res //= 2

        # ------------------------------------------------------------------
        # Middle block
        # ------------------------------------------------------------------
        mid_use_attn = current_res in attn_resolutions or True  # always attend
        self.mid_res1 = ResBlock(ch, ch, total_emb_dim, dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid_res2 = ResBlock(ch, ch, total_emb_dim, dropout)

        # ------------------------------------------------------------------
        # Build decoder (reversed)
        # ------------------------------------------------------------------
        self.up_blocks = nn.ModuleList()
        rev_mults = list(reversed(channel_mults))
        for i, mult in enumerate(rev_mults):
            out_ch = model_channels * mult
            skip_ch = skip_channels.pop() if skip_channels else 0
            use_attn = current_res in attn_resolutions
            # Use the encoder channel count from the matching level
            enc_ch = model_channels * channel_mults[len(channel_mults) - 1 - i]
            block = UpBlock(ch, enc_ch, out_ch, total_emb_dim,
                            n_res=n_res_blocks, use_attn=use_attn,
                            dropout=dropout)
            self.up_blocks.append(block)
            ch = out_ch
            current_res *= 2

        # ------------------------------------------------------------------
        # Output head
        # ------------------------------------------------------------------
        self.out_norm = _norm(ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:        (B, in_channels, H, W) noisy input.
            t:        (B,) integer timestep tensor.
            cond_emb: (B, cond_emb_dim) optional attribute / CFG embedding.

        Returns:
            (B, out_channels, H, W) predicted noise.
        """
        t_emb = self.time_embed(t)

        if cond_emb is not None and self.cond_proj is not None:
            t_emb = t_emb + self.cond_proj(cond_emb)

        # Encoder
        h = self.input_conv(x)
        all_skips: List[List[torch.Tensor]] = []

        for block in self.down_blocks:
            h, skips = block(h, t_emb)
            all_skips.append(skips)

        # Middle
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        # Decoder
        for block, skips in zip(self.up_blocks, reversed(all_skips)):
            h = block(h, skips, t_emb)

        # Output
        return self.out_conv(F.silu(self.out_norm(h)))
