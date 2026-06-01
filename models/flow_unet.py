"""U-Net backbone for conditional Flow Matching with FiLM + Cross-Attention."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Safely build GroupNorm where groups divides channels."""
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """Residual conv block with time conditioning and optional FiLM attribute conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
        attr_cond_dim: int | None = None,
    ):
        super().__init__()
        self.norm1 = _gn(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        # FiLM modulation for attribute conditioning
        if attr_cond_dim is not None:
            self.attr_film = nn.Sequential(
                nn.SiLU(),
                nn.Linear(attr_cond_dim, 2 * out_channels),
            )
            # Zero-init so FiLM starts as identity
            nn.init.zeros_(self.attr_film[-1].weight)
            nn.init.zeros_(self.attr_film[-1].bias)
        else:
            self.attr_film = None

        self.norm2 = _gn(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        attr_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = self.residual_conv(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Time conditioning (additive, unchanged)
        t_proj = self.time_mlp(t_emb)
        h = h + t_proj[:, :, None, None]

        # Attribute conditioning (FiLM: scale + shift)
        if attr_cond is not None and self.attr_film is not None:
            gamma_beta = self.attr_film(attr_cond)
            gamma, beta = gamma_beta.chunk(2, dim=1)
            h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class AttentionBlock(nn.Module):
    """Spatial self-attention block."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = _gn(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, h, w = x.shape

        x = self.norm(x)
        q = self.q(x).view(b, c, -1).permute(0, 2, 1)
        k = self.k(x).view(b, c, -1)
        v = self.v(x).view(b, c, -1).permute(0, 2, 1)

        attention = torch.bmm(q, k) * (c ** -0.5)
        attention = F.softmax(attention, dim=-1)
        out = torch.bmm(attention, v)

        out = out.permute(0, 2, 1).view(b, c, h, w)
        out = self.proj_out(out)
        return out + residual


class CrossAttentionBlock(nn.Module):
    """Cross-attention: queries from spatial features, keys/values from attribute tokens.

    Allows the model to learn spatial correspondences between attribute fields
    (e.g. "hair_color=red" should attend strongly to the top of the sprite).
    """

    def __init__(
        self,
        channels: int,
        attr_dim: int = 256,
        num_heads: int = 4,
    ):
        super().__init__()
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = _gn(channels)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(attr_dim, channels)
        self.v_proj = nn.Linear(attr_dim, channels)
        self.out_proj = nn.Linear(channels, channels)

        # Zero-init output so cross-attention starts as identity
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, attr_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) spatial features.
            attr_tokens: (B, T, attr_dim) attribute token sequence.

        Returns:
            (B, C, H, W) with cross-attention applied residually.
        """
        residual = x
        b, c, h, w = x.shape

        x = self.norm(x)
        # Flatten spatial dims: (B, C, H*W) -> (B, H*W, C)
        spatial = x.view(b, c, -1).permute(0, 2, 1)

        # Q from spatial, K/V from attr tokens
        q = self.q_proj(spatial)  # (B, H*W, C)
        k = self.k_proj(attr_tokens)  # (B, T, C)
        v = self.v_proj(attr_tokens)  # (B, T, C)

        # Reshape for multi-head: (B, S, heads, head_dim) -> (B, heads, S, head_dim)
        q = q.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, H*W, head_dim)

        # Merge heads: (B, H*W, C)
        out = out.transpose(1, 2).contiguous().view(b, -1, c)
        out = self.out_proj(out)

        # Reshape back: (B, C, H, W)
        out = out.permute(0, 2, 1).view(b, c, h, w)
        return out + residual


class UNet(nn.Module):
    """Flow-Matching U-Net with FiLM + Cross-Attention attribute conditioning.

    Attributes are injected via two pathways:
    1. FiLM: per-channel scale/shift in every ConvBlock (global, non-spatial).
    2. Cross-Attention: at attention resolutions, spatial interaction with attr tokens.
    """

    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 3,
        channel_mult: tuple[int, ...] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        time_emb_dim: int = 256,
        attention_resolutions: tuple[int, ...] = (16,),
        attr_cond_dim: int | None = None,
        attr_token_dim: int | None = None,
        cross_attn_heads: int = 4,
    ):
        super().__init__()

        attr_cond_dim = attr_cond_dim or time_emb_dim
        attr_token_dim = attr_token_dim or time_emb_dim

        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.out_channels = out_channels
        self.attr_token_dim = attr_token_dim

        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # ---- Encoder ----
        self.down_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        channels = model_channels
        self.skip_connection_channels = [channels]

        for i, mult in enumerate(channel_mult):
            out_ch = mult * model_channels

            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [
                    ConvBlock(channels, out_ch, time_emb_dim, dropout, attr_cond_dim=attr_cond_dim),
                ]
                channels = out_ch

                resolution = 64 // (2 ** i)
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(channels))
                    layers.append(CrossAttentionBlock(channels, attr_dim=attr_token_dim, num_heads=cross_attn_heads))

                self.down_blocks.append(nn.ModuleList(layers))
                self.skip_connection_channels.append(channels)

            if i != len(channel_mult) - 1:
                self.downsample_blocks.append(
                    nn.Conv2d(channels, channels, 3, stride=2, padding=1),
                )
                self.skip_connection_channels.append(channels)

        # ---- Mid ----
        self.mid_block1 = ConvBlock(channels, channels, time_emb_dim, dropout, attr_cond_dim=attr_cond_dim)
        self.mid_attn = AttentionBlock(channels)
        self.mid_cross_attn = CrossAttentionBlock(channels, attr_dim=attr_token_dim, num_heads=cross_attn_heads)
        self.mid_block2 = ConvBlock(channels, channels, time_emb_dim, dropout, attr_cond_dim=attr_cond_dim)

        # ---- Decoder ----
        self.up_blocks = nn.ModuleList()

        for i, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = mult * model_channels

            for j in range(num_res_blocks + 1):
                skip_channels = self.skip_connection_channels.pop()
                in_channels_up = channels + skip_channels

                layers: list[nn.Module] = [
                    ConvBlock(in_channels_up, out_ch, time_emb_dim, dropout, attr_cond_dim=attr_cond_dim),
                ]
                channels = out_ch

                resolution = 64 // (2 ** i)
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(channels))
                    layers.append(CrossAttentionBlock(channels, attr_dim=attr_token_dim, num_heads=cross_attn_heads))

                if i != 0 and j == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1))

                self.up_blocks.append(nn.ModuleList(layers))

        self.output_norm = _gn(channels)
        self.output_conv = nn.Conv2d(channels, self.out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attr_cond: torch.Tensor | None = None,
        attr_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        # NOTE: attr_cond is NO LONGER added to t_emb — it goes through FiLM in each ConvBlock.

        h = self.input_conv(x)
        skip_features = [h]

        # ---- Encoder ----
        down_block_idx = 0
        for i in range(len(self.channel_mult)):
            for _ in range(self.num_res_blocks):
                module_list = self.down_blocks[down_block_idx]
                down_block_idx += 1
                for layer in module_list:
                    if isinstance(layer, ConvBlock):
                        h = layer(h, t_emb, attr_cond=attr_cond)
                    elif isinstance(layer, CrossAttentionBlock):
                        if attr_tokens is not None:
                            h = layer(h, attr_tokens)
                        # else skip — equivalent to unconditional
                    else:
                        h = layer(h)  # AttentionBlock (self-attention)
                skip_features.append(h)

            if i != len(self.channel_mult) - 1:
                h = self.downsample_blocks[i](h)
                skip_features.append(h)

        # ---- Mid ----
        h = self.mid_block1(h, t_emb, attr_cond=attr_cond)
        h = self.mid_attn(h)
        if attr_tokens is not None:
            h = self.mid_cross_attn(h, attr_tokens)
        h = self.mid_block2(h, t_emb, attr_cond=attr_cond)

        # ---- Decoder ----
        for module_list in self.up_blocks:
            for layer in module_list:
                if isinstance(layer, ConvBlock):
                    skip = skip_features.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = layer(h, t_emb, attr_cond=attr_cond)
                elif isinstance(layer, CrossAttentionBlock):
                    if attr_tokens is not None:
                        h = layer(h, attr_tokens)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                else:
                    h = layer(h)  # ConvTranspose2d

        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        return h
