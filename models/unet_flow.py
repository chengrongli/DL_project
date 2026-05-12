import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """Safe GroupNorm: groups = min(max_groups, channels) divisible by channels."""
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding (Transformer-style)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """Residual conv block with additive time conditioning."""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = _gn(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm2 = _gn(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t_emb):
        residual = self.residual_conv(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        t_emb = self.time_mlp(t_emb)
        h = h + t_emb[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class AttentionBlock(nn.Module):
    """Single-head self-attention block."""
    def __init__(self, channels):
        super().__init__()
        self.norm = _gn(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        residual = x
        B, C, H, W = x.shape

        x = self.norm(x)
        q = self.q(x).view(B, C, -1).permute(0, 2, 1)
        k = self.k(x).view(B, C, -1)
        v = self.v(x).view(B, C, -1).permute(0, 2, 1)

        attention = torch.bmm(q, k) * (C ** -0.5)
        attention = F.softmax(attention, dim=-1)
        h = torch.bmm(attention, v)

        h = h.permute(0, 2, 1).view(B, C, H, W)
        h = self.proj_out(h)

        return h + residual


class UNet(nn.Module):
    """U-Net with additive time conditioning and ConvTranspose2d upsampling.

    Used by the Flow Matching pipeline (Task 1).
    """
    def __init__(self, in_channels=3, model_channels=128, out_channels=3,
                 channel_mult=(1, 2, 2, 2), num_res_blocks=2,
                 dropout=0.1, time_emb_dim=256, attention_resolutions=(16,)):
        super().__init__()

        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.out_channels = out_channels

        # Encoder
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        channels = model_channels
        self.skip_connection_channels = [channels]

        for i, mult in enumerate(channel_mult):
            out_ch = mult * model_channels

            for _ in range(num_res_blocks):
                layers = []
                layers.append(ConvBlock(channels, out_ch, time_emb_dim, dropout))
                channels = out_ch

                resolution = 64 // (2 ** i)
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(channels))

                self.down_blocks.append(nn.ModuleList(layers))
                self.skip_connection_channels.append(channels)

            if i != len(channel_mult) - 1:
                self.downsample_blocks.append(
                    nn.Conv2d(channels, channels, 3, stride=2, padding=1)
                )
                self.skip_connection_channels.append(channels)

        # Bottleneck
        self.mid_block1 = ConvBlock(channels, channels, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(channels)
        self.mid_block2 = ConvBlock(channels, channels, time_emb_dim, dropout)

        # Decoder
        self.up_blocks = nn.ModuleList()

        for i, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = mult * model_channels

            for j in range(num_res_blocks + 1):
                skip_channels = self.skip_connection_channels.pop()
                in_channels_up = channels + skip_channels

                layers = []
                layers.append(ConvBlock(in_channels_up, out_ch, time_emb_dim, dropout))
                channels = out_ch

                resolution = 64 // (2 ** i)
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(channels))

                if i != 0 and j == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1))

                self.up_blocks.append(nn.ModuleList(layers))

        # Output
        self.output_norm = _gn(channels)
        self.output_conv = nn.Conv2d(channels, self.out_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)

        h = self.input_conv(x)
        skip_features = [h]

        down_block_idx = 0
        for i in range(len(self.channel_mult)):
            for _ in range(self.num_res_blocks):
                module_list = self.down_blocks[down_block_idx]
                down_block_idx += 1
                for layer in module_list:
                    if isinstance(layer, ConvBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
                skip_features.append(h)

            if i != len(self.channel_mult) - 1:
                h = self.downsample_blocks[i](h)
                skip_features.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for module_list in self.up_blocks:
            for layer in module_list:
                if isinstance(layer, ConvBlock):
                    skip = skip_features.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = layer(h, t_emb)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                else:
                    h = layer(h)

        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)

        return h
