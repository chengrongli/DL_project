from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, levels: int) -> torch.Tensor:
        if levels <= 1:
            return x
        return torch.round(x * (levels - 1)) / (levels - 1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def ste_quantize(x: torch.Tensor, levels: int) -> torch.Tensor:
    return _STEQuantize.apply(x, levels)


class DifferentiablePixelization(nn.Module):
    def __init__(
        self,
        block_size: int = 8,
        color_levels: int = 16,
        use_ste_quant: bool = True,
    ) -> None:
        super().__init__()
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        self.block_size = block_size
        self.color_levels = color_levels
        self.use_ste_quant = use_ste_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), expected in [0, 1]
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B,C,H,W), got {x.shape}")

        b, c, h, w = x.shape
        ds_h = max(1, h // self.block_size)
        ds_w = max(1, w // self.block_size)

        small = F.interpolate(x, size=(ds_h, ds_w), mode="bilinear", align_corners=False)
        if self.color_levels > 1:
            if self.use_ste_quant:
                small = ste_quantize(small, self.color_levels)
            else:
                small = torch.round(small * (self.color_levels - 1)) / (self.color_levels - 1)
        pixel = F.interpolate(small, size=(h, w), mode="nearest")
        return pixel.clamp(0.0, 1.0)
