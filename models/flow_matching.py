"""Flow Matching utilities for pixel-character generation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatching(nn.Module):
    """Conditional flow matching on linear interpolation path.

    Training path:
        x_t = (1 - t) * x0 + t * z,   t ~ U(0, 1), z ~ N(0, I)
        target velocity u_t = z - x0

    Model predicts v_theta(x_t, t), optimized with MSE to target velocity.
    """

    def __init__(self, model: nn.Module, time_scale: float = 999.0) -> None:
        super().__init__()
        self.model = model
        self.time_scale = float(time_scale)

    def compute_loss(
        self,
        x0: torch.Tensor,
        *,
        fg_mask: torch.Tensor | None = None,
        background_weight: float = 1.0,
        foreground_weight: float = 1.0,
        alpha_weight: float = 1.0,
        return_components: bool = False,
        attr_cond: torch.Tensor | None = None,
        attr_tokens: torch.Tensor | None = None,
        cfg_dropout: float = 0.0,
    ):
        b = x0.shape[0]
        t = torch.rand(b, device=x0.device)
        z = torch.randn_like(x0)

        t_view = t.view(b, 1, 1, 1)
        x_t = (1.0 - t_view) * x0 + t_view * z
        target_v = z - x0

        # CFG dropout: independently zero out both conditioning signals
        if cfg_dropout > 0.0 and self.training:
            drop_mask = torch.rand(b, device=x0.device) < cfg_dropout
            if attr_cond is not None:
                attr_cond = attr_cond.clone()
                attr_cond[drop_mask] = 0.0
            if attr_tokens is not None:
                attr_tokens = attr_tokens.clone()
                attr_tokens[drop_mask] = 0.0

        pred_v = self.model(x_t, t * self.time_scale, attr_cond=attr_cond, attr_tokens=attr_tokens)
        sq = (pred_v - target_v) ** 2

        weights = torch.ones_like(sq)
        fg1 = None

        if fg_mask is not None:
            fg_mask = fg_mask.to(x0)
            fg1 = fg_mask[:, :1]
            if fg_mask.shape[1] != x0.shape[1]:
                repeat_factor = (x0.shape[1] + fg_mask.shape[1] - 1) // fg_mask.shape[1]
                fg_mask = fg_mask.repeat(1, repeat_factor, 1, 1)[:, : x0.shape[1]]

            bg_w = max(float(background_weight), 0.05)
            fg_w = max(float(foreground_weight), 1.0)
            weights = bg_w + (1.0 - bg_w) * fg_mask
            weights = weights * (1.0 + (fg_w - 1.0) * fg_mask)

        if x0.shape[1] >= 4:
            weights[:, 3:4] = weights[:, 3:4] * max(float(alpha_weight), 0.0)

        loss = (sq * weights).sum() / weights.sum().clamp_min(1e-6)

        if not return_components:
            return loss

        components = {
            "loss_total": loss.detach(),
            "loss_mse_raw": sq.mean().detach(),
        }
        if fg1 is not None:
            rgb = sq[:, : min(3, sq.shape[1])]
            fg_rgb = (rgb * fg1).sum() / (fg1.sum().clamp_min(1e-6) * rgb.shape[1])
            bg_rgb = (rgb * (1.0 - fg1)).sum() / ((1.0 - fg1).sum().clamp_min(1e-6) * rgb.shape[1])
            components["loss_fg_rgb"] = fg_rgb.detach()
            components["loss_bg_rgb"] = bg_rgb.detach()
        return loss, components

    @torch.no_grad()
    def sample(
        self,
        sample_shape: tuple[int, int, int, int],
        steps: int = 50,
        attr_cond: torch.Tensor | None = None,
        attr_tokens: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        device = next(self.model.parameters()).device
        x = torch.randn(sample_shape, device=device)

        for i in range(steps, 0, -1):
            t_cur = torch.full((sample_shape[0],), i / steps, device=device)
            t_next = (i - 1) / steps
            dt = t_next - (i / steps)

            if (attr_cond is not None or attr_tokens is not None) and guidance_scale != 1.0:
                v_uncond = self.model(x, t_cur * self.time_scale, attr_cond=None, attr_tokens=None)
                v_cond = self.model(x, t_cur * self.time_scale, attr_cond=attr_cond, attr_tokens=attr_tokens)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = self.model(x, t_cur * self.time_scale, attr_cond=attr_cond, attr_tokens=attr_tokens)
            x = x + dt * v

        return x.clamp(-1.0, 1.0)
