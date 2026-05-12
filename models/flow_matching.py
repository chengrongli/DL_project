"""Flow Matching utilities for pixel-character generation.

Supports both:
  - Task 1: Unconditional front+back generation (4-channel RGBA)
  - Task 2: Image-conditioned front→back generation (3-channel RGB with cond_image)
"""

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
        cond_image: torch.Tensor | None = None,
        cond_fg_mask: torch.Tensor | None = None,
        background_weight: float = 1.0,
        foreground_weight: float = 1.0,
        alpha_weight: float = 1.0,
        color_weight: float = 0.0,
        return_components: bool = False,
    ):
        b = x0.shape[0]
        t = torch.rand(b, device=x0.device)
        z = torch.randn_like(x0)

        t_view = t.view(b, 1, 1, 1)
        x_t = (1.0 - t_view) * x0 + t_view * z
        target_v = z - x0

        # Concatenate condition image if provided (Task 2)
        model_input = x_t
        if cond_image is not None:
            model_input = torch.cat([x_t, cond_image], dim=1)

        pred_v = self.model(model_input, t * self.time_scale)
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

        # Color consistency loss (Task 2: front vs back foreground colors)
        if color_weight > 0.0 and cond_image is not None and fg_mask is not None:
            # Estimate x0 from velocity prediction: x0 = x_t - t * v_pred
            x0_pred = x_t - t_view * pred_v
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            front_mask = cond_fg_mask.to(x0) if cond_fg_mask is not None else fg_mask

            front_colors = (cond_image * front_mask).sum(dim=(2, 3)) / front_mask.sum(dim=(2, 3)).clamp(min=1)
            back_colors = (x0_pred * fg_mask[:, :1]).sum(dim=(2, 3)) / fg_mask[:, :1].sum(dim=(2, 3)).clamp(min=1)

            color_loss = F.mse_loss(back_colors, front_colors.detach())
            loss = loss + color_weight * color_loss

        if not return_components:
            return loss

        components = {
            "loss_total": loss.detach(),
            "loss_mse_raw": sq.mean().detach(),
        }
        if fg1 is not None:
            rgb = sq[:, : min(3, sq.shape[1])]
            fg_rgb = (rgb * fg1).sum() / (fg1.sum().clamp_min(1e-6) * rgb.shape[1])
            bg_rgb = (rgb * (1 - fg1)).sum() / ((1 - fg1).sum().clamp_min(1e-6) * rgb.shape[1])
            components["loss_fg_rgb"] = fg_rgb.detach()
            components["loss_bg_rgb"] = bg_rgb.detach()
        return loss, components

    @torch.no_grad()
    def sample(
        self,
        sample_shape: tuple[int, int, int, int],
        steps: int = 50,
        cond_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = next(self.model.parameters()).device
        x = torch.randn(sample_shape, device=device)

        # Integrate reverse ODE from t=1 -> 0 with explicit Euler.
        for i in range(steps, 0, -1):
            t_cur = torch.full((sample_shape[0],), i / steps, device=device)
            model_input = x
            if cond_image is not None:
                model_input = torch.cat([x, cond_image], dim=1)
            v = self.model(model_input, t_cur * self.time_scale)
            t_next = (i - 1) / steps
            dt = t_next - (i / steps)
            x = x + dt * v

        return x.clamp(-1.0, 1.0)
