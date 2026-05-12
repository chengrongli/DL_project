"""
DDPM / DDIM diffusion framework.

Implements:
  - Linear and cosine beta schedules.
  - Forward (noising) process: q(x_t | x_0).
  - Reverse (denoising) process: p_θ(x_{t-1} | x_t) via the learned U-Net.
  - DDPM ancestral sampling.
  - DDIM deterministic sampling (faster, fewer steps).
  - Classifier-free guidance (CFG) at inference.

References:
  Ho et al., "Denoising Diffusion Probabilistic Models" (2020).
  Song et al., "Denoising Diffusion Implicit Models" (2021).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Beta schedules
# ---------------------------------------------------------------------------

def linear_beta_schedule(timesteps: int,
                          beta_start: float = 1e-4,
                          beta_end: float = 2e-2) -> torch.Tensor:
    """Linearly spaced betas from beta_start to beta_end."""
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 8e-3) -> torch.Tensor:
    """
    Cosine beta schedule (Nichol & Dhariwal, 2021).
    Produces smoother noise progression, especially near t=0.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0.0001, 0.9999)


# ---------------------------------------------------------------------------
# Gaussian diffusion
# ---------------------------------------------------------------------------

class GaussianDiffusion(nn.Module):
    """
    Wraps a denoising model (U-Net) in the DDPM/DDIM framework.

    Args:
        model:         The denoising U-Net.
        timesteps:     Total number of diffusion steps T.
        schedule:      Beta schedule type ("linear" or "cosine").
        loss_type:     "l1" or "l2" (pixel-space reconstruction loss).
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        schedule: str = "cosine",
        loss_type: str = "l2",
    ) -> None:
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type

        # Build schedule
        if schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register all schedule tensors as buffers (moved with .to(device))
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod",
                             alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             (1 - alphas_cumprod).sqrt())
        self.register_buffer("log_one_minus_alphas_cumprod",
                             (1 - alphas_cumprod).log())
        self.register_buffer("sqrt_recip_alphas",
                             (1.0 / alphas).sqrt())
        # Posterior variance
        posterior_variance = (
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            posterior_variance.clamp(min=1e-20).log(),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * alphas_cumprod_prev.sqrt() / (1 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1 - alphas_cumprod_prev) * alphas.sqrt() / (1 - alphas_cumprod),
        )

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def _extract(self, a: torch.Tensor, t: torch.Tensor,
                 x_shape: torch.Size) -> torch.Tensor:
        """Gather schedule values at timestep t and reshape for broadcasting."""
        b = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(b, *([1] * (len(x_shape) - 1)))

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample noisy x_t given x_0 and timestep t.
        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 − ᾱ_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_acp = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_1macp = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_acp * x_start + sqrt_1macp * noise

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        cond_emb: Optional[torch.Tensor] = None,
        cond_image: Optional[torch.Tensor] = None,
        fg_mask: Optional[torch.Tensor] = None,
        cond_fg_mask: Optional[torch.Tensor] = None,
        fg_weight: float = 6.0,
        bg_weight: float = 0.5,
        color_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute the DDPM denoising loss with optional foreground weighting
        and color consistency regularization.

        Args:
            fg_mask:       (B, 1, H, W) foreground mask for target, 1=foreground.
            cond_fg_mask:  (B, 1, H, W) foreground mask for condition image.
            fg_weight:     Foreground pixel weight multiplier.
            bg_weight:     Background pixel weight multiplier.
            color_weight:  Weight for front-back color consistency loss.

        Returns:
            Scalar loss tensor.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise)

        model_input = x_t
        if cond_image is not None:
            model_input = torch.cat([x_t, cond_image], dim=1)

        noise_pred = self.model(model_input, t, cond_emb)

        # --- Per-pixel noise prediction loss ---
        if self.loss_type == "l1":
            per_pixel = (noise_pred - noise).abs()
        else:
            per_pixel = (noise_pred - noise) ** 2

        # Foreground weighting
        if fg_mask is not None:
            fg_mask = fg_mask.to(x_start)
            weights = bg_weight + (fg_weight - bg_weight) * fg_mask
            loss = (per_pixel * weights).sum() / weights.sum().clamp(min=1e-6)
        else:
            loss = per_pixel.mean()

        # --- Color consistency loss ---
        # Estimate x0 from noise prediction to compute image-space color loss
        if color_weight > 0.0 and cond_image is not None and fg_mask is not None:
            acp_t = self._extract(self.alphas_cumprod, t, x_t.shape)
            x0_pred = (x_t - (1 - acp_t).sqrt() * noise_pred) / acp_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # Per-channel foreground color mean matching
            # front colors (from condition) vs predicted back colors
            if cond_fg_mask is not None:
                front_mask = cond_fg_mask.to(x_start)
            else:
                front_mask = fg_mask

            # Compute mean color of foreground pixels per channel
            front_colors = (cond_image * front_mask).sum(dim=(2, 3)) / front_mask.sum(dim=(2, 3)).clamp(min=1)
            back_colors = (x0_pred * fg_mask).sum(dim=(2, 3)) / fg_mask.sum(dim=(2, 3)).clamp(min=1)

            color_loss = F.mse_loss(back_colors, front_colors.detach())
            loss = loss + color_weight * color_loss

        return loss

    # ------------------------------------------------------------------
    # DDPM reverse step
    # ------------------------------------------------------------------

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None,
        cond_image: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        uncond_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior mean and variance for p_θ(x_{t-1} | x_t).

        Supports classifier-free guidance if cfg_scale > 1.
        """
        model_input = x_t
        if cond_image is not None:
            model_input = torch.cat([x_t, cond_image], dim=1)

        noise_pred = self.model(model_input, t, cond_emb)

        # Classifier-free guidance
        if cfg_scale != 1.0 and uncond_emb is not None:
            model_input_unc = x_t
            if cond_image is not None:
                model_input_unc = torch.cat([x_t, cond_image], dim=1)
            noise_pred_unc = self.model(model_input_unc, t, uncond_emb)
            noise_pred = noise_pred_unc + cfg_scale * (noise_pred - noise_pred_unc)

        # Predict x_0 from noise prediction
        sqrt_recip_acp = self._extract(
            (1.0 / self.alphas_cumprod).sqrt(), t, x_t.shape
        )
        sqrt_1m_recip_acp = self._extract(
            ((1 - self.alphas_cumprod) / self.alphas_cumprod).sqrt(), t, x_t.shape
        )
        x0_pred = sqrt_recip_acp * x_t - sqrt_1m_recip_acp * noise_pred
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        # Posterior mean
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x0_pred + coef2 * x_t

        var = self._extract(self.posterior_variance, t, x_t.shape)
        log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return mean, var, log_var

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None,
        cond_image: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        uncond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single DDPM reverse step: sample x_{t-1} ~ p_θ(x_{t-1} | x_t)."""
        mean, _, log_var = self.p_mean_variance(
            x_t, t, cond_emb, cond_image, cfg_scale, uncond_emb
        )
        noise = torch.randn_like(x_t)
        # Do not add noise at t=0
        nonzero = (t > 0).float().reshape(-1, *([1] * (x_t.ndim - 1)))
        return mean + nonzero * (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def ddpm_sample(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        cond_emb: Optional[torch.Tensor] = None,
        cond_image: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        uncond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full DDPM ancestral sampling loop.

        Args:
            shape:      (B, C, H, W) of the output tensor.
            device:     Target device.

        Returns:
            Denoised tensor in [−1, 1].
        """
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, cond_emb, cond_image, cfg_scale, uncond_emb)
        return x

    # ------------------------------------------------------------------
    # DDIM sampling (faster, deterministic)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        ddim_steps: int = 50,
        eta: float = 0.0,
        cond_emb: Optional[torch.Tensor] = None,
        cond_image: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        uncond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        DDIM sampling with `ddim_steps` NFE (much faster than DDPM).

        eta=0 → fully deterministic; eta=1 → equivalent to DDPM variance.

        Args:
            shape:       (B, C, H, W).
            ddim_steps:  Number of denoising steps (e.g. 50 or 100).
            eta:         Stochasticity coefficient.

        Returns:
            Denoised tensor in [−1, 1].
        """
        T = self.timesteps
        # Uniformly spaced timestep subsequence
        step_size = T // ddim_steps
        timestep_seq = list(reversed(range(0, T, step_size)))[:ddim_steps]

        x = torch.randn(shape, device=device)

        for i, t_val in enumerate(timestep_seq):
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            t_prev_val = timestep_seq[i + 1] if i + 1 < len(timestep_seq) else -1

            # Predict noise
            model_input = x
            if cond_image is not None:
                model_input = torch.cat([x, cond_image], dim=1)

            noise_pred = self.model(model_input, t, cond_emb)

            if cfg_scale != 1.0 and uncond_emb is not None:
                model_input_unc = x
                if cond_image is not None:
                    model_input_unc = torch.cat([x, cond_image], dim=1)
                noise_pred_unc = self.model(model_input_unc, t, uncond_emb)
                noise_pred = noise_pred_unc + cfg_scale * (noise_pred - noise_pred_unc)

            acp_t = self._extract(self.alphas_cumprod, t, x.shape)
            if t_prev_val >= 0:
                t_prev = torch.full((shape[0],), t_prev_val,
                                    device=device, dtype=torch.long)
                acp_prev = self._extract(self.alphas_cumprod, t_prev, x.shape)
            else:
                acp_prev = torch.ones_like(acp_t)

            # Predict x0
            x0_pred = (x - (1 - acp_t).sqrt() * noise_pred) / acp_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # Direction pointing to x_t
            sigma = eta * ((1 - acp_prev) / (1 - acp_t) * (1 - acp_t / acp_prev)).sqrt()
            dir_xt = (1 - acp_prev - sigma ** 2).sqrt() * noise_pred

            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = acp_prev.sqrt() * x0_pred + dir_xt + sigma * noise

        return x
