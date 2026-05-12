"""
Task 2 Training Script – Front-to-back reconstruction.

Usage:
    python train/train_task2.py --config configs/task2_config.yaml

This script trains a conditional diffusion model to map front-view sprites
to back-view sprites. Added features:
  - CFG cond-dropout during training (cfg.training.cond_drop_prob)
  - EMA of model weights for evaluation/sampling (cfg.training.ema)
  - Optional weighted sampling across multiple data sources
  - Separate OOD validation set logging (cfg.data.ood_val_dir)

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F_nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from data_code.dataset import FrontToBackDataset
from models.diffusion import GaussianDiffusion
from models.unet import UNet
from utils.visualization import save_sample_grid


# ---------------------------------------------------------------------------
# Optional: tiny PatchGAN discriminator
# ---------------------------------------------------------------------------


class PatchDiscriminator(nn.Module):
    """
    Simple 3-level PatchGAN discriminator.
    Input: (B, 3+3, H, W) – predicted back concatenated with front condition.
    """

    def __init__(self, in_channels: int = 6, ndf: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def disc_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
    real_loss = F_nn.binary_cross_entropy_with_logits(
        real_pred, torch.ones_like(real_pred)
    )
    fake_loss = F_nn.binary_cross_entropy_with_logits(
        fake_pred, torch.zeros_like(fake_pred)
    )
    return (real_loss + fake_loss) * 0.5


def gen_adv_loss(fake_pred: torch.Tensor) -> torch.Tensor:
    return F_nn.binary_cross_entropy_with_logits(
        fake_pred, torch.ones_like(fake_pred)
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_device(device_str: Optional[str]) -> torch.device:
    if device_str is None or device_str.lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        device = torch.device(device_str)
    except Exception:
        print(f"Unknown device '{device_str}', falling back to CPU.")
        return torch.device("cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")

    if device.type == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but not available; falling back to CPU.")
        return torch.device("cpu")

    return device


def build_model(cfg: dict, device: torch.device) -> GaussianDiffusion:
    unet = UNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        model_channels=cfg["model"]["model_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        n_res_blocks=cfg["model"]["n_res_blocks"],
        attn_resolutions=tuple(cfg["model"]["attn_resolutions"]),
        time_emb_dim=cfg["model"]["time_emb_dim"],
        cond_emb_dim=0,
        dropout=cfg["model"].get("dropout", 0.1),
        image_size=cfg["data"]["image_size"],
    ).to(device)

    diffusion = GaussianDiffusion(
        model=unet,
        timesteps=cfg["diffusion"]["timesteps"],
        schedule=cfg["diffusion"]["schedule"],
        loss_type=cfg["diffusion"].get("loss_type", "l2"),
    ).to(device)

    return diffusion


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------


class EMA:
    """Simple EMA for model parameters (keeps shadow copy on CPU).

    Usage:
        ema = EMA(model, decay=0.9999)
        ema.register()
        ... after each optimizer step: ema.update()
        ema.store(); ema.copy_to(); ...; ema.restore()
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().cpu().clone()

    def update(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                assert name in self.shadow
                new_avg = (1.0 - self.decay) * p.detach().cpu() + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def store(self):
        self.backup = {name: p.detach().cpu().clone() for name, p in self.model.named_parameters()}

    def copy_to(self):
        for name, p in self.model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].to(p.device))

    def restore(self):
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name].to(p.device))
        self.backup = {}

    def state_dict(self):
        return {"decay": self.decay, "shadow": {k: v.clone() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: dict):
        self.decay = float(state.get("decay", self.decay))
        shadow = state.get("shadow", {})
        for k, v in shadow.items():
            self.shadow[k] = v.clone()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_epoch(
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    optimizer_g: optim.Optimizer,
    device: torch.device,
    disc: nn.Module = None,
    optimizer_d: optim.Optimizer = None,
    lambda_adv: float = 0.01,
    scaler=None,
    fg_weight: float = 6.0,
    bg_weight: float = 0.5,
    color_weight: float = 1.0,
    cond_drop_prob: float = 0.0,
    ema: Optional[EMA] = None,
) -> dict:
    diffusion.train()
    if disc is not None:
        disc.train()

    metrics = {"loss_diff": 0.0, "loss_disc": 0.0, "loss_gen_adv": 0.0}
    n = 0

    for batch in tqdm(loader, leave=False, desc="  train"):
        cond = batch["condition"].to(device)
        target = batch["target"].to(device)
        target_alpha = batch["target_alpha"].to(device)
        B = target.shape[0]
        n += 1

        t = torch.randint(0, diffusion.timesteps, (B,), device=device)

        cond_fg = (cond.abs().sum(dim=1, keepdim=True) > 0.1).float()

        optimizer_g.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss_diff = diffusion.p_losses(
                    target, t, cond_image=cond,
                    fg_mask=target_alpha, cond_fg_mask=cond_fg,
                    fg_weight=fg_weight, bg_weight=bg_weight,
                    color_weight=color_weight,
                    cond_drop_prob=cond_drop_prob,
                )
        else:
            loss_diff = diffusion.p_losses(
                target, t, cond_image=cond,
                fg_mask=target_alpha, cond_fg_mask=cond_fg,
                fg_weight=fg_weight, bg_weight=bg_weight,
                color_weight=color_weight,
                cond_drop_prob=cond_drop_prob,
            )

        loss_g = loss_diff

        if disc is not None:
            with torch.no_grad():
                fake_back = diffusion.ddim_sample(
                    shape=(B, 3, target.shape[2], target.shape[3]),
                    device=device,
                    ddim_steps=10,
                    cond_image=cond,
                )
            fake_input = torch.cat([fake_back, cond], dim=1)
            fake_pred = disc(fake_input)
            loss_gen_adv = gen_adv_loss(fake_pred)
            loss_g = loss_g + lambda_adv * loss_gen_adv
            metrics["loss_gen_adv"] += loss_gen_adv.item()

        if scaler is not None:
            scaler.scale(loss_g).backward()
            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            scaler.step(optimizer_g)
            scaler.update()
        else:
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer_g.step()

        if ema is not None:
            ema.update()

        metrics["loss_diff"] += loss_diff.item()

        if disc is not None and optimizer_d is not None:
            optimizer_d.zero_grad()
            real_input = torch.cat([target, cond], dim=1)
            real_pred = disc(real_input)
            fake_input_d = torch.cat([fake_back.detach(), cond], dim=1)
            fake_pred_d = disc(fake_input_d)
            loss_d = disc_loss(real_pred, fake_pred_d)
            loss_d.backward()
            optimizer_d.step()
            metrics["loss_disc"] += loss_d.item()

    for k in metrics:
        metrics[k] /= max(n, 1)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task 2 – front-to-back reconstruction")
    parser.add_argument("--config", default="configs/task2_config.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _resolve_device(cfg["training"].get("device"))
    print(f"Using device: {device}")

    # Datasets
    train_ds = FrontToBackDataset(
        data_source=cfg["data"]["train_dir"],
        image_size=cfg["data"]["image_size"],
        augment=True,
        occlusion_p=cfg["data"].get("occlusion_p", 0.3),
        occlusion_intensity=cfg["data"].get("occlusion_intensity", 0.5),
        occlusion_fill=cfg["data"].get("occlusion_fill", "zero"),
        source_weights=cfg["data"].get("source_weights", None),
    )

    val_ds = FrontToBackDataset(
        data_source=cfg["data"].get("val_dir", cfg["data"]["train_dir"]),
        image_size=cfg["data"]["image_size"],
        augment=False,
    )

    ood_val_ds = None
    if cfg["data"].get("ood_val_dir"):
        ood_val_ds = FrontToBackDataset(
            data_source=cfg["data"].get("ood_val_dir"),
            image_size=cfg["data"]["image_size"],
            augment=False,
        )

    # Loader kwargs
    loader_kwargs = dict(
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )

    # Optionally use a weighted sampler built from dataset source weights
    if cfg["training"].get("use_weighted_sampler", False):
        sampler = train_ds.make_weighted_sampler(num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)

    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    ood_val_loader = None if ood_val_ds is None else DataLoader(ood_val_ds, shuffle=False, **loader_kwargs)

    # Models
    diffusion = build_model(cfg, device)

    # Quick channel consistency check
    sample = train_ds[0]
    cond_ch = int(sample["condition"].shape[0])
    target_ch = int(sample["target"].shape[0])
    expected_in = cond_ch + target_ch
    expected_out = target_ch
    if cfg["model"]["in_channels"] != expected_in or cfg["model"]["out_channels"] != expected_out:
        raise ValueError(
            "Model/data channel mismatch: "
            f"config in/out=({cfg['model']['in_channels']}, {cfg['model']['out_channels']}), "
            f"but dataset implies ({expected_in}, {expected_out}). "
            "Please update config or dataset channel settings."
        )

    optimizer_g = optim.AdamW(
        diffusion.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"].get("lr_min", 1e-6),
    )

    disc = None
    optimizer_d = None
    if cfg["training"].get("use_discriminator", False):
        disc = PatchDiscriminator().to(device)
        optimizer_d = optim.AdamW(disc.parameters(), lr=cfg["training"].get("lr_disc"))

    use_amp = cfg["training"].get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # EMA (exponential moving average)
    ema = None
    if cfg["training"].get("ema", {}).get("enabled", False):
        ema_cfg = cfg["training"].get("ema", {})
        ema = EMA(diffusion.model, decay=ema_cfg.get("decay", 0.9999))
        ema.register()

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        diffusion.load_state_dict(ckpt["diffusion"])
        optimizer_g.load_state_dict(ckpt["optimizer_g"])
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    out_dir = cfg["training"]["out_dir"]
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(out_dir, "logs"))

    fg_weight = cfg["training"].get("fg_weight", 6.0)
    bg_weight = cfg["training"].get("bg_weight", 0.5)
    color_weight = cfg["training"].get("color_weight", 1.0)

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        metrics = train_epoch(
            diffusion, train_loader, optimizer_g, device,
            disc=disc, optimizer_d=optimizer_d,
            lambda_adv=cfg["training"].get("lambda_adv", 0.01),
            scaler=scaler,
            fg_weight=fg_weight,
            bg_weight=bg_weight,
            color_weight=color_weight,
            cond_drop_prob=cfg["training"].get("cond_drop_prob", 0.0),
            ema=ema,
        )
        scheduler.step()

        for k, v in metrics.items():
            if v > 0:
                writer.add_scalar(f"train/{k}", v, epoch)
        print(
            f"Epoch {epoch:04d}  diff={metrics['loss_diff']:.5f}"
            f"  disc={metrics['loss_disc']:.5f}"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Validation
        if (epoch + 1) % cfg["training"].get("val_every", 5) == 0:
            # Optionally evaluate with EMA weights
            if ema is not None:
                ema.store()
                ema.copy_to()

            diffusion.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    cond = batch["condition"].to(device)
                    target = batch["target"].to(device)
                    target_alpha = batch["target_alpha"].to(device)
                    cond_fg = (cond.abs().sum(dim=1, keepdim=True) > 0.1).float()
                    B = target.shape[0]
                    t = torch.randint(0, diffusion.timesteps, (B,), device=device)
                    val_loss += diffusion.p_losses(
                        target, t, cond_image=cond,
                        fg_mask=target_alpha, cond_fg_mask=cond_fg,
                        fg_weight=fg_weight, bg_weight=bg_weight,
                        color_weight=color_weight,
                    ).item()
            val_loss /= max(len(val_loader), 1)
            writer.add_scalar("val/loss_diff", val_loss, epoch)
            print(f"  val_loss={val_loss:.5f}")

            # OOD validation (if provided)
            if ood_val_loader is not None:
                ood_loss = 0.0
                with torch.no_grad():
                    for batch in ood_val_loader:
                        cond = batch["condition"].to(device)
                        target = batch["target"].to(device)
                        target_alpha = batch["target_alpha"].to(device)
                        cond_fg = (cond.abs().sum(dim=1, keepdim=True) > 0.1).float()
                        B = target.shape[0]
                        t = torch.randint(0, diffusion.timesteps, (B,), device=device)
                        ood_loss += diffusion.p_losses(
                            target, t, cond_image=cond,
                            fg_mask=target_alpha, cond_fg_mask=cond_fg,
                            fg_weight=fg_weight, bg_weight=bg_weight,
                            color_weight=color_weight,
                        ).item()
                ood_loss /= max(len(ood_val_loader), 1)
                writer.add_scalar("val/ood_loss_diff", ood_loss, epoch)
                print(f"  ood_val_loss={ood_loss:.5f}")

            if ema is not None:
                ema.restore()

        # Save checkpoint
        if (epoch + 1) % cfg["training"].get("save_every", 10) == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch:04d}.pth")
            save_dict = {
                "epoch": epoch,
                "diffusion": diffusion.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
            }
            if disc is not None:
                save_dict["disc"] = disc.state_dict()
                save_dict["optimizer_d"] = optimizer_d.state_dict()
            if ema is not None:
                save_dict["ema"] = ema.state_dict()
            torch.save(save_dict, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Visual samples
        if (epoch + 1) % cfg["training"].get("sample_every", 10) == 0:
            # Use EMA for visuals if available
            if ema is not None:
                ema.store()
                ema.copy_to()

            diffusion.eval()
            try:
                sample_batch = next(iter(val_loader))
                cond_vis = sample_batch["condition"][:4].to(device)
                with torch.no_grad():
                    gen_back = diffusion.ddim_sample(
                        shape=(min(4, cond_vis.shape[0]), 3,
                               cfg["data"]["image_size"], cfg["data"]["image_size"]),
                        device=device,
                        ddim_steps=cfg["diffusion"].get("ddim_steps", 50),
                        cond_image=cond_vis,
                    )
                vis = torch.cat([cond_vis, gen_back], dim=0)
                grid_path = os.path.join(sample_dir, f"sample_epoch{epoch:04d}.png")
                save_sample_grid(vis, grid_path, nrow=4)
                print(f"  Saved samples: {grid_path}")
            except Exception as e:
                print(f"  Sample generation failed: {e}")
            finally:
                if ema is not None:
                    ema.restore()

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
