"""
Task 2 Training Script – Front-to-back reconstruction.

Usage:
    python train/train_task2.py --config configs/task2_config.yaml

The model is an image-to-image conditional diffusion U-Net that takes a
front-view sprite (3-channel) as conditioning and generates the corresponding
back-view sprite (3-channel).

The front image is channel-concatenated with the noisy back image before
entering the U-Net (so U-Net in_channels=6, out_channels=3).

Optional losses:
  - Primary: L2 noise prediction loss.
  - Perceptual / palette-consistency: added via an optional LPIPS penalty.
  - Adversarial fine-tuning: a small PatchGAN discriminator can be enabled
    via cfg["training"]["use_discriminator"] = true.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F_nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from data.dataset import FrontToBackDataset
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
            # C1
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # C2
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # C3
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # C4 – output patch map
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


def build_model(cfg: dict, device: torch.device):
    unet = UNet(
        in_channels=cfg["model"]["in_channels"],    # 6 = noisy_back(3) + front_cond(3)
        out_channels=cfg["model"]["out_channels"],  # 3 = back noise
        model_channels=cfg["model"]["model_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        n_res_blocks=cfg["model"]["n_res_blocks"],
        attn_resolutions=tuple(cfg["model"]["attn_resolutions"]),
        time_emb_dim=cfg["model"]["time_emb_dim"],
        cond_emb_dim=0,  # task 2 uses image conditioning, not attribute embedding
        dropout=cfg["model"].get("dropout", 0.1),
        image_size=cfg["data"]["image_size"],
    ).to(device)

    diffusion = GaussianDiffusion(
        model=unet,
        timesteps=cfg["diffusion"]["timesteps"],
        schedule=cfg["diffusion"]["schedule"],
        loss_type=cfg["diffusion"]["loss_type"],
    ).to(device)

    return diffusion


# ---------------------------------------------------------------------------
# Training
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
) -> dict:
    diffusion.train()
    if disc is not None:
        disc.train()

    metrics = {"loss_diff": 0.0, "loss_disc": 0.0, "loss_gen_adv": 0.0}
    n = 0

    for batch in tqdm(loader, leave=False, desc="  train"):
        cond = batch["condition"].to(device)   # front image (B, 3, H, W)
        target = batch["target"].to(device)    # back image  (B, 3, H, W)
        B = target.shape[0]
        n += 1

        t = torch.randint(0, diffusion.timesteps, (B,), device=device)

        # --- Generator / diffusion step ---
        optimizer_g.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss_diff = diffusion.p_losses(target, t, cond_image=cond)
        else:
            loss_diff = diffusion.p_losses(target, t, cond_image=cond)

        loss_g = loss_diff

        # Adversarial loss (optional, only during fine-tuning)
        if disc is not None:
            # Generate a denoised sample (expensive, use only a subset)
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

        metrics["loss_diff"] += loss_diff.item()

        # --- Discriminator step ---
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
    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Datasets
    train_ds = FrontToBackDataset(
        data_source=cfg["data"]["train_dir"],
        image_size=cfg["data"]["image_size"],
        augment=True,
        occlusion_p=cfg["data"].get("occlusion_p", 0.3),
    )
    val_ds = FrontToBackDataset(
        data_source=cfg["data"].get("val_dir", cfg["data"]["train_dir"]),
        image_size=cfg["data"]["image_size"],
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
    )

    # Models
    diffusion = build_model(cfg, device)
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
        optimizer_d = optim.AdamW(disc.parameters(), lr=cfg["training"]["lr_disc"])

    use_amp = cfg["training"].get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        diffusion.load_state_dict(ckpt["diffusion"])
        optimizer_g.load_state_dict(ckpt["optimizer_g"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    out_dir = cfg["training"]["out_dir"]
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(out_dir, "logs"))

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        metrics = train_epoch(
            diffusion, train_loader, optimizer_g, device,
            disc=disc, optimizer_d=optimizer_d,
            lambda_adv=cfg["training"].get("lambda_adv", 0.01),
            scaler=scaler,
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
            diffusion.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    cond = batch["condition"].to(device)
                    target = batch["target"].to(device)
                    B = target.shape[0]
                    t = torch.randint(0, diffusion.timesteps, (B,), device=device)
                    val_loss += diffusion.p_losses(target, t, cond_image=cond).item()
            val_loss /= len(val_loader)
            writer.add_scalar("val/loss_diff", val_loss, epoch)
            print(f"  val_loss={val_loss:.5f}")

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
            torch.save(save_dict, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Visual samples
        if (epoch + 1) % cfg["training"].get("sample_every", 10) == 0:
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
                # Save condition and generated side-by-side
                vis = torch.cat([cond_vis, gen_back], dim=0)
                grid_path = os.path.join(sample_dir, f"sample_epoch{epoch:04d}.png")
                save_sample_grid(vis, grid_path, nrow=4)
                print(f"  Saved samples: {grid_path}")
            except Exception as e:
                print(f"  Sample generation failed: {e}")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
