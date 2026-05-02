"""
Task 1 Training Script – Random front+back generation.

Usage:
    python train/train_task1.py --config configs/task1_config.yaml

The model is a conditional diffusion U-Net that jointly generates
a front+back 64×64 sprite pair (6-channel output) conditioned on
optional attribute embeddings with classifier-free guidance.

Training procedure:
  1. Load paired front/back dataset.
  2. For each batch, concatenate front and back into a 6-channel tensor.
  3. Sample random timesteps and add noise.
  4. Predict noise with U-Net (conditioned on optional attribute embedding).
  5. Compute L2 (or L1) loss on noise prediction.
  6. Log to TensorBoard; save checkpoints periodically.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from data.dataset import SpritePairDataset
from models.diffusion import GaussianDiffusion
from models.embeddings import build_lpc_attr_embedding
from models.unet import UNet
from utils.visualization import save_sample_grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device):
    """Build U-Net + diffusion wrapper from config dict."""
    unet = UNet(
        in_channels=cfg["model"]["in_channels"],      # 6 for task1
        out_channels=cfg["model"]["out_channels"],    # 6 for task1
        model_channels=cfg["model"]["model_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        n_res_blocks=cfg["model"]["n_res_blocks"],
        attn_resolutions=tuple(cfg["model"]["attn_resolutions"]),
        time_emb_dim=cfg["model"]["time_emb_dim"],
        cond_emb_dim=cfg["model"].get("cond_emb_dim", 0),
        dropout=cfg["model"].get("dropout", 0.1),
        image_size=cfg["data"]["image_size"],
    ).to(device)

    diffusion = GaussianDiffusion(
        model=unet,
        timesteps=cfg["diffusion"]["timesteps"],
        schedule=cfg["diffusion"]["schedule"],
        loss_type=cfg["diffusion"]["loss_type"],
    ).to(device)

    attr_emb = None
    if cfg["model"].get("cond_emb_dim", 0) > 0:
        attr_emb = build_lpc_attr_embedding(
            emb_dim=cfg["model"]["cond_emb_dim"],
            p_uncond=cfg["training"].get("p_uncond", 0.1),
        ).to(device)

    return diffusion, attr_emb


def train_epoch(
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    attr_emb=None,
    scaler=None,
) -> float:
    """Run one training epoch and return mean loss."""
    diffusion.train()
    total_loss = 0.0

    for batch in tqdm(loader, leave=False, desc="  train"):
        paired = batch["paired"].to(device)  # (B, 6, H, W)
        B = paired.shape[0]

        # Sample random timesteps
        t = torch.randint(0, diffusion.timesteps, (B,), device=device)

        cond_emb = None
        if attr_emb is not None:
            # In a real run, attrs would come from the batch metadata.
            # Here we pass empty dicts → attribute embedding falls back to null.
            cond_emb = attr_emb({}, force_null=False) if hasattr(attr_emb, "__call__") else None

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = diffusion.p_losses(paired, t, cond_emb=cond_emb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = diffusion.p_losses(paired, t, cond_emb=cond_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task 1 – front+back generation")
    parser.add_argument("--config", default="configs/task1_config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Datasets
    train_ds = SpritePairDataset(
        data_source=cfg["data"]["train_dir"],
        image_size=cfg["data"]["image_size"],
        augment=True,
    )
    val_ds = SpritePairDataset(
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

    # Model
    diffusion, attr_emb = build_model(cfg, device)
    all_params = list(diffusion.parameters())
    if attr_emb is not None:
        all_params += list(attr_emb.parameters())

    optimizer = optim.AdamW(
        all_params,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )

    # LR scheduler: cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"].get("lr_min", 1e-6),
    )

    # Mixed precision
    use_amp = cfg["training"].get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Resume
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        diffusion.load_state_dict(ckpt["diffusion"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Output dirs
    out_dir = cfg["training"]["out_dir"]
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(out_dir, "logs"))

    # Training loop
    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        train_loss = train_epoch(
            diffusion, train_loader, optimizer, device, attr_emb, scaler
        )
        scheduler.step()

        writer.add_scalar("loss/train", train_loss, epoch)
        print(f"Epoch {epoch:04d}  loss={train_loss:.5f}  lr={scheduler.get_last_lr()[0]:.2e}")

        # Validation loss
        if (epoch + 1) % cfg["training"].get("val_every", 5) == 0:
            diffusion.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    paired = batch["paired"].to(device)
                    B = paired.shape[0]
                    t = torch.randint(0, diffusion.timesteps, (B,), device=device)
                    loss = diffusion.p_losses(paired, t)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            writer.add_scalar("loss/val", val_loss, epoch)
            print(f"  val_loss={val_loss:.5f}")

        # Save checkpoint
        if (epoch + 1) % cfg["training"].get("save_every", 10) == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch:04d}.pth")
            save_dict: dict = {
                "epoch": epoch,
                "diffusion": diffusion.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if attr_emb is not None:
                save_dict["attr_emb"] = attr_emb.state_dict()
            torch.save(save_dict, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Generate sample images
        if (epoch + 1) % cfg["training"].get("sample_every", 10) == 0:
            diffusion.eval()
            with torch.no_grad():
                sample_shape = (
                    4,
                    cfg["model"]["out_channels"],
                    cfg["data"]["image_size"],
                    cfg["data"]["image_size"],
                )
                ddim_steps = cfg["diffusion"].get("ddim_steps", 50)
                samples = diffusion.ddim_sample(
                    shape=sample_shape,
                    device=device,
                    ddim_steps=ddim_steps,
                )
            grid_path = os.path.join(sample_dir, f"sample_epoch{epoch:04d}.png")
            save_sample_grid(samples, grid_path, nrow=2)
            print(f"  Saved samples: {grid_path}")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
