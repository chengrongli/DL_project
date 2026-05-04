"""
Task 1 Training Script – Random front+back generation using the ddpm.py framework.

Usage:
    python train/train_task1.py --config configs/task1_config.yaml

This script reuses the diffusion building blocks defined in ddpm.py to train a
6-channel (front + back) sprite model on the LPC dataset. It keeps the
configuration layout from the previous implementation while switching the core
training loop to the shared DDPM infrastructure.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ddpm import DDPM, DiffusionConfig, UNet as DDPMUNet
from data_code.dataset import SpritePairDataset
from utils.visualization import save_sample_grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_device(device_str: str | None) -> torch.device:
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


def _paired_collate_fn(batch):
    tensors = [item["paired"] for item in batch]
    return torch.stack(tensors, dim=0)


def _adjust_attention_resolutions(cfg: dict) -> tuple[int, ...]:
    raw = cfg["model"].get("attn_resolutions", [])
    if not raw:
        return tuple()
    image_size = cfg["data"].get("image_size", 32)
    scale = image_size / 32.0
    if math.isclose(scale, 1.0):
        return tuple(int(r) for r in raw)
    adjusted = []
    for r in raw:
        mapped = max(1, int(round(r / scale)))
        adjusted.append(mapped)
    return tuple(adjusted)


def train_one_epoch(
    model: DDPM,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    ema_model: DDPM,
    ema_decay: float,
    global_step: int,
    writer: SummaryWriter | None = None,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, leave=False, desc="  train")

    for batch_idx, paired in enumerate(pbar):
        images = paired.to(device)
        optimizer.zero_grad()

        loss = model.compute_losses(images)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(ema_decay).add_(param.data, alpha=1.0 - ema_decay)

        total_loss += loss.item()
        current_step = global_step + batch_idx
        pbar.set_postfix({"loss": loss.item()})
        if writer is not None:
            writer.add_scalar("loss/train_iter", loss.item(), current_step)

    avg_loss = total_loss / max(1, len(loader))
    new_step = global_step + len(loader)
    return avg_loss, new_step


@torch.no_grad()
def evaluate(model: DDPM, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    for paired in loader:
        images = paired.to(device)
        loss = model.compute_losses(images)
        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def sample_and_save(
    ema_model: DDPM,
    epoch: int,
    sample_dir: str,
    sample_shape: tuple[int, int, int, int],
    nrow: int = 4,
    upscale: int = 4,
) -> str:
    ema_model.eval()
    samples = ema_model.sample(sample_shape, progress_bar=False)
    samples = samples * 2.0 - 1.0
    save_path = os.path.join(sample_dir, f"sample_epoch{epoch:04d}.png")
    save_sample_grid(samples, save_path, nrow=nrow, upscale=upscale)
    return save_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task 1 – front+back generation")
    parser.add_argument("--config", default="configs/task1_config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _resolve_device(cfg["training"].get("device"))
    print(f"Using device: {device}")

    # Datasets
    train_ds = SpritePairDataset(
        data_source=cfg["data"]["train_dir"],
        image_size=cfg["data"]["image_size"],
        augment=True,
    )

    val_loader = None
    val_dir = cfg["data"].get("val_dir")
    if val_dir:
        val_ds = SpritePairDataset(
            data_source=val_dir,
            image_size=cfg["data"]["image_size"],
            augment=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"].get("num_workers", 4),
            pin_memory=device.type == "cuda",
            collate_fn=_paired_collate_fn,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
        collate_fn=_paired_collate_fn,
    )

    diffusion_config = DiffusionConfig(
        timesteps=cfg["diffusion"]["timesteps"],
        device=device,
    )

    attention_res = _adjust_attention_resolutions(cfg)
    model = DDPMUNet(
        in_channels=cfg["model"]["in_channels"],
        model_channels=cfg["model"]["model_channels"],
        out_channels=cfg["model"]["out_channels"],
        channel_mult=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=cfg["model"]["n_res_blocks"],
        dropout=cfg["model"].get("dropout", 0.1),
        time_emb_dim=cfg["model"]["time_emb_dim"],
        attention_resolutions=attention_res if attention_res else (16,),
    ).to(device)

    ddpm_model = DDPM(diffusion_config, model).to(device)
    ema_model = DDPM(diffusion_config, copy.deepcopy(model).to(device))
    ema_model.load_state_dict(ddpm_model.state_dict())
    for param in ema_model.parameters():
        param.requires_grad_(False)

    optimizer = optim.AdamW(
        ddpm_model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"].get("lr_min", 1e-6),
    )

    ema_decay = cfg["training"].get("ema_decay", 0.9999)

    out_dir = cfg["training"]["out_dir"]
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(out_dir, "logs"))

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        state_key = "model_state_dict" if "model_state_dict" in ckpt else "diffusion"
        ddpm_model.load_state_dict(ckpt[state_key])
        if "ema_state_dict" in ckpt:
            ema_model.load_state_dict(ckpt["ema_state_dict"])
        else:
            ema_model.load_state_dict(ddpm_model.state_dict())
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = start_epoch * len(train_loader)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    epochs = cfg["training"]["epochs"]
    val_every = cfg["training"].get("val_every", 5)
    save_every = cfg["training"].get("save_every", 10)
    sample_every = cfg["training"].get("sample_every", 10)
    sample_count = cfg["training"].get("sample_count", 8)
    sample_nrow = cfg["training"].get("sample_nrow", max(1, int(round(sample_count ** 0.5))))
    sample_upscale = cfg["training"].get("sample_upscale", 4)

    print("Starting training...")
    for epoch in range(start_epoch, epochs):
        train_loss, global_step = train_one_epoch(
            ddpm_model,
            train_loader,
            optimizer,
            device,
            ema_model,
            ema_decay,
            global_step,
            writer,
        )
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        writer.add_scalar("loss/train_epoch", train_loss, epoch)
        writer.add_scalar("lr", lr, epoch)
        print(f"Epoch {epoch + 1:04d}  loss={train_loss:.5f}  lr={lr:.2e}")

        if val_loader is not None and (epoch + 1) % val_every == 0:
            val_loss = evaluate(ddpm_model, val_loader, device)
            writer.add_scalar("loss/val", val_loss, epoch)
            print(f"  val_loss={val_loss:.5f}")

        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ddpm_epoch{epoch + 1:04d}.pth")
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": ddpm_model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(checkpoint, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        if (epoch + 1) % sample_every == 0:
            sample_shape = (
                sample_count,
                cfg["model"]["out_channels"],
                cfg["data"]["image_size"],
                cfg["data"]["image_size"],
            )
            sample_path = sample_and_save(
                ema_model,
                epoch + 1,
                sample_dir,
                sample_shape,
                nrow=sample_nrow,
                upscale=sample_upscale,
            )
            print(f"  Saved samples: {sample_path}")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
