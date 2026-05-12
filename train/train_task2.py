"""
Task 2 Training Script – Front-to-back reconstruction (Flow Matching).

Usage:
    python train/train_task2.py --config configs/task2_config.yaml

Uses Flow Matching (linear interpolation path) with a conditional U-Net.
The front image is channel-concatenated with the noisy back image before
entering the U-Net (so U-Net in_channels=6, out_channels=3).

Losses:
  - Primary: MSE on predicted velocity vs target velocity (z - x0).
  - Foreground weighting: fg pixels weighted more heavily.
  - Color consistency: front vs back foreground channel mean matching.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from data_code.dataset import FrontToBackDataset
from models.flow_matching import FlowMatching
from models.unet import UNet
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


def build_model(cfg: dict, device: torch.device) -> FlowMatching:
    unet = UNet(
        in_channels=cfg["model"]["in_channels"],    # 6 = noisy_back(3) + front_cond(3)
        out_channels=cfg["model"]["out_channels"],  # 3 = back channels
        model_channels=cfg["model"]["model_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        n_res_blocks=cfg["model"]["n_res_blocks"],
        attn_resolutions=tuple(cfg["model"]["attn_resolutions"]),
        time_emb_dim=cfg["model"]["time_emb_dim"],
        cond_emb_dim=0,  # task 2 uses image conditioning, not attribute embedding
        dropout=cfg["model"].get("dropout", 0.1),
        image_size=cfg["data"]["image_size"],
    ).to(device)

    fm = FlowMatching(
        model=unet,
        time_scale=cfg["flow"].get("time_scale", 999.0),
    ).to(device)

    return fm


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    fm: FlowMatching,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    ema_fm: FlowMatching | None = None,
    ema_decay: float = 0.999,
    scaler=None,
    fg_weight: float = 6.0,
    bg_weight: float = 0.5,
    color_weight: float = 1.0,
) -> dict:
    fm.train()

    metrics = {"loss_total": 0.0, "loss_mse_raw": 0.0, "loss_fg_rgb": 0.0, "loss_bg_rgb": 0.0}
    n = 0

    for batch in tqdm(loader, leave=False, desc="  train"):
        cond = batch["condition"].to(device)   # front image (B, 3, H, W)
        target = batch["target"].to(device)    # back image  (B, 3, H, W)
        target_alpha = batch["target_alpha"].to(device)  # (B, 1, H, W)
        B = target.shape[0]
        n += 1

        # Build front foreground mask from condition (non-black = foreground)
        cond_fg = (cond.abs().sum(dim=1, keepdim=True) > 0.1).float()

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, components = fm.compute_loss(
                    target,
                    fg_mask=target_alpha,
                    cond_image=cond,
                    cond_fg_mask=cond_fg,
                    background_weight=bg_weight,
                    foreground_weight=fg_weight,
                    color_weight=color_weight,
                    return_components=True,
                )
        else:
            loss, components = fm.compute_loss(
                target,
                fg_mask=target_alpha,
                cond_image=cond,
                cond_fg_mask=cond_fg,
                background_weight=bg_weight,
                foreground_weight=fg_weight,
                color_weight=color_weight,
                return_components=True,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), 1.0)
            optimizer.step()

        # EMA update
        if ema_fm is not None:
            with torch.no_grad():
                for ep, p in zip(ema_fm.parameters(), fm.parameters()):
                    ep.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

        for k in metrics:
            if k in components:
                metrics[k] += components[k].item()

    for k in metrics:
        metrics[k] /= max(n, 1)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task 2 – front-to-back reconstruction (Flow Matching)")
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
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )

    # Model
    fm = build_model(cfg, device)

    # Channel consistency check
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

    optimizer = optim.AdamW(
        fm.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"].get("lr_min", 1e-6),
    )

    # EMA model
    ema_decay = cfg["training"].get("ema_decay", 0.999)
    ema_fm = copy.deepcopy(fm)
    ema_fm.load_state_dict(fm.state_dict())
    for p in ema_fm.parameters():
        p.requires_grad_(False)

    use_amp = cfg["training"].get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    fg_weight = cfg["training"].get("fg_weight", 6.0)
    bg_weight = cfg["training"].get("bg_weight", 0.5)
    color_weight = cfg["training"].get("color_weight", 1.0)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        fm.load_state_dict(ckpt["flow_matching"])
        if "ema_flow_matching" in ckpt:
            ema_fm.load_state_dict(ckpt["ema_flow_matching"])
        else:
            ema_fm.load_state_dict(fm.state_dict())
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    out_dir = cfg["training"]["out_dir"]
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(out_dir, "logs"))

    sample_steps = cfg["flow"].get("sample_steps", 50)
    img_size = cfg["data"]["image_size"]

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        metrics = train_epoch(
            fm, train_loader, optimizer, device,
            ema_fm=ema_fm, ema_decay=ema_decay,
            scaler=scaler,
            fg_weight=fg_weight,
            bg_weight=bg_weight,
            color_weight=color_weight,
        )
        scheduler.step()

        for k, v in metrics.items():
            if v > 0:
                writer.add_scalar(f"train/{k}", v, epoch)
        print(
            f"Epoch {epoch:04d}  loss={metrics['loss_total']:.5f}"
            f"  fg_rgb={metrics['loss_fg_rgb']:.5f}"
            f"  bg_rgb={metrics['loss_bg_rgb']:.5f}"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Validation
        if (epoch + 1) % cfg["training"].get("val_every", 5) == 0:
            fm.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    cond = batch["condition"].to(device)
                    target = batch["target"].to(device)
                    target_alpha = batch["target_alpha"].to(device)
                    cond_fg = (cond.abs().sum(dim=1, keepdim=True) > 0.1).float()
                    loss = fm.compute_loss(
                        target,
                        fg_mask=target_alpha,
                        cond_image=cond,
                        cond_fg_mask=cond_fg,
                        background_weight=bg_weight,
                        foreground_weight=fg_weight,
                        color_weight=color_weight,
                    )
                    val_loss += loss.item()
                    n_val += 1
            val_loss /= max(n_val, 1)
            writer.add_scalar("val/loss_total", val_loss, epoch)
            print(f"  val_loss={val_loss:.5f}")

        # Save checkpoint
        if (epoch + 1) % cfg["training"].get("save_every", 10) == 0:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch:04d}.pth")
            torch.save({
                "epoch": epoch,
                "flow_matching": fm.state_dict(),
                "ema_flow_matching": ema_fm.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Visual samples
        if (epoch + 1) % cfg["training"].get("sample_every", 10) == 0:
            ema_fm.eval()
            try:
                sample_batch = next(iter(val_loader))
                cond_vis = sample_batch["condition"][:4].to(device)
                with torch.no_grad():
                    gen_back = ema_fm.sample(
                        sample_shape=(min(4, cond_vis.shape[0]), 3, img_size, img_size),
                        steps=sample_steps,
                        cond_image=cond_vis,
                    )
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
