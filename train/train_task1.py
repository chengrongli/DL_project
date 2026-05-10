"""Task1 training with Flow Matching on paired front/back sprites."""

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
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.flow_unet import UNet
from data_code.dataset import SpritePairDataset
from models.flow_matching import FlowMatching
from utils.visualization import save_sample_grid


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(name: str | None) -> torch.device:
    if name is None or name.lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def collate_fn(batch):
    return {
        "paired": torch.stack([b["paired"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
    }


def train_one_epoch(
    model: FlowMatching,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    ema_model: FlowMatching,
    ema_decay: float,
    writer: SummaryWriter,
    global_step: int,
    bg_w: float,
    fg_w: float,
    alpha_w: float,
):
    model.train()
    total = 0.0

    for i, batch in enumerate(tqdm(loader, desc="  train", leave=False)):
        x = batch["paired"].to(device)
        m = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        loss, comp = model.compute_loss(
            x,
            fg_mask=m,
            background_weight=bg_w,
            foreground_weight=fg_w,
            alpha_weight=alpha_w,
            return_components=True,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            for ep, p in zip(ema_model.parameters(), model.parameters()):
                ep.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

        total += float(loss.item())
        step = global_step + i
        writer.add_scalar("loss/train_iter", float(loss.item()), step)
        for k, v in comp.items():
            writer.add_scalar(f"loss/{k}", float(v.item()), step)

    return total / max(1, len(loader)), global_step + len(loader)


@torch.no_grad()
def evaluate(model: FlowMatching, loader: DataLoader, device: torch.device, bg_w: float, fg_w: float, alpha_w: float):
    model.eval()
    total = 0.0
    for batch in loader:
        x = batch["paired"].to(device)
        m = batch["mask"].to(device)
        loss = model.compute_loss(
            x,
            fg_mask=m,
            background_weight=bg_w,
            foreground_weight=fg_w,
            alpha_weight=alpha_w,
        )
        total += float(loss.item())
    return total / max(1, len(loader))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task1 with Flow Matching")
    parser.add_argument("--config", default="configs/task1_flow_config.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["training"].get("device", "auto"))
    print(f"Using device: {device}")

    train_ds = SpritePairDataset(
        data_source=cfg["data"]["train_dir"],
        image_size=cfg["data"]["image_size"],
        augment=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    val_loader = None
    val_dir = cfg["data"].get("val_dir")
    if val_dir:
        val_ds = SpritePairDataset(data_source=val_dir, image_size=cfg["data"]["image_size"], augment=False)
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"].get("num_workers", 4),
            pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
        )

    unet = UNet(
        in_channels=cfg["model"]["in_channels"],
        model_channels=cfg["model"]["model_channels"],
        out_channels=cfg["model"]["out_channels"],
        channel_mult=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=cfg["model"]["n_res_blocks"],
        dropout=cfg["model"].get("dropout", 0.1),
        time_emb_dim=cfg["model"]["time_emb_dim"],
        attention_resolutions=tuple(cfg["model"].get("attn_resolutions", [16])),
    ).to(device)

    flow_model = FlowMatching(unet, time_scale=cfg.get("flow", {}).get("time_scale", 999.0)).to(device)
    ema_model = FlowMatching(copy.deepcopy(unet), time_scale=cfg.get("flow", {}).get("time_scale", 999.0)).to(device)
    ema_model.load_state_dict(flow_model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = optim.AdamW(
        flow_model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"].get("lr_min", 1e-6),
    )

    out_dir = Path(cfg["training"]["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    sample_dir = out_dir / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "logs"))

    start_epoch = 0
    global_step = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        flow_model.load_state_dict(ckpt["model_state_dict"])
        if "ema_state_dict" in ckpt:
            ema_model.load_state_dict(ckpt["ema_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = start_epoch * len(train_loader)

    bg_w = cfg["training"].get("background_weight", 1.0)
    fg_w = cfg["training"].get("foreground_weight", 1.0)
    alpha_w = cfg["training"].get("alpha_weight", 1.0)

    epochs = cfg["training"]["epochs"]
    save_every = cfg["training"].get("save_every", 20)
    sample_every = cfg["training"].get("sample_every", 20)
    val_every = cfg["training"].get("val_every", 5)
    sample_count = cfg["training"].get("sample_count", 16)
    sample_steps = cfg.get("flow", {}).get("sample_steps", 50)

    print("Start Flow Matching training...")
    for epoch in range(start_epoch, epochs):
        train_loss, global_step = train_one_epoch(
            flow_model,
            train_loader,
            optimizer,
            device,
            ema_model,
            cfg["training"].get("ema_decay", 0.999),
            writer,
            global_step,
            bg_w,
            fg_w,
            alpha_w,
        )
        scheduler.step()

        writer.add_scalar("loss/train_epoch", train_loss, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
        print(f"Epoch {epoch + 1:04d} loss={train_loss:.5f}")

        if val_loader is not None and (epoch + 1) % val_every == 0:
            val_loss = evaluate(ema_model, val_loader, device, bg_w, fg_w, alpha_w)
            writer.add_scalar("loss/val", val_loss, epoch)
            print(f"  val={val_loss:.5f}")

        if (epoch + 1) % save_every == 0:
            path = ckpt_dir / f"flow_epoch{epoch + 1:04d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": flow_model.state_dict(),
                    "ema_state_dict": ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )
            print(f"  saved: {path}")

        if (epoch + 1) % sample_every == 0:
            H = cfg["data"]["image_size"]
            shape = (sample_count, cfg["model"]["out_channels"], H, 2 * H)
            samples = ema_model.sample(shape, steps=sample_steps)
            out_path = sample_dir / f"sample_epoch{epoch + 1:04d}.png"
            save_sample_grid(samples, str(out_path), nrow=max(1, int(sample_count ** 0.5)), upscale=4)
            print(f"  sample: {out_path}")

    writer.close()
    print("Flow Matching training complete.")


if __name__ == "__main__":
    main()
