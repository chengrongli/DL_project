"""Nearest-neighbor evaluation: compare generated samples against training data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.flow_unet import UNet
from models.flow_matching import FlowMatching
from data_code.dataset import SpritePairDataset


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


def to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] float tensor to [0, 255] uint8."""
    return ((tensor.clamp(-1, 1) + 1) / 2 * 255).byte()


def main() -> None:
    parser = argparse.ArgumentParser(description="Nearest-neighbor evaluation")
    parser.add_argument("--config", default="configs/task1_config.yaml")
    parser.add_argument("--ckpt", default="outputs/task1/checkpoints/flow_epoch0150.pth")
    parser.add_argument("--n_gen", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--n_train", type=int, default=500, help="Number of training samples to compare against")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--out_dir", default="outputs/task1/eval_nn")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["training"].get("device", "auto"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    unet = UNet(
        in_channels=cfg["model"]["in_channels"],
        model_channels=cfg["model"]["model_channels"],
        out_channels=cfg["model"]["out_channels"],
        channel_mult=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=cfg["model"]["n_res_blocks"],
        dropout=0.0,
        time_emb_dim=cfg["model"]["time_emb_dim"],
        attention_resolutions=tuple(cfg["model"].get("attn_resolutions", [16])),
    ).to(device)
    model = FlowMatching(unet, time_scale=cfg.get("flow", {}).get("time_scale", 999.0)).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()

    # Generate samples
    print(f"Generating {args.n_gen} samples...")
    h = cfg["data"]["image_size"]
    all_gen = []
    batch_size = 50
    for i in range(0, args.n_gen, batch_size):
        n = min(batch_size, args.n_gen - i)
        shape = (n, cfg["model"]["out_channels"], h, 2 * h)
        with torch.no_grad():
            samples = model.sample(shape, steps=args.steps)
        all_gen.append(samples.cpu())
    gen_tensors = torch.cat(all_gen, dim=0)  # (N, 4, H, 2H)

    # Load training data (use RGB channels only for comparison, more meaningful)
    print(f"Loading {args.n_train} training samples...")
    ds = SpritePairDataset(
        data_source=cfg["data"]["train_dir"],
        image_size=cfg["data"]["image_size"],
        augment=False,
    )
    train_tensors = []
    for i in tqdm(range(min(args.n_train, len(ds))), desc="Loading train"):
        sample = ds[i]
        train_tensors.append(sample["paired"][:3])  # RGB only
    train_tensors = torch.stack(train_tensors, dim=0)  # (M, 3, H, 2H)

    # Also compute train-to-train nearest neighbor distances for baseline
    gen_rgb = gen_tensors[:, :3]  # (N, 3, H, 2H)

    # Flatten to vectors
    gen_flat = gen_rgb.reshape(gen_rgb.shape[0], -1).float()  # (N, D)
    train_flat = train_tensors.reshape(train_tensors.shape[0], -1).float()  # (M, D)

    # Normalize to [0, 1] for fair L2
    gen_flat = (gen_flat + 1) / 2
    train_flat = (train_flat + 1) / 2

    # Compute pairwise L2 distances in batches
    print("Computing gen -> train nearest neighbors...")
    gen_nn_dists = []
    gen_nn_indices = []
    batch = 20
    for i in tqdm(range(0, len(gen_flat), batch), desc="Gen vs Train"):
        chunk = gen_flat[i:i+batch]  # (B, D)
        # L2 distance
        diffs = chunk.unsqueeze(1) - train_flat.unsqueeze(0)  # (B, M, D)
        dists = (diffs ** 2).sum(dim=-1).sqrt()  # (B, M)
        min_dists, min_idx = dists.min(dim=1)  # (B,)
        gen_nn_dists.append(min_dists)
        gen_nn_indices.append(min_idx)

    gen_nn_dists = torch.cat(gen_nn_dists)

    # Compute train-to-train nearest neighbor (leave-one-out)
    print("Computing train -> train nearest neighbors (baseline)...")
    train_nn_dists = []
    for i in tqdm(range(0, len(train_flat), batch), desc="Train vs Train"):
        chunk = train_flat[i:i+batch]
        diffs = chunk.unsqueeze(1) - train_flat.unsqueeze(0)
        dists = (diffs ** 2).sum(dim=-1).sqrt()
        # Set self-distance to inf
        for j in range(min(batch, len(train_flat) - i)):
            dists[j, i + j] = float("inf")
        min_dists, _ = dists.min(dim=1)
        train_nn_dists.append(min_dists)
    train_nn_dists = torch.cat(train_nn_dists)

    # Results
    print("\n" + "=" * 60)
    print("NEAREST NEIGHBOR EVALUATION RESULTS")
    print("=" * 60)
    print(f"Generated samples:       {len(gen_flat)}")
    print(f"Training samples:        {len(train_flat)}")
    print()
    print(f"Gen -> Train NN distance:")
    print(f"  Mean:   {gen_nn_dists.mean():.4f}")
    print(f"  Median: {gen_nn_dists.median():.4f}")
    print(f"  Min:    {gen_nn_dists.min():.4f}")
    print(f"  Max:    {gen_nn_dists.max():.4f}")
    print()
    print(f"Train -> Train NN distance (baseline):")
    print(f"  Mean:   {train_nn_dists.mean():.4f}")
    print(f"  Median: {train_nn_dists.median():.4f}")
    print(f"  Min:    {train_nn_dists.min():.4f}")
    print(f"  Max:    {train_nn_dists.max():.4f}")
    print()

    ratio = gen_nn_dists.mean() / train_nn_dists.mean()
    print(f"Ratio (gen_mean / train_mean): {ratio:.4f}")
    if ratio > 1.0:
        print("-> Generated samples are FARTHER from training data than training data is from itself.")
        print("   This suggests the model is NOT simply copying/memorizing training examples.")
    else:
        print("-> Generated samples are CLOSER to training data than training data is from itself.")
        print("   WARNING: Possible memorization detected!")

    # Save closest pairs for visual inspection
    print("\nSaving closest gen-train pairs for visual inspection...")
    topk_vals, topk_idx = gen_nn_dists.topk(10, largest=False)
    n_save = min(10, len(topk_idx))

    for rank in range(n_save):
        gi = topk_idx[rank].item()
        ti = gen_nn_indices[rank // batch].tolist()
        # Recompute the exact train index
        chunk_i = gi // batch
        offset_i = gi % batch
        chunk = gen_flat[gi:gi+1]
        dists = (chunk - train_flat).pow(2).sum(-1).sqrt()
        ti_best = dists.argmin().item()

        # Save generated image
        gen_img = to_uint8(gen_tensors[gi, :3])  # RGB
        gen_pil = Image.fromarray(gen_img.permute(1, 2, 0).numpy(), "RGB")
        gen_pil = gen_pil.resize((gen_pil.width * 4, gen_pil.height * 4), Image.NEAREST)
        gen_pil.save(out_dir / f"closest_{rank:02d}_gen.png")

        # Save its nearest train neighbor
        train_sample = ds[ti_best]
        train_paired = train_sample["paired"][:3]
        train_img = to_uint8(train_paired)
        train_pil = Image.fromarray(train_img.permute(1, 2, 0).numpy(), "RGB")
        train_pil = train_pil.resize((train_pil.width * 4, train_pil.height * 4), Image.NEAREST)
        train_pil.save(out_dir / f"closest_{rank:02d}_train_nn.png")

        print(f"  #{rank}: L2={topk_vals[rank]:.2f}")

    print(f"\nResults saved to {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
