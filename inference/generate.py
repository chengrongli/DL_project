"""
Task 1 Inference – Random front+back character generation.

Usage:
    python inference/generate.py \\
        --config  configs/task1_config.yaml \\
        --ckpt    outputs/task1/checkpoints/ckpt_epoch0099.pth \\
        --n       8 \\
        --steps   50 \\
        --scale   3.0 \\
        --out     outputs/task1/generated/

Outputs:
    - {out}/sample_{i:04d}_front.png  – front sprite
    - {out}/sample_{i:04d}_back.png   – back sprite
    - {out}/grid.png                  – combined overview grid
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.diffusion import GaussianDiffusion
from models.embeddings import build_lpc_attr_embedding
from models.unet import UNet
from utils.visualization import tensor_to_pil, save_sample_grid


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device, ckpt_path: str):
    unet = UNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        model_channels=cfg["model"]["model_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        n_res_blocks=cfg["model"]["n_res_blocks"],
        attn_resolutions=tuple(cfg["model"]["attn_resolutions"]),
        time_emb_dim=cfg["model"]["time_emb_dim"],
        cond_emb_dim=cfg["model"].get("cond_emb_dim", 0),
        dropout=0.0,
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
        ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    diffusion.load_state_dict(ckpt["diffusion"])
    if attr_emb is not None and "attr_emb" in ckpt:
        attr_emb.load_state_dict(ckpt["attr_emb"])

    diffusion.eval()
    if attr_emb is not None:
        attr_emb.eval()

    return diffusion, attr_emb


def generate(
    diffusion: GaussianDiffusion,
    n: int,
    image_size: int,
    device: torch.device,
    ddim_steps: int = 50,
    eta: float = 0.0,
    cfg_scale: float = 1.0,
    attr_emb=None,
    attrs: dict = None,
) -> torch.Tensor:
    """
    Generate n front+back pairs.

    Returns:
        Tensor of shape (n, 6, H, W) in [−1, 1].
        Channels [:3] are front, channels [3:] are back.
    """
    cond_emb = None
    uncond_emb = None

    if attr_emb is not None and attrs is not None:
        # Build batch of attribute index tensors
        batch_attrs = {
            k: torch.full((n,), v, dtype=torch.long, device=device)
            for k, v in attrs.items()
        }
        with torch.no_grad():
            cond_emb = attr_emb(batch_attrs, force_null=False)
            if cfg_scale != 1.0:
                uncond_emb = attr_emb.get_null_embedding(n, device)

    with torch.no_grad():
        samples = diffusion.ddim_sample(
            shape=(n, diffusion.model.out_channels, image_size, image_size),
            device=device,
            ddim_steps=ddim_steps,
            eta=eta,
            cond_emb=cond_emb,
            cfg_scale=cfg_scale,
            uncond_emb=uncond_emb,
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random front+back sprites")
    parser.add_argument("--config", default="configs/task1_config.yaml")
    parser.add_argument("--ckpt", required=True, help="Path to trained checkpoint")
    parser.add_argument("--n", type=int, default=4, help="Number of pairs to generate")
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta (0=deterministic)")
    parser.add_argument("--scale", type=float, default=1.0, help="CFG guidance scale")
    parser.add_argument("--out", default="outputs/task1/generated/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion, attr_emb = build_model(cfg, device, args.ckpt)

    image_size = cfg["data"]["image_size"]
    samples = generate(
        diffusion,
        n=args.n,
        image_size=image_size,
        device=device,
        ddim_steps=args.steps,
        eta=args.eta,
        cfg_scale=args.scale,
        attr_emb=attr_emb,
    )

    os.makedirs(args.out, exist_ok=True)

    # Split front / back channels
    front = samples[:, :3, :, :]   # first 3 channels
    back = samples[:, 3:, :, :]    # last 3 channels

    for i in range(args.n):
        front_pil = tensor_to_pil(front[i])
        back_pil = tensor_to_pil(back[i])
        front_pil.save(os.path.join(args.out, f"sample_{i:04d}_front.png"))
        back_pil.save(os.path.join(args.out, f"sample_{i:04d}_back.png"))

    # Save grid (alternating front/back)
    grid_tensor = torch.cat([front, back], dim=0)
    save_sample_grid(grid_tensor, os.path.join(args.out, "grid.png"), nrow=args.n)
    print(f"Generated {args.n} pairs → {args.out}")


if __name__ == "__main__":
    main()
