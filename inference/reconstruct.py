"""
Task 2 Inference – Front-to-back reconstruction.

Usage:
    python inference/reconstruct.py \\
        --config  configs/task2_config.yaml \\
        --ckpt    outputs/task2/checkpoints/ckpt_epoch0099.pth \\
        --input   path/to/front.png \\
        --steps   50 \\
        --out     outputs/task2/reconstructed/

    # Batch mode: reconstruct all PNGs in a directory
    python inference/reconstruct.py \\
        --config  configs/task2_config.yaml \\
        --ckpt    outputs/task2/checkpoints/ckpt_epoch0099.pth \\
        --input   path/to/front_images/ \\
        --out     outputs/task2/reconstructed/

Outputs:
    - {out}/{stem}_pred_back.png   – predicted back view
    - {out}/{stem}_side_by_side.png – front | predicted back
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.diffusion import GaussianDiffusion
from models.unet import UNet
from utils.visualization import tensor_to_pil, side_by_side


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device, ckpt_path: str) -> GaussianDiffusion:
    unet = UNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        model_channels=cfg["model"]["model_channels"],
        channel_mults=tuple(cfg["model"]["channel_mults"]),
        n_res_blocks=cfg["model"]["n_res_blocks"],
        attn_resolutions=tuple(cfg["model"]["attn_resolutions"]),
        time_emb_dim=cfg["model"]["time_emb_dim"],
        cond_emb_dim=0,
        dropout=0.0,
        image_size=cfg["data"]["image_size"],
    ).to(device)

    diffusion = GaussianDiffusion(
        model=unet,
        timesteps=cfg["diffusion"]["timesteps"],
        schedule=cfg["diffusion"]["schedule"],
        loss_type=cfg["diffusion"]["loss_type"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    diffusion.load_state_dict(ckpt["diffusion"])
    diffusion.eval()
    return diffusion


def preprocess(image_path: str, image_size: int) -> torch.Tensor:
    """Load a front image and convert to a (1, 3, H, W) tensor in [−1, 1]."""
    img = Image.open(image_path).convert("RGB").resize(
        (image_size, image_size), Image.NEAREST
    )
    t = TF.to_tensor(img) * 2.0 - 1.0  # [−1, 1]
    return t.unsqueeze(0)


def reconstruct(
    diffusion: GaussianDiffusion,
    front_tensor: torch.Tensor,
    device: torch.device,
    ddim_steps: int = 50,
    eta: float = 0.0,
    image_size: int = 64,
) -> torch.Tensor:
    """
    Generate back view conditioned on front_tensor.

    Args:
        front_tensor: (B, 3, H, W) front image tensor in [−1, 1].

    Returns:
        (B, 3, H, W) predicted back image in [−1, 1].
    """
    front_tensor = front_tensor.to(device)
    B = front_tensor.shape[0]

    with torch.no_grad():
        back_pred = diffusion.ddim_sample(
            shape=(B, 3, image_size, image_size),
            device=device,
            ddim_steps=ddim_steps,
            eta=eta,
            cond_image=front_tensor,
        )
    return back_pred


def collect_inputs(input_path: str) -> List[str]:
    """Return a list of image paths from a single file or a directory."""
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        return sorted(str(f) for f in p.glob("*.png"))
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Front-to-back reconstruction (Task 2)")
    parser.add_argument("--config", default="configs/task2_config.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input", required=True, help="Front image or directory of images")
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--out", default="outputs/task2/reconstructed/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = cfg["data"]["image_size"]

    diffusion = build_model(cfg, device, args.ckpt)
    os.makedirs(args.out, exist_ok=True)

    image_paths = collect_inputs(args.input)
    print(f"Processing {len(image_paths)} image(s)…")

    for img_path in image_paths:
        stem = Path(img_path).stem
        front_t = preprocess(img_path, image_size)
        back_t = reconstruct(diffusion, front_t, device,
                              ddim_steps=args.steps, eta=args.eta,
                              image_size=image_size)

        front_pil = tensor_to_pil(front_t[0])
        back_pil = tensor_to_pil(back_t[0])

        back_pil.save(os.path.join(args.out, f"{stem}_pred_back.png"))
        sbs = side_by_side(front_pil, back_pil)
        sbs.save(os.path.join(args.out, f"{stem}_side_by_side.png"))
        print(f"  {img_path} → {stem}_pred_back.png")

    print("Done.")


if __name__ == "__main__":
    main()
