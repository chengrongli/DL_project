"""
Task 2 Inference – Front-to-back reconstruction (Flow Matching).

Usage:
    python inference/reconstruct.py \
        --config  configs/task2_config.yaml \
        --ckpt    outputs/task2/checkpoints/ckpt_epoch0099.pth \
        --input   path/to/front.png \
        --steps   50 \
        --out     outputs/task2/reconstructed/

    # Batch mode: reconstruct all PNGs in a directory
    python inference/reconstruct.py \
        --config  configs/task2_config.yaml \
        --ckpt    outputs/task2/checkpoints/ckpt_epoch0099.pth \
        --input   path/to/front_images/ \
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

from models.flow_matching import FlowMatching
from models.unet import UNet
from utils.visualization import tensor_to_pil, side_by_side


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device, ckpt_path: str) -> FlowMatching:
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

    fm = FlowMatching(
        model=unet,
        time_scale=cfg["flow"].get("time_scale", 999.0),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    fm.load_state_dict(ckpt["ema_flow_matching"])
    fm.eval()
    return fm


def preprocess(image_path: str, image_size: int) -> torch.Tensor:
    """Load a front image and convert to a (1, 3, H, W) tensor in [-1, 1].

    Uses floodfill from corners to remove solid-color backgrounds,
    then applies alpha premultiplication to match training data.
    """
    import numpy as np

    img = Image.open(image_path).convert("RGBA").resize(
        (image_size, image_size), Image.NEAREST
    )
    arr = np.array(img)

    # Floodfill from 4 corners to find background
    h, w = arr.shape[:2]
    visited = np.zeros((h, w), dtype=bool)
    bg_mask = np.zeros((h, w), dtype=bool)
    threshold = 30  # color distance threshold

    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    for cy, cx in corners:
        if visited[cy, cx]:
            continue
        seed_color = arr[cy, cx, :3].astype(np.float32)
        stack = [(cy, cx)]
        visited[cy, cx] = True
        while stack:
            y, x = stack.pop()
            dist = np.abs(arr[y, x, :3].astype(np.float32) - seed_color).sum()
            if dist > threshold * 3:
                continue
            bg_mask[y, x] = True
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))

    # Set background transparent
    arr[bg_mask, 3] = 0

    # Alpha premultiply
    alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
    rgb = arr[:, :, :3].astype(np.float32) * alpha
    rgb = np.clip(rgb, 0, 255)
    out = np.concatenate([rgb, alpha * 255], axis=2).astype(np.uint8)
    img = Image.fromarray(out).convert("RGB")
    t = TF.to_tensor(img) * 2.0 - 1.0
    return t.unsqueeze(0)


def reconstruct(
    fm: FlowMatching,
    front_tensor: torch.Tensor,
    device: torch.device,
    steps: int = 50,
    image_size: int = 64,
) -> torch.Tensor:
    """Generate back view conditioned on front_tensor via Flow Matching ODE."""
    front_tensor = front_tensor.to(device)
    B = front_tensor.shape[0]

    with torch.no_grad():
        back_pred = fm.sample(
            sample_shape=(B, 3, image_size, image_size),
            steps=steps,
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
    parser.add_argument("--steps", type=int, default=50, help="Flow Matching ODE steps")
    parser.add_argument("--out", default="outputs/task2/reconstructed/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = cfg["data"]["image_size"]

    fm = build_model(cfg, device, args.ckpt)
    os.makedirs(args.out, exist_ok=True)

    image_paths = collect_inputs(args.input)
    print(f"Processing {len(image_paths)} image(s)}…")

    for img_path in image_paths:
        stem = Path(img_path).stem
        front_t = preprocess(img_path, image_size)
        back_t = reconstruct(fm, front_t, device,
                             steps=args.steps,
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
