"""Generate Task1 samples from a trained Flow Matching checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.unet_flow import UNet
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Flow Matching sampler")
    parser.add_argument("--config", default="configs/task1_flow_config.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--out", default="outputs/task1_flow/generated/samples.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["training"].get("device", "auto"))

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
    model = FlowMatching(unet, time_scale=cfg.get("flow", {}).get("time_scale", 999.0)).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()

    h = cfg["data"]["image_size"]
    shape = (args.n, cfg["model"]["out_channels"], h, 2 * h)
    with torch.no_grad():
        samples = model.sample(shape, steps=args.steps)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_sample_grid(samples, str(out), nrow=max(1, int(args.n ** 0.5)), upscale=4)
    print(f"Saved samples to: {out}")


if __name__ == "__main__":
    main()
