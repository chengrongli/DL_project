"""Generate conditional LPC sprites with specified attributes using Flow Matching."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.flow_unet import UNet
from models.flow_matching import FlowMatching
from models.attribute_encoder import (
    AttributeEncoder,
    encode_attributes_batch,
    ATTR_FIELDS,
    HAIR_STYLE_VOCAB,
    TORSO_TYPE_VOCAB,
    LEGS_TYPE_VOCAB,
    FEET_TYPE_VOCAB,
)
from utils.visualization import save_sample_grid

COLOR_CHOICES = [
    "black", "white", "gray", "brown", "red", "pink", "orange", "yellow",
    "green", "teal", "blue", "purple", "gold", "silver", "copper",
]


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
    parser = argparse.ArgumentParser(description="Conditional Flow Matching sampler")
    parser.add_argument("--config", default="configs/conditional_config.yaml")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--num", type=int, default=8, help="Number of samples to generate")
    parser.add_argument("--steps", type=int, default=60, help="ODE sampling steps")
    parser.add_argument("--guidance-scale", type=float, default=3.0, help="CFG guidance scale")
    parser.add_argument("--out", default="outputs/conditional/generated", help="Output directory")
    parser.add_argument("--body-type", default=None, choices=["male", "female", "teen", "child", "muscular", "adult"])
    parser.add_argument("--hair-color", default=None, choices=COLOR_CHOICES)
    parser.add_argument("--hair-style", default=None, help=f"Hair style, e.g. {', '.join(HAIR_STYLE_VOCAB[1:11])}...")
    parser.add_argument("--torso-type", default=None, choices=TORSO_TYPE_VOCAB)
    parser.add_argument("--torso-color", default=None, choices=COLOR_CHOICES)
    parser.add_argument("--legs-type", default=None, choices=LEGS_TYPE_VOCAB)
    parser.add_argument("--legs-color", default=None, choices=COLOR_CHOICES)
    parser.add_argument("--feet-type", default=None, choices=FEET_TYPE_VOCAB)
    parser.add_argument("--feet-color", default=None, choices=COLOR_CHOICES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    device = resolve_device(cfg["training"].get("device", "auto"))
    attr_cfg = cfg.get("attributes", {})

    unet = UNet(
        in_channels=cfg["model"]["in_channels"],
        model_channels=cfg["model"]["model_channels"],
        out_channels=cfg["model"]["out_channels"],
        channel_mult=tuple(cfg["model"]["channel_mults"]),
        num_res_blocks=cfg["model"]["n_res_blocks"],
        dropout=cfg["model"].get("dropout", 0.1),
        time_emb_dim=cfg["model"]["time_emb_dim"],
        attention_resolutions=tuple(cfg["model"].get("attn_resolutions", [16])),
        attr_cond_dim=attr_cfg.get("output_dim", 256),
        attr_token_dim=attr_cfg.get("token_dim", 256),
        cross_attn_heads=cfg["model"].get("cross_attn_heads", 4),
    ).to(device)

    flow_model = FlowMatching(unet, time_scale=cfg.get("flow", {}).get("time_scale", 999.0)).to(device)
    attr_encoder = AttributeEncoder(
        embed_dim=attr_cfg.get("embed_dim", 64),
        output_dim=attr_cfg.get("output_dim", 256),
        token_dim=attr_cfg.get("token_dim", 256),
        use_pos_embed=attr_cfg.get("use_pos_embed", True),
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt))
    flow_model.load_state_dict(state)
    attr_state = ckpt.get("ema_attr_encoder_state_dict", ckpt.get("attr_encoder_state_dict"))
    if attr_state is not None:
        attr_encoder.load_state_dict(attr_state)
    flow_model.eval()
    attr_encoder.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    h = cfg["data"]["image_size"]
    shape = (args.num, cfg["model"]["out_channels"], h, 2 * h)

    # Unconditional generation
    with torch.no_grad():
        samples_uncond = flow_model.sample(shape, steps=args.steps)
    save_sample_grid(samples_uncond, str(out_dir / "uncond.png"), nrow=max(1, int(args.num ** 0.5)), upscale=4)
    print(f"Saved unconditional samples to {out_dir / 'uncond.png'}")

    # Conditional generation
    attrs = {
        "body_type": args.body_type,
        "hair_color": args.hair_color,
        "hair_style": args.hair_style,
        "torso_type": args.torso_type,
        "torso_color": args.torso_color,
        "legs_type": args.legs_type,
        "legs_color": args.legs_color,
        "feet_type": args.feet_type,
        "feet_color": args.feet_color,
    }
    has_any_attr = any(v is not None for v in attrs.values())
    if has_any_attr:
        attrs_batch = [attrs] * args.num
        attr_indices = encode_attributes_batch(attrs_batch, device=device)
        attr_cond, attr_tokens = attr_encoder(attr_indices)

        with torch.no_grad():
            samples_cond = flow_model.sample(shape, steps=args.steps, attr_cond=attr_cond, attr_tokens=attr_tokens, guidance_scale=args.guidance_scale)
        desc_parts = [f"{k}={v}" for k, v in attrs.items() if v is not None]
        desc = "_".join(desc_parts) if desc_parts else "default"
        save_sample_grid(samples_cond, str(out_dir / f"cond_{desc}.png"), nrow=max(1, int(args.num ** 0.5)), upscale=4)
        print(f"Saved conditional samples ({desc}) to {out_dir / f'cond_{desc}.png'}")
    else:
        print("No attributes specified. Use --hair-color, --body-type, --torso-type, --hair-style, etc.")


if __name__ == "__main__":
    main()
