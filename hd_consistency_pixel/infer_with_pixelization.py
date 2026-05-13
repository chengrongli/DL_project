from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionPipeline,
)

from hd_consistency_pixel.pixelization import DifferentiablePixelization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with consistency LoRA + pixelization stage.")
    parser.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="low quality, blurry, bad anatomy")
    parser.add_argument("--pixel-prompt", default="pixel art, 16-bit sprite, clean outline, limited palette")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--out-dir", default="hd_consistency_pixel/outputs/infer")

    parser.add_argument("--use-controlnet-tile", action="store_true")
    parser.add_argument("--tile-controlnet-model", default="lllyasviel/control_v11f1e_sd15_tile")
    parser.add_argument("--tile-strength", type=float, default=0.75)
    parser.add_argument("--tile-control-scale", type=float, default=0.9)

    parser.add_argument("--use-diff-pixel", action="store_true")
    parser.add_argument("--pixel-block-size", type=int, default=8)
    parser.add_argument("--pixel-color-levels", type=int, default=16)
    return parser.parse_args()


def make_pair_pipeline(args: argparse.Namespace, device: torch.device):
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    pipe.load_lora_weights(args.lora_path)
    pipe.to(device)
    return pipe


def run_pair_generation(args: argparse.Namespace, device: torch.device) -> Image.Image:
    pipe = make_pair_pipeline(args, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]
    return image


def run_tile_controlnet(
    base_image: Image.Image,
    args: argparse.Namespace,
    device: torch.device,
) -> Image.Image:
    controlnet = ControlNetModel.from_pretrained(
        args.tile_controlnet_model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    pipe.load_lora_weights(args.lora_path)
    pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    image = pipe(
        prompt=args.pixel_prompt,
        negative_prompt=args.negative_prompt,
        image=base_image,
        control_image=base_image,
        strength=args.tile_strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.tile_control_scale,
        generator=generator,
    ).images[0]
    return image


def run_diff_pixel(base_image: Image.Image, args: argparse.Namespace) -> Image.Image:
    x = to_tensor(base_image).unsqueeze(0).float()
    layer = DifferentiablePixelization(
        block_size=args.pixel_block_size,
        color_levels=args.pixel_color_levels,
        use_ste_quant=True,
    )
    with torch.no_grad():
        y = layer(x).squeeze(0).cpu()
    return to_pil_image(y)


def split_front_back(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    w, h = img.size
    mid = w // 2
    front = img.crop((0, 0, mid, h))
    back = img.crop((mid, 0, w, h))
    return front, back


def save_outputs(image: Image.Image, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    front, back = split_front_back(image)
    image.save(out_dir / f"{prefix}_pair.png")
    front.save(out_dir / f"{prefix}_front.png")
    back.save(out_dir / f"{prefix}_back.png")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    pair = run_pair_generation(args, device)
    save_outputs(pair, out_dir, "01_consistency")

    final_image = pair
    if args.use_controlnet_tile:
        final_image = run_tile_controlnet(pair, args, device)
        save_outputs(final_image, out_dir, "02_tile_pixel")
    elif args.use_diff_pixel:
        final_image = run_diff_pixel(pair, args)
        save_outputs(final_image, out_dir, "02_diff_pixel")

    np.save(out_dir / "seed.npy", np.array([args.seed], dtype=np.int64))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
