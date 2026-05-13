from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(bg, image).convert("RGB")
        return image
    return image.convert("RGB")


@dataclass
class PairRow:
    paired_path: str
    front_path: str
    back_path: str
    prompt: str


class PairManifestDataset(Dataset):
    def __init__(self, manifest: str, tokenizer: CLIPTokenizer, height: int, width: int) -> None:
        self.rows: list[PairRow] = []
        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        with open(manifest, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(
                    PairRow(
                        paired_path=row.get("paired_path", ""),
                        front_path=row["front_path"],
                        back_path=row["back_path"],
                        prompt=row["prompt"],
                    )
                )
        if not self.rows:
            raise RuntimeError(f"Manifest has no rows: {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def _load_pair(self, row: PairRow) -> Image.Image:
        if row.paired_path and Path(row.paired_path).exists():
            return ensure_rgb(Image.open(row.paired_path))
        front = ensure_rgb(Image.open(row.front_path))
        back = ensure_rgb(Image.open(row.back_path))
        w, h = front.size
        pair = Image.new("RGB", (w * 2, h), (255, 255, 255))
        pair.paste(front, (0, 0))
        pair.paste(back, (w, 0))
        return pair

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        image = self.transform(self._load_pair(row))
        input_ids = self.tokenizer(
            row.prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": image, "input_ids": input_ids}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA for HD front-back consistency.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output-dir", default="hd_consistency_pixel/outputs/lora_consistency")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-train-steps", type=int, default=6000)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="hd_consistency_pixel/cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.cache_dir, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer", cache_dir=args.cache_dir)
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder", cache_dir=args.cache_dir)
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae", cache_dir=args.cache_dir)
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet", cache_dir=args.cache_dir)
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler", cache_dir=args.cache_dir)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.enable_gradient_checkpointing()
    unet = get_peft_model(unet, lora_cfg)
    trainable_params = filter(lambda p: p.requires_grad, unet.parameters())

    dataset = PairManifestDataset(
        manifest=args.manifest,
        tokenizer=tokenizer,
        height=args.height,
        width=args.width,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.train()

    steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps <= 0:
        args.max_train_steps = args.num_train_epochs * steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / steps_per_epoch)

    progress = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="train")
    global_step = 0
    for _ in range(args.num_train_epochs):
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                input_ids = batch["input_ids"].to(accelerator.device)

                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(input_ids)[0]
                pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.item():.4f}")
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(unet)
        model.save_pretrained(args.output_dir)
        print(f"Saved LoRA to {args.output_dir}")


if __name__ == "__main__":
    main()
