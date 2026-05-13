import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

# 【新增】引入最新标准的 PEFT 库
from peft import LoraConfig, get_peft_model

DEFAULT_PROMPT = (
    "masterpiece, best quality, 16-bit pixel art, classic JRPG character sprite, "
    "RPG Maker style, chibi proportions, top-down RPG perspective, "
    "clean colored pixel outlines, soft pixel shading, vibrant 16-bit color palette, "
    "solid white background, "
    "front view on left, back view on right"
)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_images(data_dir: str) -> list[Path]:
    root = Path(data_dir)
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        paths.extend(root.glob(ext))
    return sorted(paths)


class PairPromptDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: CLIPTokenizer,
        prompt: str,
        height: int,
        width: int,
    ) -> None:
        self.paths = list_images(data_dir)
        if not self.paths:
            raise ValueError(f"No images found in: {data_dir}")
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.transform = transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        
        raw_image = Image.open(path).convert("RGBA")
        background = Image.new("RGBA", raw_image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, raw_image).convert("RGB") 
        image = self.transform(image)
        
        prompt = self.prompt 
        
        input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": image, "input_ids": input_ids}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA on pairs/processed with a unified prompt.")
    parser.add_argument("--data_dir", default="pairs/processed")
    parser.add_argument("--base_model", default="Lykon/AnyLoRA")
    parser.add_argument("--output_dir", default="lora_out")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=1500)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=42)
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

    # 统一指定缓存目录为 ./my_models，避免占用系统盘空间
    cache_dir = "./my_models"
    
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer", cache_dir=cache_dir)
    text_encoder = CLIPTextModel.from_pretrained(args.base_model, subfolder="text_encoder", cache_dir=cache_dir)
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae", cache_dir=cache_dir)
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet", cache_dir=cache_dir)
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler", cache_dir=cache_dir)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 【全新逻辑】使用 PEFT 配置 LoRA
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"], # 精准命中注意力层
    )
    
    # 开启梯度检查点（省显存好习惯）
    unet.enable_gradient_checkpointing()
    # 瞬间完成大脑改造
    unet = get_peft_model(unet, lora_config)
    
    if accelerator.is_main_process:
        unet.print_trainable_parameters() # 可以在控制台看到极帅的参数占比总结

    # 只提取出需要训练的 LoRA 参数交给优化器
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    dataset = PairPromptDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=8, # 你性能强，拉到 8 会读取更快
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(lora_layers, lr=args.learning_rate)

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
    if args.max_train_steps is None or args.max_train_steps <= 0:
        args.max_train_steps = args.num_train_epochs * steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / steps_per_epoch)

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="train",
    )
    global_step = 0

    for _epoch in range(args.num_train_epochs):
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                input_ids = batch["input_ids"].to(accelerator.device)

                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

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
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    
    # 【全新保存逻辑】直接调用 PEFT 的标准保存方法
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(args.output_dir)
        print(f"🎉 训练完成！LoRA 已保存至 {args.output_dir}")

if __name__ == "__main__":
    main()