import os
import random
import torch
import numpy as np
from PIL import Image, ImageEnhance
from scipy.spatial.distance import cdist # 用于色彩锁定算法
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from peft import PeftModel

BASE_MODEL = "Lykon/AnyLoRA"
LORA_DIR = "lora_out"
PROMPT = (
    "masterpiece, best quality, 16-bit pixel art, classic JRPG character sprite, "
    "RPG Maker style, chibi proportions, top-down RPG perspective, "
    "clean colored pixel outlines, soft pixel shading, vibrant 16-bit color palette, "
    "solid white background, "
    "front view on left, back view on right"
)
NEG_PROMPT = (
    "sprite sheet, multiple characters, grid background, checkerboard, "
    "scenery, landscape, tileset, ui, text, watermark, more than two characters, "
    "thick black outline, realistic, 3d render"
)
OUTPUT_DIR = "lora_previews"
NUM_SAMPLES = 10
BASE_SEED = 8888
WIDTH = 512
HEIGHT = 256
USE_COLOR_LOCK = False
WHITE_TO_ALPHA_THRESHOLD = 240

# ==========================================
# 1. 初始化管道
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    safety_checker=None,
    torch_dtype=torch.float16,
    cache_dir="./my_models"
).to(device)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# if not os.path.isdir(LORA_DIR):
#     raise FileNotFoundError(f"LoRA directory not found: {LORA_DIR}")
# pipe.load_lora_weights(LORA_DIR)
# pipe.fuse_lora(lora_scale=1.0)
if not os.path.isdir(LORA_DIR):
    raise FileNotFoundError(f"LoRA directory not found: {LORA_DIR}")

print("🔥 正在使用 PEFT 物理强制注入 LoRA...")
pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)

# === 【新增】调节 LoRA 强度的魔法 ===
LORA_STRENGTH = 0.95  # <--- 把强度降到 60%，你可以自己改 0.5 到 0.8 试试
for name, module in pipe.unet.named_modules():
    if hasattr(module, "scaling"):
        module.scaling['default'] = LORA_STRENGTH
# ==================================

pipe.unet.merge_and_unload() 
print(f"✅ LoRA 融合完毕！当前加载强度: {LORA_STRENGTH * 100}%")

# ==========================================
# 2. 图像处理算法
# ==========================================
def crisp_pixel_downscale(pil_img, original_size=64):
    """强化对比度并降采样"""
    rgba = pil_img.convert("RGBA")
    alpha = rgba.split()[3]
    rgb = rgba.convert("RGB")

    enhancer_contrast = ImageEnhance.Contrast(rgb)
    img_contrast = enhancer_contrast.enhance(1.4)

    enhancer_sharpness = ImageEnhance.Sharpness(img_contrast)
    img_sharp = enhancer_sharpness.enhance(2.0)

    small_rgb = img_sharp.resize((original_size, original_size), Image.LANCZOS)
    small_alpha = alpha.resize((original_size, original_size), Image.NEAREST)
    rgb_array = np.array(small_rgb)

    final_rgba = np.zeros((original_size, original_size, 4), dtype=np.uint8)
    final_rgba[:, :, :3] = rgb_array
    final_rgba[:, :, 3] = np.array(small_alpha)
    return Image.fromarray(final_rgba)

def lock_colors(front_img, back_img):
    """数学级色彩锁定：强迫背面图使用正面图的调色板"""
    src_arr = np.array(front_img)
    tgt_arr = np.array(back_img)

    # 提取正面图像的所有独立 RGB 颜色（忽略透明背景）
    mask_src = src_arr[:, :, 3] > 128
    unique_colors = np.unique(src_arr[mask_src][:, :3], axis=0)

    # 如果正面全是单色（出错了），就不做映射
    if len(unique_colors) == 0:
        return back_img

    # 找到背面图像的像素点
    mask_tgt = tgt_arr[:, :, 3] > 128
    tgt_rgb = tgt_arr[mask_tgt][:, :3]

    # 计算背面每一个像素点，距离正面调色板哪个颜色最接近
    distances = cdist(tgt_rgb, unique_colors)
    closest_indices = np.argmin(distances, axis=1)
    mapped_rgb = unique_colors[closest_indices]

    # 重新组装背面图像
    out_arr = tgt_arr.copy()
    out_arr[mask_tgt, :3] = mapped_rgb

    return Image.fromarray(out_arr)

def white_to_transparent(pil_img, threshold=240):
    rgba = pil_img.convert("RGBA")
    arr = np.array(rgba)
    white_mask = (
        (arr[:, :, 0] > threshold)
        & (arr[:, :, 1] > threshold)
        & (arr[:, :, 2] > threshold)
    )
    arr[white_mask, 3] = 0
    return Image.fromarray(arr)

# ==========================================
# 4. 批量配对生成 (Paired Generation)
# ==========================================
os.makedirs("distilled", exist_ok=True)
rng = random.Random(BASE_SEED)

for idx in range(NUM_SAMPLES):
    seed = rng.randint(0, 2**31 - 1)

    composite = pipe(
        PROMPT,
        negative_prompt=NEG_PROMPT,
        guidance_scale=6.0,
        num_inference_steps=25,
        generator=torch.Generator(device=device).manual_seed(seed),
        width=WIDTH, height=HEIGHT,
    ).images[0]

    half_w = WIDTH // 2
    front_highres = composite.crop((0, 0, half_w, HEIGHT))
    back_highres = composite.crop((half_w, 0, WIDTH, HEIGHT))

    front_highres = white_to_transparent(front_highres, WHITE_TO_ALPHA_THRESHOLD)
    back_highres = white_to_transparent(back_highres, WHITE_TO_ALPHA_THRESHOLD)

    front_64 = crisp_pixel_downscale(front_highres)
    back_64_raw = crisp_pixel_downscale(back_highres)

    # ------------------ C. 色彩锁定 (Color Lock) ------------------
    # 强迫背面的颜色和正面的颜色 100% 一致！
    back_64 = lock_colors(front_64, back_64_raw) if USE_COLOR_LOCK else back_64_raw

    # ------------------ D. 保存素材 ------------------
    # 保存 64x64 原尺寸
    front_64.save(f"distilled/pair_{idx:04d}_front.png")
    back_64.save(f"distilled/pair_{idx:04d}_back.png")

    # 将两张图拼在一起，放大保存，方便你对比预览
    preview_w, preview_h = 256, 256
    paired_preview = Image.new("RGBA", (preview_w * 2, preview_h))
    paired_preview.paste(front_64.resize((preview_w, preview_h), Image.NEAREST), (0, 0))
    paired_preview.paste(back_64.resize((preview_w, preview_h), Image.NEAREST), (preview_w, 0))

    paired_preview.save(f"distilled/pair_{idx:04d}_lora_preview.png")

    print(f"Generated Perfectly Paired Sprite {idx}: seed={seed}")