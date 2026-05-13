# HD Front-Back Consistency + Pixelization

这套流程把你的新思路拆成两阶段：

1. **训练阶段（学一致性）**：在高清 front/back 人类角色配对集上训练 LoRA，只学习“左正面/右背面轮廓与设定一致”。
2. **推理阶段（做像素感）**：把生成结果交给 ControlNet-Tile（或 Pixel-Art ControlNet），也可用可微像素化层做后处理。

## 目录

- `prepare_manifest.py`: 扫描 front/back 配对并生成训练清单 CSV，同时可导出拼接图。
- `train_pair_lora.py`: 使用 diffusers + accelerate + peft 训练“正背一致性” LoRA。
- `infer_with_pixelization.py`: 推理时加载 LoRA，外挂 ControlNet-Tile 或 Differentiable Pixelization。
- `pixelization.py`: 可微像素化层（支持 straight-through 量化）。

## 数据格式

默认要求同名配对：

- `xxx_front.png`
- `xxx_back.png`

先生成 manifest：

```bash
python hd_consistency_pixel/prepare_manifest.py \
  --input-dir data/highres_pairs \
  --manifest hd_consistency_pixel/data/manifest.csv \
  --paired-dir hd_consistency_pixel/data/paired_1024x512 \
  --height 512 \
  --single-width 512
```

输出 CSV 字段：

- `char_id`
- `front_path`
- `back_path`
- `paired_path`
- `prompt`

## 训练（只学一致轮廓）

```bash
python hd_consistency_pixel/train_pair_lora.py \
  --manifest hd_consistency_pixel/data/manifest.csv \
  --base-model runwayml/stable-diffusion-v1-5 \
  --output-dir hd_consistency_pixel/outputs/lora_consistency \
  --height 512 \
  --width 1024 \
  --rank 16 \
  --max-train-steps 6000 \
  --train-batch-size 4 \
  --learning-rate 1e-4
```

建议 prompt 模板：

`high quality character sheet, full body, front view on left, back view on right, same character, same outfit details`

## 推理（外挂像素化）

### 方案 A: ControlNet-Tile（推荐）

```bash
python hd_consistency_pixel/infer_with_pixelization.py \
  --base-model runwayml/stable-diffusion-v1-5 \
  --lora-path hd_consistency_pixel/outputs/lora_consistency \
  --prompt "fantasy male knight, blue tie, same character front and back" \
  --height 512 \
  --width 1024 \
  --use-controlnet-tile \
  --tile-controlnet-model lllyasviel/control_v11f1e_sd15_tile \
  --pixel-prompt "pixel art, 16-bit sprite, clean outline, limited palette" \
  --out-dir hd_consistency_pixel/outputs/infer_tile
```

### 方案 B: Differentiable Pixelization（无 ControlNet 也能跑）

```bash
python hd_consistency_pixel/infer_with_pixelization.py \
  --base-model runwayml/stable-diffusion-v1-5 \
  --lora-path hd_consistency_pixel/outputs/lora_consistency \
  --prompt "fantasy female rogue, red scarf, same character front and back" \
  --height 512 \
  --width 1024 \
  --use-diff-pixel \
  --pixel-block-size 8 \
  --pixel-color-levels 16 \
  --out-dir hd_consistency_pixel/outputs/infer_diffpixel
```

## 说明

- 训练时不强求像素风，重点是 front/back 的结构一致、服饰元素一致。
- 像素风格放在推理阶段做，能更灵活换风格控制器，不破坏一致性模型。
