# DL Project – Controllable Pixel Character Generator

A controllable pixel-art character generator built on **LPC (Universal LPC Spritesheet Character Generator)** assets and **diffusion-based generative models**.

---

## Overview

This project implements two tasks:

| Task | Description |
|------|-------------|
| **Task 1** | Random generation of a character's **front view and back view** jointly (conditional diffusion) |
| **Task 2** | Given a **front view image**, generate the corresponding **back view** (image-to-image diffusion) |

---

## Architecture

```
DL_project/
├── data/
│   ├── spritesheet_utils.py   # LPC spritesheet parsing & pair extraction
│   ├── augmentation.py        # Paired data augmentations
│   └── dataset.py             # PyTorch Dataset classes (Task 1 & 2)
├── models/
│   ├── unet.py                # Lightweight U-Net backbone with attention
│   ├── diffusion.py           # DDPM / DDIM diffusion framework
│   └── embeddings.py          # Attribute embeddings & CFG null token
├── train/
│   ├── train_task1.py         # Task 1 training script
│   └── train_task2.py         # Task 2 training script
├── inference/
│   ├── generate.py            # Random pair generation (Task 1)
│   └── reconstruct.py         # Front-to-back reconstruction (Task 2)
├── utils/
│   ├── visualization.py       # Grid/side-by-side image utilities
│   └── metrics.py             # MSE, PSNR, SSIM, histogram distance
├── configs/
│   ├── task1_config.yaml      # Training config for Task 1
│   └── task2_config.yaml      # Training config for Task 2
├── tests/                     # Unit tests
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Place your LPC spritesheet PNG files in a directory (e.g. `data/raw_sprites/`), then extract front/back pairs:

```python
from data.dataset import build_and_save_index

pairs = build_and_save_index(
    sprite_dir="data/raw_sprites/",
    out_dir="data/pairs/train/",
    index_path="data/index_train.csv",
)
```

The extractor reads the **idle frame** (column 0) from the standard LPC rows:
- Row 2 → **front view** (walk-down direction)
- Row 0 → **back view**  (walk-up direction)

---

## Training

### Task 1 – Random Front+Back Generation

```bash
python train/train_task1.py --config configs/task1_config.yaml
```

Key settings in `configs/task1_config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.in_channels` | 6 | Front(3) + Back(3) concatenated |
| `model.out_channels` | 6 | Predict noise for both views |
| `model.cond_emb_dim` | 256 | Attribute embedding size (0 = disabled) |
| `diffusion.timesteps` | 1000 | Diffusion steps |
| `diffusion.schedule` | `cosine` | Beta schedule |
| `training.p_uncond` | 0.1 | CFG null-drop probability |

### Task 2 – Front-to-Back Reconstruction

```bash
python train/train_task2.py --config configs/task2_config.yaml
```

Key settings in `configs/task2_config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.in_channels` | 6 | Noisy back(3) + front condition(3) |
| `model.out_channels` | 3 | Predict back noise only |
| `data.occlusion_p` | 0.3 | Random occlusion augmentation probability |
| `training.use_discriminator` | false | Enable PatchGAN adversarial fine-tuning |

---

## Inference

### Task 1 – Random generation

```bash
python inference/generate.py \
    --config configs/task1_config.yaml \
    --ckpt   outputs/task1/checkpoints/ckpt_epoch0099.pth \
    --n      8 \
    --steps  50 \
    --scale  3.0 \
    --out    outputs/task1/generated/
```

Options:
- `--n`: number of pairs to generate
- `--steps`: DDIM denoising steps (fewer = faster, default 50)
- `--scale`: classifier-free guidance scale (1.0 = no guidance)
- `--eta`: DDIM stochasticity (0 = deterministic)

### Task 2 – Front-to-back reconstruction

```bash
# Single image
python inference/reconstruct.py \
    --config configs/task2_config.yaml \
    --ckpt   outputs/task2/checkpoints/ckpt_epoch0099.pth \
    --input  path/to/front.png \
    --steps  50 \
    --out    outputs/task2/reconstructed/

# Batch mode (all PNGs in a directory)
python inference/reconstruct.py \
    --config configs/task2_config.yaml \
    --ckpt   outputs/task2/checkpoints/ckpt_epoch0099.pth \
    --input  path/to/front_images/ \
    --out    outputs/task2/reconstructed/
```

---

## Evaluation

```python
from utils.metrics import evaluate_pair_batch

results = evaluate_pair_batch(fronts, pred_backs, gt_backs)
# {
#   "mse":             0.0031,
#   "psnr":            25.1,
#   "ssim":            0.87,
#   "hist_pred_front": 0.12,   # palette consistency
#   "hist_pred_gt":    0.08,
# }
```

---

## Model Details

### U-Net backbone (`models/unet.py`)

- **Encoder**: `n_levels` DownBlocks, each with `n_res_blocks` ResBlocks + stride-2 downsampling.
- **Middle**: ResBlock → Self-Attention → ResBlock.
- **Decoder**: UpBlocks with bilinear upsampling + skip connections.
- **Conditioning**: sinusoidal timestep embedding + optional attribute embedding (added via scale-shift).
- **Attention**: multi-head spatial self-attention at low-resolution feature maps (configurable via `attn_resolutions`).

### Diffusion framework (`models/diffusion.py`)

- **Forward process**: `q(x_t | x_0) = N(sqrt(a_t) * x_0, (1-a_t)*I)` with cosine schedule.
- **Reverse process**: predict noise with U-Net, compute posterior mean (DDPM), or use DDIM.
- **DDIM sampling**: deterministic (eta=0) or stochastic (eta>0), ~50 NFE for good quality.
- **Classifier-Free Guidance**: `noise* = noise_unc + s*(noise_cond - noise_unc)`.

### Attribute embeddings (`models/embeddings.py`)

- Default LPC vocabulary: body, hair, hat, outfit, legs, shoes, weapon, shield.
- CFG null embedding: with probability `p_uncond`, all attributes are replaced by a learned null vector during training.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All tests run on CPU without requiring data or a trained checkpoint.

---

## Cloud Training Tips

1. **Use `mixed_precision: true`** in the config (requires CUDA) for ~2x throughput.
2. **Increase `batch_size`** (64-128) on multi-GPU machines; use `torch.nn.DataParallel` or HuggingFace `accelerate` for multi-GPU.
3. **Start with Task 1** to validate the data pipeline; Task 2 requires the same data so no extra preprocessing is needed.
4. **Checkpoints** are saved every `save_every` epochs; resume training with `--resume path/to/ckpt.pth`.
5. **TensorBoard** logs are written to `{out_dir}/logs/`; run `tensorboard --logdir outputs/` to monitor.

---

## References

- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (2020)
- Song et al., [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (2021)
- Nichol & Dhariwal, [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (2021)
- [Universal LPC Spritesheet Character Generator](https://sanderfrenken.github.io/Universal-LPC-Spritesheet-Character-Generator/)
