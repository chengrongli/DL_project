# DL Project – Diffusion-Based LPC Character Generator

Pixel-art character generation on **Liberated Pixel Cup (LPC)** assets, powered by diffusion models. The repository contains data extraction utilities, training/inference pipelines, and evaluation helpers for two related tasks:

| Task | Description |
|------|-------------|
| **Task 1** | Generate a **front/back pair** jointly from noise (RGBA horizontal pair). Supports both DDPM and **Flow Matching**. |
| **Task 2** | Given a **front view**, diffuse the corresponding **back view** (image-to-image). |

---

## Repository layout

```
DL_project/
├── ddpm.py                  # General-purpose DDPM config/UNet/training utilities
├── configs/
│   ├── task1_config.yaml      # Task 1 DDPM training config
│   ├── task1_flow_config.yaml # Task 1 Flow Matching config
│   └── task2_config.yaml      # Task 2 training configuration
├── data_code/               # Dataset builders and augmentation utilities
│   ├── dataset.py           # SpritePairDataset & FrontToBackDataset
│   ├── augmentation.py      # Paired sprite augmentations
│   ├── spritesheet_utils.py # LPC asset parsing helpers
│   ├── repo_extractor.py    # Extract front/back pairs from LPC repos
│   ├── repo_downloader.py   # HTTP downloader / sparse clone helper
│   ├── layer_stack.py       # Manual layer composition
│   ├── random_composer.py   # Random layered sprite composer
│   └── agent_preprocess.py  # Agent-style dataset preprocessing/splitting
├── train/
│   ├── train_task1.py       # Task 1 training (DDPM)
│   ├── train_task1_flow.py  # Task 1 training (Flow Matching)
│   └── train_task2.py       # Task 2 training (legacy diffusion stack)
├── inference/
│   ├── generate.py          # Task 1 DDPM sampling
│   ├── generate_flow.py     # Task 1 Flow Matching sampling
│   └── reconstruct.py       # Task 2 reconstruction demo
├── models/                  # Legacy diffusion implementation (Task 2)
├── utils/                   # Visualisation & metric helpers
├── tests/                   # Unit tests
└── requirements.txt
```

Task 1 supports two training routes now: DDPM (`train/train_task1.py`) and Flow Matching (`train/train_task1_flow.py`). Task 2 still depends on the original `models/` modules.

---

## Environment setup

1. **Install requirements**

    ```bash README.md
    pip install -r requirements.txt
    ```

2. **(Optional) Create a conda env**

    ```bash README.md
    conda env create -f environment.yml
    conda activate dl-pixel
    ```

---

## Data preparation

A lightweight demo split (32 random composites) lives in `data/pairs/random_batch/`. For proper training use the full asset extraction or compose your own dataset.

1. **Collect LPC assets** (clone or download the [Universal LPC repository](https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator)).
2. **Extract front/back pairs** using the helper below; the extractor reads the idle frame (column 0) with rows `walk_down` (front) and `walk_up` (back).

    ```bash README.md
    python -m data_code.repo_extractor \
        --repo-root /path/to/Universal-LPC-Spritesheet-Character-Generator \
        --out-dir data/pairs/all \
        --index   data/index_all.csv \
        --patterns "*walk.png"
    ```

3. **Compose characters manually** (optional). Layer ordered sprites with `layer_stack`:

    ```bash README.md
    python -m data_code.layer_stack \
        --assets-root data/raw_assets \
        --out-dir data/pairs/custom \
        --name hero01 \
        --layers-file configs/hero01_layers.yaml
    ```

4. **Check random-data layer base size** (recommended before generation):

    ```bash README.md
    python -m data_code.random_composer \
        --assets-root data/raw_lpc_repo \
        --out-dir data/pairs/random_batch \
        --report-only
    ```

    If `body` candidates are too few, generation diversity will be bottlenecked.

5. **Generate random composites**:

    ```bash README.md
    python -m data_code.random_composer \
        --assets-root data/raw_lpc_repo \
        --out-dir data/pairs/random_batch \
        --count 4096 \
        --seed 123 \
        --palette-shift-prob 0.8
    ```

6. **Agent preprocess dataset** (split + dedup + index):

    ```bash README.md
    python -m data_code.agent_preprocess \
        --input-dir data/pairs/random_batch \
        --output-dir data/processed/task1 \
        --image-size 64 \
        --val-ratio 0.1 \
        --seed 42
    ```

7. **Visualize preprocessed data**:

    ```bash README.md
    python -m utils.visualize_preprocessed \
        --index data/processed/task1/index_train.csv \
        --out outputs/data_preview/task1_train_grid.png \
        --limit 36
    ```

All scripts emit `(front.png, back.png)` pairs compatible with `SpritePairDataset`.

---

## Training

### Task 1 – Joint front/back generation

#### Option A: DDPM

```bash README.md
python train/train_task1.py --config configs/task1_config.yaml
```

#### Option B: Flow Matching

```bash README.md
python train/train_task1_flow.py --config configs/task1_flow_config.yaml
```

Highlights from `configs/task1_config.yaml` (DDPM):

| Key | Meaning |
|-----|---------|
| `model.in_channels=4` | RGBA channels for a horizontally concatenated pair |
| `model.model_channels=64` | Base width (UNet depth adjusts via `channel_mults`) |
| `diffusion.timesteps=1000` | Linear schedule from `ddpm.py` (β₀=1e-4 → β_T=2e-2) |
| `training.batch_size=32` | Increase if memory allows; script auto-detects CUDA/MPS |
| `training.ema_decay=0.9999` | EMA applied every iteration to a shadow DDPM model |
| `training.sample_every=10` | Saves EMA samples (clamped to [-1,1]) to `outputs/task1/samples/` |

The script supports `--resume` checkpoints produced by the new DDPM loop. Logs are written to `outputs/task1/logs/` (TensorBoard compatible).

### Task 2 – Front-to-back diffusion (legacy)

Task 2 still runs on the original diffusion stack under `models/`:

```bash README.md
python train/train_task2.py --config configs/task2_config.yaml
```

Its configuration retains classifier-free guidance embeddings and optional PatchGAN fine-tuning. Once Task 1 stabilises with `ddpm.py`, the same refactor can be applied here.

---

## Inference & sampling

### Task 1 sampling

DDPM:

```bash README.md
python inference/generate.py \
    --config configs/task1_config.yaml \
    --ckpt   outputs/task1/checkpoints/ddpm_epoch0100.pth \
    --n      8 \
    --steps  50 \
    --scale  2.5 \
    --out    outputs/task1/generated
```

Flow Matching:

```bash README.md
python inference/generate_flow.py \
    --config configs/task1_flow_config.yaml \
    --ckpt   outputs/task1_flow/checkpoints/flow_epoch0200.pth \
    --n      16 \
    --steps  60 \
    --out    outputs/task1_flow/generated/samples.png
```

### Task 2 reconstruction

```bash README.md
python inference/reconstruct.py \
    --config configs/task2_config.yaml \
    --ckpt   outputs/task2/checkpoints/ckpt_epoch0099.pth \
    --input  samples/front.png \
    --steps  50 \
    --out    outputs/task2/reconstructed
```

Pass a directory to `--input` for batch processing. Optional `--compare` flag saves side-by-side grids with ground truth when available.

---

## Evaluation

Quantitative metrics (MSE, PSNR, SSIM, palette histograms) live under `utils/metrics.py`.

```python README.md
from utils.metrics import evaluate_pair_batch

metrics = evaluate_pair_batch(fronts, backs_pred, backs_gt)
print(metrics)
# {'mse': 3.2e-3, 'psnr': 25.0, 'ssim': 0.86, 'hist_pred_front': 0.11, ...}
```

Combine these with qualitative grids from `utils.visualization.save_sample_grid` for tracking progress.

---

## Troubleshooting & tips

1. **Data sanity** – ensure every front has a matching back. `SpritePairDataset` raises if pairs are missing.
2. **Mixed precision** – enable via `training.mixed_precision: true` (CUDA only) to double throughput.
3. **Memory** – reduce `batch_size` or UNet width (`model_channels`) on smaller GPUs; the DDPM UNet uses GroupNorm, so per-batch statistics remain stable.
4. **EMA checkpoints** – inference scripts prefer EMA weights; if absent, they fall back to raw weights.
5. **TensorBoard** – launch `tensorboard --logdir outputs/` for live loss curves and sample previews.

---

## References

- Ho et al., *Denoising Diffusion Probabilistic Models*, 2020.
- Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models*, 2021.
- Song et al., *Denoising Diffusion Implicit Models*, 2021.
- [Universal LPC Spritesheet Character Generator](https://sanderfrenken.github.io/Universal-LPC-Spritesheet-Character-Generator/)
