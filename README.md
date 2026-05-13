# DL Project — LPC Character Generator

基于 [Universal LPC Spritesheet Character Generator](https://sanderfrenken.github.io/Universal-LPC-Spritesheet-Character-Generator/)
的像素角色生成项目，包含两个主任务 + 一个高清一致性实验路径：

- **Task 1 — 前后视图联合生成（Flow Matching）**：从噪声同时生成一对配对的正面 / 背面像素角色。
- **Task 2 — 前视图到背视图重建（Conditional Diffusion）**：给定正面图，条件扩散模型预测对应的背面图。
- **HD Consistency + Pixelization（实验）**：先在高清 front/back 数据上训练正背一致性 LoRA，再在推理阶段外挂 ControlNet-Tile 或可微像素化层。见 `hd_consistency_pixel/README.md`。

下面的步骤假设你拿到的是一个**只包含代码、不包含数据和权重**的全新仓库，按顺序跑就能从零到训练完成。

---

## 目录

- [0. 环境准备](#0-环境准备)
- [1. 准备原始数据（front / back PNG 配对）](#1-准备原始数据front--back-png-配对)
- [2. 数据预处理（生成训练用 CSV 索引）](#2-数据预处理生成训练用-csv-索引)
- [3. Task 1 — 训练 & 采样](#3-task-1--训练--采样)
- [4. Task 2 — 训练 & 推理](#4-task-2--训练--推理)
- [5. （可选）Web Demo](#5-可选web-demo)
- [6. 仓库结构](#6-仓库结构)
- [7. 常见问题](#7-常见问题)

---

## 0. 环境准备

推荐 **Python ≥ 3.10 + CUDA 11.8**（在 A 系列 / 30 系 / 40 系卡都验证过）。

### 用 conda

```bash
conda env create -f environment.yml
conda activate oenv
```

---

## 1. 准备原始数据（front / back PNG 配对）

两个任务**共享同一份原始数据**——每个角色一组 `*_front.png` / `*_back.png`，64×64 RGBA。

### 从 LPC 上游仓库稀疏克隆 + 随机合成

1. **稀疏克隆 LPC 素材仓库**（只下身体 / 头发 / 衣服等需要的目录）：

   ```bash
   python -m data_code.repo_sparse_clone \
     --dest data/raw_lpc_repo \
     --depth 1
   ```

   完成后 `data/raw_lpc_repo/spritesheets/` 下会有 body / head / hair / torso / … 等子目录。

2. **随机合成 N 个角色**，自动抽取每张 spritesheet 第 0 列的 idle front（行 2）/ back（行 0）：

   ```bash
   python -m data_code.random_composer \
     --assets-root data/raw_lpc_repo \
     --out-dir     data/pairs/random_batch \
     --count       30000 \
     --seed        42 \
     --num-workers 0          # 0 = 自动用 (CPU-1) 个进程
   ```

   产物：`data/pairs/random_batch/char_0000_front.png`、`char_0000_back.png`、…

   > 想先看看素材池规模，可以加 `--report-only` 只打印每组层数。

### 从已有 LPC 仓库抽取每张 spritesheet 的 idle 帧

适用于你已经手动 clone 了 LPC 仓库、或者只想要“原版 spritesheet 的 front/back”而不是随机合成：

```bash
python -m data_code.repo_extractor \
  --repo-root data/raw_lpc_repo \
  --out-dir   data/pairs/lpc_repo \
  --index     data/index_lpc_repo.csv \
  --patterns  "*walk.png"
```

---

## 2. 数据预处理（生成训练用 CSV 索引）

无论你用 A / B / C 哪种方式拿到的 `*_front.png` / `*_back.png`，都需要跑一次预处理：
统一 resize 到 64×64、可选去重、并写出 `index_train.csv` / `index_val.csv`。

### Task 1 用的索引

```bash
python -m data_code.agent_preprocess \
  --input-dir  data/pairs/random_batch \
  --output-dir data/processed/task1 \
  --image-size 64 \
  --val-ratio  0.1 \
  --seed       42
```

### Task 2 用的索引

Task 2 可以**直接复用 Task 1 的索引**（两者读的都是 `front_path,back_path` 两列 CSV），
也可以单独再生成一份保持目录干净：

```bash
python -m data_code.agent_preprocess \
  --input-dir  data/pairs/random_batch \
  --output-dir data/processed/task2 \
  --image-size 64 \
  --val-ratio  0.1 \
  --seed       42
```

> 如果想复用 Task 1 的 csv，把 `configs/task2_config.yaml` 里 `data.train_dir` /
> `data.val_dir` 改成 `data/processed/task1/index_train.csv` 等即可。

预处理产物（每个 `output-dir` 下）：

```text
data/processed/taskX/
├── index_train.csv       # 训练集索引（第一列 front_path，第二列 back_path）
├── index_val.csv         # 验证集索引
├── index_all.csv
├── preprocess_stats.json
├── train_000000_front.png
├── train_000000_back.png
├── ...
```

### （可选）肉眼看一下预处理结果

```bash
python -m utils.visualize_preprocessed \
  --index data/processed/task1/index_train.csv \
  --out   outputs/preview_task1.png \
  --limit 32
```

---

## 3. Task 1 — 训练 & 采样

**模型**：4 通道（RGBA）UNet + Flow Matching。把同一个角色的 front 和 back 在宽度方向拼成
`(4, H, 2H)` 一张图直接生成，避免把 front/back 强行塞进通道维度。

### 3.1 训练

```bash
python train/train_task1.py --config configs/task1_config.yaml
```

默认配置（可在 `configs/task1_config.yaml` 调整）：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `data.image_size` | 64 | 单边图像尺寸 |
| `model.in_channels / out_channels` | 4 / 4 | RGBA |
| `model.channel_mults` | `[1,2,4,8]` | UNet 通道倍率 |
| `flow.time_scale` | 999.0 | Flow Matching 时间缩放 |
| `flow.sample_steps` | 60 | 采样积分步数 |
| `training.epochs` | 200 | |
| `training.batch_size` | 128 | |
| `training.lr` | 1e-4 | AdamW + cosine 退火到 1e-6 |
| `training.ema_decay` | 0.999 | 推理用 EMA 权重 |
| `training.foreground_weight` | 6.0 | 前景像素权重 |
| `training.background_weight` | 0.6 | 背景像素权重 |
| `training.alpha_weight` | 0.8 | alpha 通道权重 |
| `training.out_dir` | `outputs/task1_flow` | 产物目录 |

训练产物：

```text
outputs/task1_flow/
├── checkpoints/flow_epoch0050.pth
├── checkpoints/flow_epoch0100.pth
├── ...
├── samples/sample_epoch0050.png   # 训练中阶段性可视化
├── samples/...
└── logs/                          # TensorBoard 日志
```

实时观察训练曲线：

```bash
tensorboard --logdir outputs/task1_flow/logs
```

支持断点续训：

```bash
python train/train_task1.py \
  --config configs/task1_config.yaml \
  --resume outputs/task1_flow/checkpoints/flow_epoch0100.pth
```

### 3.2 采样

```bash
python inference/generate.py \
  --config configs/task1_config.yaml \
  --ckpt   outputs/task1_flow/checkpoints/flow_epoch0200.pth \
  --n      16 \
  --steps  60 \
  --out    outputs/task1_flow/generated/samples.png
```

产物是 4×4 的网格图，每个 cell 是同一角色的 `[front | back]` 横向拼图。

---

## 4. Task 2 — 训练 & 推理

**模型**：6 通道输入的 conditional UNet + DDPM/DDIM。
正面图（3ch）和加噪后的背面图（3ch）在通道维拼接进 UNet，UNet 输出 3ch 噪声预测。

### 4.1 训练

```bash
python train/train_task2.py --config configs/task2_config.yaml
```

默认配置（`configs/task2_config.yaml`）：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `data.image_size` | 64 | |
| `data.occlusion_p` | 0.3 | 训练时正面图随机遮挡概率 |
| `model.in_channels / out_channels` | 6 / 3 | noisy_back(3) + front_cond(3) → noise(3) |
| `diffusion.timesteps` | 1000 | |
| `diffusion.schedule` | `cosine` | β schedule |
| `diffusion.ddim_steps` | 50 | 验证 / 推理用 DDIM 步数 |
| `training.epochs` | 200 | |
| `training.batch_size` | 64 | |
| `training.lr` | 2e-4 | AdamW + cosine 退火 |
| `training.mixed_precision` | true | CUDA AMP |
| `training.fg_weight / bg_weight` | 6.0 / 0.5 | 前 / 背景像素加权 |
| `training.color_weight` | 1.0 | 前-背颜色一致性约束 |
| `training.use_discriminator` | false | 可选 PatchGAN 微调（diff 收敛后再开） |
| `training.out_dir` | `outputs/task2` | 产物目录 |

训练产物：

```text
outputs/task2/
├── checkpoints/ckpt_epoch0009.pth
├── checkpoints/...
├── samples/sample_epoch0009.png    # 上半行 cond 正面 / 下半行预测背面
├── samples/...
└── logs/
```

> **数据通道一致性自检**：`train_task2.py` 启动时会用 dataset 的第一个样本对一遍
> `model.in_channels / out_channels`，不匹配会直接报错而不是训到一半才 NaN。

断点续训：

```bash
python train/train_task2.py \
  --config configs/task2_config.yaml \
  --resume outputs/task2/checkpoints/ckpt_epoch0099.pth
```

### 4.2 推理（前 → 后重建）

单张：

```bash
python inference/reconstruct.py \
  --config configs/task2_config.yaml \
  --ckpt   outputs/task2/checkpoints/ckpt_epoch0199.pth \
  --input  path/to/front.png \
  --steps  50 \
  --out    outputs/task2/reconstructed/
```

整目录批量处理：

```bash
python inference/reconstruct.py \
  --config configs/task2_config.yaml \
  --ckpt   outputs/task2/checkpoints/ckpt_epoch0199.pth \
  --input  path/to/front_images/ \
  --out    outputs/task2/reconstructed/
```

每张图会输出：

- `{stem}_pred_back.png`：模型预测的背面
- `{stem}_side_by_side.png`：`[输入 front | 预测 back]` 横向拼图

> 推理脚本会先做角点 floodfill 去掉纯色背景 + alpha 预乘，与训练数据一致；
> 所以你直接喂带白底/灰底的截图也能用，但带 alpha 的 PNG 效果最好。

---

## 5. （可选）Web Demo

仓库自带一个 Flask demo，可以在浏览器里上传正面图、调用 Task 2 生成背面：

```bash
# 默认会加载 outputs/task2/checkpoints/ckpt_epoch0199.pth
python app.py
# 打开 http://localhost:5000
```

如果你训练时改了 epoch 数 / 输出目录，按需修改 `app.py` 顶部的 `CKPT_PATH`。

---

## 6. 仓库结构

```text
configs/
├── task1_config.yaml          # Task1 (Flow Matching) 配置
└── task2_config.yaml          # Task2 (Conditional Diffusion) 配置

data_code/
├── repo_sparse_clone.py       # 稀疏克隆 LPC 上游仓库
├── repo_downloader.py         # 备用：HTTP 下载工具
├── repo_extractor.py          # 从 LPC 仓库抽 idle front/back
├── random_composer.py         # 随机合成完整角色 → front/back 配对
├── layer_stack.py             # 给定层 YAML 手动叠加
├── spritesheet_utils.py       # spritesheet → front/back 帧切分
├── agent_preprocess.py        # 生成 index_train/val.csv
├── augmentation.py            # 翻转 / 颜色扰动 / 遮挡 / palette shift
├── dataset.py                 # SpritePairDataset / FrontToBackDataset
└── convert_data_other.py      # 备用：其它格式转换

models/
├── flow_matching.py           # Task1 — Flow Matching 框架
├── flow_unet.py               # Task1 — 4ch RGBA UNet
├── diffusion.py               # Task2 — GaussianDiffusion (DDPM/DDIM)
├── unet.py                    # Task2 — 6ch 条件 UNet
├── unet_flow.py               # 兼容旧实现（保留）
└── embeddings.py              # 可选属性 embedding

train/
├── train_task1.py             # Task1 训练入口（Flow Matching）
└── train_task2.py             # Task2 训练入口（Cond. Diffusion）

inference/
├── generate.py                # Task1 采样（转发到 Flow 采样器）
├── generate_flow.py           # Task1 采样实现
└── reconstruct.py             # Task2 前→后重建

utils/
├── visualization.py           # tensor → PIL / grid / side-by-side
├── visualize_preprocessed.py  # 预处理后数据集可视化
└── metrics.py

app.py                          # 可选 Flask demo
static/                         # demo 前端
```

---

## 7. 常见问题

**Q1. `FileNotFoundError: data_source must be an existing file or directory: data/processed/task2/index_train.csv`**

还没跑过 [第 2 步](#2-数据预处理生成训练用-csv-索引) 的 `agent_preprocess`。

**Q2. `Input dataset does not exist: .../data/pairs/random_batch`**

[第 1 步](#1-准备原始数据front--back-png-配对) 还没生成原始 pair。
机器上没数据就走方式 B（稀疏克隆 + 随机合成）；已经有就把 `--input-dir`
改成你 `*_front.png` 实际所在的目录。

**Q3. `Model/data channel mismatch: config in/out=(...), but dataset implies (...)`**

`task2_config.yaml` 里 `model.in_channels` 必须等于 `cond 通道 + target 通道`，
默认值 6 / 3 已经对应 RGB 数据集；只有你改了 dataset 输出通道才会触发。

**Q4. CUDA OOM**

- Task 1：把 `training.batch_size` 从 128 调到 64 / 32。
- Task 2：把 `training.batch_size` 从 64 调到 32，关 `mixed_precision` 或减小 `model_channels`。

**Q5. 想用更小的数据快速验证 pipeline**

预处理时加 `--max-samples 2000`，训练配置里把 `epochs` 改到 20、`save_every` / `sample_every` 改到 5。

**Q6. MPS / Apple Silicon**

`training.device: auto` 会自动选 MPS。Task 2 的 `mixed_precision` 只对 CUDA 生效，
MPS 下保持 `false` 即可。

---

## References

- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023.
- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
- Song et al., *Denoising Diffusion Implicit Models*, ICLR 2021.
- [Universal LPC Spritesheet Character Generator](https://sanderfrenken.github.io/Universal-LPC-Spritesheet-Character-Generator/)
