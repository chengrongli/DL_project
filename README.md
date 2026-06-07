# 像素精灵生成器 — LPC Character Generator

基于 [Universal LPC Spritesheet Character Generator](https://sanderfrenken.github.io/Universal-LPC-Spritesheet-Character-Generator/)
的像素角色生成项目，使用 **条件流匹配（Conditional Flow Matching）** 同时生成角色的正面 / 背面像素精灵图，
并支持通过属性向量控制体型、发型、服装类型与颜色。

## 功能亮点

- ✨ **属性条件生成**：指定体型、发型、上衣/下装/鞋子的类型与颜色，模型生成对应角色
- 🎲 **随机生成**：一键生成随机角色
- 🎯 **无分类器引导（CFG）**：训练时随机丢弃条件，推理时可调节引导强度
- 🧠 **FiLM + 交叉注意力**：双路径属性注入，全局调制 + 局部空间交互
- 🖼️ **正反面联合生成**：64×128 输出，一次生成完整的正反面配对

---

## 目录

- [0. 环境准备](#0-环境准备)
- [1. 准备原始数据](#1-准备原始数据)
- [2. 数据预处理](#2-数据预处理)
- [3. 训练](#3-训练)
- [4. 推理 & 采样](#4-推理--采样)
- [5. Web 演示](#5-web-演示)
- [6. 仓库结构](#6-仓库结构)
- [7. 常见问题](#7-常见问题)

---

## 0. 环境准备

推荐 **Python ≥ 3.10 + CUDA 11.8+**（RTX 30/40 系列显卡均已验证）。

```bash
conda env create -f environment.yml
conda activate oenv
```

---

## 1. 准备原始数据

### 从 LPC 上游仓库稀疏克隆 + 随机合成

1. **稀疏克隆 LPC 素材仓库**（只下载身体 / 头发 / 衣服等需要的目录）：

   ```bash
   python -m data_code.repo_sparse_clone \
     --dest data/raw_lpc_repo \
     --depth 1
   ```

2. **随机合成 N 个角色**，自动抽取每张 spritesheet 第 0 列的 idle front（行 2）/ back（行 0）：

   ```bash
   python -m data_code.random_composer \
     --assets-root data/raw_lpc_repo \
     --out-dir     data/pairs/random_batch \
     --count       30000 \
     --seed        42 \
     --num-workers 0
   ```

   > `num-workers 0` = 单进程（更强去重与缓存复用）。加 `--report-only` 可先查看素材池规模。

### 从已有 LPC 仓库抽取

```bash
python -m data_code.repo_extractor \
  --repo-root data/raw_lpc_repo \
  --out-dir   data/pairs/lpc_repo \
  --index     data/index_lpc_repo.csv \
  --patterns  "*walk.png"
```

---

## 2. 数据预处理

生成训练用 `index_train.csv` / `index_val.csv` 及 `attributes.json`：

```bash
python -m data_code.agent_preprocess \
  --input-dir  data/pairs/random_batch \
  --output-dir data/pairs/random_batch_v3 \
  --image-size 64 \
  --val-ratio  0.1 \
  --seed       42
```

预处理产物：

```text
data/pairs/random_batch_v3/
├── index_train.csv       # 训练集索引
├── index_val.csv         # 验证集索引
├── attributes.json       # 每个角色的属性标注
├── char_0000_front.png
├── char_0000_back.png
└── ...
```

> `attributes.json` 包含每个角色的 `body_type`、`hair_style`、`torso_type`、`torso_color`、
> `legs_type`、`legs_color`、`feet_type`、`feet_color` 共 8 个属性字段。

---

## 3. 训练

**模型架构**：4 通道（RGBA）UNet + Flow Matching + 属性编码器（FiLM + 交叉注意力）。

将角色的 front 和 back 在宽度方向拼成 `(4, 64, 128)` 一张图直接生成，
属性向量通过 FiLM 调制和交叉注意力注入 UNet。

```bash
python train/train_conditional.py --config configs/random_batch_v2_config.yaml
```

### 训练配置

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `data.image_size` | 64 | 单边图像尺寸 |
| `model.in_channels / out_channels` | 4 / 4 | RGBA |
| `model.channel_mults` | `[1,2,4,8]` | UNet 通道倍率 |
| `model.cross_attn_heads` | 4 | 交叉注意力头数 |
| `attributes.embed_dim` | 64 | 属性嵌入维度 |
| `attributes.output_dim` | 256 | FiLM 条件向量维度 |
| `attributes.token_dim` | 256 | 交叉注意力 token 维度 |
| `flow.time_scale` | 999.0 | Flow Matching 时间缩放 |
| `flow.sample_steps` | 60 | 采样积分步数 |
| `training.epochs` | 100 | |
| `training.batch_size` | 64 | |
| `training.lr` | 1e-4 | AdamW + cosine 退火到 1e-6 |
| `training.ema_decay` | 0.999 | 推理用 EMA 权重 |
| `training.cfg_dropout` | 0.2 | 条件丢弃概率（训练 CFG） |
| `training.guidance_scale` | 3.0 | 推理时引导强度 |

训练产物：

```text
outputs/random_batch_v3/
├── checkpoints/flow_epoch0025.pth
├── checkpoints/flow_epoch0050.pth
├── ...
├── samples/sample_epoch0025.png   # 训练中阶段性可视化
└── logs/                          # TensorBoard 日志
```

实时观察训练曲线：

```bash
tensorboard --logdir outputs/random_batch_v3/logs
```

断点续训：

```bash
python train/train_conditional.py \
  --config configs/random_batch_v2_config.yaml \
  --resume outputs/random_batch_v3/checkpoints/flow_epoch0050.pth
```

---

## 4. 推理 & 采样

### 命令行批量采样

```bash
python inference/generate_conditional.py \
  --config configs/random_batch_v2_config.yaml \
  --ckpt   outputs/random_batch_v3/checkpoints/flow_epoch0100.pth \
  --n      16 \
  --steps  60 \
  --out    outputs/generated/samples.png
```

### 属性控制采样

可以通过 `--attrs` 参数指定属性：

```bash
python inference/generate_conditional.py \
  --config configs/random_batch_v2_config.yaml \
  --ckpt   outputs/random_batch_v3/checkpoints/flow_epoch0100.pth \
  --attrs '{"body_type": "female", "hair_style": "long", "torso_type": "clothes", "torso_color": "red"}' \
  --guidance-scale 5.0 \
  --n 4 \
  --out outputs/generated/controlled.png
```

---

## 5. Web 演示

内置 Flask Web 界面，支持浏览器端交互式生成：

```bash
python app.py
# 打开 http://localhost:6006
```

### 功能

- **属性生成**：选择体型、发型、服装类型与颜色，调节引导强度和生成数量
- **随机生成**：一键生成随机角色
- **高级选项**：可调引导强度（CFG Scale）、生成数量、随机种子

### 属性词汇表

| 属性 | 可选值 |
| --- | --- |
| 体型 | 男性、女性、少年、儿童、壮硕、成人 |
| 发型 | 短发、中发、长发、马尾、编辫、卷发、刺头、刘海、双马尾、脏辫、凌乱、分头、丸子头 |
| 上衣 | 普通上衣、夹克、盔甲 |
| 下装 | 长裤、短裤、裙子、连衣裙、紧身裤、盔甲 |
| 鞋子 | 靴子、鞋子、凉鞋、盔甲 |
| 颜色 | 黑、白、灰、棕、红、粉、橙、黄、绿、青、蓝、紫、金、银、铜 |

> 如果训练时修改了输出目录或 epoch 数，按需修改 `app.py` 顶部的 `CKPT_PATH`。

---

## 6. 仓库结构

```text
configs/
├── random_batch_v2_config.yaml   # 条件 Flow Matching 训练配置
├── conditional_config.yaml       # 条件 Flow Matching 配置（变体）
└── task1_config.yaml             # 旧版无条件 Flow Matching 配置

data_code/
├── repo_sparse_clone.py          # 稀疏克隆 LPC 上游仓库
├── repo_downloader.py            # 备用：HTTP 下载工具
├── repo_extractor.py             # 从 LPC 仓库抽 idle front/back
├── random_composer.py            # 随机合成完整角色 → front/back 配对
├── layer_stack.py                # 给定层 YAML 手动叠加
├── spritesheet_utils.py          # spritesheet → front/back 帧切分
├── agent_preprocess.py           # 生成 index_train/val.csv + attributes.json
├── augmentation.py               # 翻转 / 颜色扰动 / 遮挡
├── dataset.py                    # SpritePairDataset（支持属性加载）
└── convert_data_other.py         # 备用：其它格式转换

models/
├── flow_matching.py              # Flow Matching 框架（支持条件 / 无条件）
├── flow_unet.py                  # UNet（FiLM 调制 + 交叉注意力）
├── attribute_encoder.py          # 属性编码器（嵌入 → FiLM + tokens）
├── diffusion.py                  # 旧版 DDPM/DDIM（保留）
├── unet.py                       # 旧版条件 UNet（保留）
└── embeddings.py                 # 可选属性 embedding

train/
├── train_conditional.py          # 条件 Flow Matching 训练入口
└── train_task1.py                # 旧版无条件训练入口（保留）

inference/
├── generate_conditional.py       # 条件采样（支持属性控制）
├── generate_flow.py              # 无条件采样
└── eval_nearest_neighbor.py      # 最近邻评估

app.py                            # Flask Web 演示
static/                           # Web 前端（HTML / CSS / JS）
```

---

## 7. 常见问题

**Q1. `FileNotFoundError: .../index_train.csv`**

还没跑过 [第 2 步](#2-数据预处理) 的 `agent_preprocess`。

**Q2. `Input dataset does not exist: .../data/pairs/random_batch`**

[第 1 步](#1-准备原始数据) 还没生成原始配对数据。按步骤先跑 `repo_sparse_clone` + `random_composer`。

**Q3. CUDA OOM**

把 `training.batch_size` 从 64 调到 32 / 16。

**Q4. 生成结果裸体太多**

- 使用「属性生成」标签页，手动选择上衣类型（如"普通上衣"）
- 提高引导强度（CFG Scale）到 5.0 以上
- 随机生成是无条件模式，受训练数据分布影响

**Q5. Web 页面下拉框只有"随机"**

确保通过 Flask 服务器访问（`http://localhost:6006`），不要直接打开 `static/index.html` 文件。
静态文件路径必须走 `/static/` 前缀。

**Q6. 想用更小的数据快速验证 pipeline**

预处理时加 `--max-samples 2000`，训练配置里把 `epochs` 改到 20、`save_every` / `sample_every` 改到 5。

---

## 技术架构

- **Flow Matching**：基于最优传输路径的生成模型，直接回归速度场
- **FiLM 调制**：将属性向量注入 UNet 每个卷积块，实现通道级条件控制
- **交叉注意力**：属性 token 序列与空间特征交互，学习属性-位置对应
- **无分类器引导（CFG）**：训练时随机丢弃条件，推理时放大条件信号

## 数据来源

训练数据由 [Universal LPC Spritesheet Character Generator](https://github.com/sanderfrenken/Universal-LPC-Spritesheet-Character-Generator)
的资产自动合成，包含 20,000 对正反面配对精灵图。

## 模型参数

| 参数 | 值 |
| --- | --- |
| 图像尺寸 | 64 × 64（正面+反面: 64 × 128） |
| 通道数 | 4（RGBA） |
| 采样步数 | 60（Euler ODE） |
| 属性维度 | 8（体型、发型、上衣×2、下装×2、鞋子×2） |
| 训练轮次 | 100 |

## References

- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023.
- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
- Song et al., *Denoising Diffusion Implicit Models*, ICLR 2021.
- Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018.
- Ho & Salimans, *Classifier-Free Diffusion Guidance*, NeurIPS Workshop 2022.
- [Universal LPC Spritesheet Character Generator](https://sanderfrenken.github.io/Universal-LPC-Spritesheet-Character-Generator/)
