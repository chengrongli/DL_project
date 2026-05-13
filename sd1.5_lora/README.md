# MyDistill

本项目用于训练并蒸馏像素风 RPG 角色正反面配对图（左正面、右背面）。

## 环境

- Python 3.x
- 可参考 requirement.txt 创建环境(注意不要使用oenv虚拟环境，建一个新的)
```bash
pip install -r requirement.txt
```

## 数据准备

- 原始数据放在 `data/pairs/random_batch`
- 文件命名要求：`xxx_front.png` 与 `xxx_back.png` 成对存在

## 脚本说明

### 1) preprocess.py

将同名前后视图拼接成 1 张宽度翻倍的训练图，输出到 `pairs/processed`。

- 输入：`gen/DL_project_final/data/pairs/random_batch`
- 输出：`pairs/processed`
- 规则：自动匹配 `_front` 与 `_back`，生成 `_paired`

### 2) train_lora.py

使用 `pairs/processed` 训练 LoRA，并保存到 `lora_out`。

- 默认模型：`Lykon/AnyLoRA`
- 默认提示词：见脚本内 `DEFAULT_PROMPT`
- 输入尺寸：宽 512 × 高 256
- LoRA rank 默认 8

常用参数（可选）：

- `--data_dir`：训练数据目录
- `--base_model`：基础模型
- `--output_dir`：LoRA 输出目录
- `--prompt`：统一提示词
- `--height` / `--width`：训练图尺寸
- `--train_batch_size`、`--gradient_accumulation_steps`、`--learning_rate`
- `--max_train_steps`、`--num_train_epochs`
- `--rank`、`--mixed_precision`、`--seed`

### 3) distill.py

加载基础模型与 LoRA，批量生成配对图并降采样到 64×64。

- 输入：`lora_out`
- 输出：
  - `distilled/pair_XXXX_front.png`
  - `distilled/pair_XXXX_back.png`
  - `distilled/pair_XXXX_lora_preview.png`

内置流程包含：

- 白底转透明
- 强化对比并降采样
- 正背面调色板锁定（可通过 `USE_COLOR_LOCK` 开关）

## 运行方式

```bash
python preprocess.py
python train_lora.py \
  --train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 3000 \
  --learning_rate 2e-5 \
  --mixed_precision fp16
python distill.py
```

## 推荐流程

1. 准备原始正背面素材到 `data/pairs/random_batch`
2. 运行 `preprocess.py` 生成训练对
3. 运行 `train_lora.py` 训练 LoRA
4. 运行 `distill.py` 批量生成蒸馏数据