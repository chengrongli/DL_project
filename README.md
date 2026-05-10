# DL Project – Flow Matching LPC Character Generator

本仓库已切换为 **Flow Matching first**：Task1（前后视图联合生成）不再维护 DDPM 训练/采样路径。

## What changed

- ✅ Task1 统一使用 Flow Matching
- ✅ 训练/推理主入口：
  - `train/train_task1_flow.py`
  - `inference/generate_flow.py`
- ✅ 为兼容旧命令，保留别名入口：
  - `train/train_task1.py`（转发到 Flow Matching）
  - `inference/generate.py`（转发到 Flow Matching）
- ✅ Flow 用到的 U-Net 独立到 `models/flow_unet.py`，不再依赖 `ddpm.py`

---

## Quick start

### 1) Install

```bash README.md
pip install -r requirements.txt
```

### 2) Prepare data index

```bash README.md
python -m data_code.agent_preprocess \
  --input-dir data/pairs/random_batch \
  --output-dir data/processed/task1 \
  --image-size 64 \
  --seed 42
```

### 3) Train (Task1 Flow Matching)

```bash README.md
python train/train_task1_flow.py --config configs/task1_flow_config.yaml
```

### 4) Sample

```bash README.md
python inference/generate_flow.py \
  --config configs/task1_flow_config.yaml \
  --ckpt outputs/task1_flow/checkpoints/flow_epoch0200.pth \
  --n 16 \
  --steps 60 \
  --out outputs/task1_flow/generated/samples.png
```

---

## Repository layout (cleaned)

```text
configs/
  task1_flow_config.yaml      # canonical Task1 config
  task1_config.yaml           # compatibility alias (same flow fields)
  task2_config.yaml

models/
  flow_matching.py
  flow_unet.py                # Task1 backbone (checkpoint-compatible)
  diffusion.py                # legacy stack for Task2
  unet.py                     # legacy/Task2 U-Net

train/
  train_task1_flow.py         # canonical Task1 trainer
  train_task1.py              # compatibility alias -> flow trainer
  train_task2.py

inference/
  generate_flow.py            # canonical Task1 sampler
  generate.py                 # compatibility alias -> flow sampler
  reconstruct.py
```

---

## Notes

- `configs/task1_flow_config.yaml` 是推荐配置。
- `configs/task1_config.yaml` 仅用于兼容旧路径（已改为 flow 字段）。
- Task2 当前仍是历史扩散实现，和 Task1 的 Flow Matching 代码路径已解耦。

---

## References

- Lipman et al., *Flow Matching for Generative Modeling*, 2023.
- [Universal LPC Spritesheet Character Generator](https://sanderfrenken.github.io/Universal-LPC-Spritesheet-Character-Generator/)
