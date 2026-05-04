"""
PyTorch Dataset classes for both LPC diffusion tasks.

Task 1 – SpritePairDataset
    Returns (front_tensor, back_tensor) pairs for unconditional / attribute-
    conditioned front+back generation.

Task 2 – FrontToBackDataset
    Returns (front_tensor, back_tensor) pairs where front is the conditioning
    image and back is the reconstruction target.

Both datasets accept a pre-built index file (CSV/text) or a root directory
that is scanned at construction time.
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from data_code.augmentation import (
    random_color_jitter,
    random_horizontal_flip,
    random_occlusion,
    to_tensor_pair,
)


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def _load_index(index_path: str) -> List[Tuple[str, str]]:
    """
    Load a CSV index with columns [front_path, back_path].
    Lines starting with '#' are skipped.
    """
    pairs: List[Tuple[str, str]] = []
    with open(index_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) >= 2:
                pairs.append((row[0].strip(), row[1].strip()))
    return pairs


def _scan_directory(root: str) -> List[Tuple[str, str]]:
    """
    Scan root for *_front.png / *_back.png pairs created by
    spritesheet_utils.build_pair_dataset.
    """
    front_files = sorted(Path(root).rglob("*_front.png"))
    pairs: List[Tuple[str, str]] = []
    for ff in front_files:
        bf = Path(str(ff).replace("_front.png", "_back.png"))
        if bf.exists():
            pairs.append((str(ff), str(bf)))
    return pairs


def _write_index(pairs: List[Tuple[str, str]], out_path: str) -> None:
    """Write a CSV index of (front_path, back_path) pairs."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# front_path", "back_path"])
        writer.writerows(pairs)


# ---------------------------------------------------------------------------
# Base dataset
# ---------------------------------------------------------------------------

class _BaseSpriteDataset(Dataset):
    """
    Base dataset that loads front/back image pairs from an index or directory.
    """

    def __init__(
        self,
        data_source: str,
        image_size: int = 64,
        augment: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            data_source: Either a CSV index file or a directory containing
                         *_front.png / *_back.png pairs.
            image_size:  Output spatial resolution.
            augment:     Whether to apply random augmentations.
            transform:   Optional additional transform applied to (front, back)
                         tensors after the default pipeline.
        """
        if os.path.isfile(data_source):
            self.pairs = _load_index(data_source)
        elif os.path.isdir(data_source):
            self.pairs = _scan_directory(data_source)
        else:
            raise FileNotFoundError(
                f"data_source must be an existing file or directory: {data_source}"
            )

        if not self.pairs:
            raise RuntimeError(f"No front/back pairs found in: {data_source}")

        self.image_size = image_size
        self.augment = augment
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_pair(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        front_path, back_path = self.pairs[idx]
        front = Image.open(front_path).convert("RGBA")
        back = Image.open(back_path).convert("RGBA")
        return front, back

    def _apply_augment(
        self,
        front: Image.Image,
        back: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        front, back = random_horizontal_flip(front, back, p=0.5)
        front, back = random_color_jitter(front, back)
        return front, back

    def _to_tensor(
        self,
        front: Image.Image,
        back: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return to_tensor_pair(front, back, size=self.image_size)


# ---------------------------------------------------------------------------
# Task 1 – Paired front+back generation
# ---------------------------------------------------------------------------

class SpritePairDataset(_BaseSpriteDataset):
    """
    Dataset for Task 1: unconditional / attribute-conditioned joint
    front+back generation.

    数据布局：front 和 back 水平拼接为一张 2:1 的大图，包含 4 通道（RGBA）。
    这样 UNet 的输入/输出统一为 (4, H, 2W)，避免了 6 通道版本那种
    将 front/back “塞到通道维度”的不自然方式。

    __getitem__ returns a dict:
        "front":  (4, H, W) float32 in [−1, 1]
        "back":   (4, H, W) float32 in [−1, 1]
        "paired": (4, H, 2W) 水平拼接的训练输入
        "mask":   (1, H, 2W) 二值前景 mask（主要用于诊断、或可选加权）
    """

    def __getitem__(self, idx: int):  # type: ignore[override]
        front_img, back_img = self._load_pair(idx)

        if self.augment:
            front_img, back_img = self._apply_augment(front_img, back_img)

        front_t, back_t, front_alpha, back_alpha = self._to_tensor(front_img, back_img)

        if self.transform is not None:
            front_t, back_t = self.transform(front_t, back_t)

        # 水平拼接成 (4, H, 2W)
        paired = torch.cat([front_t, back_t], dim=2)
        # alpha mask 也水平拼接，维度 (1, H, 2W)
        mask = torch.cat([front_alpha, back_alpha], dim=2)

        return {
            "front": front_t,
            "back": back_t,
            "paired": paired,
            "mask": mask,
        }


# ---------------------------------------------------------------------------
# Task 2 – Front-to-back reconstruction
# ---------------------------------------------------------------------------

class FrontToBackDataset(_BaseSpriteDataset):
    """
    Dataset for Task 2: image-to-image conditional diffusion where the
    front view is the conditioning signal and the back view is the target.

    注意：底层解码仍走 RGBA 清洗流程（用于稳健处理 PNG 透明背景），
    但 Task 2 训练默认只使用 RGB 三通道；alpha 单独作为 ``target_alpha``
    返回用于可选加权/评估。

    __getitem__ returns a dict:
        "condition":    (3, H, W)  front RGB in [−1, 1] — model input
        "target":       (3, H, W)  back RGB  in [−1, 1] — reconstruction target
        "target_alpha": (1, H, W)  binary mask of back-view foreground
    """

    def __init__(
        self,
        data_source: str,
        image_size: int = 64,
        augment: bool = True,
        occlusion_p: float = 0.3,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(data_source, image_size, augment, transform)
        self.occlusion_p = occlusion_p

    def __getitem__(self, idx: int):  # type: ignore[override]
        front_img, back_img = self._load_pair(idx)

        if self.augment:
            front_img, back_img = self._apply_augment(front_img, back_img)
            front_img = random_occlusion(front_img, p=self.occlusion_p)

        cond_t, target_t, _, target_alpha = self._to_tensor(front_img, back_img)

        # Task 2 使用 RGB 条件/目标，避免与配置中的 6=3+3 输入通道不一致。
        # alpha 作为独立 mask 返回，不丢失前景位置信息。
        cond_t = cond_t[:3, :, :]
        target_t = target_t[:3, :, :]

        if self.transform is not None:
            cond_t, target_t = self.transform(cond_t, target_t)

        return {
            "condition": cond_t,
            "target": target_t,
            "target_alpha": target_alpha,
        }


# ---------------------------------------------------------------------------
# Helper: build and save dataset index
# ---------------------------------------------------------------------------

def build_and_save_index(
    sprite_dir: str,
    out_dir: str,
    index_path: str,
) -> List[Tuple[str, str]]:
    """
    Extract front/back pairs from sprite_dir, save images to out_dir,
    and write a CSV index at index_path.

    Returns list of (front_path, back_path) tuples.
    """
    from data_code.spritesheet_utils import build_pair_dataset

    pairs = build_pair_dataset(sprite_dir, out_dir)
    _write_index(pairs, index_path)
    print(f"Wrote {len(pairs)} pairs to {index_path}")
    return pairs
