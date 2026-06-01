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
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

from data_code.augmentation import (
    front_only_background_noise_tensor,
    front_only_geometric_jitter,
    front_only_palette_perturb,
    random_color_jitter,
    random_horizontal_flip,
    random_occlusion,
    random_palette_shift,
    to_tensor_pair,
)


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def _load_index(index_path: str) -> List[Tuple[str, str]]:
    """
    Load a CSV index with columns [front_path, back_path].
    Lines starting with '#' and common header rows are skipped.
    """
    pairs: List[Tuple[str, str]] = []
    with open(index_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            c0 = row[0].strip()
            if c0.startswith("#"):
                continue

            # 跳过表头：front_path,back_path 或 # front_path,back_path
            c0_lower = c0.lower().lstrip("#").strip()
            c1_lower = row[1].strip().lower() if len(row) > 1 else ""
            if c0_lower in {"front_path", "front", "input_front"} and c1_lower in {
                "back_path",
                "back",
                "input_back",
            }:
                continue

            if len(row) >= 2:
                pairs.append((c0, row[1].strip()))
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

def _resolve_one_source(data_source: str) -> List[Tuple[str, str]]:
    if os.path.isfile(data_source):
        return _load_index(data_source)
    if os.path.isdir(data_source):
        return _scan_directory(data_source)
    raise FileNotFoundError(
        f"data_source must be an existing file or directory: {data_source}"
    )


class _BaseSpriteDataset(Dataset):
    """
    Base dataset that loads front/back image pairs from one or more sources.

    ``data_source`` accepts either:
      * a single string (CSV index file or directory), or
      * a list of strings (each interpreted as above) for multi-source mixing.

    When multiple sources are provided, each pair carries a ``source_id`` that
    can be used by samplers (e.g. ``torch.utils.data.WeightedRandomSampler``)
    to balance contributions from imbalanced datasets — essential when one
    source (e.g. LPC) dominates and causes distribution collapse.
    """

    def __init__(
        self,
        data_source: Union[str, Sequence[str]],
        image_size: int = 64,
        augment: bool = True,
        transform: Optional[Callable] = None,
        source_weights: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Args:
            data_source: Either a single CSV/directory or a list of them.
            image_size:  Output spatial resolution.
            augment:     Whether to apply random augmentations.
            transform:   Optional additional transform applied to (front, back)
                         tensors after the default pipeline.
        """
        if isinstance(data_source, (list, tuple)):
            sources = list(data_source)
        else:
            sources = [data_source]

        self.pairs: List[Tuple[str, str]] = []
        self.source_ids: List[int] = []
        self.source_names: List[str] = []
        self.source_sizes: List[int] = []

        for sid, src in enumerate(sources):
            sub_pairs = _resolve_one_source(src)
            if not sub_pairs:
                raise RuntimeError(f"No front/back pairs found in: {src}")
            self.pairs.extend(sub_pairs)
            self.source_ids.extend([sid] * len(sub_pairs))
            self.source_names.append(str(src))
            self.source_sizes.append(len(sub_pairs))

        if not self.pairs:
            raise RuntimeError(f"No front/back pairs found in: {data_source}")

        self.image_size = image_size
        self.augment = augment
        self.transform = transform
        # Optional per-source weights (floats) to bias sampling across sources.
        if source_weights is not None:
            if len(source_weights) != len(self.source_sizes):
                raise ValueError("source_weights length must match number of sources")
            self.source_weights = [float(w) for w in source_weights]
        else:
            self.source_weights = [1.0] * len(self.source_sizes)

        # Precompute per-sample weights (aligned with self.pairs) for easy use
        # with torch.utils.data.WeightedRandomSampler.
        self._sample_weights = [
            float(self.source_weights[sid]) / max(1, self.source_sizes[sid])
            for sid in self.source_ids
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_pair(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        front_path, back_path = self.pairs[idx]
        front = Image.open(front_path).convert("RGBA")
        back = Image.open(back_path).convert("RGBA")
        return front, back

    def get_sample_weights(self) -> List[float]:
        """Return a list of per-sample weights aligned with dataset indices.

        These weights are suitable to pass directly to
        ``torch.utils.data.WeightedRandomSampler`` for balanced sampling.
        """
        return list(self._sample_weights)

    def make_weighted_sampler(self, num_samples: Optional[int] = None, replacement: bool = True):
        """Build a ``WeightedRandomSampler`` using the dataset's sample weights.

        Args:
            num_samples: number of samples to draw per epoch (defaults to len(dataset)).
            replacement: whether to sample with replacement.
        """
        from torch.utils.data import WeightedRandomSampler

        if num_samples is None:
            num_samples = len(self)
        weights = torch.tensor(self._sample_weights, dtype=torch.double)
        return WeightedRandomSampler(weights, num_samples=num_samples, replacement=replacement)

    def _apply_augment(
        self,
        front: Image.Image,
        back: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        front, back = random_horizontal_flip(front, back, p=0.5)
        front, back = random_color_jitter(front, back)
        # Joint HSV/palette shift: applied to *both* sides so front/back stay
        # consistent, but the LPC palette is no longer the only one seen.
        front, back = random_palette_shift(front, back, p=0.5)
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
    将 front/back "塞到通道维度"的不自然方式。

    __getitem__ returns a dict:
        "front":  (4, H, W) float32 in [−1, 1]
        "back":   (4, H, W) float32 in [−1, 1]
        "paired": (4, H, 2W) 水平拼接的训练输入
        "mask":   (1, H, 2W) 二值前景 mask（主要用于诊断、或可选加权）
        "attributes": dict of attribute values (if attributes_json provided)
    """

    def __init__(
        self,
        data_source: Union[str, Sequence[str]],
        image_size: int = 64,
        augment: bool = True,
        transform: Optional[Callable] = None,
        source_weights: Optional[Sequence[float]] = None,
        attributes_json: Optional[str] = None,
    ) -> None:
        super().__init__(data_source, image_size, augment, transform, source_weights=source_weights)
        self._attributes: Optional[dict] = None
        if attributes_json is not None:
            import json
            with open(attributes_json, "r") as f:
                raw = json.load(f)
            # Build index: match pair filename stem → attributes
            self._attributes = {}
            for key, attrs in raw.items():
                self._attributes[key] = attrs

    def _get_attributes(self, idx: int) -> Optional[dict]:
        if self._attributes is None:
            return None
        front_path, _ = self.pairs[idx]
        # Derive key from filename: e.g. "data/.../char_0000_front.png" → "char_0000"
        stem = Path(front_path).stem
        key = stem.replace("_front", "").replace("_back", "")
        return self._attributes.get(key)

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

        result = {
            "front": front_t,
            "back": back_t,
            "paired": paired,
            "mask": mask,
        }
        attrs = self._get_attributes(idx)
        if attrs is not None:
            result["attributes"] = attrs
        return result


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
        "source_id":    int        index into ``self.source_names`` — useful
                                   for multi-source weighted sampling.

    Front-only augmentations (``front_geom_jitter_p``, ``front_palette_p``,
    ``front_bg_replace_p``) deliberately decouple the front from the back so
    the network cannot learn a per-pixel "copy front, paint back" shortcut.
    Disable them on the validation set by passing ``augment=False``.
    """

    def __init__(
        self,
        data_source: Union[str, Sequence[str]],
        image_size: int = 64,
        augment: bool = True,
        occlusion_p: float = 0.5,
        front_geom_jitter_p: float = 0.7,
        front_palette_p: float = 0.5,
        front_bg_replace_p: float = 0.4,
        occlusion_intensity: float = 0.5,
        occlusion_fill: str = "zero",
        transform: Optional[Callable] = None,
        source_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__(data_source, image_size, augment, transform, source_weights=source_weights)
        self.occlusion_p = occlusion_p
        self.front_geom_jitter_p = front_geom_jitter_p
        self.front_palette_p = front_palette_p
        self.front_bg_replace_p = front_bg_replace_p
        self.occlusion_intensity = occlusion_intensity
        self.occlusion_fill = occlusion_fill

    def __getitem__(self, idx: int):  # type: ignore[override]
        front_img, back_img = self._load_pair(idx)

        if self.augment:
            # Joint augmentations (keep front/back semantically aligned).
            front_img, back_img = self._apply_augment(front_img, back_img)

            # Front-only augmentations — break the copy-shortcut.
            front_img = random_occlusion(
                front_img,
                p=self.occlusion_p,
                intensity=getattr(self, "occlusion_intensity", 0.5),
                fill=getattr(self, "occlusion_fill", "zero"),
            )
            front_img = front_only_geometric_jitter(
                front_img, p=self.front_geom_jitter_p
            )
            front_img = front_only_palette_perturb(
                front_img, p=self.front_palette_p
            )

        cond_t, target_t, cond_alpha, target_alpha = self._to_tensor(front_img, back_img)

        # Task 2 使用 RGB 条件/目标，避免与配置中的 6=3+3 输入通道不一致。
        # alpha 作为独立 mask 返回，不丢失前景位置信息。
        cond_t = cond_t[:3, :, :]
        target_t = target_t[:3, :, :]

        # Tensor-stage front-only background replacement. Done *after*
        # premultiply so the random background actually survives into the
        # final RGB tensor (PIL-stage replacement would be re-zeroed by
        # the alpha-premultiply inside _rgba_to_rgba_tensor).
        if self.augment and self.front_bg_replace_p > 0.0:
            cond_t = front_only_background_noise_tensor(
                cond_t, cond_alpha, p=self.front_bg_replace_p
            )

        if self.transform is not None:
            cond_t, target_t = self.transform(cond_t, target_t)

        return {
            "condition": cond_t,
            "target": target_t,
            "target_alpha": target_alpha,
            "source_id": int(self.source_ids[idx]),
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
