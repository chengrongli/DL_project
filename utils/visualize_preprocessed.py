"""Visualize preprocessed front/back pairs as a grid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

from data_code.augmentation import to_tensor_pair
from utils.visualization import save_sample_grid
import torch
from PIL import Image


def _load_index(index_path: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    with index_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#") or row[0] == "front_path":
                continue
            if len(row) >= 2:
                pairs.append((Path(row[0]), Path(row[1])))
    return pairs


def visualize_pairs(index_path: str, out_path: str, *, limit: int = 32, image_size: int = 64) -> str:
    pairs = _load_index(Path(index_path).expanduser().resolve())
    if not pairs:
        raise RuntimeError(f"No pairs in index: {index_path}")

    tiles = []
    for front_path, back_path in pairs[:limit]:
        front = Image.open(front_path).convert("RGBA")
        back = Image.open(back_path).convert("RGBA")
        front_t, back_t, _, _ = to_tensor_pair(front, back, size=image_size)
        pair = torch.cat([front_t, back_t], dim=2)
        tiles.append(pair)

    batch = torch.stack(tiles, dim=0)
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    save_sample_grid(batch, str(out), nrow=max(1, int(len(tiles) ** 0.5)), upscale=4)
    return str(out)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a preprocessed paired dataset index")
    parser.add_argument("--index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out = visualize_pairs(args.index, args.out, limit=args.limit, image_size=args.image_size)
    print(f"Saved visualization to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
