"""Convert the `data_other` dataset layout into SpritePair-compatible PNGs.

The Kaggle "Pixel Characters" dataset (and similar mirrors) stores each
character frame under numeric direction folders::

    data_other/
        0/   # back view
        1/   # left  view (unused)
        2/   # front view
        3/   # right view (unused)

This script pairs images with identical filenames from the "0" and "2"
subfolders, resizes them to a uniform square resolution (default 64×64), and
saves front/back PNGs into a target directory compatible with
:dataclass:`SpritePairDataset`.

Usage
-----

    python -m data_code.convert_data_other \
        --root data/data_other \
        --out-dir data/pairs/data_other_pairs \
        --image-size 64 \
        --index data/pairs/data_other_pairs/index.csv

A CSV index is optional but recommended when the dataset grows large.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

BACK_FOLDER = "0"
FRONT_FOLDER = "2"
SUPPORTED_EXTENSIONS = {".png", ".webp"}


def _collect_images(folder: Path) -> dict[Tuple[str, str], Path]:
    mapping: dict[Tuple[str, str], Path] = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            key = (path.stem, path.suffix.lower())
            mapping[key] = path
    return mapping


def _resize_and_save(src: Path, dst: Path, image_size: int) -> None:
    img = Image.open(src).convert("RGBA")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.NEAREST)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)


def convert_data_other(
    root: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    image_size: int = 64,
    index_path: Optional[str | os.PathLike[str]] = None,
    limit: Optional[int] = None,
    overwrite: bool = True,
) -> int:
    """Convert the dataset located at ``root`` into paired PNGs."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")

    back_dir = root_path / BACK_FOLDER
    front_dir = root_path / FRONT_FOLDER
    if not back_dir.exists() or not front_dir.exists():
        raise RuntimeError(
            f"Expected subfolders '{BACK_FOLDER}' (back) and '{FRONT_FOLDER}' (front) under {root_path}"
        )

    back_images = _collect_images(back_dir)
    front_images = _collect_images(front_dir)
    keys = sorted(back_images.keys() & front_images.keys(), key=lambda k: k[0])
    if not keys:
        raise RuntimeError("No matching filenames found between back and front folders.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        for suffix in ("_front.png", "_back.png"):
            if any((out_path / f"char_{0:05d}{suffix}").exists() for _ in range(1)):
                raise FileExistsError(
                    f"Output directory {out_path} already contains paired PNGs; pass overwrite=True to replace."
                )

    index_entries: List[Tuple[str, str]] = []
    total = 0

    for idx, key in enumerate(keys):
        if limit is not None and total >= limit:
            break
        stem, suffix = key
        front_src = front_images[key]
        back_src = back_images[key]

        front_dst = out_path / f"char_{total:05d}_front.png"
        back_dst = out_path / f"char_{total:05d}_back.png"

        _resize_and_save(front_src, front_dst, image_size)
        _resize_and_save(back_src, back_dst, image_size)

        index_entries.append((str(front_dst), str(back_dst)))
        total += 1

    if index_path is not None:
        index_file = Path(index_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        with index_file.open("w", encoding="utf-8") as f:
            f.write("front_path,back_path\n")
            for front, back in index_entries:
                f.write(f"{front},{back}\n")

    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert data_other dataset into paired PNGs")
    parser.add_argument("--root", required=True, help="Path to data_other directory (containing 0/ and 2/ folders)")
    parser.add_argument("--out-dir", required=True, help="Destination directory for paired PNGs")
    parser.add_argument("--image-size", type=int, default=64, help="Output image size (default 64)")
    parser.add_argument("--index", help="Optional CSV index path")
    parser.add_argument("--limit", type=int, help="Optional max number of characters to convert")
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if output directory already has PNGs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total = convert_data_other(
        root=args.root,
        out_dir=args.out_dir,
        image_size=args.image_size,
        index_path=args.index,
        limit=args.limit,
        overwrite=not args.no_overwrite,
    )
    print(f"Converted {total} character pairs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
