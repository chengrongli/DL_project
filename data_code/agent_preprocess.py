"""Agent-style preprocessing for front/back LPC pair datasets.

This script standardizes pair filenames, optionally deduplicates samples,
creates train/val CSV indices, and writes dataset statistics for downstream
training (DDPM / Flow Matching).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image


@dataclass(frozen=True)
class PairRecord:
    front: Path
    back: Path


def _scan_pairs(root: Path) -> List[PairRecord]:
    pairs: List[PairRecord] = []
    for front in sorted(root.rglob("*_front.png")):
        back = Path(str(front).replace("_front.png", "_back.png"))
        if back.exists():
            pairs.append(PairRecord(front=front, back=back))
    return pairs


def _pair_hash(front: Image.Image, back: Image.Image) -> str:
    h = hashlib.sha1()
    h.update(front.tobytes())
    h.update(back.tobytes())
    return h.hexdigest()


def _load_resize_rgba(path: Path, image_size: int) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.NEAREST)
    return img


def _write_index(path: Path, items: Sequence[Tuple[Path, Path]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["front_path", "back_path"])
        for front, back in items:
            writer.writerow([str(front), str(back)])


def preprocess_pairs(
    input_dir: str,
    output_dir: str,
    *,
    image_size: int = 64,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_samples: int | None = None,
    deduplicate: bool = True,
) -> dict:
    src_root = Path(input_dir).expanduser().resolve()
    out_root = Path(output_dir).expanduser().resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {src_root}")

    pairs = _scan_pairs(src_root)
    if not pairs:
        raise RuntimeError(f"No *_front.png/*_back.png pairs found in: {src_root}")

    if max_samples is not None:
        pairs = pairs[: max(0, max_samples)]

    rng = random.Random(seed)
    rng.shuffle(pairs)

    out_root.mkdir(parents=True, exist_ok=True)
    processed: List[Tuple[Path, Path]] = []
    seen_hash = set()
    dropped_duplicates = 0

    for idx, rec in enumerate(pairs):
        front = _load_resize_rgba(rec.front, image_size)
        back = _load_resize_rgba(rec.back, image_size)

        if deduplicate:
            sig = _pair_hash(front, back)
            if sig in seen_hash:
                dropped_duplicates += 1
                continue
            seen_hash.add(sig)

        name = f"pair_{len(processed):06d}"
        front_out = out_root / f"{name}_front.png"
        back_out = out_root / f"{name}_back.png"
        front.save(front_out)
        back.save(back_out)
        processed.append((front_out, back_out))

    if not processed:
        raise RuntimeError("All samples were filtered out during preprocessing.")

    rng.shuffle(processed)
    n_val = int(round(len(processed) * val_ratio))
    n_val = min(max(n_val, 1), max(1, len(processed) - 1)) if len(processed) > 1 else 0

    val_items = processed[:n_val]
    train_items = processed[n_val:]

    _write_index(out_root / "index_all.csv", processed)
    _write_index(out_root / "index_train.csv", train_items)
    _write_index(out_root / "index_val.csv", val_items)

    stats = {
        "input_dir": str(src_root),
        "output_dir": str(out_root),
        "input_pairs": len(pairs),
        "processed_pairs": len(processed),
        "dropped_duplicates": dropped_duplicates,
        "image_size": image_size,
        "train_pairs": len(train_items),
        "val_pairs": len(val_items),
        "val_ratio": val_ratio,
        "seed": seed,
    }

    stats_path = out_root / "preprocess_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess LPC front/back pair dataset")
    parser.add_argument("--input-dir", required=True, help="Directory containing raw *_front.png/*_back.png")
    parser.add_argument("--output-dir", required=True, help="Directory to write processed data")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--no-deduplicate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    stats = preprocess_pairs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
        deduplicate=not args.no_deduplicate,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
