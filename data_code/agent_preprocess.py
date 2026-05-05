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


def _process_source(
    source_dir: Path,
    output_dir: Path,
    *,
    image_size: int,
    seed: int,
    max_samples: int | None,
    deduplicate: bool,
    name_prefix: str,
) -> tuple[List[Tuple[Path, Path]], int, int]:
    pairs = _scan_pairs(source_dir)
    if not pairs:
        raise RuntimeError(f"No *_front.png/*_back.png pairs found in: {source_dir}")

    if max_samples is not None:
        pairs = pairs[: max(0, max_samples)]

    rng = random.Random(seed)
    rng.shuffle(pairs)

    processed: List[Tuple[Path, Path]] = []
    seen_hash = set()
    dropped_duplicates = 0

    for rec in pairs:
        front = _load_resize_rgba(rec.front, image_size)
        back = _load_resize_rgba(rec.back, image_size)

        if deduplicate:
            sig = _pair_hash(front, back)
            if sig in seen_hash:
                dropped_duplicates += 1
                continue
            seen_hash.add(sig)

        name = f"{name_prefix}_{len(processed):06d}"
        front_out = output_dir / f"{name}_front.png"
        back_out = output_dir / f"{name}_back.png"
        front.save(front_out)
        back.save(back_out)
        processed.append((front_out, back_out))

    if not processed:
        raise RuntimeError(f"All samples were filtered out for source: {source_dir}")

    return processed, len(pairs), dropped_duplicates


def preprocess_pairs(
    input_dir: str,
    output_dir: str,
    *,
    val_input_dir: str | None = None,
    image_size: int = 64,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_samples: int | None = None,
    max_val_samples: int | None = None,
    deduplicate: bool = True,
) -> dict:
    src_root = Path(input_dir).expanduser().resolve()
    out_root = Path(output_dir).expanduser().resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {src_root}")

    val_root = Path(val_input_dir).expanduser().resolve() if val_input_dir else None
    if val_root is not None and not val_root.exists():
        raise FileNotFoundError(f"Validation dataset does not exist: {val_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    train_items, train_input_pairs, train_dropped_duplicates = _process_source(
        source_dir=src_root,
        output_dir=out_root,
        image_size=image_size,
        seed=seed,
        max_samples=max_samples,
        deduplicate=deduplicate,
        name_prefix="train",
    )

    if val_root is not None:
        val_items, val_input_pairs, val_dropped_duplicates = _process_source(
            source_dir=val_root,
            output_dir=out_root,
            image_size=image_size,
            seed=seed + 10007,
            max_samples=max_val_samples,
            deduplicate=deduplicate,
            name_prefix="val",
        )
    else:
        rng = random.Random(seed)
        rng.shuffle(train_items)
        n_val = int(round(len(train_items) * val_ratio))
        n_val = min(max(n_val, 1), max(1, len(train_items) - 1)) if len(train_items) > 1 else 0
        val_items = train_items[:n_val]
        train_items = train_items[n_val:]
        val_input_pairs = n_val
        val_dropped_duplicates = 0

    all_items = [*train_items, *val_items]

    _write_index(out_root / "index_all.csv", all_items)
    _write_index(out_root / "index_train.csv", train_items)
    _write_index(out_root / "index_val.csv", val_items)

    stats = {
        "input_dir": str(src_root),
        "val_input_dir": str(val_root) if val_root is not None else None,
        "output_dir": str(out_root),
        "train_input_pairs": train_input_pairs,
        "val_input_pairs": val_input_pairs,
        "processed_pairs": len(all_items),
        "train_dropped_duplicates": train_dropped_duplicates,
        "val_dropped_duplicates": val_dropped_duplicates,
        "image_size": image_size,
        "train_pairs": len(train_items),
        "val_pairs": len(val_items),
        "val_ratio": val_ratio if val_root is None else None,
        "seed": seed,
    }

    stats_path = out_root / "preprocess_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess LPC front/back pair dataset")
    parser.add_argument("--input-dir", required=True, help="Directory containing raw *_front.png/*_back.png")
    parser.add_argument("--output-dir", required=True, help="Directory to write processed data")
    parser.add_argument("--val-input-dir", help="Optional separate validation directory")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Used only when --val-input-dir is not provided")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--no-deduplicate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    stats = preprocess_pairs(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        val_input_dir=args.val_input_dir,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
        max_val_samples=args.max_val_samples,
        deduplicate=not args.no_deduplicate,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
