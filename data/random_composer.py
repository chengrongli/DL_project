"""Randomly compose complete LPC characters from downloaded layers.

Given a local LPC asset directory (for example, the result of the sparse
clone), this script samples random combinations of body/head/hair/torso/
legs/feet layers and exports the idle front/back images for each
composition.  It builds on the same `compose_layers` helper used by the
manual layer stack utility, but removes the need to hand-write YAML.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from data.spritesheet_utils import compose_layers, save_pair


# Order matters: lower indices are composited first (background).
DEFAULT_LAYER_ORDER: Sequence[str] = (
    "body",
    "legs",
    "torso",
    "head",
    "hair",
    "feet",
)

# Glob patterns are relative to the assets root provided by the user.
LAYER_PATTERNS: Dict[str, Sequence[str]] = {
    # Required base body.
    "body": ("spritesheets/body/bodies/**/walk.png",),
    # Clothing layers.
    "torso": (
        "spritesheets/torso/**/walk.png",
    ),
    "legs": (
        "spritesheets/legs/**/walk.png",
    ),
    "feet": (
        "spritesheets/feet/**/walk.png",
    ),
    # Head and hair layers.
    "head": (
        "spritesheets/head/heads/**/walk.png",
    ),
    "hair": (
        "spritesheets/hair/**/walk.png",
    ),
}


@dataclass(frozen=True)
class LayerChoice:
    group: str
    path: Path


def _gather_candidates(root: Path, patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(
            p
            for p in root.glob(pattern)
            if p.is_file() and p.suffix.lower() == ".png" and p.name == "walk.png"
        )
    # Deduplicate while preserving order
    unique: List[Path] = []
    seen = set()
    for file in sorted(files):  # sort for determinism before dedupe
        try:
            rel = file.relative_to(root)
        except ValueError:
            rel = file
        if rel not in seen:
            unique.append(file)
            seen.add(rel)
    return unique


def _build_layer_pool(
    assets_root: Path,
    groups: Sequence[str],
) -> Dict[str, List[Path]]:
    pool: Dict[str, List[Path]] = {}
    for group in groups:
        patterns = LAYER_PATTERNS.get(group)
        if not patterns:
            continue
        candidates = _gather_candidates(assets_root, patterns)
        if candidates:
            pool[group] = candidates
    return pool


def random_compose_batch(
    assets_root: str,
    out_dir: str,
    *,
    count: int,
    groups: Sequence[str] = DEFAULT_LAYER_ORDER,
    seed: Optional[int] = None,
    column: int = 0,
    prefix: str = "char",
) -> List[LayerChoice]:
    root = Path(assets_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Assets root does not exist: {root}")

    rng = random.Random(seed)
    layer_pool = _build_layer_pool(root, groups)

    if "body" not in layer_pool:
        raise RuntimeError(
            "No body layers found. Ensure your assets include 'spritesheets/body/bodies/**/walk.png'."
        )

    selections: List[LayerChoice] = []

    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    for idx in range(count):
        chosen_layers: List[LayerChoice] = []
        for group in groups:
            candidates = layer_pool.get(group)
            if not candidates:
                continue  # group optional or missing in dataset
            path = rng.choice(candidates)
            chosen_layers.append(LayerChoice(group=group, path=path))

        # Compose and save
        layer_paths = [str(choice.path) for choice in chosen_layers]
        front, back = compose_layers(layer_paths, col=column)
        base_name = f"{prefix}_{idx:04d}"
        save_pair(front, back, str(out_path), base_name)

        selections.extend(chosen_layers)

    return selections


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly compose LPC characters from local assets")
    parser.add_argument("--assets-root", required=True, help="Root directory containing downloaded spritesheets")
    parser.add_argument("--out-dir", required=True, help="Output directory for composed front/back pairs")
    parser.add_argument("--count", type=int, default=16, help="Number of random compositions to generate")
    parser.add_argument("--prefix", default="char", help="Filename prefix for generated pairs")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--column", type=int, default=0, help="Spritesheet column index to extract (default: 0)")
    parser.add_argument(
        "--groups",
        nargs="*",
        help="Layer groups to include (default order: body legs torso head hair feet)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    groups = tuple(args.groups) if args.groups else DEFAULT_LAYER_ORDER
    random_compose_batch(
        assets_root=args.assets_root,
        out_dir=args.out_dir,
        count=args.count,
        groups=groups,
        seed=args.seed,
        column=args.column,
        prefix=args.prefix,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
