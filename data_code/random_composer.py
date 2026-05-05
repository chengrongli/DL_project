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
from typing import Dict, List, Optional, Sequence

from data_code.augmentation import random_palette_shift
from data_code.spritesheet_utils import compose_layers, save_pair


DEFAULT_LAYER_ORDER: Sequence[str] = (
    "body",
    "legs",
    "torso",
    "head",
    "hair",
    "feet",
)

LAYER_PATTERNS: Dict[str, Sequence[str]] = {
    # body 在 LPC 仓库中分布很分散，仅匹配 bodies/** 容易把基座限制得过小。
    # 保留旧路径 + 全量兜底路径，提升结构多样性。
    "body": (
        "spritesheets/body/bodies/**/walk.png",
        "spritesheets/body/**/walk.png",
    ),
    "torso": ("spritesheets/torso/**/walk.png",),
    "legs": ("spritesheets/legs/**/walk.png",),
    "feet": ("spritesheets/feet/**/walk.png",),
    "head": (
        "spritesheets/head/heads/**/walk.png",
        "spritesheets/head/**/walk.png",
    ),
    "hair": ("spritesheets/hair/**/walk.png",),
}


MAX_RESAMPLE_TRIES = 12


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
    unique: List[Path] = []
    seen = set()
    for file in sorted(files):
        try:
            rel = file.relative_to(root)
        except ValueError:
            rel = file
        if rel not in seen:
            unique.append(file)
            seen.add(rel)
    return unique


def _build_layer_pool(assets_root: Path, groups: Sequence[str]) -> Dict[str, List[Path]]:
    pool: Dict[str, List[Path]] = {}
    for group in groups:
        patterns = LAYER_PATTERNS.get(group)
        if not patterns:
            continue
        candidates = _gather_candidates(assets_root, patterns)
        if candidates:
            pool[group] = candidates
    return pool


def summarize_layer_pool(assets_root: str, groups: Sequence[str] = DEFAULT_LAYER_ORDER) -> Dict[str, int]:
    root = Path(assets_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Assets root does not exist: {root}")
    pool = _build_layer_pool(root, groups)
    return {g: len(pool.get(g, [])) for g in groups}


def random_compose_batch(
    assets_root: str,
    out_dir: str,
    *,
    count: int,
    groups: Sequence[str] = DEFAULT_LAYER_ORDER,
    seed: Optional[int] = None,
    column: int = 0,
    prefix: str = "char",
    palette_shift_prob: float = 0.0,
    palette_h: float = 0.08,
    palette_s: float = 0.2,
    palette_v: float = 0.2,
) -> List[LayerChoice]:
    root = Path(assets_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Assets root does not exist: {root}")

    rng = random.Random(seed)
    layer_pool = _build_layer_pool(root, groups)

    if "body" not in layer_pool:
        raise RuntimeError(
            "No body layers found. Ensure your assets include 'spritesheets/body/**/walk.png'."
        )

    pool_summary = {g: len(layer_pool.get(g, [])) for g in groups}
    summary_str = ", ".join(f"{k}={v}" for k, v in pool_summary.items())
    print(f"[random_composer] layer pool summary: {summary_str}")

    selections: List[LayerChoice] = []

    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    used_signatures = set()

    for idx in range(count):
        chosen_layers: List[LayerChoice] = []
        signature = None

        # 尽量避免重复组合，提升有效多样性。
        for _ in range(MAX_RESAMPLE_TRIES):
            trial: List[LayerChoice] = []
            for group in groups:
                candidates = layer_pool.get(group)
                if not candidates:
                    continue
                path = rng.choice(candidates)
                trial.append(LayerChoice(group=group, path=path))

            trial_sig = tuple(str(item.path) for item in trial)
            if trial_sig not in used_signatures:
                chosen_layers = trial
                signature = trial_sig
                break

            # 如果实在抽不到新组合，接受重复，避免死循环。
            chosen_layers = trial
            signature = trial_sig

        if signature is not None:
            used_signatures.add(signature)

        layer_paths = [str(choice.path) for choice in chosen_layers]
        front, back = compose_layers(layer_paths, col=column)

        if palette_shift_prob > 0.0 and rng.random() < palette_shift_prob:
            front, back = random_palette_shift(
                front,
                back,
                p=1.0,
                max_h=palette_h,
                max_s=palette_s,
                max_v=palette_v,
            )

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
    parser.add_argument("--palette-shift-prob", type=float, default=0.0,
                        help="Probability of applying a random palette shift to each composition")
    parser.add_argument("--palette-h", type=float, default=0.08,
                        help="Maximum hue shift (normalized 0-1 range; default 0.08 ≈ 30°)")
    parser.add_argument("--palette-s", type=float, default=0.2,
                        help="Maximum saturation scaling factor (fraction)")
    parser.add_argument("--palette-v", type=float, default=0.2,
                        help="Maximum value scaling factor (fraction)")
    parser.add_argument(
        "--groups",
        nargs="*",
        help="Layer groups to include (default order: body legs torso head hair feet)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only print available layer counts and exit",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    groups = tuple(args.groups) if args.groups else DEFAULT_LAYER_ORDER
    if args.report_only:
        summary = summarize_layer_pool(args.assets_root, groups)
        summary_str = ", ".join(f"{k}={v}" for k, v in summary.items())
        print(f"[random_composer] {summary_str}")
        return 0

    random_compose_batch(
        assets_root=args.assets_root,
        out_dir=args.out_dir,
        count=args.count,
        groups=groups,
        seed=args.seed,
        column=args.column,
        prefix=args.prefix,
        palette_shift_prob=args.palette_shift_prob,
        palette_h=args.palette_h,
        palette_s=args.palette_s,
        palette_v=args.palette_v,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
