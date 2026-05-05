"""Randomly compose complete LPC characters from downloaded layers.

Given a local LPC asset directory (for example, the result of the sparse
clone), this script samples random combinations of body/head/hair/torso/
legs/feet layers and exports the idle front/back images for each
composition.  It builds on the same `compose_layers` helper used by the
manual layer stack utility, but removes the need to hand-write YAML.
"""

from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from data_code.augmentation import random_palette_shift
from data_code.spritesheet_utils import compose_layers, save_pair


DEFAULT_LAYER_ORDER: Sequence[str] = (
    "body",
    "legs",
    "feet",
    "torso",
    "dress",
    "arms",
    "neck",
    "head",
    "eyes",
    "facial",
    "hair",
    "hat",
    "shoulders",
)

REQUIRED_GROUPS: Set[str] = {"body", "legs", "feet", "torso", "head"}
GROUP_SAMPLING_PROB: Dict[str, float] = {
    "dress": 0.22,
    "arms": 0.32,
    "neck": 0.45,
    "eyes": 1.0,     # 眼睛强制保留，避免“无脸感”
    "facial": 0.22,  # 面部饰品降低概率，减少遮挡/混乱
    "hair": 0.88,
    "hat": 0.22,     # 帽子显著降低，避免遮住脸
    "shoulders": 0.18,
    # 预留：手工传 groups 时可选用
    "tail": 0.1,
    "wings": 0.03,
}

# 避免随机到残缺/伤残层，生成“缺胳膊少腿”的异常角色。
PATH_BLOCKLIST = (
    "wound",
    "prosthesis",
    "wheelchair",
    "blood",
    "corpse",
    "injury",
)

GROUP_BLOCKLIST: Dict[str, Sequence[str]] = {
    # 脸部优先“正常人像”
    "head": (
        "alien", "boarman", "frankenstein", "goblin", "jack", "lizard",
        "minotaur", "mouse", "orc", "pig", "rabbit", "rat", "sheep",
        "skeleton", "troll", "zombie",
    ),
    "eyes": ("cyclops",),
    "facial": ("masks", "patches"),
    "hat": ("horns", "skull", "bone", "mask"),
}

BODY_COMPAT_TOKENS = ("male", "female", "teen", "child", "muscular", "pregnant", "adult", "lizard")

def _walk_patterns(prefix: str) -> Sequence[str]:
    # 同时覆盖:
    #   1) .../walk.png
    #   2) .../walk/<variant>.png
    return (f"{prefix}/**/walk.png", f"{prefix}/**/walk/*.png")


LAYER_PATTERNS: Dict[str, Sequence[str]] = {
    # base body 只允许完整 bodies，tail/wings 另作可选层
    "body": _walk_patterns("spritesheets/body/bodies"),
    "torso": _walk_patterns("spritesheets/torso"),
    "legs": _walk_patterns("spritesheets/legs"),
    "feet": _walk_patterns("spritesheets/feet"),
    # 头部限定到 heads，避免随机到“只有耳朵/附加件”导致脸异常
    "head": _walk_patterns("spritesheets/head/heads/human"),
    "hair": _walk_patterns("spritesheets/hair"),
    # 眼睛限定 human 子集，剔除 cyclops
    "eyes": _walk_patterns("spritesheets/eyes/human"),
    # 面部配件限定较干净子集
    "facial": (
        *_walk_patterns("spritesheets/facial/glasses"),
        *_walk_patterns("spritesheets/facial/earrings"),
        *_walk_patterns("spritesheets/facial/monocle"),
    ),
    "hat": _walk_patterns("spritesheets/hat"),
    "neck": _walk_patterns("spritesheets/neck"),
    "shoulders": _walk_patterns("spritesheets/shoulders"),
    "dress": _walk_patterns("spritesheets/dress"),
    "arms": _walk_patterns("spritesheets/arms"),
    "tail": _walk_patterns("spritesheets/body/tail"),
    "wings": _walk_patterns("spritesheets/body/wings"),
}


MAX_RESAMPLE_TRIES = 18
MAX_QUALITY_RETRY = 24
MIN_ALPHA_PIXELS = 550
MIN_BBOX_HEIGHT = 34
MIN_BBOX_WIDTH = 16
MIN_LARGEST_COMPONENT_RATIO = 0.72


@dataclass(frozen=True)
class LayerChoice:
    group: str
    path: Path


def _is_blocked_path(path: Path) -> bool:
    text = str(path).lower()
    return any(token in text for token in PATH_BLOCKLIST)


def _is_group_allowed(group: str, path: Path) -> bool:
    text = str(path).lower()
    blocked = GROUP_BLOCKLIST.get(group, ())
    return not any(token in text for token in blocked)


def _gather_candidates(root: Path, patterns: Sequence[str], group: str) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(
            p
            for p in root.glob(pattern)
            if p.is_file()
            and p.suffix.lower() == ".png"
            and not _is_blocked_path(p)
            and _is_group_allowed(group, p)
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
        candidates = _gather_candidates(assets_root, patterns, group)
        if candidates:
            pool[group] = candidates
    return pool


def _infer_body_tokens(body_path: Path) -> Set[str]:
    parts = {p.lower() for p in body_path.parts}
    return {tok for tok in BODY_COMPAT_TOKENS if tok in parts}


def _filter_compatible(candidates: List[Path], body_tokens: Set[str]) -> List[Path]:
    if not candidates or not body_tokens:
        return candidates

    filtered: List[Path] = []
    for p in candidates:
        parts = {x.lower() for x in p.parts}
        tokens_in_path = {tok for tok in BODY_COMPAT_TOKENS if tok in parts}
        if not tokens_in_path or tokens_in_path.intersection(body_tokens):
            filtered.append(p)
    return filtered if filtered else candidates


def _largest_component_ratio(mask: np.ndarray) -> float:
    # 64x64 小图上用轻量 DFS 即可，不依赖 scipy。
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)
    total = int(mask.sum())
    if total <= 0:
        return 0.0

    best = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = 1
            count = 0
            while stack:
                cy, cx = stack.pop()
                count += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        stack.append((ny, nx))
            if count > best:
                best = count

    return float(best) / float(total)


def _passes_quality(front, back) -> bool:
    fa = np.array(front.getchannel("A"))
    ba = np.array(back.getchannel("A"))
    fmask = fa > 0
    bmask = ba > 0

    if int(fmask.sum()) < MIN_ALPHA_PIXELS or int(bmask.sum()) < MIN_ALPHA_PIXELS:
        return False

    def _bbox_ok(mask: np.ndarray) -> bool:
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return False
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        return w >= MIN_BBOX_WIDTH and h >= MIN_BBOX_HEIGHT

    if not (_bbox_ok(fmask) and _bbox_ok(bmask)):
        return False

    # 最大连通块占比过低，通常意味着角色被拆散（断臂/悬浮碎片过多）
    if _largest_component_ratio(fmask) < MIN_LARGEST_COMPONENT_RATIO:
        return False
    if _largest_component_ratio(bmask) < MIN_LARGEST_COMPONENT_RATIO:
        return False

    return True


def summarize_layer_pool(assets_root: str, groups: Sequence[str] = DEFAULT_LAYER_ORDER) -> Dict[str, int]:
    root = Path(assets_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Assets root does not exist: {root}")
    pool = _build_layer_pool(root, groups)
    return {g: len(pool.get(g, [])) for g in groups}


def _sample_layer_combo(
    rng: random.Random,
    layer_pool: Dict[str, List[Path]],
    groups: Sequence[str],
) -> tuple[List[LayerChoice], tuple[str, ...]]:
    trial: List[LayerChoice] = []
    body_candidates = layer_pool.get("body", [])
    if not body_candidates:
        raise RuntimeError("No body candidates found in layer pool")

    body_path = rng.choice(body_candidates)
    trial.append(LayerChoice(group="body", path=body_path))
    body_tokens = _infer_body_tokens(body_path)

    for group in groups:
        if group == "body":
            continue
        candidates = layer_pool.get(group)
        if not candidates:
            continue

        if group not in REQUIRED_GROUPS:
            p_keep = GROUP_SAMPLING_PROB.get(group, 0.5)
            if rng.random() > p_keep:
                continue

        compatible = _filter_compatible(candidates, body_tokens)
        path = rng.choice(compatible)
        trial.append(LayerChoice(group=group, path=path))

    trial_sig = tuple(str(item.path) for item in trial)
    return trial, trial_sig


def _compose_one_sample(
    idx: int,
    layer_pool: Dict[str, List[Path]],
    groups: Sequence[str],
    out_dir: Path,
    *,
    seed: Optional[int],
    prefix: str,
    column: int,
    palette_shift_prob: float,
    palette_h: float,
    palette_s: float,
    palette_v: float,
    enforce_unique: bool = True,
) -> List[LayerChoice]:
    local_seed = (seed or 0) + idx * 1009 + 17
    rng = random.Random(local_seed)

    chosen_layers: List[LayerChoice] = []
    used_local = set()
    accepted_front = None
    accepted_back = None

    for _ in range(MAX_QUALITY_RETRY):
        for _ in range(MAX_RESAMPLE_TRIES):
            trial, trial_sig = _sample_layer_combo(rng, layer_pool, groups)
            if not enforce_unique or trial_sig not in used_local:
                chosen_layers = trial
                used_local.add(trial_sig)
                break
            chosen_layers = trial

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

        if _passes_quality(front, back):
            accepted_front, accepted_back = front, back
            break

    if accepted_front is None or accepted_back is None:
        layer_paths = [str(choice.path) for choice in chosen_layers]
        accepted_front, accepted_back = compose_layers(layer_paths, col=column)

    base_name = f"{prefix}_{idx:04d}"
    save_pair(accepted_front, accepted_back, str(out_dir), base_name)
    return chosen_layers


def _compose_one_sample_worker(args: tuple) -> Tuple[int, List[LayerChoice]]:
    (
        idx,
        layer_pool,
        groups,
        out_dir,
        seed,
        prefix,
        column,
        palette_shift_prob,
        palette_h,
        palette_s,
        palette_v,
    ) = args

    layers = _compose_one_sample(
        idx=idx,
        layer_pool=layer_pool,
        groups=groups,
        out_dir=Path(out_dir),
        seed=seed,
        prefix=prefix,
        column=column,
        palette_shift_prob=palette_shift_prob,
        palette_h=palette_h,
        palette_s=palette_s,
        palette_v=palette_v,
        enforce_unique=False,
    )
    return idx, layers


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
    num_workers: int = 0,
) -> List[LayerChoice]:
    root = Path(assets_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Assets root does not exist: {root}")

    rng = random.Random(seed)
    layer_pool = _build_layer_pool(root, groups)

    missing_required = [g for g in REQUIRED_GROUPS if g in groups and g not in layer_pool]
    if missing_required:
        raise RuntimeError(
            "Missing required layer groups: " + ", ".join(sorted(missing_required))
        )

    pool_summary = {g: len(layer_pool.get(g, [])) for g in groups}
    summary_str = ", ".join(f"{k}={v}" for k, v in pool_summary.items())
    print(f"[random_composer] layer pool summary: {summary_str}")

    selections: List[LayerChoice] = []

    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    if num_workers <= 0:
        num_workers = max(1, min((os.cpu_count() or 1) - 1, count))

    if num_workers == 1:
        used_signatures = set()
        for idx in range(count):
            # 单线程保留全局去重行为
            local_rng = random.Random((seed or 0) + idx * 997 + 13)
            chosen_layers: List[LayerChoice] = []
            accepted_front = None
            accepted_back = None

            for _ in range(MAX_QUALITY_RETRY):
                for _ in range(MAX_RESAMPLE_TRIES):
                    trial, trial_sig = _sample_layer_combo(local_rng, layer_pool, groups)
                    if trial_sig not in used_signatures:
                        chosen_layers = trial
                        used_signatures.add(trial_sig)
                        break
                    chosen_layers = trial

                layer_paths = [str(choice.path) for choice in chosen_layers]
                front, back = compose_layers(layer_paths, col=column)

                if palette_shift_prob > 0.0 and local_rng.random() < palette_shift_prob:
                    front, back = random_palette_shift(
                        front,
                        back,
                        p=1.0,
                        max_h=palette_h,
                        max_s=palette_s,
                        max_v=palette_v,
                    )

                if _passes_quality(front, back):
                    accepted_front, accepted_back = front, back
                    break

            if accepted_front is None or accepted_back is None:
                layer_paths = [str(choice.path) for choice in chosen_layers]
                accepted_front, accepted_back = compose_layers(layer_paths, col=column)

            base_name = f"{prefix}_{idx:04d}"
            save_pair(accepted_front, accepted_back, str(out_path), base_name)
            selections.extend(chosen_layers)
    else:
        print(f"[random_composer] parallel mode enabled: workers={num_workers}")
        task_args = [
            (
                idx,
                layer_pool,
                tuple(groups),
                str(out_path),
                seed,
                prefix,
                column,
                palette_shift_prob,
                palette_h,
                palette_s,
                palette_v,
            )
            for idx in range(count)
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for idx, chosen_layers in ex.map(_compose_one_sample_worker, task_args):
                _ = idx
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
        help="Layer groups to include (default: website-like full character stack)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only print available layer counts and exit",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Parallel workers (0=auto, 1=single-process)",
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
        num_workers=args.num_workers,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
