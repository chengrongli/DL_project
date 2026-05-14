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
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

from data_code.augmentation import random_palette_shift
from data_code.spritesheet_utils import extract_front_back, load_spritesheet, save_pair


DEFAULT_LAYER_ORDER: Sequence[str] = (
    "shadow",
    "body",
    "tail",
    "wings",
    "legs",
    "feet",
    "torso",
    "dress",
    "arms",
    "hands",
    "neck",
    "head",
    "eyes",
    "facial",
    "hair",
    "hat",
    "shoulders",
    "cape",
    "backpack",
    "backpack_cargo",
    "quiver",
    "shield",
    "weapon",
    "tools",
)

REQUIRED_GROUPS: Set[str] = {"body", "legs", "feet", "torso", "head"}
GROUP_SAMPLING_PROB: Dict[str, float] = {
    "shadow": 1.0,
    "dress": 0.22,
    "arms": 0.32,
    "hands": 0.52,
    "neck": 0.45,
    "eyes": 1.0,     # 眼睛强制保留，避免“无脸感”
    "facial": 0.5,   # 提高面部配件出现率（含眼镜）
    "hair": 0.9,
    "hat": 0.30,     # 提升帽子出现率，配合 helmet 偏好
    "shoulders": 0.18,
    "cape": 0.2,
    "backpack": 0.2,
    "backpack_cargo": 0.12,
    "quiver": 0.12,
    "shield": 0.26,
    "weapon": 0.46,
    "tools": 0.15,
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
    "body": ("skeleton", "zombie"),
    # 脸部优先“正常人像”
    "head": (
        "alien", "boarman", "frankenstein", "goblin", "jack", "lizard",
        "minotaur", "mouse", "orc", "pig", "rabbit", "rat", "sheep",
        "skeleton", "troll", "zombie",
    ),
    "eyes": ("cyclops",),
    "facial": ("masks", "patches"),
    "hair": (
        "spiked_liberty",
        "balding",
        "curls_large_xlong",
        "xlong_wavy",
        "relm_xlong",
        "relm_ponytail",
        "long_band",
        "longhawk",
        "topknot_short",
    ),
    "hat": ("horns", "skull", "bone", "mask"),
    "cape": ("solid_behind", "tattered_behind"),
    "backpack": ("basket_contents",),
}

BODY_COMPAT_TOKENS = (
    "male", "female", "teen", "child", "muscular", "pregnant", "adult", "lizard",
    "small", "elderly", "gaunt", "plump",
)
STRICT_BODY_COMPAT_GROUPS: Set[str] = {"head", "eyes", "hair", "neck", "torso", "legs", "feet", "dress", "shadow"}

def _walk_patterns(prefix: str) -> Sequence[str]:
    # 同时覆盖:
    #   1) .../walk.png
    #   2) .../walk/<variant>.png
    return (f"{prefix}/**/walk.png", f"{prefix}/**/walk/*.png")


LAYER_PATTERNS: Dict[str, Sequence[str]] = {
    "shadow": _walk_patterns("spritesheets/shadow"),
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
    "hands": _walk_patterns("spritesheets/hands"),
    "dress": _walk_patterns("spritesheets/dress"),
    "arms": _walk_patterns("spritesheets/arms"),
    "cape": _walk_patterns("spritesheets/cape"),
    "backpack": _walk_patterns("spritesheets/backpack"),
    "backpack_cargo": _walk_patterns("spritesheets/backpack/basket_contents"),
    "quiver": _walk_patterns("spritesheets/quiver"),
    "shield": _walk_patterns("spritesheets/shield"),
    "weapon": _walk_patterns("spritesheets/weapon"),
    "tools": _walk_patterns("spritesheets/tools"),
    "tail": _walk_patterns("spritesheets/body/tail"),
    "wings": _walk_patterns("spritesheets/body/wings"),
}


MAX_RESAMPLE_TRIES = 18
MAX_QUALITY_RETRY = 40
MIN_ALPHA_PIXELS = 550
MIN_BBOX_HEIGHT = 34
MIN_BBOX_WIDTH = 16
MIN_LARGEST_COMPONENT_RATIO = 0.72
MIN_EYE_ALPHA_PIXELS = 10
MIN_EYE_VISIBLE_RATIO = 0.65
MAX_HAIR_FACE_OVERLAP_RATIO = 0.40
MIN_HAIR_FRONT_PIXELS = 90
MAX_HAIR_FRONT_PIXELS = 950
MIN_HAIR_HEAD_CONTRAST = 22.0
MIN_TOP_HAIR_COVER_PIXELS = 60
MIN_FRONT_BACK_TOTAL_RATIO = 0.62
MIN_REQUIRED_LAYER_SIDE_PIXELS = 4

HANDHELD_EXCLUSIVE_GROUPS: Set[str] = {"weapon", "tools"}
FACE_OCCLUDER_GROUPS: Set[str] = {"hair", "hat", "facial", "weapon", "tools", "shield"}
REQUIRED_BIDIR_GROUPS: Set[str] = {"body", "legs", "feet", "torso", "head"}
STRICT_CLOTH_GROUPS: Set[str] = {"torso", "dress", "arms", "legs", "feet"}
GROUP_MIN_SIDE_PIXELS: Dict[str, int] = {
    "body": 80,
    "head": 70,
    "torso": 40,
    "legs": 30,
    "feet": 20,
}
SAFE_FALLBACK_GROUPS: Set[str] = {"shadow", "body", "legs", "feet", "torso", "head", "eyes", "neck", "hair"}
GROUP_DEPENDENCIES: Dict[str, Set[str]] = {
    "backpack_cargo": {"backpack"},
}

COLOR_TOKENS: Tuple[str, ...] = (
    "bluegray", "charcoal", "lavender", "maroon", "purple", "orange", "yellow",
    "green", "forest", "teal", "navy", "blue", "sky", "red", "pink", "rose",
    "brown", "leather", "walnut", "gray", "black", "white", "gold", "silver",
    "copper", "bronze", "iron", "steel", "tin",
)
NEUTRAL_COLORS: Set[str] = {"black", "white", "gray", "charcoal", "silver", "iron", "steel", "tin", "leather", "walnut", "brown"}
COLOR_FAMILY: Dict[str, str] = {
    "blue": "cool", "navy": "cool", "sky": "cool", "bluegray": "cool", "teal": "cool", "green": "cool", "forest": "cool",
    "purple": "cool", "lavender": "cool",
    "red": "warm", "orange": "warm", "yellow": "warm", "pink": "warm", "rose": "warm", "maroon": "warm",
    "gold": "metal", "bronze": "metal", "copper": "metal", "silver": "metal", "iron": "metal", "steel": "metal", "tin": "metal",
    "black": "neutral", "white": "neutral", "gray": "neutral", "charcoal": "neutral", "brown": "earth", "leather": "earth", "walnut": "earth",
}
CLOTH_COLOR_GROUPS: Set[str] = {"torso", "dress", "legs", "feet", "cape", "hat", "backpack", "backpack_cargo"}
STRICT_OBJECT_GROUPS: Set[str] = {"weapon", "tools", "shield", "quiver", "backpack", "backpack_cargo", "cape"}
REPAIR_REMOVAL_PRIORITY: Tuple[str, ...] = (
    "backpack_cargo",
    "weapon",
    "tools",
    "shield",
    "quiver",
    "cape",
    "backpack",
    "hat",
    "facial",
    "shoulders",
    "tail",
    "wings",
    "dress",
    "arms",
    "neck",
    "hair",
)


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
    if any(token in text for token in blocked):
        return False
    return not _is_problematic_one_sided_path(group, path)


def _extract_color_token(path: Path) -> Optional[str]:
    text = str(path).lower()
    for token in COLOR_TOKENS:
        if token in text:
            return token
    return None


def _is_color_compatible(color: Optional[str], dominant: Optional[str]) -> bool:
    if color is None or dominant is None:
        return True
    if color == dominant:
        return True
    if color in NEUTRAL_COLORS or dominant in NEUTRAL_COLORS:
        return True
    f1 = COLOR_FAMILY.get(color)
    f2 = COLOR_FAMILY.get(dominant)
    if f1 is None or f2 is None:
        return True
    if f1 == f2:
        return True
    # 允许冷色+金属、暖色+金属这类常见搭配。
    return {"metal", f1, f2} in ({"metal", "cool"}, {"metal", "warm"}, {"metal", "earth"})


def _is_problematic_one_sided_path(group: str, path: Path) -> bool:
    if group not in {"hair", "cape", "backpack"}:
        return False
    try:
        sheet = load_spritesheet(str(path))
        front, back = extract_front_back(sheet, col=0)
    except Exception:
        return False
    fa = int((np.array(front.getchannel("A")) > 0).sum())
    ba = int((np.array(back.getchannel("A")) > 0).sum())
    hi = max(fa, ba)
    lo = min(fa, ba)
    return hi >= 260 and lo <= 3


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


def _build_body_compat_pool(
    layer_pool: Dict[str, List[Path]],
    groups: Sequence[str],
) -> Dict[Path, Dict[str, List[Path]]]:
    body_candidates = layer_pool.get("body", [])
    result: Dict[Path, Dict[str, List[Path]]] = {}
    for body_path in body_candidates:
        body_tokens = _infer_body_tokens(body_path)
        body_style = _infer_body_style(body_path)
        by_group: Dict[str, List[Path]] = {}
        for group in groups:
            candidates = layer_pool.get(group, [])
            if not candidates:
                continue
            compatible = _filter_compatible(
                candidates,
                body_tokens,
                allow_fallback=(group not in STRICT_BODY_COMPAT_GROUPS),
            )
            if group == "head":
                strict_head = _filter_head_compatible(compatible, body_style)
                if strict_head:
                    compatible = strict_head
            if not compatible:
                if group in STRICT_BODY_COMPAT_GROUPS:
                    if group in REQUIRED_GROUPS:
                        compatible = candidates
                    else:
                        continue
                else:
                    compatible = candidates
            by_group[group] = compatible
        result[body_path] = by_group
    return result


def _infer_body_style(body_path: Path) -> str:
    parts = [p.lower() for p in body_path.parts]
    for style in ("child", "teen", "female", "male", "muscular", "pregnant"):
        if style in parts:
            return style
    return "adult"


def _infer_body_tokens(body_path: Path) -> Set[str]:
    parts = {p.lower() for p in body_path.parts}
    tokens = {tok for tok in BODY_COMPAT_TOKENS if tok in parts}
    if {"male", "female", "pregnant", "muscular"}.intersection(tokens):
        tokens.add("adult")
    if "pregnant" in tokens:
        tokens.add("female")
    if "muscular" in tokens:
        tokens.add("male")
    if "teen" in tokens:
        tokens.add("small")
    return tokens


def _filter_compatible(
    candidates: List[Path],
    body_tokens: Set[str],
    *,
    allow_fallback: bool = True,
) -> List[Path]:
    if not candidates or not body_tokens:
        return candidates

    filtered: List[Path] = []
    for p in candidates:
        parts = {x.lower() for x in p.parts}
        tokens_in_path = {tok for tok in BODY_COMPAT_TOKENS if tok in parts}
        if not tokens_in_path or tokens_in_path.intersection(body_tokens):
            filtered.append(p)
    if filtered:
        return filtered
    return candidates if allow_fallback else []


def _filter_head_compatible(candidates: List[Path], body_style: str) -> List[Path]:
    if not candidates:
        return candidates

    def _ok(path: Path) -> bool:
        s = str(path).lower()
        if body_style in {"female", "pregnant"}:
            return "/female" in s and "elderly" not in s
        if body_style in {"male", "muscular"}:
            return "/male" in s and "elderly" not in s
        if body_style == "child":
            return "/child/" in s or "_small/" in s
        if body_style == "teen":
            return "_small/" in s
        return "elderly" not in s

    out = [p for p in candidates if _ok(p)]
    return out if out else candidates


def _prefer_non_foreground(candidates: List[Path], group: str) -> List[Path]:
    if group not in {"hair", "hat", "facial", "weapon", "tools", "shield"} or not candidates:
        return candidates
    non_fg = [
        p
        for p in candidates
        if "/fg/" not in str(p).lower() and "foreground" not in str(p).lower()
    ]
    return non_fg if non_fg else candidates


def _prefer_semantic_candidates(rng: random.Random, candidates: List[Path], group: str) -> List[Path]:
    if not candidates:
        return candidates
    lower = [(p, str(p).lower()) for p in candidates]
    if group == "hair":
        colorful = [
            p
            for p, _ in lower
            if (c := _extract_color_token(p)) is not None and c not in NEUTRAL_COLORS
        ]
        if colorful and rng.random() < 0.82:
            return colorful
    if group == "facial":
        glasses = [p for p, s in lower if "/facial/glasses/" in s]
        if glasses and rng.random() < 0.9:
            return glasses
    if group == "hat":
        helmets = [p for p, s in lower if "/hat/helmet/" in s or "helmet" in s or "/helm" in s]
        if helmets and rng.random() < 0.58:
            return helmets
    return candidates


def _dominant_cloth_color(trial: List[LayerChoice]) -> Optional[str]:
    votes: Dict[str, int] = {}
    for layer in trial:
        if layer.group not in CLOTH_COLOR_GROUPS and layer.group != "hair":
            continue
        c = _extract_color_token(layer.path)
        if c is None:
            continue
        votes[c] = votes.get(c, 0) + 1
    if not votes:
        return None
    return max(votes.items(), key=lambda kv: kv[1])[0]


def _prefer_color_harmony(
    candidates: List[Path],
    *,
    group: str,
    dominant_color: Optional[str],
) -> List[Path]:
    if not candidates:
        return candidates
    if group not in CLOTH_COLOR_GROUPS:
        return candidates
    filtered = [p for p in candidates if _is_color_compatible(_extract_color_token(p), dominant_color)]
    return filtered if filtered else candidates


def _choose_candidate(
    rng: random.Random,
    candidates: List[Path],
    group_usage: Dict[Path, int],
    diversity_strength: float,
) -> Path:
    if not candidates:
        raise RuntimeError("Cannot choose from empty candidates")
    if len(candidates) == 1:
        return candidates[0]

    if diversity_strength <= 0.0 or rng.random() > diversity_strength:
        return rng.choice(candidates)

    sample_size = min(64, len(candidates))
    subset = rng.sample(candidates, sample_size) if sample_size < len(candidates) else candidates
    min_used = min(group_usage.get(path, 0) for path in subset)
    least_used = [path for path in subset if group_usage.get(path, 0) == min_used]
    return rng.choice(least_used)


def _compose_layers_cached(
    layer_paths: List[str],
    *,
    col: int,
    tile_cache: Dict[Tuple[str, int], Tuple[Image.Image, Image.Image]],
) -> Tuple[Image.Image, Image.Image]:
    front_canvas = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    back_canvas = Image.new("RGBA", (64, 64), (0, 0, 0, 0))

    for path in layer_paths:
        tiles = _get_cached_tiles(path, col=col, tile_cache=tile_cache)

        front_canvas = Image.alpha_composite(front_canvas, tiles[0])
        back_canvas = Image.alpha_composite(back_canvas, tiles[1])

    return front_canvas, back_canvas


def _get_cached_tiles(
    path: str,
    *,
    col: int,
    tile_cache: Dict[Tuple[str, int], Tuple[Image.Image, Image.Image]],
) -> Tuple[Image.Image, Image.Image]:
    key = (path, col)
    tiles = tile_cache.get(key)
    if tiles is None:
        sheet = load_spritesheet(path)
        tiles = extract_front_back(sheet, col=col)
        tile_cache[key] = tiles
    return tiles


def _passes_face_visibility(
    chosen_layers: List[LayerChoice],
    *,
    column: int,
    tile_cache: Dict[Tuple[str, int], Tuple[Image.Image, Image.Image]],
) -> bool:
    eyes_index = -1
    eyes_front: Optional[Image.Image] = None
    for idx, layer in enumerate(chosen_layers):
        if layer.group != "eyes":
            continue
        eyes_index = idx
        eyes_front, _ = _get_cached_tiles(str(layer.path), col=column, tile_cache=tile_cache)
        break
    if eyes_index < 0 or eyes_front is None:
        return False

    eye_mask = np.array(eyes_front.getchannel("A")) > 0
    eye_pixels = int(eye_mask.sum())
    if eye_pixels < MIN_EYE_ALPHA_PIXELS:
        return False

    occluder_mask = np.zeros_like(eye_mask, dtype=bool)
    for layer in chosen_layers[eyes_index + 1:]:
        if layer.group not in FACE_OCCLUDER_GROUPS:
            continue
        front_tile, _ = _get_cached_tiles(str(layer.path), col=column, tile_cache=tile_cache)
        occluder_mask |= (np.array(front_tile.getchannel("A")) > 0)

    overlap = int((eye_mask & occluder_mask).sum())
    visible_ratio = 1.0 - (float(overlap) / float(eye_pixels))
    return visible_ratio >= MIN_EYE_VISIBLE_RATIO


def _passes_hair_sanity(
    chosen_layers: List[LayerChoice],
    *,
    column: int,
    tile_cache: Dict[Tuple[str, int], Tuple[Image.Image, Image.Image]],
) -> bool:
    hair_layer = next((layer for layer in chosen_layers if layer.group == "hair"), None)
    if hair_layer is None:
        return True

    head_layer = next((layer for layer in chosen_layers if layer.group == "head"), None)
    eyes_layer = next((layer for layer in chosen_layers if layer.group == "eyes"), None)
    if head_layer is None or eyes_layer is None:
        return False

    hair_front, _ = _get_cached_tiles(str(hair_layer.path), col=column, tile_cache=tile_cache)
    head_front, _ = _get_cached_tiles(str(head_layer.path), col=column, tile_cache=tile_cache)
    eyes_front, _ = _get_cached_tiles(str(eyes_layer.path), col=column, tile_cache=tile_cache)

    hair_mask = np.array(hair_front.getchannel("A")) > 0
    hair_pixels = int(hair_mask.sum())
    if hair_pixels < MIN_HAIR_FRONT_PIXELS:
        return False
    if hair_pixels > MAX_HAIR_FRONT_PIXELS:
        return False

    face_mask = (np.array(head_front.getchannel("A")) > 0) | (np.array(eyes_front.getchannel("A")) > 0)
    face_pixels = int(face_mask.sum())
    if face_pixels <= 0:
        return False
    overlap = int((hair_mask & face_mask).sum())
    if (float(overlap) / float(face_pixels)) > MAX_HAIR_FACE_OVERLAP_RATIO:
        return False

    # 头发与头部基底颜色太接近时，看起来会像“光头”。
    hair_rgb = np.array(hair_front.convert("RGBA"))[:, :, :3]
    head_rgb = np.array(head_front.convert("RGBA"))[:, :, :3]
    head_mask = np.array(head_front.getchannel("A")) > 0
    if int(head_mask.sum()) <= 0:
        return False
    ys, xs = np.where(head_mask)
    y_mid = int((ys.min() + ys.max()) / 2)
    top_region = np.zeros_like(head_mask, dtype=bool)
    top_region[: y_mid + 1, :] = True
    top_cover = int((hair_mask & head_mask & top_region).sum())
    if top_cover < MIN_TOP_HAIR_COVER_PIXELS:
        return False

    hair_mean = hair_rgb[hair_mask].mean(axis=0)
    head_mean = head_rgb[head_mask].mean(axis=0)
    contrast = float(np.linalg.norm(hair_mean - head_mean))
    return contrast >= MIN_HAIR_HEAD_CONTRAST


def _passes_front_back_consistency(
    chosen_layers: List[LayerChoice],
    front: Image.Image,
    back: Image.Image,
    *,
    column: int,
    tile_cache: Dict[Tuple[str, int], Tuple[Image.Image, Image.Image]],
) -> bool:
    front_total = int((np.array(front.getchannel("A")) > 0).sum())
    back_total = int((np.array(back.getchannel("A")) > 0).sum())
    low = min(front_total, back_total)
    high = max(front_total, back_total)
    if high <= 0 or (float(low) / float(high)) < MIN_FRONT_BACK_TOTAL_RATIO:
        return False

    for layer in chosen_layers:
        front_tile, back_tile = _get_cached_tiles(str(layer.path), col=column, tile_cache=tile_cache)
        fa = int((np.array(front_tile.getchannel("A")) > 0).sum())
        ba = int((np.array(back_tile.getchannel("A")) > 0).sum())
        group = layer.group

        if group in REQUIRED_BIDIR_GROUPS:
            min_side = max(MIN_REQUIRED_LAYER_SIDE_PIXELS, GROUP_MIN_SIDE_PIXELS.get(group, 0))
            if fa < min_side or ba < min_side:
                return False
            continue

        # 对衣物类组更严格：不允许明显“正面大片，背面近乎无”或反过来。
        if group in STRICT_CLOTH_GROUPS:
            lo = min(fa, ba)
            hi = max(fa, ba)
            if hi >= 36 and lo <= 2:
                return False
            continue

        if group in STRICT_OBJECT_GROUPS:
            lo = min(fa, ba)
            hi = max(fa, ba)
            if hi >= 28 and lo <= 1:
                return False
            if hi >= 20 and (float(lo) / float(hi)) < 0.1:
                return False

    return True


def _passes_color_harmony(chosen_layers: List[LayerChoice]) -> bool:
    colors: List[str] = []
    for layer in chosen_layers:
        if layer.group not in CLOTH_COLOR_GROUPS:
            continue
        c = _extract_color_token(layer.path)
        if c is not None:
            colors.append(c)
    if len(colors) <= 1:
        return True

    uniq = sorted(set(colors))
    if len(uniq) > 4:
        return False

    dominant = max(set(colors), key=colors.count)
    bad = 0
    for c in uniq:
        if not _is_color_compatible(c, dominant):
            bad += 1
    return bad <= 1


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


def _passes_all_constraints(
    chosen_layers: List[LayerChoice],
    front: Image.Image,
    back: Image.Image,
    *,
    column: int,
    tile_cache: Dict[Tuple[str, int], Tuple[Image.Image, Image.Image]],
) -> bool:
    return (
        _passes_quality(front, back)
        and _passes_face_visibility(chosen_layers, column=column, tile_cache=tile_cache)
        and _passes_hair_sanity(chosen_layers, column=column, tile_cache=tile_cache)
        and _passes_color_harmony(chosen_layers)
        and _passes_front_back_consistency(chosen_layers, front, back, column=column, tile_cache=tile_cache)
    )


def _repair_layers_for_constraints(
    chosen_layers: List[LayerChoice],
    *,
    column: int,
    tile_cache: Dict[Tuple[str, int], Tuple[Image.Image, Image.Image]],
) -> Tuple[List[LayerChoice], Image.Image, Image.Image, bool]:
    repaired = list(chosen_layers)
    present_groups = {x.group for x in repaired}
    required_keep = REQUIRED_GROUPS.union({"eyes"})
    if "hair" in present_groups:
        required_keep = required_keep.union({"hair"})
    while True:
        layer_paths = [str(choice.path) for choice in repaired]
        front, back = _compose_layers_cached(layer_paths, col=column, tile_cache=tile_cache)
        if _passes_all_constraints(repaired, front, back, column=column, tile_cache=tile_cache):
            return repaired, front, back, True

        removed = False
        for grp in REPAIR_REMOVAL_PRIORITY:
            if grp in required_keep:
                continue
            idx = next((i for i, item in enumerate(repaired) if item.group == grp), None)
            if idx is not None:
                repaired.pop(idx)
                removed = True
                break
        if not removed:
            return repaired, front, back, False


def _attach_valid_hair_if_possible(
    layers: List[LayerChoice],
    *,
    body_path: Optional[Path],
    layer_pool: Dict[str, List[Path]],
    compat_pool_by_body: Dict[Path, Dict[str, List[Path]]],
    column: int,
    tile_cache: Dict[Tuple[str, int], Tuple[Image.Image, Image.Image]],
    rng: random.Random,
) -> List[LayerChoice]:
    if any(x.group == "hair" for x in layers):
        return layers
    if body_path is None:
        return layers
    candidates = compat_pool_by_body.get(body_path, {}).get("hair") or layer_pool.get("hair", [])
    if not candidates:
        return layers
    candidates = _prefer_non_foreground(candidates, "hair")
    candidates = _prefer_semantic_candidates(rng, candidates, "hair")

    pick_pool = rng.sample(candidates, min(len(candidates), 48)) if len(candidates) > 48 else list(candidates)
    for hp in pick_pool:
        trial = list(layers) + [LayerChoice(group="hair", path=hp)]
        if _passes_hair_sanity(trial, column=column, tile_cache=tile_cache) and _passes_face_visibility(
            trial,
            column=column,
            tile_cache=tile_cache,
        ):
            return trial
    return layers


def summarize_layer_pool(assets_root: str, groups: Sequence[str] = DEFAULT_LAYER_ORDER) -> Dict[str, int]:
    root = Path(assets_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Assets root does not exist: {root}")
    pool = _build_layer_pool(root, groups)
    return {g: len(pool.get(g, [])) for g in groups}


def _sample_layer_combo(
    rng: random.Random,
    layer_pool: Dict[str, List[Path]],
    compat_pool_by_body: Dict[Path, Dict[str, List[Path]]],
    groups: Sequence[str],
    usage_counts: Dict[str, Dict[Path, int]],
    diversity_strength: float,
) -> tuple[List[LayerChoice], tuple[str, ...]]:
    trial: List[LayerChoice] = []
    body_candidates = layer_pool.get("body", [])
    if not body_candidates:
        raise RuntimeError("No body candidates found in layer pool")

    body_path = _choose_candidate(
        rng,
        body_candidates,
        usage_counts.get("body", {}),
        diversity_strength,
    )
    trial.append(LayerChoice(group="body", path=body_path))
    chosen_groups = {"body"}
    compatible_pool = compat_pool_by_body.get(body_path, {})

    for group in groups:
        if group == "body":
            continue
        candidates = compatible_pool.get(group) or layer_pool.get(group)
        if not candidates:
            continue
        candidates = _prefer_non_foreground(candidates, group)
        candidates = _prefer_semantic_candidates(rng, candidates, group)
        dominant_color = _dominant_cloth_color(trial)
        candidates = _prefer_color_harmony(candidates, group=group, dominant_color=dominant_color)

        if group not in REQUIRED_GROUPS:
            p_keep = GROUP_SAMPLING_PROB.get(group, 0.5)
            if rng.random() > p_keep:
                continue

        deps = GROUP_DEPENDENCIES.get(group)
        if deps and not deps.issubset(chosen_groups):
            continue

        if group in HANDHELD_EXCLUSIVE_GROUPS and chosen_groups.intersection(HANDHELD_EXCLUSIVE_GROUPS):
            continue
        if group == "hat" and "hair" in chosen_groups and rng.random() < 0.70:
            continue
        if group == "facial" and ("hair" in chosen_groups or "hat" in chosen_groups) and rng.random() < 0.15:
            continue

        path = _choose_candidate(
            rng,
            candidates,
            usage_counts.get(group, {}),
            diversity_strength,
        )
        trial.append(LayerChoice(group=group, path=path))
        chosen_groups.add(group)

    # 降低“光头”出现：若没抽到 hair，则在兼容池里强制补一层 hair（若存在）。
    if "hair" in groups and "hair" not in chosen_groups:
        hair_candidates = compatible_pool.get("hair") or layer_pool.get("hair")
        if hair_candidates:
            hair_candidates = _prefer_non_foreground(hair_candidates, "hair")
            hair_candidates = _prefer_semantic_candidates(rng, hair_candidates, "hair")
            hair_path = _choose_candidate(
                rng,
                hair_candidates,
                usage_counts.get("hair", {}),
                diversity_strength,
            )
            trial.append(LayerChoice(group="hair", path=hair_path))
            chosen_groups.add("hair")

    trial_sig = tuple(str(item.path) for item in trial)
    return trial, trial_sig


def _compose_one_sample(
    idx: int,
    layer_pool: Dict[str, List[Path]],
    compat_pool_by_body: Dict[Path, Dict[str, List[Path]]],
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
    diversity_strength: float,
    usage_counts: Dict[str, Dict[Path, int]],
    tile_cache: Dict[Tuple[str, int], Tuple[Image.Image, Image.Image]],
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
            trial, trial_sig = _sample_layer_combo(
                rng,
                layer_pool,
                compat_pool_by_body,
                groups,
                usage_counts,
                diversity_strength,
            )
            if not enforce_unique or trial_sig not in used_local:
                chosen_layers = trial
                used_local.add(trial_sig)
                break
            chosen_layers = trial

        layer_paths = [str(choice.path) for choice in chosen_layers]
        front, back = _compose_layers_cached(
            layer_paths,
            col=column,
            tile_cache=tile_cache,
        )

        if palette_shift_prob > 0.0 and rng.random() < palette_shift_prob:
            front, back = random_palette_shift(
                front,
                back,
                p=1.0,
                max_h=palette_h,
                max_s=palette_s,
                max_v=palette_v,
            )
        elif any(x.group == "hair" for x in chosen_layers) and rng.random() < 0.4:
            # 额外提升发色多样性（轻度），避免整体都落在少量固定发色。
            front, back = random_palette_shift(
                front,
                back,
                p=1.0,
                max_h=max(palette_h, 0.1),
                max_s=max(palette_s, 0.18),
                max_v=max(palette_v, 0.16),
            )

        if _passes_all_constraints(chosen_layers, front, back, column=column, tile_cache=tile_cache):
            accepted_front, accepted_back = front, back
            break

    if accepted_front is None or accepted_back is None:
        repaired_layers, repaired_front, repaired_back, ok = _repair_layers_for_constraints(
            chosen_layers,
            column=column,
            tile_cache=tile_cache,
        )
        if ok:
            chosen_layers = repaired_layers
            accepted_front, accepted_back = repaired_front, repaired_back
        else:
            safe_layers = [choice for choice in repaired_layers if choice.group in SAFE_FALLBACK_GROUPS]
            if safe_layers:
                repaired_layers = safe_layers
            body_path = next((x.path for x in repaired_layers if x.group == "body"), None)
            repaired_layers = _attach_valid_hair_if_possible(
                repaired_layers,
                body_path=body_path,
                layer_pool=layer_pool,
                compat_pool_by_body=compat_pool_by_body,
                column=column,
                tile_cache=tile_cache,
                rng=rng,
            )
            layer_paths = [str(choice.path) for choice in repaired_layers]
            accepted_front, accepted_back = _compose_layers_cached(
                layer_paths,
                col=column,
                tile_cache=tile_cache,
            )
            chosen_layers = repaired_layers

    base_name = f"{prefix}_{idx:04d}"
    save_pair(accepted_front, accepted_back, str(out_dir), base_name)
    for layer in chosen_layers:
        usage_counts.setdefault(layer.group, {})
        usage_counts[layer.group][layer.path] = usage_counts[layer.group].get(layer.path, 0) + 1
    return chosen_layers


def _compose_one_sample_worker(args: tuple) -> Tuple[int, List[LayerChoice]]:
    (
        idx,
        layer_pool,
        compat_pool_by_body,
        groups,
        out_dir,
        seed,
        prefix,
        column,
        palette_shift_prob,
        palette_h,
        palette_s,
        palette_v,
        diversity_strength,
    ) = args

    local_usage: DefaultDict[str, Dict[Path, int]] = defaultdict(dict)
    local_cache: Dict[Tuple[str, int], Tuple[Image.Image, Image.Image]] = {}
    layers = _compose_one_sample(
        idx=idx,
        layer_pool=layer_pool,
        compat_pool_by_body=compat_pool_by_body,
        groups=groups,
        out_dir=Path(out_dir),
        seed=seed,
        prefix=prefix,
        column=column,
        palette_shift_prob=palette_shift_prob,
        palette_h=palette_h,
        palette_s=palette_s,
        palette_v=palette_v,
        diversity_strength=diversity_strength,
        usage_counts=local_usage,
        tile_cache=local_cache,
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
    palette_shift_prob: float = 0.55,
    palette_h: float = 0.12,
    palette_s: float = 0.20,
    palette_v: float = 0.18,
    diversity_strength: float = 0.8,
    num_workers: int = 0,
) -> List[LayerChoice]:
    root = Path(assets_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Assets root does not exist: {root}")

    layer_pool = _build_layer_pool(root, groups)
    compat_pool_by_body = _build_body_compat_pool(layer_pool, groups)

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
        # 默认优先单进程：可复用缓存、保留全局去重与全局多样性采样。
        num_workers = 1

    if num_workers == 1:
        used_signatures = set()
        usage_counts: DefaultDict[str, Dict[Path, int]] = defaultdict(dict)
        tile_cache: Dict[Tuple[str, int], Tuple[Image.Image, Image.Image]] = {}
        for idx in range(count):
            # 单线程保留全局去重行为
            local_rng = random.Random((seed or 0) + idx * 997 + 13)
            chosen_layers: List[LayerChoice] = []
            accepted_front = None
            accepted_back = None

            for _ in range(MAX_QUALITY_RETRY):
                for _ in range(MAX_RESAMPLE_TRIES):
                    trial, trial_sig = _sample_layer_combo(
                        local_rng,
                        layer_pool,
                        compat_pool_by_body,
                        groups,
                        usage_counts,
                        diversity_strength,
                    )
                    if trial_sig not in used_signatures:
                        chosen_layers = trial
                        used_signatures.add(trial_sig)
                        break
                    chosen_layers = trial

                layer_paths = [str(choice.path) for choice in chosen_layers]
                front, back = _compose_layers_cached(
                    layer_paths,
                    col=column,
                    tile_cache=tile_cache,
                )

                if palette_shift_prob > 0.0 and local_rng.random() < palette_shift_prob:
                    front, back = random_palette_shift(
                        front,
                        back,
                        p=1.0,
                        max_h=palette_h,
                        max_s=palette_s,
                        max_v=palette_v,
                    )
                elif any(x.group == "hair" for x in chosen_layers) and local_rng.random() < 0.4:
                    front, back = random_palette_shift(
                        front,
                        back,
                        p=1.0,
                        max_h=max(palette_h, 0.1),
                        max_s=max(palette_s, 0.18),
                        max_v=max(palette_v, 0.16),
                    )

                if _passes_all_constraints(chosen_layers, front, back, column=column, tile_cache=tile_cache):
                    accepted_front, accepted_back = front, back
                    break

            if accepted_front is None or accepted_back is None:
                repaired_layers, repaired_front, repaired_back, ok = _repair_layers_for_constraints(
                    chosen_layers,
                    column=column,
                    tile_cache=tile_cache,
                )
                if ok:
                    chosen_layers = repaired_layers
                    accepted_front, accepted_back = repaired_front, repaired_back
                else:
                    safe_layers = [choice for choice in repaired_layers if choice.group in SAFE_FALLBACK_GROUPS]
                    if safe_layers:
                        repaired_layers = safe_layers
                    body_path = next((x.path for x in repaired_layers if x.group == "body"), None)
                    repaired_layers = _attach_valid_hair_if_possible(
                        repaired_layers,
                        body_path=body_path,
                        layer_pool=layer_pool,
                        compat_pool_by_body=compat_pool_by_body,
                        column=column,
                        tile_cache=tile_cache,
                        rng=local_rng,
                    )
                    layer_paths = [str(choice.path) for choice in repaired_layers]
                    accepted_front, accepted_back = _compose_layers_cached(
                        layer_paths,
                        col=column,
                        tile_cache=tile_cache,
                    )
                    chosen_layers = repaired_layers

            base_name = f"{prefix}_{idx:04d}"
            save_pair(accepted_front, accepted_back, str(out_path), base_name)
            for layer in chosen_layers:
                usage_counts[layer.group][layer.path] = usage_counts[layer.group].get(layer.path, 0) + 1
            selections.extend(chosen_layers)
    else:
        print(f"[random_composer] parallel mode enabled: workers={num_workers}")
        task_args = [
            (
                idx,
                layer_pool,
                compat_pool_by_body,
                tuple(groups),
                str(out_path),
                seed,
                prefix,
                column,
                palette_shift_prob,
                palette_h,
                palette_s,
                palette_v,
                diversity_strength,
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
    parser.add_argument("--palette-shift-prob", type=float, default=0.55,
                        help="Probability of applying a random palette shift to each composition")
    parser.add_argument("--palette-h", type=float, default=0.12,
                        help="Maximum hue shift (normalized 0-1 range; default 0.12 ≈ 43°)")
    parser.add_argument("--palette-s", type=float, default=0.20,
                        help="Maximum saturation scaling factor (fraction)")
    parser.add_argument("--palette-v", type=float, default=0.18,
                        help="Maximum value scaling factor (fraction)")
    parser.add_argument(
        "--diversity-strength",
        type=float,
        default=0.8,
        help="Diversity bias in [0,1]: higher means prefer less-used layers (default: 0.8)",
    )
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
        diversity_strength=max(0.0, min(1.0, args.diversity_strength)),
        num_workers=args.num_workers,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
