"""Randomly compose complete LPC characters from downloaded layers.

Given a local LPC asset directory (for example, the result of the sparse
clone), this script samples random combinations of body/head/hair/torso/
legs/feet layers and exports the idle front/back images for each
composition.  It builds on the same `compose_layers` helper used by the
manual layer stack utility, but removes the need to hand-write YAML.
"""

from __future__ import annotations

import argparse
import json
import os
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
    # "wings",    # banned: front/back asymmetry
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
    # "weapon",   # banned: inconsistent facing direction
    # "tools",    # banned: same issue
)

REQUIRED_GROUPS: Set[str] = {"body", "legs", "feet", "torso", "head"}

# Z-order for compositing: lower = drawn first (further back).
# Groups not listed default to 50 (mid-layer).
GROUP_Z_ORDER: Dict[str, int] = {
    "shadow": 0,
    "hair_bg": 5,       # long hair behind body
    "body": 10,
    "tail": 15,
    "legs": 20,
    "feet": 25,
    "torso": 35,
    "dress": 40,
    "arms": 50,
    "hands": 55,
    "neck": 60,
    "head": 70,
    "eyes": 75,
    "facial": 80,
    "hair": 85,
    "hat": 90,
    "shoulders": 95,
    "cape": 100,
    "backpack": 105,
    "backpack_cargo": 106,
    "quiver": 110,
    "shield": 115,
}
GROUP_SAMPLING_PROB: Dict[str, float] = {
    "shadow": 1.0,
    "dress": 0.22,
    "arms": 0.32,
    "hands": 0.52,
    "neck": 0.45,
    "eyes": 1.0,     # 眼睛强制保留，避免"无脸感"
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

# 避免随机到残缺/伤残层，生成"缺胳膊少腿"的异常角色。
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
    # 脸部优先"正常人像"
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
        "half_up",
        "loose",
        "shoulderl",
        "shoulderr",
        "single",
        "extensions",
        "idol",
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

# Torso subfolders that only cover the waist area (belts, bandages, aprons).
# Female bodies should not use these as the sole torso layer.
_TORSO_MINIMAL_SUBFOLDERS: Set[str] = {"aprons", "bandage", "waist"}

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
    # 头部限定到 heads，避免随机到"只有耳朵/附加件"导致脸异常
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

# Raw color tokens found in LPC asset paths — used for path-based extraction.
COLOR_TOKENS: Tuple[str, ...] = (
    "bluegray", "charcoal", "lavender", "maroon", "purple", "orange", "yellow",
    "green", "forest", "teal", "navy", "blue", "sky", "red", "pink", "rose",
    "brown", "leather", "walnut", "gray", "black", "white", "gold", "silver",
    "copper", "bronze", "iron", "steel", "tin",
    "slate", "crimson", "peach", "coral", "tan", "cream", "olive", "amber",
)
# Normalise raw tokens to 15 simplified colours for training.
COLOR_NORMALIZE: Dict[str, str] = {
    "bluegray": "blue", "charcoal": "gray", "lavender": "purple", "maroon": "red",
    "forest": "green", "navy": "blue", "sky": "blue", "rose": "red",
    "leather": "brown", "walnut": "brown", "iron": "silver", "steel": "silver",
    "tin": "silver", "bronze": "copper",
    "slate": "gray", "crimson": "red", "peach": "pink", "coral": "red",
    "tan": "brown", "cream": "white", "olive": "green", "amber": "orange",
}
SIMPLIFIED_COLORS: Tuple[str, ...] = (
    "black", "white", "gray", "brown", "red", "pink", "orange", "yellow",
    "green", "teal", "blue", "purple", "gold", "silver", "copper",
)
NEUTRAL_COLORS: Set[str] = {"black", "white", "gray", "silver", "brown"}

# Torso subfolder → merged torso type
TORSO_TYPE_MAP: Dict[str, str] = {
    "clothes": "clothes",
    "jacket": "jacket",
    "armour": "armour",
    "chainmail": "armour",
    "aprons": "bare",
    "bandage": "bare",
    "waist": "bare",
}

COLOR_FAMILY: Dict[str, str] = {
    "blue": "cool", "teal": "cool", "green": "cool",
    "purple": "cool",
    "red": "warm", "orange": "warm", "yellow": "warm", "pink": "warm",
    "gold": "metal", "silver": "metal", "copper": "metal",
    "black": "neutral", "white": "neutral", "gray": "neutral", "brown": "earth",
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


def _normalize_color(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    return COLOR_NORMALIZE.get(raw, raw)


# Fine-grained hair folder names → 14 simplified categories.
HAIR_STYLE_MAP: Dict[str, str] = {
    "buzzcut": "short", "flat_top_fade": "short", "flat_top_straight": "short",
    "shorthawk": "short", "pixie": "short", "high_and_tight": "short",
    "balding": "short", "relm_short": "short",
    "bob": "medium", "bob_side_part": "medium", "page": "medium", "page2": "medium",
    "plain": "medium", "lob": "medium", "sara": "medium", "natural": "medium",
    "long": "long", "long_straight": "long", "long_center_part": "long",
    "long_messy": "long", "long_messy2": "long", "long_tied": "long",
    "long_band": "long", "xlong": "long", "wavy": "long", "xlong_wavy": "long",
    "curtains_long": "long", "relm_xlong": "long",
    "ponytail": "ponytail", "ponytail2": "ponytail", "high_ponytail": "ponytail",
    "relm_ponytail": "ponytail",
    "braid": "braid", "braid2": "braid",
    "curly_long": "curly", "curly_short": "curly", "curly_short2": "curly",
    "curls_large": "curly", "curls_large_xlong": "curly", "afro": "curly",
    "jewfro": "curly",
    "spiked": "spiked", "spiked2": "spiked", "spiked_beehive": "spiked",
    "spiked_liberty": "spiked", "spiked_liberty2": "spiked",
    "spiked_porcupine": "spiked", "longhawk": "spiked",
    "bangs": "bangs", "bangslong": "bangs", "bangslong2": "bangs",
    "bangsshort": "bangs", "parted_side_bangs": "bangs",
    "parted_side_bangs2": "bangs",
    "pigtails": "pigtails", "pigtails_bangs": "pigtails", "bunches": "pigtails",
    "dreadlocks_long": "dreadlocks", "dreadlocks_short": "dreadlocks",
    "cornrows": "dreadlocks", "twists_fade": "dreadlocks",
    "twists_straight": "dreadlocks",
    "messy": "messy", "messy1": "messy", "messy2": "messy", "messy3": "messy",
    "bedhead": "messy", "unkempt": "messy", "halfmessy": "messy",
    "cowlick": "messy", "cowlick_tall": "messy", "mop": "messy",
    "parted": "parted", "parted2": "parted", "parted3": "parted",
    "curtains": "parted", "swoop": "parted", "swoop_side": "parted",
    "bangs_bun": "bun", "princess": "bun",
}

LEGS_TYPE_MAP: Dict[str, str] = {
    "pants": "pants", "pants2": "pants",
    "formal": "pants", "formal_striped": "pants", "cuffed": "pants",
    "pantaloons": "pants",
    "shorts": "shorts",
    "skirts": "skirt",
    "leggings": "leggings", "leggings2": "leggings", "hose": "leggings",
    "armour": "armour", "fur": "armour",
}

FEET_TYPE_MAP: Dict[str, str] = {
    "boots": "boots",
    "shoes": "shoes", "slippers": "shoes", "socks": "shoes", "accessory": "shoes",
    "sandals": "sandals",
    "armour": "armour", "hoofs": "armour",
}


def _extract_hair_style(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    for i, p in enumerate(parts):
        if p == "hair" and i + 1 < len(parts):
            return HAIR_STYLE_MAP.get(parts[i + 1], parts[i + 1])
    return None


def _extract_torso_subfolder(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    for i, p in enumerate(parts):
        if p == "torso" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _extract_legs_subfolder(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    for i, p in enumerate(parts):
        if p == "legs" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _extract_feet_subfolder(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    for i, p in enumerate(parts):
        if p == "feet" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _detect_skin_color(body_path: Path) -> Optional[str]:
    """Detect skin color from the body layer by sampling the torso region."""
    try:
        sheet = load_spritesheet(str(body_path))
        body_front, _ = extract_front_back(sheet)
    except Exception:
        return None

    arr = np.array(body_front.convert("RGBA"))
    alpha = arr[:, :, 3] > 128

    h, w = alpha.shape
    top = h // 3
    bot = 2 * h // 3
    region = np.zeros_like(alpha)
    region[top:bot, :] = True
    mask = alpha & region

    if not mask.any():
        mask = alpha

    pixels = arr[mask][:, :3].astype(float)
    if len(pixels) == 0:
        return None

    brightness = pixels.mean(axis=1)
    keep = (brightness > 20) & (brightness < 240)
    pixels = pixels[keep]
    if len(pixels) == 0:
        return None

    median_rgb = np.median(pixels, axis=0)

    best_color = None
    best_dist = float("inf")
    for name, ref in _SIMPLIFIED_COLOR_RGB.items():
        d = sum((float(median_rgb[i]) - ref[i]) ** 2 for i in range(3))
        if d < best_dist:
            best_dist = d
            best_color = name
    return best_color


def _extract_attributes(chosen_layers: List[LayerChoice]) -> dict:
    attrs: dict = {
        "body_type": "adult",
        "hair_color": None,
        "hair_style": None,
        "torso_type": "bare",
        "torso_color": None,
        "legs_type": "pants",
        "legs_color": None,
        "feet_type": "shoes",
        "feet_color": None,
    }
    body_path = None
    has_dress = False
    for layer in chosen_layers:
        g = layer.group
        p = layer.path
        if g == "body":
            style = _infer_body_style(p)
            attrs["body_type"] = "adult" if style == "pregnant" else style
            body_path = p
        elif g == "hair":
            attrs["hair_color"] = None  # filled later by pixel detection
            attrs["hair_style"] = _extract_hair_style(p)
        elif g == "torso":
            raw_sub = _extract_torso_subfolder(p)
            torso_type = TORSO_TYPE_MAP.get(raw_sub, "bare")
            attrs["torso_type"] = torso_type
            if torso_type != "bare":
                attrs["torso_color"] = _normalize_color(_extract_color_token(p))
        elif g == "legs":
            raw_sub = _extract_legs_subfolder(p)
            attrs["legs_type"] = LEGS_TYPE_MAP.get(raw_sub, "pants")
            attrs["legs_color"] = _normalize_color(_extract_color_token(p))
        elif g == "feet":
            raw_sub = _extract_feet_subfolder(p)
            attrs["feet_type"] = FEET_TYPE_MAP.get(raw_sub, "shoes")
            attrs["feet_color"] = _normalize_color(_extract_color_token(p))
        elif g == "dress":
            has_dress = True

    # Dress covers the legs area.
    if has_dress:
        attrs["legs_type"] = "dress"

    if attrs["torso_type"] == "bare" and body_path is not None:
        attrs["torso_color"] = _detect_skin_color(body_path)

    return attrs


@dataclass(frozen=True)
class LayerChoice:
    group: str
    path: Path


def _rotate_tile_hue(tile: Image.Image, hue_delta: float) -> Image.Image:
    """Rotate hue of all non-transparent pixels in a single tile."""
    import numpy as np

    rgba = np.array(tile.convert("RGBA"))
    mask = rgba[:, :, 3] > 0
    if not mask.any():
        return tile

    pixels = rgba[mask, :3].astype(np.float32) / 255.0
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    v = cmax
    s = np.where(cmax > 1e-6, delta / cmax, 0.0)

    h = np.zeros_like(r)
    mr = (cmax == r) & (delta > 1e-6)
    mg = (cmax == g) & (delta > 1e-6) & ~mr
    mb = (cmax == b) & (delta > 1e-6) & ~mr & ~mg
    h[mr] = ((g[mr] - b[mr]) / delta[mr]) % 6.0
    h[mg] = ((b[mg] - r[mg]) / delta[mg]) + 2.0
    h[mb] = ((r[mb] - g[mb]) / delta[mb]) + 4.0
    h /= 6.0
    h = (h + hue_delta) % 1.0

    i = (h * 6.0).astype(int) % 6
    f = (h * 6.0) - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    rgb = np.zeros_like(pixels)
    for idx, (rv, gv, bv) in enumerate([(v,t,p), (q,v,p), (p,v,t), (p,q,v), (t,p,v), (v,p,q)]):
        sel = i == idx
        rgb[sel, 0] = rv[sel]
        rgb[sel, 1] = gv[sel]
        rgb[sel, 2] = bv[sel]

    rgba[mask, :3] = (rgb * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _rotate_hair_hue(
    img: Image.Image,
    hair_path: Path,
    hue_delta: float,
    column: int = 0,
    tile_cache: Optional[Dict] = None,
    is_back: bool = False,
) -> Image.Image:
    """Rotate hue of hair pixels only in the composed image."""
    import numpy as np

    sheet = load_spritesheet(str(hair_path))
    hair_front, hair_back = extract_front_back(sheet)
    hair_tile = hair_back if is_back else hair_front

    hair_alpha = np.array(hair_tile.getchannel("A")) > 0
    if not hair_alpha.any():
        return img

    rgba = np.array(img.convert("RGBA"))
    mask = hair_alpha & (rgba[:, :, 3] > 0)
    if not mask.any():
        return img

    pixels = rgba[mask, :3].astype(np.float32) / 255.0

    # Vectorised HSV hue rotation using numpy
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Value
    v = cmax
    # Saturation
    s = np.where(cmax > 0, delta / cmax, 0.0)
    # Hue
    h = np.zeros_like(r)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0) & ~mask_r
    mask_b = (cmax == b) & (delta > 0) & ~mask_r & ~mask_g
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0
    h /= 6.0  # normalise to [0, 1]

    h = (h + hue_delta) % 1.0

    # HSV to RGB (vectorised)
    i = (h * 6.0).astype(int) % 6
    f = (h * 6.0) - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    rgb = np.zeros_like(pixels)
    for idx, (rv, gv, bv) in enumerate([(v,t,p), (q,v,p), (p,v,t), (p,q,v), (t,p,v), (v,p,q)]):
        sel = i == idx
        rgb[sel, 0] = rv[sel]
        rgb[sel, 1] = gv[sel]
        rgb[sel, 2] = bv[sel]

    rgba[mask, :3] = (rgb * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def _detect_hair_color(front_img: Image.Image, hair_path: Path) -> Optional[str]:
    """Detect hair colour from the composed front image using the hair tile alpha mask."""
    import numpy as np

    try:
        sheet = load_spritesheet(str(hair_path))
        hair_tile, _ = extract_front_back(sheet)
    except Exception:
        return None

    hair_arr = np.array(hair_tile.convert("RGBA"))
    hair_alpha = hair_arr[:, :, 3] > 128
    if not hair_alpha.any():
        return None

    front_arr = np.array(front_img.convert("RGBA").resize(hair_tile.size, Image.NEAREST))
    pixels = front_arr[hair_alpha][:, :3].astype(float)

    if len(pixels) == 0:
        return None

    # Filter out extreme shadows / highlights
    brightness = pixels.mean(axis=1)
    mask = (brightness > 20) & (brightness < 240)
    pixels = pixels[mask]
    if len(pixels) == 0:
        return None

    # Median colour
    median_rgb = np.median(pixels, axis=0)

    # Map to nearest simplified colour using Lab distance
    try:
        from colorsys import rgb_to_lab_approx
    except ImportError:
        pass

    # Simple RGB distance fallback — good enough for 15 coarse categories
    best_color = None
    best_dist = float("inf")
    for name, ref in _SIMPLIFIED_COLOR_RGB.items():
        d = sum((float(median_rgb[i]) - ref[i]) ** 2 for i in range(3))
        if d < best_dist:
            best_dist = d
            best_color = name
    return best_color


_SIMPLIFIED_COLOR_RGB: Dict[str, Tuple[int, int, int]] = {
    "black": (30, 30, 30), "white": (235, 235, 235), "gray": (128, 128, 128),
    "brown": (139, 90, 43), "red": (200, 30, 30), "pink": (255, 150, 180),
    "orange": (230, 130, 30), "yellow": (230, 220, 50), "green": (50, 160, 50),
    "teal": (0, 160, 160), "blue": (50, 80, 200), "purple": (140, 50, 180),
    "gold": (220, 180, 40), "silver": (180, 180, 190), "copper": (180, 100, 50),
}


def _is_blocked_path(path: Path) -> bool:
    text = str(path).lower()
    return any(token in text for token in PATH_BLOCKLIST)


def _is_group_allowed(group: str, path: Path) -> bool:
    # Match blocklist tokens against individual path PARTS using EXACT match.
    # Substring matching on the full path string caused false positives,
    # e.g. "rat" matching "universal-lpc-spritesheet-character-generator".
    parts = {p.lower() for p in path.parts}
    blocked = GROUP_BLOCKLIST.get(group, ())
    if parts.intersection(blocked):
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


def _build_lpc_compat_index(
    sheet_defs_root: str | Path,
    assets_root: str | Path,
    body_types: Sequence[str] = ("male", "female", "muscular", "teen", "child", "pregnant"),
) -> Dict[str, Dict[str, Set[Path]]]:
    """Build a precise body-type → allowed asset paths index from LPC sheet_definitions.

    Reads every JSON file in sheet_definitions/, extracts which body types each
    asset defines paths for, then resolves those paths against the actual asset
    directory.

    Returns: {group_name: {body_type: set_of_allowed_paths}}
    """
    import json as _json

    defs = Path(sheet_defs_root)
    assets = Path(assets_root)
    result: Dict[str, Dict[str, Set[Path]]] = {}

    # Map LPC sheet_definitions category names to our layer group names.
    # Some categories (dress, jacket, shirts, etc.) all map to "torso" in our system.
    CATEGORY_TO_GROUP = {
        "hair": "hair",
        "torso": "torso",
        "legs": "legs",
        "feet": "feet",
        "head": "head",
        "headwear": "hat",
        "body": "body",
    }

    def _collect_json_files(directory: Path) -> list:
        """Recursively collect all non-meta JSON files."""
        out = []
        if not directory.exists():
            return out
        for root, dirs, files in os.walk(directory):
            for f in sorted(files):
                if f.endswith(".json") and not f.startswith("meta_"):
                    out.append(Path(root) / f)
        return out

    def _extract_paths_for_bodytype(data: dict, body_type: str, category_dir: Path) -> Set[Path]:
        """Extract actual file paths from layer definitions for a given body type."""
        paths = set()
        # Check layer_1, layer_2, etc.
        for key, val in data.items():
            if not key.startswith("layer"):
                continue
            if not isinstance(val, dict):
                continue
            rel_path = val.get(body_type)
            if rel_path is None:
                continue
            # rel_path is like "hair/bangs/adult/" — resolve to actual files
            abs_dir = assets / rel_path
            if abs_dir.exists():
                for p in abs_dir.rglob("*.png"):
                    paths.add(p.resolve())
        return paths

    # Walk each category directory
    for category_name in ("hair", "torso", "legs", "feet", "head"):
        cat_dir = defs / category_name
        group = CATEGORY_TO_GROUP.get(category_name, category_name)

        if group not in result:
            result[group] = {bt: set() for bt in body_types}

        for json_file in _collect_json_files(cat_dir):
            try:
                data = _json.load(open(json_file))
            except Exception:
                continue

            for bt in body_types:
                resolved = _extract_paths_for_bodytype(data, bt, cat_dir)
                result[group][bt].update(resolved)

    return result


def _build_body_compat_pool(
    layer_pool: Dict[str, List[Path]],
    groups: Sequence[str],
) -> Dict[Path, Dict[str, List[Path]]]:
    body_candidates = layer_pool.get("body", [])

    # Try to load LPC sheet_definitions for precise filtering
    lpc_defs = Path(__file__).resolve().parent.parent / "Universal-LPC-Spritesheet-Character-Generator" / "sheet_definitions"
    assets_root = layer_pool.get("body", [Path(".")])[0]
    # Walk up to find assets root (parent of spritesheets/)
    for p in [assets_root] + list(assets_root.parents):
        if (p / "spritesheets").exists():
            assets_root = p
            break

    lpc_index = None
    if lpc_defs.exists():
        try:
            lpc_index = _build_lpc_compat_index(lpc_defs, assets_root / "spritesheets")
        except Exception:
            lpc_index = None

    result: Dict[Path, Dict[str, List[Path]]] = {}
    for body_path in body_candidates:
        body_style = _infer_body_style(body_path)
        by_group: Dict[str, List[Path]] = {}
        for group in groups:
            candidates = layer_pool.get(group, [])
            if not candidates:
                continue

            compatible = candidates

            # Use precise LPC index if available
            if lpc_index is not None and group in lpc_index:
                allowed = lpc_index[group].get(body_style, set())
                if not allowed:
                    # muscular shares male assets in LPC
                    if body_style == "muscular":
                        allowed = lpc_index[group].get("male", set())
                    # pregnant shares female assets
                    elif body_style == "pregnant":
                        allowed = lpc_index[group].get("female", set())

                if allowed:
                    filtered = [c for c in candidates if c.resolve() in allowed]
                    if filtered:
                        compatible = filtered

            # Head: additional gender-specific filtering
            if group == "head":
                strict_head = _filter_head_compatible(compatible, body_style)
                if strict_head:
                    compatible = strict_head

            # Blocklist
            compatible = [c for c in compatible if _is_group_allowed(group, c)]

            if not compatible:
                if group in STRICT_BODY_COMPAT_GROUPS:
                    if group in REQUIRED_GROUPS:
                        compatible = [c for c in candidates if _is_group_allowed(group, c)]
                    else:
                        continue
                else:
                    compatible = [c for c in candidates if _is_group_allowed(group, c)]
            by_group[group] = compatible
        result[body_path] = by_group
    return result


def _infer_body_style(body_path: Path) -> str:
    parts = [p.lower() for p in body_path.parts]
    for style in ("child", "teen", "female", "male", "muscular"):
        if style in parts:
            return style
    if "pregnant" in parts:
        return "female"
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
        tokens.add("adult")
        tokens.add("male")
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


def _shift_tile_y(tile: Image.Image, offset: int) -> Image.Image:
    """Shift a tile up (negative) or down (positive) by *offset* pixels."""
    w, h = tile.size
    result = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    result.paste(tile, (0, offset))
    return result


def _compose_layers_cached(
    layer_paths: List[str],
    *,
    col: int,
    tile_cache: Dict[Tuple[str, int], Tuple[Image.Image, Image.Image]],
    layer_groups: Optional[Sequence[str]] = None,
    group_y_offsets: Optional[Dict[str, int]] = None,
) -> Tuple[Image.Image, Image.Image]:
    front_canvas = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    back_canvas = Image.new("RGBA", (64, 64), (0, 0, 0, 0))

    for i, path in enumerate(layer_paths):
        tiles = _get_cached_tiles(path, col=col, tile_cache=tile_cache)
        ft, bt = tiles[0], tiles[1]

        # Apply per-group y-offset (e.g. shift hair up for child bodies).
        if layer_groups and group_y_offsets and i < len(layer_groups):
            y_off = group_y_offsets.get(layer_groups[i], 0)
            if y_off != 0:
                ft = _shift_tile_y(ft, y_off)
                bt = _shift_tile_y(bt, y_off)

        front_canvas = Image.alpha_composite(front_canvas, ft)
        back_canvas = Image.alpha_composite(back_canvas, bt)

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

    # Child bodies use a y-shift to reposition hair, so the raw tile
    # overlap ratio is misleadingly high.  Skip the overlap check for
    # child and rely on the shift + colour contrast instead.
    body_layer = next((layer for layer in chosen_layers if layer.group == "body"), None)
    is_child = body_layer is not None and _infer_body_style(body_layer.path) == "child"

    hair_front, _ = _get_cached_tiles(str(hair_layer.path), col=column, tile_cache=tile_cache)
    head_front, _ = _get_cached_tiles(str(head_layer.path), col=column, tile_cache=tile_cache)
    eyes_front, _ = _get_cached_tiles(str(eyes_layer.path), col=column, tile_cache=tile_cache)

    hair_mask = np.array(hair_front.getchannel("A")) > 0
    hair_pixels = int(hair_mask.sum())
    if hair_pixels < MIN_HAIR_FRONT_PIXELS:
        return False
    if hair_pixels > MAX_HAIR_FRONT_PIXELS:
        return False

    if not is_child:
        face_mask = (np.array(head_front.getchannel("A")) > 0) | (np.array(eyes_front.getchannel("A")) > 0)
        face_pixels = int(face_mask.sum())
        if face_pixels <= 0:
            return False
        overlap = int((hair_mask & face_mask).sum())
        if (float(overlap) / float(face_pixels)) > MAX_HAIR_FACE_OVERLAP_RATIO:
            return False

    # 头发与头部基底颜色太接近时，看起来会像"光头"。
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

        # 对衣物类组更严格：不允许明显"正面大片，背面近乎无"或反过来。
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


def _child_hair_y_offset(chosen_layers: List[LayerChoice]) -> Optional[Dict[str, int]]:
    """Return a y-offset dict for hair+hat if body is child, else None."""
    body = next((l for l in chosen_layers if l.group == "body"), None)
    if body and _infer_body_style(body.path) == "child":
        offsets: Dict[str, int] = {"hair": -4}
        if any(l.group == "hat" for l in chosen_layers):
            offsets["hat"] = -4
        return offsets
    return None


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
        groups = [l.group for l in repaired]
        y_off = _child_hair_y_offset(repaired)
        front, back = _compose_layers_cached(
            layer_paths, col=column, tile_cache=tile_cache,
            layer_groups=groups, group_y_offsets=y_off,
        )
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
            # Last resort for child: drop hair (child hair tiles have
            # inherently high overlap and the y-shift may not be enough).
            # For adults, keep hair to avoid baldness.
            body_l = next((l for l in repaired if l.group == "body"), None)
            is_child_body = body_l is not None and _infer_body_style(body_l.path) == "child"
            if is_child_body:
                hair_idx = next((i for i, item in enumerate(repaired) if item.group == "hair"), None)
                if hair_idx is not None:
                    repaired.pop(hair_idx)
                    present_groups.discard("hair")
                    required_keep.discard("hair")
                    continue
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


def _add_hair_bg_layer(trial: list, hair_path: Path) -> None:
    """Add a hair_bg layer if the hair tile has a dual fg/bg structure.

    LPC dual-layer hair styles have fg/ (drawn above head) and bg/ (drawn behind
    body). If we picked a fg/ tile, find the matching bg/ tile and add it as
    group "hair_bg". If we picked a bg/ tile, find the matching fg/ tile and
    swap it into "hair" while the bg/ becomes "hair_bg".
    """
    p = hair_path
    parts = [x.lower() for x in p.parts]

    if "fg" in parts:
        # Have fg, need bg
        bg_path = Path(str(p).replace("/fg/", "/bg/"))
        if bg_path.exists():
            trial.append(LayerChoice(group="hair_bg", path=bg_path))
    elif "bg" in parts:
        # Have bg, need fg — swap: fg becomes the main hair, bg becomes hair_bg
        fg_path = Path(str(p).replace("/bg/", "/fg/"))
        if fg_path.exists():
            # Remove the bg entry we just added as "hair", re-add as hair_bg
            trial[:] = [l for l in trial if not (l.group == "hair" and l.path == hair_path)]
            trial.append(LayerChoice(group="hair", path=fg_path))
            trial.append(LayerChoice(group="hair_bg", path=hair_path))


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

    # Exclude child and teen bodies — their small size causes face/hair rendering issues.
    body_candidates = [
        p for p in body_candidates
        if _infer_body_style(p) not in ("child", "teen")
    ]
    if not body_candidates:
        raise RuntimeError("No non-child/teen body candidates found")

    body_path = _choose_candidate(
        rng,
        body_candidates,
        usage_counts.get("body", {}),
        diversity_strength,
    )
    trial.append(LayerChoice(group="body", path=body_path))
    chosen_groups = {"body"}
    compatible_pool = compat_pool_by_body.get(body_path, {})

    body_style = _infer_body_style(body_path)

    for group in groups:
        if group == "body":
            continue
        candidates = compatible_pool.get(group) or layer_pool.get(group)
        if not candidates:
            continue
        candidates = _prefer_non_foreground(candidates, group)
        candidates = _prefer_semantic_candidates(rng, candidates, group)

        # Female bodies: exclude minimal torso layers (aprons/bandage/waist)
        # so they always get proper upper-body clothing.
        if group == "torso" and body_style in ("female", "pregnant"):
            clothed = [
                p for p in candidates
                if _extract_torso_subfolder(p) not in _TORSO_MINIMAL_SUBFOLDERS
            ]
            if clothed:
                candidates = clothed

        # Male and muscular bodies should not wear dresses.
        if group == "dress" and body_style in ("male", "muscular"):
            continue

        # Exclude feminine hairstyles for male bodies.
        if group == "hair" and body_style in ("male", "muscular"):
            _FEMININE_HAIR_KEYWORDS = {"pigtails", "bunches"}
            candidates = [
                p for p in candidates
                if not any(kw in str(p).lower() for kw in _FEMININE_HAIR_KEYWORDS)
            ]
            if not candidates:
                continue

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
        if group == "hat" and "hair" in chosen_groups:
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

        # Dual-layer hair: if hair path is in fg/ or bg/ subdir,
        # add the complementary bg/ or fg/ tile as hair_bg group.
        if group == "hair":
            _add_hair_bg_layer(trial, path)

    #降低光头出现
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
            # Also add bg layer for dual-layer hair
            _add_hair_bg_layer(trial, hair_path)

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

        # Pre-rotate hair tile hue for child bodies so colour is uniform.
        child_hue_key = None
        child_hair_pre = next((l for l in chosen_layers if l.group == "hair"), None)
        child_body_pre = next((l for l in chosen_layers if l.group == "body"), None)
        if child_hair_pre and child_body_pre and _infer_body_style(child_body_pre.path) == "child":
            ch_hue = rng.uniform(-0.15, 0.15)
            ch_sheet = load_spritesheet(str(child_hair_pre.path))
            ch_hf, ch_hb = extract_front_back(ch_sheet)
            ch_hf = _rotate_tile_hue(ch_hf, ch_hue)
            ch_hb = _rotate_tile_hue(ch_hb, ch_hue)
            child_hue_key = (str(child_hair_pre.path), column)
            tile_cache[child_hue_key] = (ch_hf, ch_hb)

        y_off = _child_hair_y_offset(chosen_layers)
        # Sort layers by z-order so hair_bg renders behind body, etc.
        chosen_layers.sort(key=lambda l: GROUP_Z_ORDER.get(l.group, 50))
        layer_paths = [str(choice.path) for choice in chosen_layers]
        groups = [l.group for l in chosen_layers]
        front, back = _compose_layers_cached(
            layer_paths,
            col=column,
            tile_cache=tile_cache,
            layer_groups=groups,
            group_y_offsets=y_off,
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

        if _passes_all_constraints(chosen_layers, front, back, column=column, tile_cache=tile_cache):
            accepted_front, accepted_back = front, back
            if child_hue_key is not None:
                tile_cache.pop(child_hue_key, None)
            break

        if child_hue_key is not None:
            tile_cache.pop(child_hue_key, None)

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
            repaired_layers.sort(key=lambda l: GROUP_Z_ORDER.get(l.group, 50))
            layer_paths = [str(choice.path) for choice in repaired_layers]
            y_off2 = _child_hair_y_offset(repaired_layers)
            groups2 = [l.group for l in repaired_layers]
            accepted_front, accepted_back = _compose_layers_cached(
                layer_paths,
                col=column,
                tile_cache=tile_cache,
                layer_groups=groups2,
                group_y_offsets=y_off2,
            )
            chosen_layers = repaired_layers

    # Hair hue rotation REMOVED — was causing hair to become invisible
    # or mismatch with the pixel-detected hair_color label.

    base_name = f"{prefix}_{idx:04d}"
    save_pair(accepted_front, accepted_back, str(out_dir), base_name)
    for layer in chosen_layers:
        usage_counts.setdefault(layer.group, {})
        usage_counts[layer.group][layer.path] = usage_counts[layer.group].get(layer.path, 0) + 1
    return chosen_layers


def _compose_one_sample_worker(args: tuple) -> Tuple[int, List[LayerChoice], dict]:
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
    attrs = _extract_attributes(layers)

    # Fix hair_color: pixel-based detection when path extraction fails
    if attrs.get("hair_color") is None:
        hair_path = next((l.path for l in layers if l.group == "hair"), None)
        if hair_path is not None:
            front_path = Path(out_dir) / f"{prefix}_{idx:04d}_front.png"
            if front_path.exists():
                front_img = Image.open(front_path).convert("RGBA")
                attrs["hair_color"] = _detect_hair_color(front_img, hair_path)

    return idx, layers, attrs


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

    attributes_data: Dict[str, dict] = {}

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

                # Pre-rotate hair tile hue
                seq_hair_pre = next((l for l in chosen_layers if l.group == "hair"), None)
                seq_rotated_key = None
                if seq_hair_pre is not None:
                    seq_hue = local_rng.uniform(-0.15, 0.15)
                    seq_sheet = load_spritesheet(str(seq_hair_pre.path))
                    seq_hf, seq_hb = extract_front_back(seq_sheet)
                    seq_hf = _rotate_tile_hue(seq_hf, seq_hue)
                    seq_hb = _rotate_tile_hue(seq_hb, seq_hue)
                    seq_rotated_key = (str(seq_hair_pre.path), column)
                    tile_cache[seq_rotated_key] = (seq_hf, seq_hb)

                chosen_layers.sort(key=lambda l: GROUP_Z_ORDER.get(l.group, 50))
                layer_paths = [str(choice.path) for choice in chosen_layers]
                seq_y_off = _child_hair_y_offset(chosen_layers)
                seq_groups = [l.group for l in chosen_layers]
                front, back = _compose_layers_cached(
                    layer_paths,
                    col=column,
                    tile_cache=tile_cache,
                    layer_groups=seq_groups,
                    group_y_offsets=seq_y_off,
                )

                if seq_rotated_key is not None:
                    del tile_cache[seq_rotated_key]

                if palette_shift_prob > 0.0 and local_rng.random() < palette_shift_prob:
                    front, back = random_palette_shift(
                        front,
                        back,
                        p=1.0,
                        max_h=palette_h,
                        max_s=palette_s,
                        max_v=palette_v,
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
                    repaired_layers.sort(key=lambda l: GROUP_Z_ORDER.get(l.group, 50))
                    layer_paths = [str(choice.path) for choice in repaired_layers]
                    fb_y_off = _child_hair_y_offset(repaired_layers)
                    fb_groups = [l.group for l in repaired_layers]
                    accepted_front, accepted_back = _compose_layers_cached(
                        layer_paths,
                        col=column,
                        tile_cache=tile_cache,
                        layer_groups=fb_groups,
                        group_y_offsets=fb_y_off,
                    )
                    chosen_layers = repaired_layers

            # Hair hue rotation REMOVED — was causing hair to become invisible
            # or mismatch with the pixel-detected hair_color label.

            base_name = f"{prefix}_{idx:04d}"
            save_pair(accepted_front, accepted_back, str(out_path), base_name)
            attrs = _extract_attributes(chosen_layers)
            if attrs.get("hair_color") is None:
                hair_path = next((l.path for l in chosen_layers if l.group == "hair"), None)
                if hair_path is not None:
                    front_path = Path(str(out_path)) / f"{prefix}_{idx:04d}_front.png"
                    if front_path.exists():
                        front_img = Image.open(front_path).convert("RGBA")
                        attrs["hair_color"] = _detect_hair_color(front_img, hair_path)
            attributes_data[base_name] = attrs
            # Record chosen layers for debugging
            attrs["_layers"] = [{"group": l.group, "path": str(l.path)} for l in chosen_layers]
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
            for idx, chosen_layers, attrs in ex.map(_compose_one_sample_worker, task_args):
                base_name = f"{prefix}_{idx:04d}"
                attrs["_layers"] = [{"group": l.group, "path": str(l.path)} for l in chosen_layers]
                attributes_data[base_name] = attrs
                selections.extend(chosen_layers)

    if attributes_data:
        attrs_path = out_path / "attributes.json"
        with open(attrs_path, "w") as f:
            json.dump(attributes_data, f, indent=2)
        print(f"[random_composer] saved attributes for {len(attributes_data)} samples to {attrs_path}")

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
