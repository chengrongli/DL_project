"""
LPC Spritesheet utilities.

Parses Universal LPC Spritesheet Character Generator assets to extract
front/back sprite pairs for training.

LPC spritesheet layout (standard 64×64 tile grid):
  Row 0: walk-up   (back)
  Row 1: walk-left
  Row 2: walk-down (front)
  Row 3: walk-right

Each row contains WALK_FRAMES (9) columns of animation frames.
We use frame index 0 (stand-still) from each direction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# Standard LPC tile dimensions
TILE_W = 64
TILE_H = 64

# Row indices for each walk direction in a standard LPC sheet
ROW_WALK_UP = 0      # back view
ROW_WALK_LEFT = 1
ROW_WALK_DOWN = 2    # front view
ROW_WALK_RIGHT = 3

# Column index for the idle (stand-still) frame
COL_IDLE = 0


def load_spritesheet(path: str) -> Image.Image:
    """Load a spritesheet image from disk and convert to RGBA."""
    img = Image.open(path).convert("RGBA")
    return img


def extract_tile(
    sheet: Image.Image,
    row: int,
    col: int,
    tile_w: int = TILE_W,
    tile_h: int = TILE_H,
) -> Image.Image:
    """
    Extract a single tile from a spritesheet.

    Args:
        sheet: Full spritesheet image.
        row:   Tile row index (0-based).
        col:   Tile column index (0-based).
        tile_w: Tile width in pixels.
        tile_h: Tile height in pixels.

    Returns:
        RGBA image of the extracted tile.
    """
    x = col * tile_w
    y = row * tile_h
    tile = sheet.crop((x, y, x + tile_w, y + tile_h))
    return tile


def extract_front_back(
    sheet: Image.Image,
    tile_w: int = TILE_W,
    tile_h: int = TILE_H,
    col: int = COL_IDLE,
) -> Tuple[Image.Image, Image.Image]:
    """
    Extract the front-view and back-view tiles from an LPC spritesheet.

    Returns:
        (front_tile, back_tile) both as RGBA PIL Images.
    """
    front = extract_tile(sheet, ROW_WALK_DOWN, col, tile_w, tile_h)
    back = extract_tile(sheet, ROW_WALK_UP, col, tile_w, tile_h)
    return front, back


def compose_layers(
    layer_paths: List[str],
    base_size: Tuple[int, int] = (TILE_W, TILE_H),
    col: int = COL_IDLE,
) -> Tuple[Image.Image, Image.Image]:
    """
    Composite multiple LPC layer spritesheets into a single front/back pair.

    Layer paths are composited in order (first layer is bottom-most).

    Args:
        layer_paths: Ordered list of spritesheet file paths.
        base_size:   Output image size (width, height).
        col:         Column index (animation frame).

    Returns:
        (front_composite, back_composite) as RGBA PIL Images.
    """
    front_canvas = Image.new("RGBA", base_size, (0, 0, 0, 0))
    back_canvas = Image.new("RGBA", base_size, (0, 0, 0, 0))

    for path in layer_paths:
        sheet = load_spritesheet(path)
        front_tile, back_tile = extract_front_back(sheet, base_size[0], base_size[1], col)
        front_canvas = Image.alpha_composite(front_canvas, front_tile)
        back_canvas = Image.alpha_composite(back_canvas, back_tile)

    return front_canvas, back_canvas


def collect_spritesheet_pairs(
    root_dir: str,
    extensions: Tuple[str, ...] = (".png",),
) -> List[str]:
    """
    Recursively collect all spritesheet file paths under root_dir.

    Returns a sorted list of absolute file paths.
    """
    root = Path(root_dir)
    paths: List[str] = []
    for ext in extensions:
        paths.extend(str(p) for p in root.rglob(f"*{ext}"))
    return sorted(paths)


def spritesheet_to_pair(
    path: str,
    col: int = COL_IDLE,
) -> Optional[Tuple[Image.Image, Image.Image]]:
    """
    Load a single spritesheet and return its (front, back) idle tiles.

    Returns None if the file cannot be opened or is too small.
    """
    try:
        sheet = load_spritesheet(path)
    except Exception:
        return None

    min_h = (max(ROW_WALK_DOWN, ROW_WALK_UP) + 1) * TILE_H
    min_w = (col + 1) * TILE_W
    w, h = sheet.size
    if w < min_w or h < min_h:
        return None

    return extract_front_back(sheet, TILE_W, TILE_H, col)


def save_pair(
    front: Image.Image,
    back: Image.Image,
    out_dir: str,
    stem: str,
) -> Tuple[str, str]:
    """Save a front/back pair to disk and return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    front_path = os.path.join(out_dir, f"{stem}_front.png")
    back_path = os.path.join(out_dir, f"{stem}_back.png")
    front.save(front_path)
    back.save(back_path)
    return front_path, back_path


def build_pair_dataset(
    sprite_dir: str,
    out_dir: str,
    col: int = COL_IDLE,
) -> List[Tuple[str, str]]:
    """
    Scan sprite_dir for all spritesheets, extract idle front/back pairs,
    save them under out_dir, and return a list of (front_path, back_path).

    Args:
        sprite_dir: Root directory containing LPC spritesheets.
        out_dir:    Output directory for extracted pair images.
        col:        Column index (animation frame) to extract.

    Returns:
        List of (front_image_path, back_image_path) tuples.
    """
    paths = collect_spritesheet_pairs(sprite_dir)
    pairs: List[Tuple[str, str]] = []

    for path in paths:
        result = spritesheet_to_pair(path, col)
        if result is None:
            continue
        front, back = result
        stem = Path(path).stem
        fp, bp = save_pair(front, back, out_dir, stem)
        pairs.append((fp, bp))

    return pairs
