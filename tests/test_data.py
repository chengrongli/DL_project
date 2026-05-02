"""
Tests for data utilities: spritesheet parsing, augmentation, and datasets.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.spritesheet_utils import (
    TILE_W,
    TILE_H,
    ROW_WALK_DOWN,
    ROW_WALK_UP,
    extract_tile,
    extract_front_back,
    spritesheet_to_pair,
    save_pair,
    build_pair_dataset,
)
from data.augmentation import (
    random_horizontal_flip,
    random_color_jitter,
    random_occlusion,
    to_tensor_pair,
)
from data.dataset import SpritePairDataset, FrontToBackDataset, _write_index
from data.repo_extractor import extract_pairs_from_repo


# ---------------------------------------------------------------------------
# Helpers: generate fake spritesheets
# ---------------------------------------------------------------------------

def _make_spritesheet(rows: int = 4, cols: int = 9,
                       tile_w: int = TILE_W, tile_h: int = TILE_H) -> Image.Image:
    """Create a synthetic RGBA spritesheet with distinct color per row."""
    w = cols * tile_w
    h = rows * tile_h
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    colors = [
        (200, 100, 50, 255),   # row 0 – walk up (back)
        (50, 200, 100, 255),   # row 1 – walk left
        (100, 50, 200, 255),   # row 2 – walk down (front)
        (180, 180, 50, 255),   # row 3 – walk right
    ]
    for r in range(rows):
        for c in range(cols):
            tile = Image.new("RGBA", (tile_w, tile_h), colors[r % len(colors)])
            img.paste(tile, (c * tile_w, r * tile_h))
    return img


def _save_spritesheet(tmp_dir: str, name: str = "test_sheet") -> str:
    sheet = _make_spritesheet()
    path = os.path.join(tmp_dir, f"{name}.png")
    sheet.save(path)
    return path


# ---------------------------------------------------------------------------
# spritesheet_utils
# ---------------------------------------------------------------------------

def test_extract_tile_size():
    sheet = _make_spritesheet()
    tile = extract_tile(sheet, row=0, col=0)
    assert tile.size == (TILE_W, TILE_H)


def test_extract_front_back_distinct():
    sheet = _make_spritesheet()
    front, back = extract_front_back(sheet)
    # Front (row 2) and back (row 0) should have different colors
    assert front.size == (TILE_W, TILE_H)
    assert back.size == (TILE_W, TILE_H)
    front_px = front.getpixel((TILE_W // 2, TILE_H // 2))
    back_px = back.getpixel((TILE_W // 2, TILE_H // 2))
    assert front_px != back_px, "Front and back rows should differ in color"


def test_spritesheet_to_pair_with_file(tmp_path):
    path = _save_spritesheet(str(tmp_path))
    result = spritesheet_to_pair(path)
    assert result is not None
    front, back = result
    assert front.size == (TILE_W, TILE_H)
    assert back.size == (TILE_W, TILE_H)


def test_spritesheet_to_pair_missing_file():
    result = spritesheet_to_pair("/nonexistent/path/sheet.png")
    assert result is None


def test_save_pair(tmp_path):
    sheet = _make_spritesheet()
    front, back = extract_front_back(sheet)
    fp, bp = save_pair(front, back, str(tmp_path), "test")
    assert os.path.isfile(fp)
    assert os.path.isfile(bp)


def test_build_pair_dataset(tmp_path):
    sprite_dir = tmp_path / "sprites"
    sprite_dir.mkdir()
    out_dir = tmp_path / "pairs"

    # Create 3 fake spritesheets
    for i in range(3):
        _save_spritesheet(str(sprite_dir), f"sheet_{i:03d}")

    pairs = build_pair_dataset(str(sprite_dir), str(out_dir))
    assert len(pairs) == 3
    for fp, bp in pairs:
        assert os.path.isfile(fp)
        assert os.path.isfile(bp)

# ---------------------------------------------------------------------------
# repo_extractor
# ---------------------------------------------------------------------------

def test_extract_pairs_from_repo(tmp_path):
    repo_root = tmp_path / "repo"
    male_dir = repo_root / "spritesheets" / "body" / "bodies" / "male"
    armor_dir = repo_root / "spritesheets" / "torso" / "armor"
    male_dir.mkdir(parents=True)
    armor_dir.mkdir(parents=True)

    sheet = _make_spritesheet()
    sheet.save(str(male_dir / "walk.png"))
    sheet.save(str(armor_dir / "walk.png"))

    out_dir = tmp_path / "out"
    index_path = tmp_path / "pairs.csv"

    result = extract_pairs_from_repo(
        repo_root=str(repo_root),
        out_dir=str(out_dir),
        index_path=str(index_path),
        include_patterns=("walk.png",),
    )

    assert result.successful == 2
    assert not result.failed_paths

    front_paths = {
        out_dir / "spritesheets" / "body" / "bodies" / "male" / "walk_front.png",
        out_dir / "spritesheets" / "torso" / "armor" / "walk_front.png",
    }

    for fp in front_paths:
        assert fp.exists()

    import csv

    with index_path.open(newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row and not row[0].startswith("#")]

    assert len(rows) == 2
    indexed_fronts = {row[0] for row in rows}
    assert all(str(fp) in indexed_fronts for fp in front_paths)


# ---------------------------------------------------------------------------
# augmentation
# ---------------------------------------------------------------------------

def test_horizontal_flip_deterministic():
    import random
    random.seed(0)
    front = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    back = Image.new("RGBA", (64, 64), (0, 0, 255, 255))
    f2, b2 = random_horizontal_flip(front, back, p=1.0)  # always flip
    # After flip, left pixel should equal original right pixel
    assert f2.getpixel((0, 0)) == front.getpixel((63, 0))


def test_color_jitter_preserves_size():
    front = Image.new("RGB", (64, 64), (100, 150, 200))
    back = Image.new("RGB", (64, 64), (50, 100, 150))
    f2, b2 = random_color_jitter(front, back)
    assert f2.size == front.size
    assert b2.size == back.size


def test_random_occlusion_shape():
    img = Image.new("RGBA", (64, 64), (100, 100, 100, 255))
    result = random_occlusion(img, p=1.0)  # always occlude
    assert result.size == img.size


def test_to_tensor_pair_range():
    front = Image.new("RGB", (64, 64), (128, 64, 200))
    back = Image.new("RGB", (64, 64), (0, 0, 0))
    ft, bt = to_tensor_pair(front, back, size=64)
    assert ft.shape == (3, 64, 64)
    assert bt.shape == (3, 64, 64)
    assert ft.min() >= -1.0 - 1e-5
    assert ft.max() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

def _make_pair_dir(tmp_path, n: int = 5):
    """Create a directory with n front/back PNG pairs."""
    d = tmp_path / "pairs"
    d.mkdir()
    for i in range(n):
        front = Image.new("RGBA", (64, 64), (i * 20, 100, 200, 255))
        back = Image.new("RGBA", (64, 64), (200, 100, i * 20, 255))
        front.save(str(d / f"char_{i:03d}_front.png"))
        back.save(str(d / f"char_{i:03d}_back.png"))
    return str(d)


def test_sprite_pair_dataset_len(tmp_path):
    d = _make_pair_dir(tmp_path)
    ds = SpritePairDataset(d, image_size=32, augment=False)
    assert len(ds) == 5


def test_sprite_pair_dataset_shapes(tmp_path):
    d = _make_pair_dir(tmp_path)
    ds = SpritePairDataset(d, image_size=32, augment=False)
    item = ds[0]
    assert item["front"].shape == (3, 32, 32)
    assert item["back"].shape == (3, 32, 32)
    assert item["paired"].shape == (6, 32, 32)


def test_sprite_pair_dataset_range(tmp_path):
    d = _make_pair_dir(tmp_path)
    ds = SpritePairDataset(d, image_size=32, augment=False)
    item = ds[0]
    assert item["front"].min() >= -1.0 - 1e-5
    assert item["front"].max() <= 1.0 + 1e-5


def test_front_to_back_dataset_shapes(tmp_path):
    d = _make_pair_dir(tmp_path)
    ds = FrontToBackDataset(d, image_size=32, augment=False)
    item = ds[0]
    assert item["condition"].shape == (3, 32, 32)
    assert item["target"].shape == (3, 32, 32)


def test_dataset_from_csv_index(tmp_path):
    d = _make_pair_dir(tmp_path)
    # Build CSV index manually
    import csv
    index_path = str(tmp_path / "index.csv")
    pairs = [
        (str(tmp_path / "pairs" / f"char_{i:03d}_front.png"),
         str(tmp_path / "pairs" / f"char_{i:03d}_back.png"))
        for i in range(5)
    ]
    _write_index(pairs, index_path)

    ds = SpritePairDataset(index_path, image_size=32, augment=False)
    assert len(ds) == 5


def test_dataset_missing_source():
    with pytest.raises(FileNotFoundError):
        SpritePairDataset("/nonexistent/path/", image_size=32)
