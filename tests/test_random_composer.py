from __future__ import annotations

from pathlib import Path

from PIL import Image

from data.random_composer import random_compose_batch

TILE = 64


def _make_sheet(body_color, front_color):
    sheet = Image.new("RGBA", (TILE, TILE * 4), (0, 0, 0, 0))
    for row, color in ((0, body_color), (2, front_color)):
        tile = Image.new("RGBA", (TILE, TILE), color)
        sheet.paste(tile, (0, row * TILE))
    return sheet


def _write_walk_png(path: Path, body_color, front_color):
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet = _make_sheet(body_color, front_color)
    sheet.save(path)


def test_random_composer_generates_pairs(tmp_path: Path):
    assets = tmp_path / "assets"

    _write_walk_png(
        assets / "spritesheets/body/bodies/male/walk.png",
        (200, 200, 200, 255),
        (210, 210, 210, 255),
    )
    _write_walk_png(
        assets / "spritesheets/head/heads/human/male/walk.png",
        (150, 50, 50, 255),
        (180, 80, 80, 255),
    )
    _write_walk_png(
        assets / "spritesheets/hair/bangslong/adult/walk.png",
        (0, 0, 0, 0),
        (50, 150, 50, 200),
    )
    _write_walk_png(
        assets / "spritesheets/torso/clothes/longsleeve/longsleeve/male/walk.png",
        (0, 0, 0, 0),
        (50, 50, 150, 255),
    )
    _write_walk_png(
        assets / "spritesheets/legs/pants/male/walk.png",
        (0, 0, 0, 0),
        (80, 80, 200, 255),
    )
    _write_walk_png(
        assets / "spritesheets/feet/shoes/leather/male/walk.png",
        (0, 0, 0, 0),
        (30, 30, 30, 255),
    )

    out_dir = tmp_path / "output"
    random_compose_batch(
        assets_root=str(assets),
        out_dir=str(out_dir),
        count=3,
        seed=123,
        prefix="sample",
    )

    for idx in range(3):
        front = out_dir / f"sample_{idx:04d}_front.png"
        back = out_dir / f"sample_{idx:04d}_back.png"
        assert front.exists()
        assert back.exists()


def test_random_composer_handles_missing_optional_group(tmp_path: Path):
    assets = tmp_path / "assets"
    _write_walk_png(
        assets / "spritesheets/body/bodies/male/walk.png",
        (200, 200, 200, 255),
        (210, 210, 210, 255),
    )
    _write_walk_png(
        assets / "spritesheets/head/heads/human/male/walk.png",
        (150, 50, 50, 255),
        (180, 80, 80, 255),
    )

    out_dir = tmp_path / "output"
    random_compose_batch(
        assets_root=str(assets),
        out_dir=str(out_dir),
        count=1,
        groups=("body", "head"),
        prefix="minimal",
    )

    assert (out_dir / "minimal_0000_front.png").exists()
    assert (out_dir / "minimal_0000_back.png").exists()
