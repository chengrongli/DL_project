from __future__ import annotations

from pathlib import Path

from PIL import Image

from data.layer_stack import compose_character

TILE = 64


def _make_sheet(back_color, front_color):
    sheet = Image.new("RGBA", (TILE, TILE * 4), (0, 0, 0, 0))
    for row, color in [(0, back_color), (2, front_color)]:
        tile = Image.new("RGBA", (TILE, TILE), color)
        sheet.paste(tile, (0, row * TILE))
    return sheet


def test_compose_character(tmp_path: Path):
    root = tmp_path / "assets"
    body_dir = root / "spritesheets" / "body"
    body_dir.mkdir(parents=True)

    base_sheet = _make_sheet((10, 10, 200, 255), (20, 20, 220, 255))
    base_path = body_dir / "base.png"
    base_sheet.save(base_path)

    overlay_sheet = _make_sheet((0, 0, 0, 0), (50, 200, 50, 255))
    overlay_path = body_dir / "overlay.png"
    overlay_sheet.save(overlay_path)

    front_path, back_path = compose_character(
        assets_root=str(root),
        layers=[
            "spritesheets/body/base.png",
            "spritesheets/body/overlay.png",
        ],
        out_dir=str(tmp_path / "output"),
        name="hero",
    )

    front_img = Image.open(front_path)
    back_img = Image.open(back_path)

    assert front_img.getpixel((TILE // 2, TILE // 2)) == (50, 200, 50, 255)
    assert back_img.getpixel((TILE // 2, TILE // 2)) == (10, 10, 200, 255)
