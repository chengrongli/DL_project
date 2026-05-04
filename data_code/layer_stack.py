"""Compose full front/back character sprites from selected LPC layers.

This utility mirrors the browser generator by stacking multiple
spritesheet layers in a fixed order and extracting the idle front/back
frame. Supply either repeated `--layer` arguments or a YAML/JSON file
listing the relative spritesheet paths you want to combine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

from data_code.spritesheet_utils import compose_layers, save_pair


def _load_layers_from_file(path: Path) -> List[str]:
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to read YAML layer files. Install it (pip install PyYAML)."
            )
        data = yaml.safe_load(path.read_text())
    else:
        data = json.loads(path.read_text())

    if isinstance(data, dict):
        layers = data.get("layers")
    else:
        layers = data

    if not isinstance(layers, list):
        raise ValueError("Layer config must be a list or a mapping with a 'layers' key")

    normalized: List[str] = []
    for entry in layers:
        if not isinstance(entry, str):
            raise ValueError("Layer entries must be strings with relative paths")
        normalized.append(entry)
    return normalized


def compose_character(
    assets_root: str,
    layers: Sequence[str],
    out_dir: str,
    name: str,
    column: int = 0,
) -> tuple[Path, Path]:
    root = Path(assets_root).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    full_paths = []
    for rel in layers:
        abs_path = root / rel
        if not abs_path.is_file():
            raise FileNotFoundError(f"Layer file not found: {abs_path}")
        full_paths.append(str(abs_path))

    front, back = compose_layers(full_paths, col=column)
    front_path, back_path = save_pair(front, back, str(out_path), name)
    return Path(front_path), Path(back_path)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose LPC character from selected layers")
    parser.add_argument("--assets-root", required=True, help="Root directory containing downloaded spritesheets")
    parser.add_argument("--out-dir", required=True, help="Where to save the composed front/back PNGs")
    parser.add_argument("--name", required=True, help="Base filename for the composed pair")
    parser.add_argument("--column", type=int, default=0, help="Spritesheet column index to extract (default: 0)")
    parser.add_argument(
        "--layer",
        action="append",
        dest="layers",
        help="Relative path to a spritesheet layer (can be specified multiple times)",
    )
    parser.add_argument(
        "--layers-file",
        help="YAML/JSON file listing layer paths (use when many layers are needed)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    layers: List[str] = []
    if args.layers_file:
        layers.extend(_load_layers_from_file(Path(args.layers_file)))
    if args.layers:
        layers.extend(args.layers)
    if not layers:
        raise SystemExit("No layers provided. Use --layer or --layers-file.")

    compose_character(
        assets_root=args.assets_root,
        layers=layers,
        out_dir=args.out_dir,
        name=args.name,
        column=args.column,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
