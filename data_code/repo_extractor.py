"""Utilities to extract paired front/back sprites from the
Universal LPC Spritesheet Character Generator repository.

This module scans a cloned copy of the upstream repository, gathers
spritesheets that contain the standard four walking directions, and
exports the idle frame (column 0) for the front (row 2) and back (row 0)
views. The extracted pairs are saved to disk while mirroring the original
folder structure so that filenames remain unique. A CSV index
(front_path, back_path) can optionally be written alongside the images.

Example
-------

```
python -m data.repo_extractor \
    --repo-root /path/to/Universal-LPC-Spritesheet-Character-Generator \
    --out-dir data/pairs/lpc_repo \
    --index data/index_lpc_repo.csv \
    --patterns "*walk.png" "*run.png"
```
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from data_code.spritesheet_utils import spritesheet_to_pair


DEFAULT_SEARCH_DIRS: Tuple[str, ...] = ("spritesheets",)
DEFAULT_PATTERNS: Tuple[str, ...] = ("*walk.png",)


@dataclass
class ExtractionResult:
    """Summary of the extraction process."""

    pairs: List[Tuple[str, str]]
    total_candidates: int
    failed_paths: List[str]

    @property
    def successful(self) -> int:
        return len(self.pairs)

    @property
    def skipped(self) -> int:
        return self.total_candidates - self.successful


def _collect_candidates(
    repo_root: Path,
    search_dirs: Sequence[str],
    patterns: Sequence[str],
) -> List[Path]:
    """Return unique, sorted spritesheet paths matching the given patterns."""
    candidates: set[Path] = set()
    for rel_dir in search_dirs:
        base = repo_root / rel_dir
        if not base.is_dir():
            continue
        for pattern in patterns:
            for path in base.rglob(pattern):
                if path.is_file():
                    candidates.add(path)
    return sorted(candidates)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_pair(
    front,  # PIL.Image.Image
    back,
    out_dir: Path,
    repo_root: Path,
    src_path: Path,
) -> Tuple[Path, Path]:
    """Save a front/back pair mirroring the source directory structure."""
    rel = src_path.relative_to(repo_root)
    front_path = out_dir / rel.parent / f"{rel.stem}_front.png"
    back_path = out_dir / rel.parent / f"{rel.stem}_back.png"
    _ensure_parent(front_path)
    _ensure_parent(back_path)
    front.save(front_path)
    back.save(back_path)
    return front_path, back_path


def _write_index(pairs: Sequence[Tuple[str, str]], index_path: Path) -> None:
    """Write a CSV index of front/back image paths."""
    _ensure_parent(index_path)
    with index_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# front_path", "back_path"])
        writer.writerows(pairs)


def extract_pairs_from_repo(
    repo_root: str,
    out_dir: str,
    index_path: Optional[str] = None,
    *,
    include_patterns: Sequence[str] = DEFAULT_PATTERNS,
    search_dirs: Sequence[str] = DEFAULT_SEARCH_DIRS,
    column: int = 0,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> ExtractionResult:
    """Extract idle front/back pairs from an LPC repository clone."""
    root = Path(repo_root).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    idx_path = Path(index_path).expanduser().resolve() if index_path else None

    candidates = _collect_candidates(root, search_dirs, include_patterns)

    pairs: List[Tuple[str, str]] = []
    failed: List[str] = []

    for sprite_path in candidates:
        result = spritesheet_to_pair(str(sprite_path), col=column)
        if result is None:
            failed.append(str(sprite_path))
            continue

        front_img, back_img = result
        if dry_run:
            front_path = out / sprite_path.name
            back_path = out / sprite_path.name
        else:
            front_path, back_path = _save_pair(front_img, back_img, out, root, sprite_path)

        pairs.append((str(front_path), str(back_path)))

        if limit is not None and len(pairs) >= limit:
            break

    if idx_path and not dry_run:
        _write_index(pairs, idx_path)

    return ExtractionResult(pairs=pairs, total_candidates=len(candidates), failed_paths=failed)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract front/back sprite pairs from the LPC repo")
    parser.add_argument("--repo-root", required=True, help="Path to the Universal LPC repository clone")
    parser.add_argument("--out-dir", required=True, help="Directory to write extracted front/back pairs")
    parser.add_argument("--index", help="Optional CSV index file to write")
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=list(DEFAULT_PATTERNS),
        help="Glob patterns (relative) used to select spritesheets (default: *walk.png)",
    )
    parser.add_argument(
        "--search-dirs",
        nargs="*",
        default=list(DEFAULT_SEARCH_DIRS),
        help="Relative directories to search within the repository (default: spritesheets)",
    )
    parser.add_argument("--column", type=int, default=0, help="Spritesheet column index to extract (default: 0)")
    parser.add_argument("--limit", type=int, help="Optional limit on the number of pairs to extract")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without writing any files")
    return parser.parse_args(argv)


def _format_summary(result: ExtractionResult) -> str:
    return (
        f"Total candidates: {result.total_candidates}\n"
        f"Successful: {result.successful}\n"
        f"Skipped: {result.skipped}\n"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = extract_pairs_from_repo(
        repo_root=args.repo_root,
        out_dir=args.out_dir,
        index_path=args.index,
        include_patterns=args.patterns,
        search_dirs=args.search_dirs,
        column=args.column,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    print(_format_summary(result))

    if result.failed_paths:
        print("Skipped files (layout mismatch):")
        for path in result.failed_paths:
            print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
