"""Sparse clone helper for the Universal LPC spritesheet repository.

This utility wraps a few git commands to fetch only the sprite assets
needed for training.  It uses partial clone + sparse checkout so that
only selected directories are downloaded instead of the entire repo.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

DEFAULT_REPO_URL = "https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator.git"
DEFAULT_BRANCH = "master"
DEFAULT_PATHS: Sequence[str] = (
    "spritesheets/body",
    "spritesheets/head",
    "spritesheets/hair",
    "spritesheets/torso",
    "spritesheets/legs",
    "spritesheets/feet",
    "spritesheets/arms",
    "spritesheets/shoulders",
    "spritesheets/hands",
    "spritesheets/dress",
    "spritesheets/cape",
    "spritesheets/hat",
    "spritesheets/eyes",
    "spritesheets/facial",
    "spritesheets/neck",
    "spritesheets/backpack",
    "spritesheets/quiver",
    "spritesheets/shield",
    "spritesheets/weapon",
    "spritesheets/tools",
    "spritesheets/shadow",
)


def _read_paths_file(path: Path) -> List[str]:
    paths: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        paths.append(stripped)
    return paths


def prepare_sparse_paths(
    *,
    extra_paths: Sequence[str] = (),
    paths_file: Optional[str] = None,
    include_defaults: bool = True,
) -> List[str]:
    paths: List[str] = []
    if include_defaults:
        paths.extend(DEFAULT_PATHS)
    if paths_file:
        paths.extend(_read_paths_file(Path(paths_file)))
    paths.extend(extra_paths)

    # Preserve order while removing duplicates
    seen = set()
    deduped = []
    for p in paths:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    return deduped


def sparse_clone(
    *,
    repo_url: str = DEFAULT_REPO_URL,
    branch: str = DEFAULT_BRANCH,
    dest_dir: str,
    paths: Sequence[str],
    depth: int = 1,
    force: bool = False,
) -> None:
    dest = Path(dest_dir).expanduser().resolve()
    if dest.exists():
        if force:
            shutil.rmtree(dest)
        else:
            raise FileExistsError(f"Destination already exists: {dest}. Use --force to overwrite.")

    clone_cmd = [
        "git",
        "clone",
        "--filter=blob:none",
        "--sparse",
        "--depth",
        str(depth),
        "--branch",
        branch,
        repo_url,
        str(dest),
    ]
    subprocess.run(clone_cmd, check=True)

    sparse_set_cmd = ["git", "sparse-checkout", "set", *paths]
    subprocess.run(sparse_set_cmd, cwd=str(dest), check=True)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse-clone the LPC spritesheet repository")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--dest", required=True, help="Destination directory for the clone")
    parser.add_argument("--path", action="append", dest="paths", default=[], help="Additional path to include")
    parser.add_argument("--paths-file", help="Text file listing extra paths (one per line)")
    parser.add_argument("--no-defaults", action="store_true", help="Do not include the default sprite directories")
    parser.add_argument("--depth", type=int, default=1, help="Shallow clone depth (default: 1)")
    parser.add_argument("--force", action="store_true", help="Delete destination if it already exists")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    sparse_paths = prepare_sparse_paths(
        extra_paths=args.paths,
        paths_file=args.paths_file,
        include_defaults=not args.no_defaults,
    )
    if not sparse_paths:
        raise SystemExit("No sparse paths specified")

    sparse_clone(
        repo_url=args.repo_url,
        branch=args.branch,
        dest_dir=args.dest,
        paths=sparse_paths,
        depth=args.depth,
        force=args.force,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
