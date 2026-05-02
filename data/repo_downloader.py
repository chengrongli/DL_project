"""Selective downloader for the Universal LPC spritesheet repository.

This module allows downloading only the spritesheet files that are
needed for training (for example the `walk.png` sheets) without cloning
the entire upstream repository. It can use either the GitHub Contents
API _or_ a single `git/trees` call to recursively traverse the desired
directories and fetch matching files.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import requests


GITHUB_API = "https://api.github.com"
DEFAULT_OWNER = "LiberatedPixelCup"
DEFAULT_REPO = "Universal-LPC-Spritesheet-Character-Generator"
DEFAULT_REF = "master"
DEFAULT_ROOT_DIRS: Sequence[str] = ("spritesheets",)
DEFAULT_PATTERNS: Sequence[str] = ("walk.png",)
DEFAULT_TIMEOUT = 30


@dataclass
class DownloadSummary:
    """Accumulates statistics for a download run."""

    downloaded: List[str] = field(default_factory=list)
    skipped_existing: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def total_candidates(self) -> int:
        return len(self.downloaded) + len(self.skipped_existing) + len(self.errors)


class _GitHubClient:
    def __init__(self, token: Optional[str] = None) -> None:
        self.session = requests.Session()
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "lpc-downloader"
        }
        if token:
            headers["Authorization"] = f"token {token}"
        self.session.headers.update(headers)

    def list_directory(self, owner: str, repo: str, path: str, ref: str) -> List[dict]:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
        response = self.session.get(url, params={"ref": ref}, timeout=DEFAULT_TIMEOUT)
        if response.status_code == 404:
            return []
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and data.get("type") == "file":
            # When path is a file, GitHub returns a dict instead of list
            return [data]
        if not isinstance(data, list):
            raise ValueError(f"Unexpected API payload for {path}: {data!r}")
        return data

    def fetch_tree(self, owner: str, repo: str, ref: str) -> dict:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{ref}"
        response = self.session.get(url, params={"recursive": 1}, timeout=DEFAULT_TIMEOUT)
        if response.status_code == 404:
            return {"tree": [], "truncated": False}
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or "tree" not in data:
            raise ValueError(f"Unexpected tree payload: {data!r}")
        return data

    def download_file(self, url: str) -> bytes:
        response = self.session.get(url, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        return response.content


def _matches_patterns(rel_path: Path, patterns: Sequence[str]) -> bool:
    rel_str = rel_path.as_posix()
    for pattern in patterns:
        if fnmatch.fnmatch(rel_str, pattern):
            return True
        if fnmatch.fnmatch(rel_path.name, pattern):
            return True
    return False


def _iter_remote_files_contents_api(
    client: _GitHubClient,
    owner: str,
    repo: str,
    ref: str,
    base_path: str,
    patterns: Sequence[str],
    current_path: Optional[str] = None,
) -> Iterable[dict]:
    path = base_path if current_path is None else f"{base_path}/{current_path}".strip("/")
    entries = client.list_directory(owner, repo, path, ref)
    for entry in entries:
        entry_path = entry.get("path")
        if not entry_path:
            continue
        rel_path = Path(entry_path)
        if entry["type"] == "dir":
            sub_rel = rel_path.relative_to(base_path)
            yield from _iter_remote_files_contents_api(
                client,
                owner,
                repo,
                ref,
                base_path,
                patterns,
                current_path=sub_rel.as_posix(),
            )
        elif entry["type"] == "file":
            rel_to_base = rel_path.relative_to(base_path)
            if _matches_patterns(rel_to_base, patterns):
                yield entry


def _iter_remote_files_tree(
    client: _GitHubClient,
    owner: str,
    repo: str,
    ref: str,
    root_dirs: Sequence[str],
    patterns: Sequence[str],
) -> Iterable[dict]:
    tree = client.fetch_tree(owner, repo, ref)
    if tree.get("truncated"):
        yield {"path": "__git_tree_truncated__", "error": "Git tree API response truncated"}
        return

    for node in tree.get("tree", []):
        if node.get("type") != "blob":
            continue
        path = node.get("path")
        if not path:
            continue
        for root in root_dirs:
            if path == root:
                rel = Path("")
            elif path.startswith(f"{root}/"):
                rel = Path(path).relative_to(root)
            else:
                continue
            if _matches_patterns(rel, patterns):
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
                yield {"path": path, "download_url": raw_url}
            break


def download_selected_assets(
    *,
    owner: str = DEFAULT_OWNER,
    repo: str = DEFAULT_REPO,
    ref: str = DEFAULT_REF,
    root_dirs: Sequence[str] = DEFAULT_ROOT_DIRS,
    patterns: Sequence[str] = DEFAULT_PATTERNS,
    out_dir: str,
    token: Optional[str] = None,
    overwrite: bool = False,
    base_url: Optional[str] = None,
    use_tree_api: bool = False,
) -> DownloadSummary:
    """Download selected spritesheet files from GitHub or a mirror.

    Args:
        base_url: Optional HTTP base used instead of GitHub's direct download URLs.
                  Useful when pulling from the live demo site (GitHub Pages).
        use_tree_api: Fetch file list via a single `git/trees` call (lower API usage).
    """
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    client = _GitHubClient(token)
    summary = DownloadSummary()

    if use_tree_api:
        entries = list(_iter_remote_files_tree(client, owner, repo, ref, root_dirs, patterns))
    else:
        entries = []
        for root in root_dirs:
            entries.extend(
                _iter_remote_files_contents_api(client, owner, repo, ref, root, patterns)
            )

    for entry in entries:
        remote_path = entry.get("path")
        if not remote_path:
            continue
        if entry.get("error"):
            summary.errors[remote_path] = entry["error"]
            continue

        if base_url:
            download_url = f"{base_url.rstrip('/')}/{remote_path}"
        else:
            download_url = entry.get("download_url")
            if not download_url:
                summary.errors[remote_path] = "Missing download_url"
                continue

        local_path = out_path / remote_path
        if local_path.exists() and not overwrite:
            summary.skipped_existing.append(str(local_path))
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            local_path.write_bytes(client.download_file(download_url))
            summary.downloaded.append(str(local_path))
        except Exception as exc:  # pragma: no cover - defensive
            summary.errors[remote_path] = str(exc)

    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selective LPC asset downloader")
    parser.add_argument("--out-dir", required=True, help="Destination directory for downloaded files")
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument(
        "--root-dirs",
        nargs="*",
        default=list(DEFAULT_ROOT_DIRS),
        help="Root directories to traverse (default: spritesheets)",
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=list(DEFAULT_PATTERNS),
        help="Filename patterns to match (default: walk.png)",
    )
    parser.add_argument("--token", help="GitHub personal access token (optional)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument(
        "--base-url",
        help="Override the download base URL (e.g. GitHub Pages mirror)",
    )
    parser.add_argument(
        "--use-tree",
        action="store_true",
        help="Use a single git/trees call instead of per-directory listings",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    token = args.token or os.getenv("GITHUB_TOKEN")
    summary = download_selected_assets(
        owner=args.owner,
        repo=args.repo,
        ref=args.ref,
        root_dirs=args.root_dirs,
        patterns=args.patterns,
        out_dir=args.out_dir,
        token=token,
        overwrite=args.overwrite,
        base_url=args.base_url,
        use_tree_api=args.use_tree,
    )

    print(f"Downloaded: {len(summary.downloaded)}")
    if summary.skipped_existing:
        print(f"Skipped (existing): {len(summary.skipped_existing)}")
    if summary.errors:
        print(f"Errors: {len(summary.errors)}")
        for path, msg in summary.errors.items():
            print(f"  {path}: {msg}")
    return 0 if not summary.errors else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
