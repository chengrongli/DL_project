from __future__ import annotations

from pathlib import Path
from typing import Dict
from unittest.mock import patch

import pytest

from data_code import repo_downloader


class MockResponse:
    def __init__(self, *, json_data=None, content=None, status_code=200):
        self._json = json_data
        self.content = content
        self.status_code = status_code

    def json(self):
        if self._json is None:
            raise ValueError("No JSON payload")
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class MockSession:
    def __init__(
        self,
        api_payloads: Dict[str, list],
        file_payloads: Dict[str, bytes],
        tree_payload: Optional[dict] = None,
    ):
        self.headers = {}
        self._api_payloads = api_payloads
        self._file_payloads = file_payloads
        self._tree_payload = tree_payload or {"tree": [], "truncated": False}

    def get(self, url, params=None, timeout=None):  # noqa: D401 - mimic requests
        if "api.github.com" in url:
            if "/git/trees/" in url:
                return MockResponse(json_data=self._tree_payload)
            path = url.split("/contents/")[1]
            payload = self._api_payloads.get(path, [])
            return MockResponse(json_data=payload)
        if url in self._file_payloads:
            return MockResponse(content=self._file_payloads[url])
        raise AssertionError(f"Unexpected URL: {url}")


def test_download_selected_assets(tmp_path: Path):
    api_payloads = {
        "spritesheets": [
            {
                "type": "dir",
                "path": "spritesheets/body",
            }
        ],
        "spritesheets/body": [
            {
                "type": "file",
                "path": "spritesheets/body/walk.png",
                "name": "walk.png",
                "download_url": "https://example.com/walk.png",
            },
            {
                "type": "file",
                "path": "spritesheets/body/run.png",
                "name": "run.png",
                "download_url": "https://example.com/run.png",
            },
        ],
    }
    file_payloads = {
        "https://example.com/walk.png": b"walk-bytes",
        "https://example.com/run.png": b"run-bytes",
    }

    mock_session = MockSession(api_payloads, file_payloads)

    with patch.object(repo_downloader.requests, "Session", return_value=mock_session):
        summary = repo_downloader.download_selected_assets(
            owner="owner",
            repo="repo",
            ref="main",
            root_dirs=("spritesheets",),
            patterns=("walk.png",),
            out_dir=str(tmp_path),
        )

    expected_file = tmp_path / "spritesheets" / "body" / "walk.png"
    assert expected_file.exists()
    assert expected_file.read_bytes() == b"walk-bytes"

    assert len(summary.downloaded) == 1
    assert summary.skipped_existing == []
    # run.png should not be downloaded because it doesn't match the pattern
    other_file = tmp_path / "spritesheets" / "body" / "run.png"
    assert not other_file.exists()


def test_download_selected_assets_with_tree_api(tmp_path: Path):
    tree_payload = {
        "tree": [
            {"path": "spritesheets/body/walk.png", "type": "blob"},
            {"path": "spritesheets/body/run.png", "type": "blob"},
            {"path": "README.md", "type": "blob"},
        ],
        "truncated": False,
    }
    file_payloads = {
        "https://raw.githubusercontent.com/owner/repo/main/spritesheets/body/walk.png": b"walk-bytes",
    }
    mock_session = MockSession({}, file_payloads, tree_payload=tree_payload)

    with patch.object(repo_downloader.requests, "Session", return_value=mock_session):
        summary = repo_downloader.download_selected_assets(
            owner="owner",
            repo="repo",
            ref="main",
            root_dirs=("spritesheets",),
            patterns=("walk.png",),
            out_dir=str(tmp_path),
            use_tree_api=True,
        )

    expected_file = tmp_path / "spritesheets" / "body" / "walk.png"
    assert expected_file.exists()
    assert expected_file.read_bytes() == b"walk-bytes"
    assert len(summary.downloaded) == 1


def test_download_selected_assets_with_base_url(tmp_path: Path):
    api_payloads = {
        "spritesheets": [
            {
                "type": "dir",
                "path": "spritesheets/body",
            }
        ],
        "spritesheets/body": [
            {
                "type": "file",
                "path": "spritesheets/body/walk.png",
                "name": "walk.png",
                "download_url": None,
            }
        ],
    }
    mirror_base = "https://mirror.example.com/Universal-LPC"
    file_payloads = {
        f"{mirror_base}/spritesheets/body/walk.png": b"walk-bytes",
    }

    mock_session = MockSession(api_payloads, file_payloads)

    with patch.object(repo_downloader.requests, "Session", return_value=mock_session):
        summary = repo_downloader.download_selected_assets(
            owner="owner",
            repo="repo",
            ref="main",
            root_dirs=("spritesheets",),
            patterns=("walk.png",),
            out_dir=str(tmp_path),
            base_url=mirror_base,
        )

    expected_file = tmp_path / "spritesheets" / "body" / "walk.png"
    assert expected_file.exists()
    assert expected_file.read_bytes() == b"walk-bytes"
    assert len(summary.downloaded) == 1
