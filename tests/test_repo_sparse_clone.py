from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from data import repo_sparse_clone


def test_prepare_sparse_paths_defaults_and_extra(tmp_path: Path):
    paths_file = tmp_path / "paths.txt"
    paths_file.write_text("# comment\nextra/one\nextra/two\n")
    result = repo_sparse_clone.prepare_sparse_paths(
        extra_paths=["spritesheets/custom"],
        paths_file=str(paths_file),
        include_defaults=False,
    )
    assert result == ["extra/one", "extra/two", "spritesheets/custom"]


def test_sparse_clone_commands(tmp_path: Path):
    dest = tmp_path / "lpc"
    paths = ["spritesheets/body", "spritesheets/head"]
    calls: List[tuple] = []

    def fake_run(cmd, cwd=None, check=None):
        calls.append((tuple(cmd), cwd))

    with patch.object(repo_sparse_clone.subprocess, "run", side_effect=fake_run):
        repo_sparse_clone.sparse_clone(dest_dir=str(dest), paths=paths, force=False)

    clone_cmd, clone_cwd = calls[0]
    set_cmd, set_cwd = calls[1]

    assert clone_cmd[:3] == ("git", "clone", "--filter=blob:none")
    assert clone_cwd is None
    assert set_cmd[:3] == ("git", "sparse-checkout", "set")
    assert set_cmd[3:] == tuple(paths)
    assert set_cwd == str(dest)


def test_sparse_clone_force_overwrites(tmp_path: Path):
    dest = tmp_path / "existing"
    dest.mkdir()

    with patch.object(repo_sparse_clone.subprocess, "run") as mock_run:
        repo_sparse_clone.sparse_clone(dest_dir=str(dest), paths=["spritesheets/body"], force=True)

    assert mock_run.call_count == 2
    assert not dest.exists()


def test_sparse_clone_errors_when_dest_exists(tmp_path: Path):
    dest = tmp_path / "existing"
    dest.mkdir()

    with pytest.raises(FileExistsError):
        repo_sparse_clone.sparse_clone(dest_dir=str(dest), paths=["spritesheets/body"], force=False)
