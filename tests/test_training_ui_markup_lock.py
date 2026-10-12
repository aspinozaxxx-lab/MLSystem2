from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlsystem2.training_ui_api._markup_lock import markup_read_lock, markup_write_lock


def test_worker_process_cannot_read_half_published_dataset(tmp_path: Path) -> None:
    root = tmp_path / "clone"
    root.mkdir()
    first, second = root / "first", root / "second"
    first.write_text("old")
    second.write_text("old")
    config = SimpleNamespace(mlmarkup_root=root, mlmarkup_editor_root=root)
    child_code = """
import sys
from pathlib import Path
from types import SimpleNamespace
from mlsystem2.training_ui_api._markup_lock import markup_read_lock
root = Path(sys.argv[1])
config = SimpleNamespace(mlmarkup_root=root, mlmarkup_editor_root=root)
print('ready', flush=True)
with markup_read_lock(config):
    print((root / 'first').read_text() + '|' + (root / 'second').read_text(), flush=True)
"""
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    child = None
    try:
        with markup_write_lock(config):
            first.write_text("new")
            child = subprocess.Popen(
                [sys.executable, "-c", child_code, str(root)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            assert child.stdout.readline().strip() == "ready"
            with pytest.raises(subprocess.TimeoutExpired):
                child.communicate(timeout=0.15)
            second.write_text("new")
        output, errors = child.communicate(timeout=10)
        assert child.returncode == 0, errors
        assert output.strip() == "new|new"
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate()


def test_nested_catalog_reads_inside_publication_do_not_deadlock(tmp_path: Path) -> None:
    config = SimpleNamespace(mlmarkup_root=tmp_path, mlmarkup_editor_root=tmp_path)
    with markup_write_lock(config), markup_read_lock(config), markup_write_lock(config):
        with markup_read_lock(config):
            pass
    with markup_read_lock(config):
        with pytest.raises(RuntimeError, match="публикацию"):
            with markup_write_lock(config):
                pass


def test_missing_working_source_does_not_create_directories_for_reading(tmp_path: Path) -> None:
    root = tmp_path / "missing-parent" / "clone"
    config = SimpleNamespace(mlmarkup_root=root, mlmarkup_editor_root=root)
    with markup_read_lock(config):
        assert not root.is_dir()
    assert not root.parent.exists()
