from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mlsystem2.training_ui_api._datasets import list_classes, list_datasets


def test_list_classes_uses_git_history_for_dataset_update_dates(tmp_path: Path) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_a_main = mlmarkup_root / "class_a" / "main"
    class_b_main = mlmarkup_root / "class_b" / "main"
    class_b_test = mlmarkup_root / "class_b" / "test"
    class_a_main.mkdir(parents=True)
    class_b_main.mkdir(parents=True)
    class_b_test.mkdir(parents=True)
    (class_a_main / "class_a.txt").write_text("scene-a\n", encoding="utf-8")
    (class_a_main / "class_a.geojson").write_text("{}", encoding="utf-8")
    (class_b_main / "class_b.txt").write_text("scene-b\n", encoding="utf-8")
    (class_b_main / "class_b.geojson").write_text("{}", encoding="utf-8")
    (class_b_test / "class_b_test.txt").write_text("scene-b-test\n", encoding="utf-8")
    (class_b_test / "class_b_test.geojson").write_text("{}", encoding="utf-8")

    _git(["init"], mlmarkup_root)
    _git(["config", "user.email", "test@example.com"], mlmarkup_root)
    _git(["config", "user.name", "Test User"], mlmarkup_root)
    _git(["add", "."], mlmarkup_root)
    _git(
        ["commit", "-m", "initial datasets"],
        mlmarkup_root,
        date="2024-01-02T10:20:30+00:00",
    )
    (class_b_test / "class_b_test.geojson").write_text('{"updated":true}', encoding="utf-8")
    _git(["add", "class_b/test/class_b_test.geojson"], mlmarkup_root)
    _git(
        ["commit", "-m", "update class b"],
        mlmarkup_root,
        date="2024-03-04T05:06:07+00:00",
    )

    classes = {item.key: item for item in list_classes(mlmarkup_root)}
    datasets = {item.key: item for item in list_datasets(mlmarkup_root)}

    assert classes["class_a"].updated_at == datetime(2024, 1, 2, 10, 20, 30, tzinfo=timezone.utc)
    assert classes["class_b"].updated_at == datetime(2024, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
    assert [item.key for item in classes["class_b"].variants] == ["class_b\\main", "class_b\\test"]
    assert datasets["class_a\\main"].name == "class_a\\main"
    assert datasets["class_a\\main"].class_name == "class_a"
    assert datasets["class_a\\main"].variant_name == "main"
    assert datasets["class_b\\test"].updated_at == datetime(2024, 3, 4, 5, 6, 7, tzinfo=timezone.utc)


def _git(args: list[str], cwd: Path, *, date: str | None = None) -> None:
    env = os.environ.copy()
    if date is not None:
        env["GIT_AUTHOR_DATE"] = date
        env["GIT_COMMITTER_DATE"] = date
    try:
        subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError:
        pytest.skip("git не установлен")
