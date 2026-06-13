from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mlsystem2.training_ui_api._datasets import list_classes, list_datasets, list_image_folders


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
    initial_datasets = {item.key: item for item in list_datasets(mlmarkup_root)}
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
    assert datasets["class_a\\main"].version == initial_datasets["class_a\\main"].version
    assert datasets["class_b\\main"].version == initial_datasets["class_b\\main"].version
    assert datasets["class_b\\test"].version != initial_datasets["class_b\\test"].version
    assert datasets["class_b\\test"].version.startswith("git:")


def test_list_image_folders_returns_direct_raster_folder_counts(tmp_path: Path) -> None:
    images_root = tmp_path / "prepared_images"
    first = images_root / "kanopus" / "Olskij"
    second = images_root / "kanopus" / "irkutsk"
    empty_parent = images_root / "kanopus"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "scene-1.tif").touch()
    (first / "scene-2.tiff").touch()
    (first / "readme.txt").touch()
    (second / "scene-3.tif").touch()

    folders = list_image_folders(images_root)

    assert [(item.key, item.image_count) for item in folders] == [
        ("kanopus/irkutsk", 1),
        ("kanopus/Olskij", 2),
    ]
    assert empty_parent.exists()


def test_list_datasets_counts_images_from_folder_entries_with_dedup(tmp_path: Path) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    dataset = mlmarkup_root / "Вырубки" / "main"
    dataset.mkdir(parents=True)
    (dataset / "scenes.txt").write_text("irkutsk\nKV3_100.L2.PMS.SCN01.tif\nlost\n", encoding="utf-8")
    (dataset / "annotation.geojson").write_text("{}", encoding="utf-8")
    images_root = tmp_path / "prepared_images"
    scene_dir = images_root / "kanopus" / "irkutsk"
    scene_dir.mkdir(parents=True)
    (scene_dir / "KV3_100.L2.PMS.SCN01_cog.tif").touch()
    (scene_dir / "KV3_101.L2.PMS.SCN02.tif").touch()

    datasets = {item.key: item for item in list_datasets(mlmarkup_root, images_root)}

    assert datasets["Вырубки\\main"].image_count == 2


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
