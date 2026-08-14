from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mlsystem2.training_ui_api._datasets import (
    find_image_folder,
    list_classes,
    list_datasets,
    list_image_folders,
    resolve_scenes_file_images,
)


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
    assert [item.key for item in classes["class_b"].datasets] == [
        "class_b\\main",
        "class_b\\test",
    ]
    assert datasets["class_a\\main"].name == "class_a\\main"
    assert datasets["class_a\\main"].class_name == "class_a"
    assert datasets["class_a\\main"].dataset_name == "main"
    assert datasets["class_b\\test"].updated_at == datetime(2024, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
    assert datasets["class_a\\main"].version == initial_datasets["class_a\\main"].version
    assert datasets["class_b\\main"].version == initial_datasets["class_b\\main"].version
    assert datasets["class_b\\test"].version != initial_datasets["class_b\\test"].version
    assert datasets["class_b\\test"].version.startswith("git:")


def test_list_datasets_uses_atomic_release_metadata_without_git(tmp_path: Path) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    dataset = mlmarkup_root / "Реки" / "test"
    dataset.mkdir(parents=True)
    (dataset / "Olskij_SCN06.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )
    release_commit = "a" * 40
    dataset_commit = "b" * 40
    (mlmarkup_root / ".mlsystem2-release").write_text(release_commit + "\n", encoding="utf-8")
    (mlmarkup_root / ".mlsystem2-release-metadata.json").write_text(
        json.dumps(
            {
                "release_commit": release_commit,
                "datasets": {
                    "Реки/test": {
                        "commit": dataset_commit,
                        "committed_at": "2026-08-09T10:20:30+00:00",
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = {item.key: item for item in list_datasets(mlmarkup_root)}["Реки\\test"]

    assert result.version == f"git:{dataset_commit}"
    assert result.updated_at == datetime(2026, 8, 9, 10, 20, 30, tzinfo=timezone.utc)


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
    ortho = images_root / "orto" / "ryazan"
    ortho.mkdir(parents=True)
    (ortho / "ortho-1.tif").touch()
    ignored = images_root / "sentinel" / "region"
    ignored.mkdir(parents=True)
    (ignored / "ignored.tif").touch()

    folders = list_image_folders(images_root)

    assert [(item.key, item.image_count, item.imagery_type) for item in folders] == [
        ("kanopus/irkutsk", 1, "kanopus"),
        ("kanopus/Olskij", 2, "kanopus"),
        ("orto/ryazan", 1, "ortho"),
    ]
    assert empty_parent.exists()
    assert find_image_folder(images_root, "orto/ryazan", "ortho") is not None
    assert find_image_folder(images_root, "orto/ryazan", "kanopus") is None


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


def test_scene_relative_path_selects_only_the_exact_duplicate_filename(
    tmp_path: Path,
) -> None:
    images_root = tmp_path / "prepared_images" / "orto"
    first = images_root / "first" / "shared.tif"
    second = images_root / "second" / "shared.tif"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()
    scenes = tmp_path / "scenes.txt"
    scenes.write_text("first/shared.tif\n", encoding="utf-8")

    assert resolve_scenes_file_images(scenes, images_root) == [first.resolve()]


def test_list_datasets_splits_positive_and_hard_negative_geojson(tmp_path: Path) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    dataset = mlmarkup_root / "Вырубки" / "main"
    dataset.mkdir(parents=True)
    (dataset / "scenes.txt").write_text("scene-1\n", encoding="utf-8")
    (dataset / "annotation.geojson").write_text("{}", encoding="utf-8")
    (dataset / "hard_negative.geojson").write_text("{}", encoding="utf-8")

    datasets = {item.key: item for item in list_datasets(mlmarkup_root)}

    assert datasets["Вырубки\\main"].annotation_file == str(dataset / "annotation.geojson")
    assert datasets["Вырубки\\main"].hard_negative_annotation_file == str(
        dataset / "hard_negative.geojson"
    )
    assert datasets["Вырубки\\main"].diagnostics == []


def test_list_datasets_detects_per_image_format_and_counts_geojson(
    tmp_path: Path,
) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    dataset = mlmarkup_root / "Реки" / "test"
    dataset.mkdir(parents=True)
    (dataset / "Olskij_SCN06.geojson").write_text("{}", encoding="utf-8")
    (dataset / "Olskij_SCN07.geojson").write_text("{}", encoding="utf-8")
    (dataset / "Olskij_SCN06_footprint.geojson").write_text("{}", encoding="utf-8")
    (dataset / "Olskij_SCN07_footprint.geojson").write_text("{}", encoding="utf-8")
    images_root = tmp_path / "prepared_images"
    images = images_root / "kanopus" / "Olskij"
    images.mkdir(parents=True)
    (images / "SCN06.tif").touch()
    (images / "SCN07.tif").touch()

    info = {
        item.key: item for item in list_datasets(mlmarkup_root, images_root)
    }["Реки\\test"]

    assert info.format == "per_image"
    assert info.annotations_dir == str(dataset)
    assert info.scenes_file is None
    assert info.annotation_file is None
    assert info.image_count == 2
    assert info.diagnostics == []


def test_list_datasets_reports_ambiguous_positive_geojson(tmp_path: Path) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    dataset = mlmarkup_root / "Вырубки" / "main"
    dataset.mkdir(parents=True)
    (dataset / "scenes.txt").write_text("scene-1\n", encoding="utf-8")
    (dataset / "a.geojson").write_text("{}", encoding="utf-8")
    (dataset / "b.geojson").write_text("{}", encoding="utf-8")
    (dataset / "hard_negative.geojson").write_text("{}", encoding="utf-8")

    datasets = {item.key: item for item in list_datasets(mlmarkup_root)}

    assert datasets["Вырубки\\main"].annotation_file is None
    assert datasets["Вырубки\\main"].hard_negative_annotation_file == str(
        dataset / "hard_negative.geojson"
    )
    assert datasets["Вырубки\\main"].diagnostics


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
