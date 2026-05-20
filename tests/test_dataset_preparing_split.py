from __future__ import annotations

from pathlib import Path

from mlsystem2.dataset_preparing._object_counts import SceneObjectCount
from mlsystem2.dataset_preparing._scene_matching import (
    filter_existing_scenes,
    index_image_files,
    read_scene_list,
)
from mlsystem2.dataset_preparing._split import split_train_val_by_object_counts


def test_read_scene_list_ignores_empty_lines_comments_and_keeps_first_column(tmp_path: Path) -> None:
    path = tmp_path / "scenes.txt"
    path.write_text("\n# комментарий\nscene_a.tif\tmetadata\n  scene_b  \n", encoding="utf-8")

    assert read_scene_list(path) == ["scene_a.tif", "scene_b"]


def test_filter_existing_scenes_matches_filename_stem_casefold_and_normalized(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "SCENE_A.tif").write_text("", encoding="utf-8")
    (images / "scene_b.tiff").write_text("", encoding="utf-8")
    (images / "scene-cog.tif").write_text("", encoding="utf-8")

    result = filter_existing_scenes(
        ["scene_a.tif", "scene_b", "scene", "missing"],
        index_image_files(images),
    )

    assert result.existing_scenes == ["scene_a.tif", "scene_b", "scene"]
    assert result.missing_scenes == ["missing"]
    assert result.scene_to_image["scene_b"].name == "scene_b.tiff"
    assert result.scene_to_image["scene"].name == "scene-cog.tif"


def test_split_is_deterministic_and_has_no_overlap() -> None:
    rows = [
        SceneObjectCount("a", None, 10),
        SceneObjectCount("b", None, 5),
        SceneObjectCount("c", None, 1),
        SceneObjectCount("d", None, 0),
        SceneObjectCount("e", None, 0),
    ]

    first = split_train_val_by_object_counts(rows, target_val_fraction=0.2, seed=7)
    second = split_train_val_by_object_counts(rows, target_val_fraction=0.2, seed=7)

    assert [item.scene_name for item in first.train] == [item.scene_name for item in second.train]
    assert [item.scene_name for item in first.val] == [item.scene_name for item in second.val]
    assert not ({item.scene_name for item in first.train} & {item.scene_name for item in first.val})


def test_split_puts_at_least_one_object_in_val_when_possible() -> None:
    rows = [
        SceneObjectCount("large", None, 200),
        SceneObjectCount("medium", None, 120),
        SceneObjectCount("small_a", None, 90),
        SceneObjectCount("small_b", None, 80),
        SceneObjectCount("zero_a", None, 0),
        SceneObjectCount("zero_b", None, 0),
    ]

    split = split_train_val_by_object_counts(rows, target_val_fraction=0.1, seed=12)

    assert split.summary["val_objects"] > 0
    assert split.summary["train_objects"] > 0
    assert not ({item.scene_name for item in split.train} & {item.scene_name for item in split.val})
