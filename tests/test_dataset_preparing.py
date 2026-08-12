from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from mlsystem2.dataset_preparing.api import prepare_dataset, resolve_scene_images
from mlsystem2.dataset_preparing.contracts import (
    DatasetClassRequest,
    DatasetPreparationRequest,
    SceneImageResolutionRequest,
)


def test_prepare_dataset_returns_independent_scenes(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _write_raster(images / "scene_a.tif", 1, 0)
    _write_raster(images / "scene_b.tif", 2, 4)
    _write_raster(images / "scene_c.tif", 3, 8)
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text("scene_a\nscene_b\nscene_c\n", encoding="utf-8")
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation(annotation_file, ["scene_a.tif", "scene_a.tif", "scene_b.tif", "scene_c.tif"])

    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images),
            scenes_file=str(scenes_file),
            annotation_file=str(annotation_file),
            val_fraction=0.5,
        )
    )

    assert result.report.status == "ok", result.report.errors
    assert result.dataset is not None
    assert result.dataset.format == "legacy_binary"
    assert [scene.scene_id for scene in result.dataset.scenes] == [
        "scene_a",
        "scene_b",
        "scene_c",
    ]
    assert all(scene.annotation_file is None for scene in result.dataset.scenes)
    assert not any("vrt" in key for key in type(result.dataset).model_fields)
    for scene in result.dataset.scenes:
        with rasterio.open(scene.image_path) as raster:
            assert raster.count == 1


def test_prepare_dataset_report_counts_hard_negative_objects(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _write_raster(images / "scene_a.tif", 1, 0)
    _write_raster(images / "scene_b.tif", 2, 4)
    _write_raster(images / "scene_c.tif", 3, 8)
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text("scene_a\nscene_b\nscene_c\n", encoding="utf-8")
    annotation_file = tmp_path / "annotations.geojson"
    hard_negative_file = tmp_path / "hard_negative.geojson"
    _write_annotation(annotation_file, ["scene_a.tif", "scene_a.tif", "scene_b.tif", "scene_c.tif"])
    _write_annotation(hard_negative_file, ["scene_b.tif", "scene_c.tif", "scene_c.tif"])

    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images),
            scenes_file=str(scenes_file),
            annotation_file=str(annotation_file),
            hard_negative_annotation_file=str(hard_negative_file),
            val_fraction=0.5,
        )
    )

    assert result.report.status == "ok"
    assert result.dataset is not None
    assert result.dataset.hard_negative_annotation_file == hard_negative_file.resolve().as_posix()
    assert result.report.positive_objects == 4
    assert result.report.hard_negative_objects == 3
    assert result.report.objects_total == 7
    assert "train_scenes_count" not in result.report.model_dump()
    scene_by_id = {scene.scene_id: scene for scene in result.report.scenes}
    assert scene_by_id["scene_a"].positive_objects == 2
    assert scene_by_id["scene_a"].hard_negative_objects == 0
    assert scene_by_id["scene_a"].object_count == 2
    assert scene_by_id["scene_c"].positive_objects == 1
    assert scene_by_id["scene_c"].hard_negative_objects == 2
    assert scene_by_id["scene_c"].object_count == 3
    assert "split" not in result.report.scenes[0].model_dump()


def test_prepare_dataset_multiclass_merges_scenes_and_assigns_class_ids(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _write_raster(images / "scene_a.tif", 1, 0)
    _write_raster(images / "scene_b.tif", 2, 4)
    _write_raster(images / "scene_c.tif", 3, 8)
    class_a_scenes = tmp_path / "class_a.txt"
    class_b_scenes = tmp_path / "class_b.txt"
    class_c_scenes = tmp_path / "class_c.txt"
    class_a_scenes.write_text("scene_a\nscene_b\n", encoding="utf-8")
    class_b_scenes.write_text("scene_b\nscene_c\n", encoding="utf-8")
    class_c_scenes.write_text("", encoding="utf-8")
    class_a_annotation = tmp_path / "class_a.geojson"
    class_b_annotation = tmp_path / "class_b.geojson"
    class_c_annotation = tmp_path / "class_c.geojson"
    class_a_hard_negative = tmp_path / "class_a_hard_negative.geojson"
    _write_annotation(class_a_annotation, ["scene_a.tif", "scene_b.tif"])
    _write_annotation(class_b_annotation, ["scene_b.tif", "scene_c.tif"])
    _write_annotation(class_c_annotation, ["scene_a.tif"])
    _write_annotation(class_a_hard_negative, ["scene_c.tif", "scene_c.tif"])

    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images),
            classes=[
                DatasetClassRequest(
                    slug="class_a",
                    name="Класс А",
                    scenes_file=str(class_a_scenes),
                    annotation_file=str(class_a_annotation),
                    hard_negative_annotation_file=str(class_a_hard_negative),
                ),
                DatasetClassRequest(
                    slug="class_b",
                    name="Класс Б",
                    scenes_file=str(class_b_scenes),
                    annotation_file=str(class_b_annotation),
                ),
                DatasetClassRequest(
                    slug="class_c",
                    name="Класс В",
                    scenes_file=str(class_c_scenes),
                    annotation_file=str(class_c_annotation),
                ),
            ],
            val_fraction=0.5,
        )
    )

    assert result.report.status == "ok"
    assert result.report.scenes_total == 3
    assert result.report.scenes_found == 3
    assert result.report.positive_objects == 5
    assert result.report.hard_negative_objects == 2
    assert result.report.objects_total == 7
    assert result.dataset is not None
    assert result.dataset.annotation_file is None
    assert result.dataset.class_annotations[0].hard_negative_annotation_file == (
        class_a_hard_negative.resolve().as_posix()
    )
    assert [item.class_id for item in result.dataset.class_annotations] == [1, 2, 3]
    assert [item.slug for item in result.dataset.class_annotations] == [
        "class_a",
        "class_b",
        "class_c",
    ]
    assert [item.priority for item in result.dataset.class_annotations] == [0, 0, 0]
    assert result.dataset.format == "legacy_multiclass"
    assert len(result.dataset.scenes) == 3


def test_prepare_dataset_tile_mode_keeps_all_binary_scenes(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for index, scene in enumerate(("scene_a", "scene_b", "scene_c", "scene_d")):
        _write_raster(images / f"{scene}.tif", index + 1, index * 4)
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text("scene_a\nscene_b\nscene_c\nscene_d\n", encoding="utf-8")
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation(annotation_file, ["scene_a.tif", "scene_c.tif"])

    request = DatasetPreparationRequest(
        images_dir=str(images),
        scenes_file=str(scenes_file),
        annotation_file=str(annotation_file),
        val_fraction=0.5,
    )
    first = prepare_dataset(request)
    second = prepare_dataset(request)

    assert first.report.status == "ok"
    assert first.dataset is not None
    assert _scene_ids(first) == _scene_ids(second)
    assert [scene.model_dump() for scene in first.dataset.scenes] == [
        scene.model_dump() for scene in second.dataset.scenes
    ]
    assert set(_scene_ids(first)) == {"scene_a", "scene_b", "scene_c", "scene_d"}
    assert all(scene.image_path is not None for scene in first.report.scenes)


def test_prepare_dataset_tile_mode_keeps_all_multiclass_scenes(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for index, scene in enumerate(("scene_a", "scene_b", "scene_c")):
        _write_raster(images / f"{scene}.tif", index + 1, index * 4)
    class_a_scenes = tmp_path / "class_a.txt"
    class_b_scenes = tmp_path / "class_b.txt"
    class_a_scenes.write_text("scene_a\nscene_b\nscene_c\n", encoding="utf-8")
    class_b_scenes.write_text("", encoding="utf-8")
    class_a_annotation = tmp_path / "class_a.geojson"
    class_b_annotation = tmp_path / "class_b.geojson"
    _write_annotation(class_a_annotation, ["scene_a.tif"])
    _write_annotation(class_b_annotation, [])

    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images),
            classes=[
                DatasetClassRequest(
                    slug="class_a",
                    name="Класс А",
                    scenes_file=str(class_a_scenes),
                    annotation_file=str(class_a_annotation),
                ),
                DatasetClassRequest(
                    slug="class_b",
                    name="Класс Б",
                    scenes_file=str(class_b_scenes),
                    annotation_file=str(class_b_annotation),
                ),
            ],
            val_fraction=0.5,
        )
    )

    assert result.report.status == "ok"
    assert result.dataset is not None
    assert [item.slug for item in result.dataset.class_annotations] == ["class_a", "class_b"]
    assert set(_scene_ids(result)) >= {"scene_a"}
    assert len(_scene_ids(result)) == 3


def test_prepare_dataset_keeps_scenes_with_different_resolution_and_grid_independent(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _write_raster(images / "scene_a.tif", 1, 0, crs="EPSG:3857", pixel_size=1.0)
    _write_raster(
        images / "scene_b.tif",
        2,
        2.25,
        crs="EPSG:3857",
        pixel_size=0.5,
        top=4.25,
    )
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text("scene_a\nscene_b\n", encoding="utf-8")
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation(annotation_file, ["scene_a.tif", "scene_b.tif"])

    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images),
            scenes_file=str(scenes_file),
            annotation_file=str(annotation_file),
            val_fraction=0.5,
        )
    )

    assert result.report.status == "ok"
    assert result.dataset is not None
    assert [Path(scene.image_path).name for scene in result.dataset.scenes] == [
        "scene_a.tif",
        "scene_b.tif",
    ]
    with rasterio.open(result.dataset.scenes[0].image_path) as first:
        assert first.res == (1.0, 1.0)
    with rasterio.open(result.dataset.scenes[1].image_path) as second:
        assert second.res == (0.5, 0.5)


def test_prepare_dataset_does_not_merge_overlapping_scenes(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _write_raster(images / "lower.tif", 10, 0)
    _write_masked_raster_with_invalid_white_edge(images / "upper.tif")
    _write_raster(images / "val_scene.tif", 20, 8)
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text("lower\nupper\nval_scene\n", encoding="utf-8")
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation(
        annotation_file,
        ["lower.tif"] + ["upper.tif"] * 100 + ["val_scene.tif"] * 25,
    )

    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images),
            scenes_file=str(scenes_file),
            annotation_file=str(annotation_file),
            val_fraction=0.2,
        )
    )

    assert result.report.status == "ok"
    assert result.dataset is not None
    assert [Path(scene.image_path).name for scene in result.dataset.scenes] == [
        "lower.tif",
        "upper.tif",
        "val_scene.tif",
    ]
    with rasterio.open(result.dataset.scenes[0].image_path) as dataset:
        data = dataset.read(1, window=((0, 1), (0, 1)), masked=False)
        assert int(data[0, 0]) == 10


def test_prepare_per_image_dataset_matches_names_and_counts_roles(tmp_path: Path) -> None:
    images = tmp_path / "images"
    first_folder = images / "Ольский"
    second_folder = images / "Магадан"
    first_folder.mkdir(parents=True)
    second_folder.mkdir(parents=True)
    first_image = first_folder / "SCN06.tif"
    second_image = second_folder / "SCN07.tiff"
    _write_raster(first_image, 1, 0)
    _write_raster(second_image, 2, 4)
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    _write_per_image_annotation(
        annotations / "Ольский_SCN06.geojson",
        [None, "hard_negative", "hard_negative"],
    )
    _write_per_image_annotation(
        annotations / "Магадан_SCN07.geojson",
        ["positive"],
    )

    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images),
            annotations_dir=str(annotations),
            val_fraction=0.2,
        )
    )

    assert result.report.status == "ok"
    assert result.report.scenes_total == 2
    assert result.report.scenes_found == 2
    assert result.report.positive_objects == 2
    assert result.report.hard_negative_objects == 2
    assert result.dataset is not None
    assert result.dataset.format == "per_image_binary"
    assert [scene.scene_id for scene in result.dataset.scenes] == [
        "Магадан/SCN07",
        "Ольский/SCN06",
    ]
    assert all(scene.annotation_file is not None for scene in result.dataset.scenes)


def test_prepare_per_image_dataset_accepts_empty_feature_collection(tmp_path: Path) -> None:
    images = tmp_path / "images" / "folder"
    images.mkdir(parents=True)
    _write_raster(images / "scene.tif", 1, 0)
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    _write_per_image_annotation(annotations / "folder_scene.geojson", [])

    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images.parent),
            annotations_dir=str(annotations),
            val_fraction=0.2,
        )
    )

    assert result.report.status == "ok"
    assert result.report.objects_total == 0
    assert result.dataset is not None


def test_prepare_per_image_dataset_reports_missing_and_ambiguous_tiff(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    first = images / "one" / "same"
    second = images / "two" / "same"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    _write_raster(first / "scene.tif", 1, 0)
    _write_raster(second / "scene.tif", 2, 4)
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    _write_per_image_annotation(annotations / "same_scene.geojson", [])
    _write_per_image_annotation(annotations / "missing_scene.geojson", [])

    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images),
            annotations_dir=str(annotations),
            val_fraction=0.2,
        )
    )

    assert result.dataset is None
    assert result.report.status == "error"
    assert result.report.scenes_total == 2
    assert result.report.scenes_found == 0
    assert result.report.missing_files == ["missing_scene.geojson"]
    assert any("неоднозначно" in error for error in result.report.errors)


@pytest.mark.parametrize(
    ("crs", "geometry_type", "error_fragment"),
    [
        ("EPSG:4326", "Polygon", "не совпадает"),
        ("EPSG:3857", "Point", "Polygon или MultiPolygon"),
    ],
)
def test_prepare_per_image_dataset_validates_crs_and_geometry(
    tmp_path: Path,
    crs: str,
    geometry_type: str,
    error_fragment: str,
) -> None:
    images = tmp_path / "images" / "folder"
    images.mkdir(parents=True)
    _write_raster(images / "scene.tif", 1, 0)
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    geometry = (
        {"type": "Point", "coordinates": [1, 1]}
        if geometry_type == "Point"
        else {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        }
    )
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": [
            {"type": "Feature", "properties": {}, "geometry": geometry}
        ],
    }
    (annotations / "folder_scene.geojson").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images.parent),
            annotations_dir=str(annotations),
            val_fraction=0.2,
        )
    )

    assert result.dataset is None
    assert any(error_fragment in error for error in result.report.errors)


def test_prepare_per_image_dataset_rejects_empty_directory(tmp_path: Path) -> None:
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()

    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images),
            annotations_dir=str(annotations),
            val_fraction=0.2,
        )
    )

    assert result.dataset is None
    assert any("ни одной сопоставленной сцены" in error for error in result.report.errors)


def test_prepare_dataset_expands_folder_scene_entry(tmp_path: Path) -> None:
    images = tmp_path / "images"
    folder = images / "kanopus" / "irkutsk"
    folder.mkdir(parents=True)
    _write_raster(folder / "scene_a.tif", 1, 0)
    _write_raster(folder / "scene_b.tif", 2, 4)
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text("irkutsk\n", encoding="utf-8")
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation(annotation_file, ["scene_a.tif", "scene_b.tif"])

    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images),
            scenes_file=str(scenes_file),
            annotation_file=str(annotation_file),
            val_fraction=0.5,
        )
    )

    assert result.report.status == "ok"
    assert result.dataset is not None
    assert result.report.scenes_total == 2
    assert result.report.scenes_found == 2
    assert set(_scene_ids(result)) == {
        "kanopus/irkutsk/scene_a.tif",
        "kanopus/irkutsk/scene_b.tif",
    }


def test_prepare_dataset_resolves_ambiguous_scene_by_annotation_geometry(tmp_path: Path) -> None:
    images = tmp_path / "images"
    far_folder = images / "far"
    near_folder = images / "near"
    far_folder.mkdir(parents=True)
    near_folder.mkdir(parents=True)
    _write_raster(far_folder / "scene_a.tif", 1, 1_000_000)
    _write_raster(near_folder / "scene_a.tif", 2, 0)
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text("scene_a\n", encoding="utf-8")
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation(annotation_file, ["scene_a.tif"])

    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images),
            scenes_file=str(scenes_file),
            annotation_file=str(annotation_file),
            val_fraction=0.5,
        )
    )

    assert result.report.status == "ok"
    assert result.dataset is not None
    assert result.report.scenes[0].image_path is not None
    assert result.report.scenes[0].image_path.endswith("/near/scene_a.tif")


def test_resolve_scene_images_keeps_extensionless_dotted_scene_exact(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    requested = images / "Канопус.PMS.SCN02.tif"
    (images / "Канопус.PMS.SCN01.tif").touch()
    requested.touch()
    (images / "Канопус.PMS.SCN03.tif").touch()
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text("Канопус.PMS.SCN02\n", encoding="utf-8")

    resolution = resolve_scene_images(
        SceneImageResolutionRequest(
            images_dir=str(images),
            scenes_file=str(scenes_file),
        )
    )

    assert [Path(item.image_path) for item in resolution.images] == [requested]
    assert resolution.missing_scenes == []
    assert resolution.ambiguous_scenes == {}


def test_training_pipeline_accepts_unicode_relative_scene_path_without_extension(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    duplicate_name = "Канопус PMS.SCN02.tif"
    other = images / "Качугский" / duplicate_name
    selected = images / "Ольхонский район" / duplicate_name
    other.parent.mkdir(parents=True)
    selected.parent.mkdir(parents=True)
    _write_raster(other, 1, 1_000_000)
    _write_raster(selected, 2, 0)
    scene_id = "Ольхонский район/Канопус PMS.SCN02"
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text(f"{scene_id}\n", encoding="utf-8")
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation(annotation_file, [duplicate_name])

    resolution = resolve_scene_images(
        SceneImageResolutionRequest(
            images_dir=str(images),
            scenes_file=str(scenes_file),
        )
    )
    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images),
            scenes_file=str(scenes_file),
            annotation_file=str(annotation_file),
            val_fraction=0.5,
        )
    )

    assert [Path(item.image_path) for item in resolution.images] == [selected]
    assert resolution.missing_scenes == []
    assert resolution.ambiguous_scenes == {}
    assert result.report.status == "ok"
    assert result.dataset is not None
    assert [scene.scene_id for scene in result.report.scenes] == [scene_id]
    assert result.report.scenes[0].image_path == selected.resolve().as_posix()


def test_resolve_scene_images_uses_annotation_for_duplicate_stem(tmp_path: Path) -> None:
    images = tmp_path / "images"
    far = images / "far" / "scene_a.tif"
    near = images / "near" / "scene_a.tif"
    far.parent.mkdir(parents=True)
    near.parent.mkdir(parents=True)
    _write_raster(far, 1, 1_000_000)
    _write_raster(near, 2, 0)
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text("scene_a\n", encoding="utf-8")
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation(annotation_file, ["scene_a.tif"])

    resolution = resolve_scene_images(
        SceneImageResolutionRequest(
            images_dir=str(images),
            scenes_file=str(scenes_file),
            annotation_files=[str(annotation_file)],
        )
    )

    assert [Path(item.image_path) for item in resolution.images] == [near]
    assert resolution.ambiguous_scenes == {}


def test_resolve_scene_images_reports_duplicate_stem_without_annotation(tmp_path: Path) -> None:
    images = tmp_path / "images"
    first = images / "first" / "scene_a.tif"
    second = images / "second" / "scene_a.tif"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text("scene_a\n", encoding="utf-8")

    resolution = resolve_scene_images(
        SceneImageResolutionRequest(
            images_dir=str(images),
            scenes_file=str(scenes_file),
        )
    )

    assert resolution.images == []
    assert resolution.ambiguous_scenes == {
        "scene_a": [first.resolve().as_posix(), second.resolve().as_posix()]
    }


def test_prepare_dataset_resolves_ambiguous_scene_by_hard_negative_geometry(tmp_path: Path) -> None:
    images = tmp_path / "images"
    far_folder = images / "far"
    near_folder = images / "near"
    far_folder.mkdir(parents=True)
    near_folder.mkdir(parents=True)
    _write_raster(far_folder / "scene_a.tif", 1, 1_000_000)
    _write_raster(near_folder / "scene_a.tif", 2, 0)
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text("scene_a\n", encoding="utf-8")
    annotation_file = tmp_path / "annotations.geojson"
    hard_negative_file = tmp_path / "hard_negative.geojson"
    _write_annotation(annotation_file, [])
    _write_geometry_annotation(
        hard_negative_file,
        [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    )

    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images),
            scenes_file=str(scenes_file),
            annotation_file=str(annotation_file),
            hard_negative_annotation_file=str(hard_negative_file),
            val_fraction=0.5,
        )
    )

    assert result.report.status == "ok"
    assert result.dataset is not None
    assert result.report.scenes[0].image_path is not None
    assert result.report.scenes[0].image_path.endswith("/near/scene_a.tif")


def test_prepare_dataset_reports_error_when_scene_is_missing(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _write_raster(images / "scene_a.tif", 1, 0)
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text("scene_a\nmissing_scene\n", encoding="utf-8")
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation(annotation_file, ["scene_a.tif"])

    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images),
            scenes_file=str(scenes_file),
            annotation_file=str(annotation_file),
            val_fraction=0.5,
        )
    )

    assert result.report.status == "error"
    assert result.dataset is None
    assert result.report.missing_files


def test_prepare_dataset_reports_error_when_crs_is_missing(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _write_raster(images / "scene_a.tif", 1, 0)
    _write_raster(images / "scene_b.tif", 2, 4, crs=None)
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text("scene_a\nscene_b\n", encoding="utf-8")
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation(annotation_file, ["scene_a.tif", "scene_b.tif"])

    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images),
            scenes_file=str(scenes_file),
            annotation_file=str(annotation_file),
            val_fraction=0.5,
        )
    )

    assert result.report.status == "error"
    assert result.dataset is None
    assert any("CRS" in error for error in result.report.errors)


def test_prepare_dataset_accepts_all_valid_mask_without_nodata(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _write_raster(images / "scene_a.tif", 1, 0)
    _write_raster(images / "scene_b.tif", 2, 4, nodata=None)
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text("scene_a\nscene_b\n", encoding="utf-8")
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation(annotation_file, ["scene_a.tif", "scene_b.tif"])

    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images),
            scenes_file=str(scenes_file),
            annotation_file=str(annotation_file),
            val_fraction=0.5,
            expected_band_count=1,
            expected_dtype="uint8",
        )
    )

    assert result.report.status == "ok"
    assert result.dataset is not None
    assert result.report.band_count == 1
    assert result.report.dtypes == ["uint8"]


def test_prepare_dataset_enforces_expected_band_count_and_dtype(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    _write_raster(images / "scene_a.tif", 1, 0, count=3)
    _write_raster(images / "scene_b.tif", 2, 4, count=3)
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text("scene_a\nscene_b\n", encoding="utf-8")
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation(annotation_file, ["scene_a.tif", "scene_b.tif"])

    result = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images),
            scenes_file=str(scenes_file),
            annotation_file=str(annotation_file),
            val_fraction=0.5,
            expected_band_count=4,
            expected_dtype="uint16",
        )
    )

    assert result.report.status == "error"
    assert result.dataset is None
    assert any("должен содержать 4 каналов" in error for error in result.report.errors)
    assert any("должны иметь dtype uint16" in error for error in result.report.errors)


def test_prepare_per_image_multiclass_validates_manifest_and_feature_classes(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    _write_raster(images / "scene_a.tif", 1, 0)
    classes = [
        {"id": 1, "slug": "first", "name": "Первый", "color": "#F59E0B", "priority": 100},
        {"id": 2, "slug": "second", "name": "Второй", "color": "#8B5CF6", "priority": 0},
    ]
    (annotations / ".mlsystem2-dataset.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "multiclass",
                "combined": False,
                "classes": classes,
                "sources": [],
                "scene_ids": ["scene_a"],
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "_mlsystem2_schema_version": 1,
        "_mlsystem2_task": "multiclass",
        "_mlsystem2_classes": classes,
        "features": [
            {
                "type": "Feature",
                "id": "feature-1",
                "properties": {
                    "_mlsystem2_role": "positive",
                    "_mlsystem2_class": "second",
                    "_mlsystem2_origin_key": "origin-1",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            },
            {
                "type": "Feature",
                "id": "feature-2",
                "properties": {
                    "_mlsystem2_role": "hard_negative",
                    "_mlsystem2_origin_key": "origin-2",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]],
                },
            },
        ],
    }
    annotation_path = annotations / "images_scene_a.geojson"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    request = DatasetPreparationRequest(
        images_dir=str(images),
        annotations_dir=str(annotations),
        val_fraction=0.5,
    )
    result = prepare_dataset(request)

    assert result.report.status == "ok", result.report.errors
    assert result.dataset is not None
    assert result.dataset.format == "per_image_multiclass"
    assert [item.slug for item in result.dataset.classes] == ["first", "second"]
    assert result.report.class_counts == {"first": 0, "second": 1}
    assert result.report.hard_negative_objects == 1

    payload["features"][0]["properties"]["_mlsystem2_class"] = "unknown"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    invalid = prepare_dataset(request)
    assert invalid.report.status == "error"
    assert any("неизвестный класс" in error for error in invalid.report.errors)


def _write_raster(
    path: Path,
    value: int,
    left: float,
    *,
    crs: str | None = "EPSG:3857",
    pixel_size: float = 1.0,
    top: float = 4.0,
    nodata: int | None = 0,
    count: int = 1,
    dtype: str = "uint8",
) -> None:
    data = np.full((count, 4, 4), value, dtype=np.dtype(dtype))
    transform = from_origin(left, top, pixel_size, pixel_size)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(data)


def _write_masked_raster_with_invalid_white_edge(path: Path) -> None:
    data = np.full((1, 4, 4), 50, dtype=np.uint8)
    data[:, 0, :] = 255
    mask = np.full((4, 4), 255, dtype=np.uint8)
    mask[0, :] = 0
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK="YES"):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=4,
            height=4,
            count=1,
            dtype="uint8",
            crs="EPSG:3857",
            transform=from_origin(0, 4, 1, 1),
            tiled=True,
        ) as dataset:
            dataset.write(data)
            dataset.write_mask(mask)


def _write_annotation(path: Path, scene_names: list[str]) -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"image_name": scene_name},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
            for scene_name in scene_names
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_geometry_annotation(path: Path, coordinates: list[list[list[float]]]) -> None:
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": coordinates},
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_per_image_annotation(path: Path, roles: list[str | None]) -> None:
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [
            {
                "type": "Feature",
                "id": index,
                "properties": (
                    {} if role is None else {"_mlsystem2_role": role}
                ),
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5], [0, 0]]
                    ],
                },
            }
            for index, role in enumerate(roles)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _scene_ids(result) -> list[str]:
    return [scene.scene_id for scene in result.report.scenes]
