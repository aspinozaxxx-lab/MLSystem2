from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from mlsystem2.dataset_preparing.api import prepare_dataset
from mlsystem2.dataset_preparing.contracts import DatasetPreparationRequest


def test_prepare_dataset_builds_in_memory_vrt_xml(tmp_path: Path) -> None:
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

    assert result.report.status == "ok"
    assert result.dataset is not None
    assert result.dataset.train_vrt_xml.startswith("<VRTDataset")
    assert "<VRTDataset" in result.dataset.val_vrt_xml
    assert max(
        result.dataset.train_vrt_xml.count("<SourceFilename"),
        result.dataset.val_vrt_xml.count("<SourceFilename"),
    ) >= 2
    _assert_vrt_reads(result.dataset.train_vrt_xml)
    _assert_vrt_reads(result.dataset.val_vrt_xml)


def test_prepare_dataset_builds_vrt_for_different_resolution_and_grid(
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
    assert "<VRTDataset" in result.dataset.train_vrt_xml
    assert "<VRTDataset" in result.dataset.val_vrt_xml


def test_prepare_dataset_vrt_uses_source_masks_for_overlap(tmp_path: Path) -> None:
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
    with MemoryFile(result.dataset.train_vrt_xml.encode("utf-8")) as memory_file:
        with memory_file.open() as dataset:
            data = dataset.read(1, window=((0, 1), (0, 1)), masked=False)
    assert int(data[0, 0]) == 10


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


def test_prepare_dataset_reports_error_when_nodata_is_missing(tmp_path: Path) -> None:
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
        )
    )

    assert result.report.status == "error"
    assert result.dataset is None
    assert any("nodata" in error for error in result.report.errors)


def _write_raster(
    path: Path,
    value: int,
    left: float,
    *,
    crs: str | None = "EPSG:3857",
    pixel_size: float = 1.0,
    top: float = 4.0,
    nodata: int | None = 0,
) -> None:
    data = np.full((1, 4, 4), value, dtype=np.uint8)
    transform = from_origin(left, top, pixel_size, pixel_size)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint8",
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


def _assert_vrt_reads(vrt_xml: str) -> None:
    with MemoryFile(vrt_xml.encode("utf-8")) as memory_file:
        with memory_file.open() as dataset:
            data = dataset.read(1, window=((0, 1), (0, 1)))
    assert data.shape == (1, 1)
