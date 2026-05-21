from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

from mlsystem2.cli.prepare_images_for_vrt import prepare_images_for_vrt


def test_prepare_images_for_vrt_keeps_band_count(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    prepared_dir = tmp_path / "prepared"
    raw_dir.mkdir()
    _write_raw_raster_with_white_border(raw_dir / "scene.tif", count=3)

    report = prepare_images_for_vrt(raw_dir, prepared_dir, tmp_path / "report.json")

    assert report["status"] == "ok"
    item = report["files"][0]
    assert item["source_count"] == 3
    assert item["output_count"] == 3
    assert item["source_nodata"] == 0
    assert item["output_nodata"] == 0
    assert item["has_alpha"] is False
    assert item["has_internal_mask"] is False
    assert item["has_sidecar_msk"] is False
    assert item["is_cog_check"] is True
    with rasterio.open(prepared_dir / "scene.tif") as dataset:
        assert dataset.count == 3
        assert dataset.nodata == 0
        assert ColorInterp.alpha not in dataset.colorinterp


def test_prepare_images_for_vrt_processes_multiple_files_with_workers(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    prepared_dir = tmp_path / "prepared"
    raw_dir.mkdir()
    _write_raw_raster_with_white_border(raw_dir / "scene_a.tif", count=3)
    _write_raw_raster_with_white_border(raw_dir / "nested" / "scene_b.tiff", count=3)

    report = prepare_images_for_vrt(raw_dir, prepared_dir, tmp_path / "report.json", workers=2)

    assert report["status"] == "ok"
    assert report["input_count"] == 2
    assert report["output_count"] == 2
    assert report["error_count"] == 0
    assert report["workers"] == 2
    assert (prepared_dir / "scene_a.tif").is_file()
    assert (prepared_dir / "nested" / "scene_b.tiff").is_file()
    assert all(item["is_cog_check"] is True for item in report["files"])
    assert all(item["has_alpha"] is False for item in report["files"])
    assert all(item["has_sidecar_msk"] is False for item in report["files"])


def test_prepare_images_for_vrt_writes_epsg3857_nodata_without_mask(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    prepared_dir = tmp_path / "prepared"
    raw_dir.mkdir()
    _write_raw_raster_with_white_border(raw_dir / "nested" / "scene.tif", count=1)

    report = prepare_images_for_vrt(raw_dir, prepared_dir, tmp_path / "report.json")

    output_path = prepared_dir / "nested" / "scene.tif"
    assert report["status"] == "ok"
    item = report["files"][0]
    assert item["has_internal_mask"] is False
    assert item["has_sidecar_msk"] is False
    assert not Path(str(output_path) + ".msk").exists()
    with rasterio.open(output_path) as dataset:
        assert dataset.crs == "EPSG:3857"
        assert dataset.nodata == 0
        assert dataset.count == 1


def test_prepare_images_for_vrt_preserves_source_colorinterp_without_alpha(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    prepared_dir = tmp_path / "prepared"
    raw_dir.mkdir()
    _write_raw_raster_with_white_border(
        raw_dir / "scene.tif",
        count=4,
        colorinterp=(
            ColorInterp.gray,
            ColorInterp.undefined,
            ColorInterp.undefined,
            ColorInterp.undefined,
        ),
    )

    report = prepare_images_for_vrt(raw_dir, prepared_dir, tmp_path / "report.json")

    assert report["status"] == "ok"
    item = report["files"][0]
    assert item["output_count"] == 4
    assert item["output_nodata"] == 0
    assert item["source_colorinterp"] == ["gray", "undefined", "undefined", "undefined"]
    assert item["output_colorinterp"] == ["gray", "undefined", "undefined", "undefined"]
    assert item["is_cog_check"] is True
    with rasterio.open(prepared_dir / "scene.tif") as dataset:
        assert dataset.count == 4
        assert dataset.nodata == 0
        assert ColorInterp.alpha not in dataset.colorinterp
        assert dataset.colorinterp == (
            ColorInterp.gray,
            ColorInterp.undefined,
            ColorInterp.undefined,
            ColorInterp.undefined,
        )


def test_prepare_images_for_vrt_preserves_descriptions_and_band_tags(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    prepared_dir = tmp_path / "prepared"
    raw_dir.mkdir()
    descriptions = ("red band", "green band", "blue band", "nir band")
    _write_raw_raster_with_white_border(
        raw_dir / "scene.tif",
        count=4,
        colorinterp=(
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.undefined,
        ),
        descriptions=descriptions,
        band_tags={4: {"name": "nir"}},
    )

    report = prepare_images_for_vrt(raw_dir, prepared_dir, tmp_path / "report.json")

    assert report["status"] == "ok"
    item = report["files"][0]
    assert item["source_descriptions"] == list(descriptions)
    assert item["output_descriptions"] == list(descriptions)
    assert item["has_internal_mask"] is False
    with rasterio.open(prepared_dir / "scene.tif") as dataset:
        assert dataset.descriptions == descriptions
        assert dataset.tags(4)["name"] == "nir"


def test_prepare_images_for_vrt_replaces_source_alpha_colorinterp(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    prepared_dir = tmp_path / "prepared"
    raw_dir.mkdir()
    _write_raw_raster_with_white_border(
        raw_dir / "scene.tif",
        count=4,
        colorinterp=(
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        ),
    )

    report = prepare_images_for_vrt(raw_dir, prepared_dir, tmp_path / "report.json")

    assert report["status"] == "ok"
    item = report["files"][0]
    assert item["source_had_alpha_colorinterp"] is True
    assert item["source_colorinterp"] == ["red", "green", "blue", "alpha"]
    assert item["output_count"] == 4
    assert item["output_colorinterp"] == ["red", "green", "blue", "undefined"]
    assert item["has_alpha"] is False
    assert item["is_cog_check"] is True
    with rasterio.open(prepared_dir / "scene.tif") as dataset:
        assert dataset.count == 4
        assert ColorInterp.alpha not in dataset.colorinterp


def test_prepare_images_for_vrt_errors_without_source_nodata(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    prepared_dir = tmp_path / "prepared"
    raw_dir.mkdir()
    _write_raw_raster_with_white_border(raw_dir / "scene.tif", count=4, nodata=None)

    report = prepare_images_for_vrt(raw_dir, prepared_dir, tmp_path / "report.json")

    assert report["status"] == "error"
    assert report["output_count"] == 0
    item = report["files"][0]
    assert item["status"] == "error"
    assert "nodata" in item["error"]
    assert not (prepared_dir / "scene.tif").exists()


def _write_raw_raster_with_white_border(
    path: Path,
    *,
    count: int,
    colorinterp: tuple[ColorInterp, ...] | None = None,
    descriptions: tuple[str, ...] | None = None,
    band_tags: dict[int, dict[str, str]] | None = None,
    nodata: int | None = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((count, 8, 8), 20, dtype=np.uint8)
    if nodata is not None:
        data[:, 0, :] = nodata
        data[:, -1, :] = nodata
        data[:, :, 0] = nodata
        data[:, :, -1] = nodata
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=count,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(40.0, 50.0, 0.0001, 0.0001),
        nodata=nodata,
    ) as dataset:
        dataset.write(data)
        if colorinterp is not None:
            dataset.colorinterp = colorinterp
        if descriptions is not None:
            for band_index, description in enumerate(descriptions, start=1):
                dataset.set_band_description(band_index, description)
        if band_tags is not None:
            for band_index, tags in band_tags.items():
                dataset.update_tags(band_index, **tags)
