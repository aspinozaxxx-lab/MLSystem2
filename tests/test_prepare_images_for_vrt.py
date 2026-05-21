from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from mlsystem2.cli.prepare_images_for_vrt import prepare_images_for_vrt


def test_prepare_images_for_vrt_keeps_band_count(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    prepared_dir = tmp_path / "prepared"
    raw_dir.mkdir()
    _write_raw_raster_with_white_border(raw_dir / "scene.tif", count=3)

    report = prepare_images_for_vrt(raw_dir, prepared_dir, tmp_path / "report.json")

    assert report["status"] == "ok"
    with rasterio.open(prepared_dir / "scene.tif") as dataset:
        assert dataset.count == 3


def test_prepare_images_for_vrt_writes_epsg3857_and_internal_mask(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    prepared_dir = tmp_path / "prepared"
    raw_dir.mkdir()
    _write_raw_raster_with_white_border(raw_dir / "nested" / "scene.tif", count=1)

    report = prepare_images_for_vrt(raw_dir, prepared_dir, tmp_path / "report.json")

    output_path = prepared_dir / "nested" / "scene.tif"
    assert report["status"] == "ok"
    assert not Path(str(output_path) + ".msk").exists()
    with rasterio.open(output_path) as dataset:
        assert dataset.crs == "EPSG:3857"
        mask = dataset.dataset_mask()
        assert mask.min() == 0
        assert mask.max() == 255


def _write_raw_raster_with_white_border(path: Path, *, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((count, 8, 8), 20, dtype=np.uint8)
    data[:, 0, :] = 255
    data[:, -1, :] = 255
    data[:, :, 0] = 255
    data[:, :, -1] = 255
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
    ) as dataset:
        dataset.write(data)
