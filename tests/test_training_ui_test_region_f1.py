"""Nodata свободной области исключается из F1 после любой постобработки."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from mlsystem2.training_ui_api._pseudo_runner import run_test_sample_f1


def test_region_f1_ignores_prediction_outside_valid_area(tmp_path: Path):
    image_path, mask_path, prediction_path = (tmp_path / name for name in ("снимок.tif", "маска.png", "прогноз.npy"))
    valid = np.zeros((8, 8), dtype=np.uint8)
    valid[2:6, 3:5] = 255
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(image_path, "w", driver="GTiff", width=8, height=8,
        count=4, dtype="uint8", nodata=0, crs="EPSG:3857", transform=from_origin(0, 8, 1, 1), ALPHA="NO") as dst:
        dst.write(np.broadcast_to(valid, (4, 8, 8)))
        dst.write_mask(valid)
    with rasterio.open(mask_path, "w", driver="PNG", width=8, height=8, count=1, dtype="uint8") as dst:
        dst.write((valid > 0).astype(np.uint8), 1)
    np.save(prediction_path, np.ones((8, 8), dtype=np.uint8))
    report = run_test_sample_f1({"operation": "test_sample_f1", "run_root": str(tmp_path / "проверка"),
        "tiles": [{"index": 1, "image_path": str(image_path), "mask_path": str(mask_path),
                   "precomputed_prediction_path": str(prediction_path), "clip_to_valid_data": True}]})
    assert report["status"] == "ok", report
    assert report["true_positive"] == 8
    assert report["false_positive"] == report["false_negative"] == 0
