from __future__ import annotations

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin

from mlsystem2.training_ui_api._pseudo_runner import (
    _final_status,
    _find_images,
    _image_index,
    _features_from_mask,
)


def test_features_from_mask_writes_geojson_coordinates_in_wgs84() -> None:
    mask = np.zeros((2, 2), dtype=np.uint8)
    mask[0, 0] = 1
    transform = from_origin(11469928.363425, 6873318.079527, 3.4240042653603187, 3.4240042653603293)

    features = _features_from_mask(
        mask,
        transform,
        CRS.from_epsg(3857),
        (3.4240042653603187, 3.4240042653603293),
        "scene-1",
        {"class_key": "deforestation", "class_name": "Вырубки"},
    )

    assert len(features) == 1
    xs = [point[0] for point in features[0]["geometry"]["coordinates"][0]]
    ys = [point[1] for point in features[0]["geometry"]["coordinates"][0]]
    assert all(103.0 < x < 103.1 for x in xs)
    assert all(52.3 < y < 52.5 for y in ys)
    assert features[0]["properties"]["_crs"] == "EPSG:3857"
    assert features[0]["properties"]["_x_res"] == 3.4240042653603187


def test_find_images_accepts_txt_scene_forms_and_dataset_folders(tmp_path) -> None:
    scene_dir = tmp_path / "kanopus" / "irkutsk"
    scene_dir.mkdir(parents=True)
    first = scene_dir / "KV3_100.L2.PMS.SCN01_cog.tif"
    second = scene_dir / "KV3_101.L2.PMS.SCN02.tif"
    first.touch()
    second.touch()

    index = _image_index(tmp_path)

    assert _find_images("./KV3_100.L2.PMS.SCN01_cog.tif", index) == [first]
    assert _find_images("KV3_100.L2.PMS.SCN01.tif", index) == [first]
    assert _find_images("irkutsk", index) == [first, second]


def test_final_status_errors_when_no_scenes_processed() -> None:
    reports = [{"scene_id": "irkutsk", "status": "missing_image", "feature_count": 0}]

    assert _final_status(reports, failures=[], missing=["irkutsk"]) == "error"
    assert _final_status([{"status": "ok"}], failures=[], missing=["lost"]) == "partial"
    assert _final_status([{"status": "ok"}], failures=[], missing=[]) == "ok"
