from __future__ import annotations

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import shape

from mlsystem2.training_ui_api._pseudo_runner import (
    _collect_scene_inputs,
    _final_status,
    _features_from_mask,
    _find_images,
    _image_index,
    _postprocess_mask,
    _select_postprocess_profile,
    _summary,
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
    assert features[0]["properties"]["postprocess_profile"] == "none"
    assert features[0]["properties"]["postprocess_level"] == 1


def test_select_postprocess_profile_uses_unique_image_count_boundaries() -> None:
    assert _select_postprocess_profile(0).name == "none"
    assert _select_postprocess_profile(5).name == "none"
    assert _select_postprocess_profile(6).name == "detail_v2"
    assert _select_postprocess_profile(50).name == "detail_v2"
    assert _select_postprocess_profile(51).name == "strong"


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


def test_collect_scene_inputs_deduplicates_found_rasters(tmp_path) -> None:
    scene_dir = tmp_path / "kanopus" / "irkutsk"
    scene_dir.mkdir(parents=True)
    first = scene_dir / "KV3_100.L2.PMS.SCN01_cog.tif"
    second = scene_dir / "KV3_101.L2.PMS.SCN02.tif"
    first.touch()
    second.touch()
    index = _image_index(tmp_path)

    inputs, missing = _collect_scene_inputs(
        ["irkutsk", "KV3_100.L2.PMS.SCN01.tif", "lost"],
        index,
    )

    assert [item.image_path for item in inputs] == [first, second]
    assert inputs[0].request_scenes == ("irkutsk", "KV3_100.L2.PMS.SCN01.tif")
    assert inputs[1].request_scenes == ("irkutsk",)
    assert missing == ["lost"]


def test_detail_v2_filters_small_polygons_and_holes_in_metric_crs() -> None:
    profile = _select_postprocess_profile(6)
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[10:100, 10:100] = 1
    mask[40:60, 40:60] = 0
    mask[0:10, 0:10] = 1

    processed_mask = _postprocess_mask(mask, profile)
    features = _features_from_mask(
        processed_mask,
        from_origin(0, 120, 1, 1),
        CRS.from_epsg(3857),
        (1.0, 1.0),
        "scene-1",
        {"class_key": "deforestation", "class_name": "Вырубки"},
        postprocess_profile=profile,
    )

    assert len(features) == 1
    geometry = shape(features[0]["geometry"])
    assert geometry.geom_type == "Polygon"
    assert len(geometry.interiors) == 0
    assert features[0]["properties"]["postprocess_profile"] == "detail_v2"
    assert features[0]["properties"]["postprocess_level"] == 2


def test_strong_profile_applies_binary_closing() -> None:
    profile = _select_postprocess_profile(51)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:22] = 1
    mask[10:40, 24:36] = 1

    processed_mask = _postprocess_mask(mask, profile)

    assert processed_mask[20, 22] == 1
    assert processed_mask[20, 23] == 1


def test_summary_reports_unique_image_count_and_postprocess_profile(tmp_path) -> None:
    profile = _select_postprocess_profile(51)

    summary = _summary(
        {},
        scenes=["a", "b"],
        status="ok",
        output_geojson=tmp_path / "pseudo_markup.geojson",
        started=0.0,
        scene_reports=[{"status": "ok"}],
        failures=[],
        missing=[],
        feature_count=3,
        unique_image_count=51,
        postprocess_profile=profile,
    )

    assert summary["input_scene_count"] == 2
    assert summary["unique_image_count"] == 51
    assert summary["postprocess_profile"] == "strong"
    assert summary["postprocess_level"] == 3
    assert summary["postprocess_params"]["simplify_m"] == 30.0


def test_final_status_errors_when_no_scenes_processed() -> None:
    reports = [{"scene_id": "irkutsk", "status": "missing_image", "feature_count": 0}]

    assert _final_status(reports, failures=[], missing=["irkutsk"]) == "error"
    assert _final_status([{"status": "ok"}], failures=[], missing=["lost"]) == "partial"
    assert _final_status([{"status": "ok"}], failures=[], missing=[]) == "ok"
