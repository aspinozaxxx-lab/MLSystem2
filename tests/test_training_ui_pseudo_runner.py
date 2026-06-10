from __future__ import annotations

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import box, shape

from mlsystem2.training_ui_api._pseudo_runner import (
    _completed_request_scene_count,
    _collect_scene_inputs,
    _final_status,
    _features_from_mask,
    _find_images,
    _image_index,
    _merge_overlapping_features,
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
    assert inputs[0].request_scene_count == 2
    assert inputs[1].request_scenes == ("irkutsk",)
    assert inputs[1].request_scene_count == 1
    assert missing == ["lost"]


def test_completed_request_scene_count_uses_original_txt_rows() -> None:
    assert _completed_request_scene_count(
        [
            {"status": "missing_image"},
            {"status": "ok", "request_scenes": ["a", "b"], "request_scene_count": 3},
            {"status": "failed", "request_scenes": ["c"]},
        ]
    ) == 5


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


def test_merge_overlapping_features_dissolves_intersections() -> None:
    features = [
        _geojson_feature(box(0, 0, 2, 2), "scene-1"),
        _geojson_feature(box(1, 1, 3, 3), "scene-2"),
    ]

    merged = _merge_overlapping_features(features)

    assert len(merged) == 1
    geometry = shape(merged[0]["geometry"])
    assert round(geometry.area, 6) == 7.0
    assert merged[0]["properties"]["scene_id"] == "scene-1"
    assert merged[0]["properties"]["source_scene_ids"] == ["scene-1", "scene-2"]
    assert merged[0]["properties"]["merged_feature_count"] == 2


def test_merge_overlapping_features_dissolves_touching_polygons() -> None:
    features = [
        _geojson_feature(box(0, 0, 1, 1), "scene-1"),
        _geojson_feature(box(1, 0, 2, 1), "scene-2"),
    ]

    merged = _merge_overlapping_features(features)

    assert len(merged) == 1
    geometry = shape(merged[0]["geometry"])
    assert geometry.geom_type == "Polygon"
    assert round(geometry.area, 6) == 2.0


def test_merge_overlapping_features_keeps_disjoint_polygons_separate() -> None:
    features = [
        _geojson_feature(box(0, 0, 1, 1), "scene-1"),
        _geojson_feature(box(3, 0, 4, 1), "scene-2"),
    ]

    merged = _merge_overlapping_features(features)

    assert len(merged) == 2
    assert [item["properties"]["source_scene_ids"] for item in merged] == [
        ["scene-1"],
        ["scene-2"],
    ]
    assert [item["properties"]["merged_feature_count"] for item in merged] == [1, 1]


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
        feature_count=2,
        feature_count_before_merge=3,
        unique_image_count=51,
        postprocess_profile=profile,
    )

    assert summary["input_scene_count"] == 2
    assert summary["unique_image_count"] == 51
    assert summary["feature_count_before_merge"] == 3
    assert summary["feature_count"] == 2
    assert summary["postprocess_profile"] == "strong"
    assert summary["postprocess_level"] == 3
    assert summary["postprocess_params"]["simplify_m"] == 30.0
    assert summary["postprocess_merge_overlaps"] is True
    assert summary["postprocess_merge_policy"] == "overlap_or_touch"


def test_final_status_errors_when_no_scenes_processed() -> None:
    reports = [{"scene_id": "irkutsk", "status": "missing_image", "feature_count": 0}]

    assert _final_status(reports, failures=[], missing=["irkutsk"]) == "error"
    assert _final_status([{"status": "ok"}], failures=[], missing=["lost"]) == "partial"
    assert _final_status([{"status": "ok"}], failures=[], missing=[]) == "ok"


def _geojson_feature(geometry, scene_id: str) -> dict:
    return {
        "type": "Feature",
        "geometry": geometry.__geo_interface__,
        "properties": {
            "_crs": "EPSG:4326",
            "_x_res": 1.0,
            "_y_res": 1.0,
            "scene_id": scene_id,
            "class_key": "deforestation",
            "class_name": "Вырубки",
            "source_model": "model",
            "postprocess_profile": "none",
            "postprocess_level": 1,
        },
    }
