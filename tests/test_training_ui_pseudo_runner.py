from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import box, shape

from mlsystem2.training_ui_api import _pseudo_runner
from mlsystem2.training_ui_api._pseudo_runner import (
    PSEUDO_INFERENCE_BACKEND,
    _completed_image_count,
    _collect_scene_inputs,
    _final_status,
    _features_from_mask,
    _find_images,
    _filter_compact_features,
    _image_index,
    _merge_overlapping_features,
    _postprocess_mask,
    _postprocess_profile_from_config,
    _select_postprocess_profile,
    _summary,
    _write_pseudo_progress,
    run_pseudo_markup,
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


def test_postprocess_profile_accepts_template_overrides() -> None:
    profile = _postprocess_profile_from_config(
        _select_postprocess_profile(51),
        {
            "postprocess.min_area_m2": 10000.0,
            "postprocess.min_hole_area_m2": 5000.0,
            "postprocess.simplify_m": 15.0,
            "postprocess.filter_compact_objects.enabled": True,
            "postprocess.filter_compact_objects.min_isoperimetric_quotient": 0.25,
            "postprocess.filter_compact_objects.max_bbox_ratio": 3.5,
        },
    )

    assert profile.name == "strong"
    assert profile.min_area_m2 == 10000.0
    assert profile.min_hole_area_m2 == 5000.0
    assert profile.simplify_m == 15.0
    assert profile.filter_compact_min_isoperimetric_quotient == 0.25
    assert profile.filter_compact_max_bbox_ratio == 3.5


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


def test_completed_image_count_uses_finished_image_reports() -> None:
    assert _completed_image_count(
        [
            {"status": "missing_image"},
            {"status": "ok", "request_scenes": ["a", "b"], "request_scene_count": 3},
            {"status": "failed", "request_scenes": ["c"]},
            {"status": "running"},
        ]
    ) == 3


def test_pseudo_progress_counts_folder_entry_as_found_images(tmp_path) -> None:
    progress_path = tmp_path / "progress.json"

    _write_pseudo_progress(
        progress_path,
        total=2,
        started=0.0,
        scene_reports=[
            {
                "status": "ok",
                "request_scenes": ["irkutsk"],
                "request_scene_count": 1,
            }
        ],
        failures=[],
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["current"] == 1
    assert payload["total"] == 2


def test_pseudo_progress_counts_unique_found_images_plus_missing(tmp_path) -> None:
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
    progress_path = tmp_path / "progress.json"

    _write_pseudo_progress(
        progress_path,
        total=len(inputs) + len(missing),
        started=0.0,
        scene_reports=[
            {"status": "missing_image"},
            {
                "status": "ok",
                "request_scenes": ["irkutsk", "KV3_100.L2.PMS.SCN01.tif"],
                "request_scene_count": 2,
            },
            {"status": "failed", "request_scenes": ["irkutsk"], "request_scene_count": 1},
        ],
        failures=[{"scene_id": "KV3_101.L2.PMS.SCN02", "error": "boom"}],
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert len(inputs) == 2
    assert missing == ["lost"]
    assert payload["current"] == 3
    assert payload["total"] == 3


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


def test_strong_profile_does_not_apply_binary_closing() -> None:
    profile = _select_postprocess_profile(51)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:22] = 1
    mask[10:40, 24:36] = 1

    processed_mask = _postprocess_mask(mask, profile)

    assert processed_mask[20, 22] == 0
    assert processed_mask[20, 23] == 0


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


def test_filter_compact_features_removes_lake_like_objects() -> None:
    profile = _postprocess_profile_from_config(
        _select_postprocess_profile(51),
        {
            "postprocess.filter_compact_objects.enabled": True,
            "postprocess.filter_compact_objects.min_isoperimetric_quotient": 0.25,
            "postprocess.filter_compact_objects.max_bbox_ratio": 3.5,
        },
    )
    features = [
        _geojson_feature(box(30.00, 50.00, 30.01, 50.01), "lake"),
        _geojson_feature(box(30.00, 50.02, 30.12, 50.025), "river"),
    ]

    filtered = _filter_compact_features(features, profile)

    assert [item["properties"]["scene_id"] for item in filtered] == ["river"]


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
    assert summary["inference_backend"] == PSEUDO_INFERENCE_BACKEND
    assert summary["triton_model"] is None
    assert summary["unique_image_count"] == 51
    assert summary["feature_count_before_merge"] == 3
    assert summary["feature_count"] == 2
    assert summary["postprocess_profile"] == "strong"
    assert summary["postprocess_level"] == 3
    assert summary["postprocess_params"]["mask_min_object_pixels"] == 48
    assert summary["postprocess_params"]["mask_min_hole_pixels"] == 48
    assert summary["postprocess_params"]["min_area_m2"] == 3000.0
    assert summary["postprocess_params"]["min_hole_area_m2"] == 5000.0
    assert summary["postprocess_params"]["simplify_m"] == 15.0
    assert "binary_closing_radius" not in summary["postprocess_params"]
    assert summary["postprocess_merge_overlaps"] is True
    assert summary["postprocess_merge_policy"] == "overlap_or_touch"


def test_run_pseudo_markup_uses_local_checkpoint_and_releases_cuda(tmp_path, monkeypatch) -> None:
    config = _pseudo_runner_config(tmp_path)
    fake_torch = _FakeTorch()
    fake_model = _FakeModel()
    load_requests = []

    def fake_load_checkpoint(request):
        load_requests.append(request)
        return SimpleNamespace(model=SimpleNamespace(model=fake_model))

    def fake_infer_scene(**kwargs):
        assert kwargs["torch"] is fake_torch
        assert kwargs["model"] is fake_model
        return []

    monkeypatch.setattr(_pseudo_runner, "_torch", lambda: fake_torch)
    monkeypatch.setattr(_pseudo_runner, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(_pseudo_runner, "_infer_scene", fake_infer_scene)

    report = run_pseudo_markup(config)

    assert report["status"] == "ok"
    assert report["inference_backend"] == "pytorch_one_off"
    assert report["triton_model"] is None
    assert report["processed"] == 1
    assert load_requests
    assert str(load_requests[0].checkpoint_uri).endswith("best.pt")
    assert fake_model.device == "cuda"
    assert fake_model.eval_called is True
    assert fake_torch.cuda.empty_cache_calls == 1


def test_run_pseudo_markup_releases_cuda_on_checkpoint_error(tmp_path, monkeypatch) -> None:
    config = _pseudo_runner_config(tmp_path)
    fake_torch = _FakeTorch()

    def fake_load_checkpoint(request):
        del request
        raise RuntimeError("load failed")

    monkeypatch.setattr(_pseudo_runner, "_torch", lambda: fake_torch)
    monkeypatch.setattr(_pseudo_runner, "load_checkpoint", fake_load_checkpoint)

    report = run_pseudo_markup(config)

    assert report["status"] == "error"
    assert report["inference_backend"] == "pytorch_one_off"
    assert report["triton_model"] is None
    assert report["failures"][0]["stage"] == "load_checkpoint"
    assert fake_torch.cuda.empty_cache_calls == 1
    output = json.loads((tmp_path / "pseudo_markup.geojson").read_text(encoding="utf-8"))
    assert output == {"type": "FeatureCollection", "features": []}


def test_pseudo_runner_keeps_triton_registration_out_of_one_off_path() -> None:
    source = Path(_pseudo_runner.__file__).read_text(encoding="utf-8")

    assert "load_checkpoint(" in source
    for forbidden in ("build_triton_model_export_zip", "model_repository", "model_archive", "tritonclient"):
        assert forbidden not in source


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


def _pseudo_runner_config(tmp_path) -> dict[str, object]:
    images_root = tmp_path / "images"
    images_root.mkdir()
    (images_root / "scene-1.tif").touch()
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text("scene-1\n", encoding="utf-8")
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    return {
        "run_root": str(tmp_path / "run"),
        "output_geojson": str(tmp_path / "pseudo_markup.geojson"),
        "report_path": str(tmp_path / "report.json"),
        "scenes_file": str(scenes_file),
        "images_root": str(images_root),
        "local_checkpoint_path": str(checkpoint_path),
        "threshold": 0.5,
        "tile_size": 32,
        "stride": 32,
        "batch_size": 1,
        "device": "cuda",
        "class_key": "deforestation",
        "class_name": "Вырубки",
    }


class _FakeModel:
    def __init__(self) -> None:
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


class _FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return True

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()

    def device(self, value: str) -> str:
        return value
