from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import box, shape
from shapely.ops import transform as shapely_transform

from mlsystem2.training_ui_api import _pseudo_runner
from mlsystem2.training_ui_api._external_models import ExternalTestPrediction
from mlsystem2.training_ui_api._pseudo_runner import (
    _InferencePauseController,
    PSEUDO_INFERENCE_BACKEND,
    _completed_image_count,
    _final_status,
    _features_from_mask,
    _geometry_postprocessor,
    _geometry_to_metric,
    _infer_test_tile_mask,
    _merge_overlapping_features,
    _multiclass_pixel_counts,
    _postprocess_mask,
    _postprocess_geometry,
    _postprocess_profile_from_config,
    _resolve_feature_type_conflicts,
    _resolve_scene_inputs,
    _select_postprocess_profile,
    _summary,
    _structured_multiclass_metrics,
    _write_pseudo_progress,
    run_pseudo_markup,
    run_test_sample_f1,
)


def test_inference_pause_controller_moves_model_to_cpu_and_resumes(tmp_path: Path) -> None:
    control_dir = tmp_path / "control"
    control_dir.mkdir()
    request_path = control_dir / "pause.request"
    request_path.write_text("pause-token\n", encoding="utf-8")
    moved_to: list[str] = []
    emptied: list[bool] = []
    model = SimpleNamespace(to=lambda device: moved_to.append(str(device)))
    torch = SimpleNamespace(
        device=lambda value: value,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: emptied.append(True),
        ),
    )
    controller = _InferencePauseController(torch, [model], "cuda", str(control_dir))

    thread = threading.Thread(target=controller.pause_if_requested)
    thread.start()
    marker_path = control_dir / "paused"
    deadline = time.monotonic() + 2
    while not marker_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert marker_path.read_text(encoding="utf-8").strip() == "pause-token"
    assert moved_to == ["cpu"]
    assert emptied == [True]

    request_path.unlink()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert moved_to == ["cpu", "cuda"]
    assert not marker_path.exists()


def test_multiclass_wrong_type_is_fp_and_fn_but_foreground_match() -> None:
    truth = np.asarray([[1, 2]], dtype=np.uint8)
    predicted = np.asarray([[2, 1]], dtype=np.uint8)
    pixel = _multiclass_pixel_counts(truth, predicted, [1, 2])
    schema = [
        {"id": 1, "slug": "first", "name": "Первый", "color": "#F59E0B", "priority": 100},
        {"id": 2, "slug": "second", "name": "Второй", "color": "#8B5CF6", "priority": 0},
    ]
    metrics = _structured_multiclass_metrics(
        pixel,
        {
            1: {"true_positive": 0, "false_positive": 1, "false_negative": 1},
            2: {"true_positive": 0, "false_positive": 1, "false_negative": 1},
        },
        schema,
        foreground_pixel={"true_positive": 2, "false_positive": 0, "false_negative": 0},
        foreground_objects={"true_positive": 2, "false_positive": 0, "false_negative": 0},
    )

    assert pixel[1] == {"true_positive": 0, "false_positive": 1, "false_negative": 1}
    assert pixel[2] == {"true_positive": 0, "false_positive": 1, "false_negative": 1}
    assert metrics["pixel"]["macro"]["f1"] == 0.0
    assert metrics["pixel"]["micro"]["false_positive"] == 2
    assert metrics["pixel"]["micro"]["false_negative"] == 2
    assert metrics["pixel"]["foreground"]["f1"] == 1.0
    assert metrics["objects"]["macro"]["f1"] == 0.0


def test_native_multiclass_checkpoint_accepts_canonical_identifiers_without_retraining() -> None:
    loaded = SimpleNamespace(
        model=SimpleNamespace(spec=SimpleNamespace(output_channels=3)),
        artifact=SimpleNamespace(
            metadata={
                "task": "multiclass",
                "class_schema": [
                    {
                        "id": 1,
                        "slug": "type_legacy_1",
                        "name": "Переувлажнение",
                        "color": "#112233",
                        "priority": 10,
                    },
                    {
                        "id": 2,
                        "slug": "type_legacy_2",
                        "name": "Заболачивание",
                        "color": "#445566",
                        "priority": 20,
                    },
                ],
                "val_best_threshold": 0.4,
            }
        ),
    )
    canonical = [
        {
            "id": 1,
            "slug": "floodings",
            "name": "Переувлажнение",
            "color": "#AABBCC",
            "priority": 30,
        },
        {
            "id": 2,
            "slug": "swampings",
            "name": "Заболачивание",
            "color": "#DDEEFF",
            "priority": 40,
        },
    ]

    task, object_types, threshold = _pseudo_runner._native_model_contract(
        loaded,
        {"class_schema": canonical},
    )

    assert task == "multiclass"
    assert object_types == canonical
    assert threshold == pytest.approx(0.4)


def test_features_from_mask_writes_geojson_coordinates_in_wgs84() -> None:
    mask = np.zeros((2, 2), dtype=np.uint8)
    mask[0, 0] = 1
    confidence_map = np.asarray([[0.83, 0.1], [0.2, 0.3]], dtype=np.float32)
    transform = from_origin(11469928.363425, 6873318.079527, 3.4240042653603187, 3.4240042653603293)

    features = _features_from_mask(
        mask,
        transform,
        CRS.from_epsg(3857),
        (3.4240042653603187, 3.4240042653603293),
        "scene-1",
        {"class_key": "deforestation", "class_name": "Вырубки"},
        confidence_map=confidence_map,
    )

    assert len(features) == 1
    xs = [point[0] for point in features[0]["geometry"]["coordinates"][0]]
    ys = [point[1] for point in features[0]["geometry"]["coordinates"][0]]
    assert all(103.0 < x < 103.1 for x in xs)
    assert all(52.3 < y < 52.5 for y in ys)
    assert features[0]["properties"]["_crs"] == "EPSG:3857"
    assert features[0]["properties"]["_x_res"] == 3.4240042653603187
    assert features[0]["properties"]["class_id"] == "deforestation"
    assert features[0]["properties"]["postprocess_profile"] == "none"
    assert features[0]["properties"]["postprocess_level"] == 1
    assert features[0]["properties"]["confidence"] == pytest.approx(0.83)


def test_multiclass_features_keep_parent_class_and_object_type() -> None:
    features = _features_from_mask(
        np.asarray([[1, 2]], dtype=np.uint8),
        from_origin(30.0, 60.0, 0.01, 0.01),
        CRS.from_epsg(4326),
        (0.01, 0.01),
        "scene-1",
        {
            "class_key": "combined-dataset",
            "class_name": "Комбинированный класс",
            "object_types": [
                {
                    "id": 1,
                    "slug": "first",
                    "name": "Первый",
                    "color": "#F59E0B",
                    "priority": 100,
                },
                {
                    "id": 2,
                    "slug": "second",
                    "name": "Второй",
                    "color": "#8B5CF6",
                    "priority": 0,
                },
            ],
        },
    )

    assert len(features) == 2
    assert {item["properties"]["class_id"] for item in features} == {
        "combined-dataset"
    }
    assert {
        (
            item["properties"]["object_type_id"],
            item["properties"]["object_type_slug"],
            item["properties"]["object_type_color"],
        )
        for item in features
    } == {(1, "first", "#F59E0B"), (2, "second", "#8B5CF6")}


def test_aoi_inference_expands_windows_until_internal_object_is_complete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "scene.tif"
    image = np.ones((4, 32, 64), dtype=np.uint8)
    image[0, 8:24, 4:45] = 9
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=64,
        height=32,
        count=4,
        dtype="uint8",
        nodata=0,
        crs="EPSG:4326",
        transform=from_origin(0, 32, 1, 1),
    ) as dataset:
        dataset.write(image)

    monkeypatch.setattr(
        _pseudo_runner,
        "_predict_tile",
        lambda _torch, _model, tile, **_kwargs: (
            (tile[0] == 9).astype(np.uint8),
            (tile[0] == 9).astype(np.float32),
        ),
    )

    performance: dict[str, object] = {}
    features = _pseudo_runner._infer_scene(
        torch=object(),
        model=object(),
        image_path=image_path,
        scene="scene",
        config={"class_key": "rivers"},
        tile_size=16,
        stride=16,
        batch_size=1,
        threshold=0.5,
        device="cpu",
        postprocess_profile=_select_postprocess_profile(1),
        aoi_wgs84=box(20, 10, 24, 22),
        metrics=performance,
    )

    assert len(features) == 1
    assert shape(features[0]["geometry"]).bounds == pytest.approx((4.0, 8.0, 45.0, 24.0))
    assert float(performance["gpu_sec"]) >= 0.0


def test_window_grid_rejects_stride_that_would_leave_internal_gaps() -> None:
    with pytest.raises(RuntimeError, match="появляются разрывы"):
        list(_pseudo_runner._windows(100, 100, tile_size=16, stride=17))


def test_context_windows_start_outside_all_raster_edges() -> None:
    windows = list(
        _pseudo_runner._windows(
            8,
            8,
            tile_size=8,
            stride=4,
            context=2,
        )
    )

    assert [
        (int(window.col_off), int(window.row_off)) for window in windows
    ] == [(-2, -2), (2, -2), (-2, 2), (2, 2)]


def test_pseudo_markup_discards_predictions_from_input_context_frame(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "context.tif"
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=4,
        dtype="uint8",
        nodata=0,
        crs="EPSG:3857",
        transform=from_origin(0, 8, 1, 1),
    ) as dataset:
        dataset.write(np.ones((4, 8, 8), dtype=np.uint8))

    def predict_only_frame(_torch, _model, image, **_kwargs):
        frame = np.ones(image.shape[-2:], dtype=np.uint8)
        frame[2:-2, 2:-2] = 0
        return frame, frame.astype(np.float32)

    monkeypatch.setattr(_pseudo_runner, "_predict_tile", predict_only_frame)

    result = _infer_test_tile_mask(
        torch=object(),
        model=object(),
        input_channels=4,
        image_path=image_path,
        tile_size=8,
        stride=4,
        context=2,
        threshold=0.5,
        device="cpu",
        postprocess_profile=_select_postprocess_profile(0),
    )

    assert np.count_nonzero(result) == 0


def test_test_tile_is_fully_covered_by_checkpoint_sized_windows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "full-area.tif"
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=13,
        height=11,
        count=4,
        dtype="uint8",
        nodata=0,
        crs="EPSG:3857",
        transform=from_origin(0, 11, 1, 1),
    ) as dataset:
        dataset.write(np.ones((4, 11, 13), dtype=np.uint8))

    input_shapes: list[tuple[int, ...]] = []

    def predict_every_pixel(_torch, _model, image, **_kwargs):
        input_shapes.append(image.shape)
        prediction = np.ones(image.shape[-2:], dtype=np.uint8)
        return prediction, prediction.astype(np.float32)

    monkeypatch.setattr(_pseudo_runner, "_predict_tile", predict_every_pixel)

    result = _infer_test_tile_mask(
        torch=object(),
        model=object(),
        input_channels=4,
        image_path=image_path,
        tile_size=8,
        stride=4,
        context=2,
        threshold=0.5,
        device="cpu",
        postprocess_profile=_select_postprocess_profile(0),
    )

    assert result.shape == (11, 13)
    assert np.all(result == 1)
    assert input_shapes == [(4, 8, 8)] * 12


def test_pseudo_runner_accepts_only_rgb_or_rgba_for_three_channel_checkpoint(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.tif"
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 8, 1, 1),
    ) as dataset:
        dataset.write(np.zeros((3, 8, 8), dtype=np.uint8))

    with rasterio.open(image_path) as dataset:
        assert _pseudo_runner._validate_raster_input_channels(dataset, image_path, 3) == (1, 2, 3)
        with pytest.raises(RuntimeError, match="должен содержать 4 каналов"):
            _pseudo_runner._validate_raster_input_channels(dataset, image_path, 4)

    rgba_path = tmp_path / "rgba.tif"
    with rasterio.open(
        rgba_path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=4,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 8, 1, 1),
    ) as dataset:
        dataset.write(np.zeros((4, 8, 8), dtype=np.uint8))

    with rasterio.open(rgba_path) as dataset:
        assert _pseudo_runner._validate_raster_input_channels(dataset, rgba_path, 3) == (1, 2, 3)
        assert _pseudo_runner._validate_raster_input_channels(dataset, rgba_path, 4) == (1, 2, 3, 4)

    five_channel_path = tmp_path / "five-channels.tif"
    with rasterio.open(
        five_channel_path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=5,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 8, 1, 1),
    ) as dataset:
        dataset.write(np.zeros((5, 8, 8), dtype=np.uint8))

    with rasterio.open(five_channel_path) as dataset:
        with pytest.raises(RuntimeError, match="получено 5"):
            _pseudo_runner._validate_raster_input_channels(dataset, five_channel_path, 3)


def test_pseudo_runner_reads_first_three_rgba_channels_for_rgb_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "rgba.tif"
    image = np.stack(
        [np.full((8, 8), value, dtype=np.uint8) for value in (11, 22, 33, 44)]
    )
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=4,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 8, 1, 1),
    ) as dataset:
        dataset.write(image)

    received_images: list[np.ndarray] = []

    def fake_predict_tile(_torch, _model, tile, *, threshold, device):
        del threshold, device
        received_images.append(tile.copy())
        return (
            np.zeros((8, 8), dtype=np.uint8),
            np.zeros((8, 8), dtype=np.float32),
        )

    monkeypatch.setattr(_pseudo_runner, "_predict_tile", fake_predict_tile)
    result = _infer_test_tile_mask(
        torch=object(),
        model=object(),
        input_channels=3,
        image_path=image_path,
        tile_size=8,
        stride=8,
        threshold=0.5,
        device="cpu",
        postprocess_profile=_select_postprocess_profile(0),
    )

    assert np.array_equal(result, np.zeros((8, 8), dtype=np.uint8))
    assert len(received_images) == 1
    assert received_images[0].shape == (3, 8, 8)
    assert [np.unique(channel).item() for channel in received_images[0]] == [11.0, 22.0, 33.0]


def test_parallel_prefetch_closes_rasterio_handles_in_owner_threads(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.tif"
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=3,
        dtype="uint8",
        nodata=0,
        crs="EPSG:3857",
        transform=from_origin(0, 16, 1, 1),
    ) as output:
        output.write(np.ones((3, 16, 16), dtype=np.uint8))

    metrics: dict[str, object] = {}
    with _pseudo_runner._InferenceRasterReader(
        image_path,
        input_channels=3,
        channel_mapping="rgb",
        source_imagery_type="ortho",
        target_resolution_m=1.0,
        metrics=metrics,
    ) as reader:
        assert reader.dataset is not None
        tiles = list(
            _pseudo_runner._prefetched_tiles(
                [
                    rasterio.windows.Window(0, 0, 8, 8),
                    rasterio.windows.Window(8, 0, 8, 8),
                    rasterio.windows.Window(0, 8, 8, 8),
                    rasterio.windows.Window(8, 8, 8, 8),
                ],
                dataset=reader.dataset,
                input_indexes=(1, 2, 3),
                nodata=0,
                tile_size=8,
                raster_reader=reader,
                read_workers=4,
                prefetch_batches=2,
                batch_size=2,
            )
        )

    assert [window for window, _ in tiles] == [
        rasterio.windows.Window(0, 0, 8, 8),
        rasterio.windows.Window(8, 0, 8, 8),
        rasterio.windows.Window(0, 8, 8, 8),
        rasterio.windows.Window(8, 8, 8, 8),
    ]
    assert all(tile is not None and tile.shape == (3, 8, 8) for _, tile in tiles)
    assert float(metrics["resampling_sec"]) >= 0.0


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
            "postprocess.smooth.enabled": True,
            "postprocess.smooth.iterations": 1,
            "postprocess.smooth.offset": 0.125,
            "postprocess.simplify_m": 1.0,
            "postprocess.filter_compact_objects.enabled": True,
            "postprocess.filter_compact_objects.min_isoperimetric_quotient": 0.25,
            "postprocess.filter_compact_objects.max_bbox_ratio": 3.5,
        },
    )

    assert profile.name == "strong"
    assert profile.min_area_m2 == 10000.0
    assert profile.min_hole_area_m2 == 5000.0
    assert profile.smooth_iterations == 1
    assert profile.smooth_offset == 0.125
    assert profile.simplify_m == 1.0
    assert profile.filter_compact_mode == "remove_compact"
    assert profile.filter_compact_min_isoperimetric_quotient == 0.25
    assert profile.filter_compact_max_bbox_ratio == 3.5


def test_geometry_postprocessor_is_absent_for_none_profile() -> None:
    assert _geometry_postprocessor(_select_postprocess_profile(1)) is None


def test_geometry_postprocessor_applies_configured_vector_filters() -> None:
    profile = replace(_select_postprocess_profile(1), min_area_m2=50.0)
    postprocessor = _geometry_postprocessor(profile)

    assert postprocessor is not None
    assert postprocessor(box(0, 0, 10, 10), "EPSG:32637").area == 100.0
    assert postprocessor(box(0, 0, 5, 5), "EPSG:32637").is_empty


def test_find_images_accepts_txt_scene_forms_and_dataset_folders(tmp_path) -> None:
    scene_dir = tmp_path / "kanopus" / "irkutsk"
    scene_dir.mkdir(parents=True)
    first = scene_dir / "KV3_100.L2.PMS.SCN01_cog.tif"
    second = scene_dir / "KV3_101.L2.PMS.SCN02.tif"
    first.touch()
    second.touch()

    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text(
        "./KV3_100.L2.PMS.SCN01_cog.tif\n"
        "KV3_100.L2.PMS.SCN01.tif\n"
        "irkutsk\n",
        encoding="utf-8",
    )

    inputs, missing, input_scene_count = _resolve_scene_inputs(
        {"images_root": str(tmp_path), "scenes_file": str(scenes_file)}
    )

    assert [item.image_path for item in inputs] == [first, second]
    assert inputs[0].request_scenes == (
        "./KV3_100.L2.PMS.SCN01_cog.tif",
        "KV3_100.L2.PMS.SCN01.tif",
        "irkutsk",
    )
    assert inputs[1].request_scenes == ("irkutsk",)
    assert missing == []
    assert input_scene_count == 3


def test_find_images_prefers_exact_relative_path_for_duplicate_filename(
    tmp_path: Path,
) -> None:
    images_root = tmp_path / "images"
    first = images_root / "first" / "shared.tif"
    second = images_root / "second" / "shared.tif"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()

    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text("first/shared.tif\n", encoding="utf-8")

    inputs, missing, input_scene_count = _resolve_scene_inputs(
        {"images_root": str(images_root), "scenes_file": str(scenes_file)}
    )

    assert [item.image_path for item in inputs] == [first]
    assert missing == []
    assert input_scene_count == 1


def test_collect_scene_inputs_deduplicates_found_rasters(tmp_path) -> None:
    scene_dir = tmp_path / "kanopus" / "irkutsk"
    scene_dir.mkdir(parents=True)
    first = scene_dir / "KV3_100.L2.PMS.SCN01_cog.tif"
    second = scene_dir / "KV3_101.L2.PMS.SCN02.tif"
    first.touch()
    second.touch()
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text(
        "irkutsk\nKV3_100.L2.PMS.SCN01.tif\nlost\n",
        encoding="utf-8",
    )

    inputs, missing, _input_scene_count = _resolve_scene_inputs(
        {"images_root": str(tmp_path), "scenes_file": str(scenes_file)}
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
    scenes_file = tmp_path / "scenes.txt"
    scenes_file.write_text(
        "irkutsk\nKV3_100.L2.PMS.SCN01.tif\nlost\n",
        encoding="utf-8",
    )
    inputs, missing, _input_scene_count = _resolve_scene_inputs(
        {"images_root": str(tmp_path), "scenes_file": str(scenes_file)}
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
    mask[2:10, 2:10] = 1

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
    features[0]["properties"]["confidence"] = 0.72
    features[1]["properties"]["confidence"] = 0.91

    merged = _merge_overlapping_features(features)

    assert len(merged) == 1
    geometry = shape(merged[0]["geometry"])
    assert round(geometry.area, 6) == 7.0
    assert merged[0]["properties"]["scene_id"] == "scene-1"
    assert merged[0]["properties"]["source_scene_ids"] == ["scene-1", "scene-2"]
    assert merged[0]["properties"]["merged_feature_count"] == 2
    assert merged[0]["properties"]["confidence"] == 0.91


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


def test_multiclass_conflicts_use_confidence_then_priority() -> None:
    schema = [
        {"id": 1, "slug": "first", "name": "Первый", "color": "#F59E0B", "priority": 100},
        {"id": 2, "slug": "second", "name": "Второй", "color": "#8B5CF6", "priority": 0},
    ]
    features = [
        _typed_geojson_feature(box(0, 0, 2, 2), class_id=1, confidence=0.8),
        _typed_geojson_feature(box(1, 0, 3, 2), class_id=2, confidence=0.9),
        _typed_geojson_feature(box(4, 0, 6, 2), class_id=1, confidence=0.7),
        _typed_geojson_feature(box(5, 0, 7, 2), class_id=2, confidence=0.7),
    ]

    resolved = _resolve_feature_type_conflicts(features, schema)

    areas_by_id: dict[int, float] = {1: 0.0, 2: 0.0}
    for feature in resolved:
        areas_by_id[int(feature["properties"]["object_type_id"])] += shape(
            feature["geometry"]
        ).area
    assert areas_by_id == {1: 6.0, 2: 6.0}


def test_multiclass_conflict_resolution_does_not_union_disjoint_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = [
        {"id": 1, "slug": "first", "name": "Первый", "color": "#F59E0B", "priority": 100},
        {"id": 2, "slug": "second", "name": "Второй", "color": "#8B5CF6", "priority": 0},
    ]
    features = [
        _typed_geojson_feature(
            box(index * 3, 0, index * 3 + 1, 1),
            class_id=(index % 2) + 1,
            confidence=0.5,
        )
        for index in range(250)
    ]
    calls = 0
    real_unary_union = _pseudo_runner.unary_union

    def counted_unary_union(geometries):
        nonlocal calls
        calls += 1
        return real_unary_union(geometries)

    monkeypatch.setattr(_pseudo_runner, "unary_union", counted_unary_union)

    resolved = _resolve_feature_type_conflicts(features, schema)

    assert len(resolved) == len(features)
    assert calls == 0


def test_geometry_postprocess_removes_compact_objects_regardless_of_area() -> None:
    profile = replace(
        _postprocess_profile_from_config(
            _select_postprocess_profile(51),
            {
                "postprocess.filter_compact_objects.enabled": True,
                "postprocess.filter_compact_objects.min_isoperimetric_quotient": 0.25,
                "postprocess.filter_compact_objects.max_bbox_ratio": 3.5,
            },
        ),
        mask_min_object_pixels=None,
        mask_min_hole_pixels=None,
        min_area_m2=None,
        min_hole_area_m2=None,
        simplify_m=None,
    )
    small_square = _postprocess_geometry(box(0, 0, 10, 10), "EPSG:32637", profile)
    large_square = _postprocess_geometry(box(0, 0, 1000, 1000), "EPSG:32637", profile)
    elongated = _postprocess_geometry(box(0, 0, 100, 10), "EPSG:32637", profile)

    assert small_square.is_empty
    assert large_square.is_empty
    assert not elongated.is_empty


def test_geometry_postprocess_inverse_mode_keeps_only_compact_objects() -> None:
    profile = replace(
        _postprocess_profile_from_config(
            _select_postprocess_profile(51),
            {
                "postprocess.filter_compact_objects.enabled": True,
                "postprocess.filter_compact_objects.mode": "keep_compact",
                "postprocess.filter_compact_objects.min_isoperimetric_quotient": 0.25,
                "postprocess.filter_compact_objects.max_bbox_ratio": 3.5,
            },
        ),
        mask_min_object_pixels=None,
        mask_min_hole_pixels=None,
        min_area_m2=None,
        min_hole_area_m2=None,
        simplify_m=None,
    )
    small_square = _postprocess_geometry(box(0, 0, 10, 10), "EPSG:32637", profile)
    large_square = _postprocess_geometry(box(0, 0, 1000, 1000), "EPSG:32637", profile)
    elongated = _postprocess_geometry(box(0, 0, 100, 10), "EPSG:32637", profile)

    assert not small_square.is_empty
    assert not large_square.is_empty
    assert elongated.is_empty


def test_geometry_postprocess_smooths_before_simplifying() -> None:
    profile = replace(
        _select_postprocess_profile(1),
        smooth_iterations=1,
        smooth_offset=0.125,
        simplify_m=1.0,
    )
    polygon = shape(
        {
            "type": "Polygon",
            "coordinates": [[
                [0, 0],
                [0, 10],
                [5, 10],
                [5, 20],
                [15, 20],
                [15, 0],
                [0, 0],
            ]],
        }
    )

    result = _postprocess_geometry(polygon, "EPSG:32637", profile)

    assert result.is_valid
    assert result != polygon
    assert len(result.exterior.coords) <= len(polygon.exterior.coords) * 2


def test_geometry_postprocess_reprojects_web_mercator_to_local_utm() -> None:
    geometry = box(11546954, 6801527, 11552171, 6806744)

    metric_geometry, to_source = _geometry_to_metric(geometry, "EPSG:3857")

    assert to_source is not None
    assert metric_geometry.bounds != geometry.bounds
    restored = shapely_transform(to_source, metric_geometry)
    assert restored.equals_exact(geometry, tolerance=1e-6)


@pytest.mark.parametrize(
    ("rows", "columns"),
    [
        (slice(0, 12), slice(8, 20)),
        (slice(20, 32), slice(8, 20)),
        (slice(8, 20), slice(0, 12)),
        (slice(8, 20), slice(20, 32)),
    ],
)
def test_test_tile_keeps_compact_prediction_touching_raster_boundary(
    tmp_path,
    monkeypatch,
    rows,
    columns,
) -> None:
    prediction = np.zeros((32, 32), dtype=np.uint8)
    prediction[rows, columns] = 1
    result = _infer_fixed_test_prediction(
        tmp_path,
        monkeypatch,
        prediction,
        _compact_filter_profile(),
    )

    assert np.array_equal(result, prediction)


def test_test_tile_filters_same_compact_prediction_inside_raster(
    tmp_path,
    monkeypatch,
) -> None:
    prediction = np.zeros((32, 32), dtype=np.uint8)
    prediction[8:20, 8:20] = 1
    result = _infer_fixed_test_prediction(
        tmp_path,
        monkeypatch,
        prediction,
        _compact_filter_profile(),
    )

    assert not np.any(result)


def test_test_tile_keeps_small_prediction_touching_raster_boundary(
    tmp_path,
    monkeypatch,
) -> None:
    prediction = np.zeros((32, 32), dtype=np.uint8)
    prediction[:3, :3] = 1
    profile = replace(
        _select_postprocess_profile(6),
        min_area_m2=1000.0,
        min_hole_area_m2=None,
        simplify_m=None,
    )
    result = _infer_fixed_test_prediction(tmp_path, monkeypatch, prediction, profile)

    assert np.array_equal(result, prediction)


def test_test_tile_keeps_compact_prediction_within_one_pixel_of_boundary(
    tmp_path,
    monkeypatch,
) -> None:
    prediction = np.zeros((32, 32), dtype=np.uint8)
    prediction[1:5, 10:14] = 1

    result = _infer_fixed_test_prediction(
        tmp_path,
        monkeypatch,
        prediction,
        _compact_filter_profile(),
    )

    assert np.array_equal(result, prediction)


def _compact_filter_profile():
    return replace(
        _postprocess_profile_from_config(
            _select_postprocess_profile(6),
            {
                "postprocess.filter_compact_objects.enabled": True,
                "postprocess.filter_compact_objects.min_isoperimetric_quotient": 0.25,
                "postprocess.filter_compact_objects.max_bbox_ratio": 3.5,
            },
        ),
        mask_min_object_pixels=None,
        mask_min_hole_pixels=None,
        min_area_m2=1.0,
        min_hole_area_m2=None,
        simplify_m=None,
    )


def _infer_fixed_test_prediction(tmp_path, monkeypatch, prediction, profile) -> np.ndarray:
    image_path = tmp_path / "tile.tif"
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=4,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 32, 1, 1),
    ) as dataset:
        dataset.write(np.ones((4, 32, 32), dtype=np.uint8))
    monkeypatch.setattr(
        _pseudo_runner,
        "_predict_tile",
        lambda *args, **kwargs: (prediction, prediction.astype(np.float32)),
    )
    return _infer_test_tile_mask(
        torch=object(),
        model=object(),
        image_path=image_path,
        tile_size=32,
        stride=32,
        threshold=0.5,
        device="cpu",
        postprocess_profile=profile,
    )


def test_summary_reports_unique_image_count_and_postprocess_profile(tmp_path) -> None:
    profile = _select_postprocess_profile(51)

    summary = _summary(
        {},
        input_scene_count=2,
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


def test_run_pseudo_markup_uses_external_torchscript_adapter(tmp_path, monkeypatch) -> None:
    config = _pseudo_runner_config(tmp_path)
    config["external_model"] = _external_oks_manifest()
    config["threshold"] = None
    fake_torch = _FakeTorch()
    loaded_external = SimpleNamespace(torch=fake_torch)
    calls: dict[str, object] = {}

    def fake_load_external(archive_path, manifest, *, device, scratch_root):
        calls["archive_path"] = archive_path
        calls["manifest"] = manifest
        calls["device"] = device
        calls["scratch_root"] = scratch_root
        return loaded_external

    def fake_predict(loaded, **kwargs):
        assert loaded is loaded_external
        calls["prediction"] = kwargs
        return []

    monkeypatch.setattr(_pseudo_runner, "load_external_model", fake_load_external)
    monkeypatch.setattr(_pseudo_runner, "predict_external_scene", fake_predict)
    monkeypatch.setattr(
        _pseudo_runner,
        "load_checkpoint",
        lambda request: pytest.fail(f"Нативный loader вызван для внешней модели: {request}"),
    )

    report = run_pseudo_markup(config)

    assert report["status"] == "ok"
    assert report["processed"] == 1
    assert report["source"]["threshold"] is None
    assert calls["device"] == "cuda"
    assert calls["manifest"].adapter == "oks_multiclass_footprints"
    assert calls["prediction"]["scene"] == "scene-1"
    assert calls["prediction"]["geometry_postprocessor"] is None
    assert fake_torch.cuda.empty_cache_calls == 1


def test_test_sample_f1_sums_tiles_with_identical_geographic_bounds_independently(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    transform = from_origin(0, 4, 1, 1)
    tiles = []
    true_masks = [
        np.asarray([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
        np.asarray([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
    ]
    predictions = [
        np.asarray([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
        np.asarray([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
    ]
    for index, true_mask in enumerate(true_masks, start=1):
        image_path = tmp_path / f"tile_{index:03d}.tif"
        mask_path = tmp_path / f"tile_{index:03d}_mask.png"
        with rasterio.open(
            image_path,
            "w",
            driver="GTiff",
            width=4,
            height=4,
            count=4,
            dtype="uint8",
            crs="EPSG:3857",
            transform=transform,
        ) as dataset:
            dataset.write(np.ones((4, 4, 4), dtype=np.uint8))
        with rasterio.open(
            mask_path,
            "w",
            driver="PNG",
            width=4,
            height=4,
            count=1,
            dtype="uint8",
        ) as dataset:
            dataset.write(true_mask * 255, 1)
        tiles.append({"index": index, "image_path": str(image_path), "mask_path": str(mask_path)})

    fake_torch = _FakeTorch()
    fake_model = _FakeModel()
    monkeypatch.setattr(_pseudo_runner, "_torch", lambda: fake_torch)
    monkeypatch.setattr(
        _pseudo_runner,
        "load_checkpoint",
        lambda request: SimpleNamespace(model=SimpleNamespace(model=fake_model)),
    )
    monkeypatch.setattr(
        _pseudo_runner,
        "_infer_test_tile_mask",
        lambda **kwargs: predictions[int(Path(kwargs["image_path"]).stem[-3:]) - 1],
    )

    report = run_test_sample_f1(
        {
            "operation": "test_sample_f1",
            "run_root": str(tmp_path / "run"),
            "local_checkpoint_path": str(checkpoint_path),
            "threshold": 0.5,
            "tile_size": 4,
            "stride": 4,
            "device": "cuda",
            "postprocess_profile": "strong",
            "test_f1_evaluator_version": 2,
            "tiles": tiles,
        }
    )

    assert report["status"] == "ok"
    assert report["processed"] == 2
    assert report["true_positive"] == 2
    assert report["false_positive"] == 1
    assert report["false_negative"] == 1
    assert report["object_true_positive"] == 2
    assert report["object_false_positive"] == 0
    assert report["object_false_negative"] == 0
    assert report["postprocess_profile"] == "strong"
    assert report["test_f1_evaluator_version"] == 2
    assert report["preserve_boundary_components"] is True
    assert fake_torch.cuda.empty_cache_calls == 1


def test_test_sample_f1_prefers_checkpoint_window_over_test_tile_size(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    image_path = tmp_path / "tile.tif"
    mask_path = tmp_path / "tile_mask.png"
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=13,
        height=11,
        count=4,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 11, 1, 1),
    ) as dataset:
        dataset.write(np.ones((4, 11, 13), dtype=np.uint8))
    with rasterio.open(
        mask_path,
        "w",
        driver="PNG",
        width=13,
        height=11,
        count=1,
        dtype="uint8",
    ) as dataset:
        dataset.write(np.zeros((11, 13), dtype=np.uint8), 1)

    fake_torch = _FakeTorch()
    fake_model = _FakeModel()
    monkeypatch.setattr(_pseudo_runner, "_torch", lambda: fake_torch)
    monkeypatch.setattr(
        _pseudo_runner,
        "load_checkpoint",
        lambda _request: SimpleNamespace(
            model=SimpleNamespace(
                model=fake_model,
                spec=SimpleNamespace(input_channels=4, output_channels=1),
            ),
            artifact=SimpleNamespace(
                metadata={
                    "sample_size": 8,
                    "inference_context": 2,
                    "inference_core_size": 4,
                    "confidence_threshold": 0.5,
                    "task": "binary",
                }
            ),
        ),
    )
    received: dict[str, int] = {}

    def infer(**kwargs):
        received.update(
            tile_size=int(kwargs["tile_size"]),
            context=int(kwargs["context"]),
            stride=int(kwargs["stride"]),
        )
        return np.zeros((11, 13), dtype=np.uint8)

    monkeypatch.setattr(_pseudo_runner, "_infer_test_tile_mask", infer)

    report = run_test_sample_f1(
        {
            "operation": "test_sample_f1",
            "run_root": str(tmp_path / "run-checkpoint-window"),
            "local_checkpoint_path": str(checkpoint_path),
            "tile_size": 13,
            "context": 0,
            "stride": 13,
            "device": "cuda",
            "tiles": [
                {
                    "index": 1,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                }
            ],
        }
    )

    assert report["status"] == "ok"
    assert received == {"tile_size": 8, "context": 2, "stride": 4}
    assert report["inference_tile_size"] == 8
    assert report["inference_context"] == 2
    assert report["inference_stride"] == 4
    assert report["inference_core_size"] == 4
    assert report["tiles"][0]["image_width"] == 13
    assert report["tiles"][0]["image_height"] == 11
    assert report["tiles"][0]["inference_window_count"] == 12


def test_checkpoint_window_falls_back_when_legacy_metadata_values_are_empty() -> None:
    loaded = SimpleNamespace(
        artifact=SimpleNamespace(
            metadata={"sample_size": None, "inference_context": None}
        )
    )

    assert _pseudo_runner._checkpoint_inference_window(
        loaded,
        tile_size=1024,
        context=0,
        stride=512,
    ) == (1024, 0, 512)


def test_managed_test_sample_f1_uses_each_class_sample_and_macro_average(
    tmp_path: Path,
) -> None:
    masks = [
        np.asarray(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
        np.asarray(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    ]
    predictions = [
        np.asarray(
            [
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [2, 2, 2, 2],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
        np.asarray(
            [
                [2, 2, 2, 2],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    ]
    tiles = []
    for index, (mask, prediction) in enumerate(zip(masks, predictions, strict=True), start=1):
        mask_path = tmp_path / f"managed_{index}_mask.png"
        prediction_path = tmp_path / f"managed_{index}_prediction.npy"
        with rasterio.open(
            mask_path,
            "w",
            driver="PNG",
            width=4,
            height=4,
            count=1,
            dtype="uint8",
        ) as dataset:
            dataset.write(mask * 255, 1)
        np.save(prediction_path, prediction)
        tiles.append(
            {
                "index": index,
                "image_path": str(tmp_path / f"managed_{index}.tif"),
                "mask_path": str(mask_path),
                "precomputed_prediction_path": str(prediction_path),
                "target_class_id": index,
                "target_class_slug": f"class_{index}",
                "test_sample_id": f"sample-{index}",
                "source_tile_index": 1,
            }
        )

    report = run_test_sample_f1(
        {
            "operation": "test_sample_f1",
            "run_root": str(tmp_path / "managed-run"),
            "task": "multiclass",
            "object_types": [
                {"id": 1, "slug": "class_1", "name": "Первый", "color": "#112233"},
                {"id": 2, "slug": "class_2", "name": "Второй", "color": "#445566"},
            ],
            "managed_test_samples": True,
            "test_samples": [{"sample_id": "sample-1"}, {"sample_id": "sample-2"}],
            "tiles": tiles,
        }
    )

    assert report["status"] == "ok"
    assert report["true_positive"] == 5
    assert report["false_positive"] == 3
    assert report["false_negative"] == 4
    pixel = report["metrics"]["pixel"]
    assert pixel["per_class"]["class_1"]["f1"] == pytest.approx(2 / 3)
    assert pixel["per_class"]["class_2"]["f1"] == pytest.approx(0.4)
    assert pixel["macro"]["f1"] == pytest.approx((2 / 3 + 0.4) / 2)
    assert report["metrics"]["objects"]["macro"]["f1"] == pytest.approx(0.5)
    assert report["metrics"]["aggregation"] == "macro"
    assert report["metrics"]["test_samples"] == [
        {"sample_id": "sample-1"},
        {"sample_id": "sample-2"},
    ]


def test_test_sample_f1_passes_external_instance_ids(tmp_path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "model.zip"
    checkpoint_path.write_bytes(b"archive")
    image_path = tmp_path / "tile.tif"
    image_path.touch()
    mask_path = tmp_path / "tile_mask.png"
    with rasterio.open(
        mask_path,
        "w",
        driver="PNG",
        width=4,
        height=1,
        count=1,
        dtype="uint8",
    ) as dataset:
        dataset.write(np.ones((1, 4), dtype=np.uint8) * 255, 1)
    fake_torch = _FakeTorch()
    loaded_external = SimpleNamespace(torch=fake_torch)
    predicted_instances = np.asarray([[10, 10, 20, 20]], dtype=np.int64)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        _pseudo_runner,
        "load_external_model",
        lambda *args, **kwargs: loaded_external,
    )

    def fake_external_prediction(*args, **kwargs):
        captured["geometry_postprocessor"] = kwargs["geometry_postprocessor"]
        return ExternalTestPrediction(
            mask=(predicted_instances > 0).astype(np.uint8),
            instances=predicted_instances,
        )

    monkeypatch.setattr(
        _pseudo_runner,
        "predict_external_test_tile",
        fake_external_prediction,
    )

    def fake_object_f1(request):
        captured["instances"] = request.y_pred_instances
        return SimpleNamespace(true_positive=2, false_positive=0, false_negative=0)

    monkeypatch.setattr(_pseudo_runner, "compute_object_f1", fake_object_f1)

    report = run_test_sample_f1(
        {
            "operation": "test_sample_f1",
            "run_root": str(tmp_path / "run"),
            "local_checkpoint_path": str(checkpoint_path),
            "external_model": _external_zu_manifest(),
            "threshold": 0.0,
            "device": "cuda",
            "tiles": [
                {
                    "index": 1,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                }
            ],
        }
    )

    assert report["status"] == "ok"
    assert np.array_equal(captured["instances"], predicted_instances)
    assert captured["geometry_postprocessor"] is None
    assert report["object_true_positive"] == 2
    assert fake_torch.cuda.empty_cache_calls == 1


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


def _typed_geojson_feature(
    geometry,
    *,
    class_id: int,
    confidence: float,
) -> dict:
    return {
        "type": "Feature",
        "geometry": geometry.__geo_interface__,
        "properties": {
            "object_type_id": class_id,
            "confidence": confidence,
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


def _external_oks_manifest() -> dict[str, object]:
    return {
        "version": 1,
        "adapter": "oks_multiclass_footprints",
        "artifact_path": "models/model.zip",
        "archive_sha256": "0" * 64,
        "model_member": "oks/1/model.pt",
        "model_root": "oks",
        "input_channels": 3,
        "target_resolution_m": 0.6,
        "tile_size": 1024,
        "stride": 768,
        "context": 128,
        "score_threshold": None,
        "min_area_m2": 30.0,
        "min_hole_area_m2": 50.0,
        "nms_iou_threshold": None,
        "nms_relative_intersection": None,
        "max_shift_m": 50.0,
        "shift_iterations": 50,
        "shift_confidence": 0.2,
        "correction_confidence": 0.05,
    }


def _external_zu_manifest() -> dict[str, object]:
    return {
        "version": 1,
        "adapter": "detectron2_instances",
        "artifact_path": "models/model.zip",
        "archive_sha256": "0" * 64,
        "model_member": "zu/1/model.pt",
        "model_root": "zu",
        "input_channels": 3,
        "target_resolution_m": 0.15,
        "tile_size": 1884,
        "stride": 628,
        "context": 628,
        "score_threshold": 0.0,
        "min_area_m2": 50.0,
        "min_hole_area_m2": 10.0,
        "nms_iou_threshold": 0.75,
        "nms_relative_intersection": 0.75,
        "max_shift_m": None,
        "shift_iterations": None,
        "shift_confidence": None,
        "correction_confidence": None,
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
