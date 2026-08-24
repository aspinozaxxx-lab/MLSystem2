from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping
import yaml

from mlsystem2.training_ui_api import _geoalert_compose_runner, _model_export, _worker
from mlsystem2.training_ui_api._geoalert_runner import (
    _effective_postprocess_config,
    _extract_model_archive,
    _pipeline_bricks,
    _rasterize_scene_prediction,
)
from mlsystem2.training_ui_api._templates import (
    INFERENCE_BASE_DEFAULT_CONFIG,
    initial_inference_templates,
)
from mlsystem2.training_ui_api.contracts import TrainingUIAPIError
from mlsystem2.training_ui_api._inference_backend import (
    GEOALERT_INFERENCE_BACKEND,
    PYTORCH_INFERENCE_BACKEND,
    inference_backend_for_imagery,
)
from mlsystem2.training_ui_api._pseudo_runner import run_test_sample_f1


def test_geoalert_pause_unloads_triton_and_loads_it_on_resume(
    tmp_path: Path,
    monkeypatch,
) -> None:
    control_dir = tmp_path / "control"
    control_dir.mkdir()
    request_path = control_dir / "pause.request"
    request_path.write_text("pause-token\n", encoding="utf-8")
    actions: list[tuple[str, str]] = []
    ready: list[str] = []
    monkeypatch.setattr(
        _geoalert_compose_runner,
        "_triton_repository_action",
        lambda _url, model, action: actions.append((model, action)),
    )
    monkeypatch.setattr(
        _geoalert_compose_runner,
        "_wait_for_triton_model",
        lambda _url, model: ready.append(model),
    )

    thread = threading.Thread(
        target=_geoalert_compose_runner._pause_if_requested,
        args=(
            {
                "control_dir": str(control_dir),
                "triton_http_url": "http://triton",
                "triton_model_name": "secondary-model",
            },
        ),
    )
    thread.start()
    marker_path = control_dir / "paused"
    deadline = time.monotonic() + 2
    while not marker_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert marker_path.read_text(encoding="utf-8").strip() == "pause-token"
    assert actions == [("secondary-model", "unload")]

    request_path.unlink()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert actions == [
        ("secondary-model", "unload"),
        ("secondary-model", "load"),
    ]
    assert ready == ["secondary-model"]
    assert not marker_path.exists()


def test_ortho_uses_geoalert_backend_and_kanopus_keeps_compatible_backend() -> None:
    assert inference_backend_for_imagery("ortho") == GEOALERT_INFERENCE_BACKEND
    assert inference_backend_for_imagery("kanopus") == PYTORCH_INFERENCE_BACKEND
    assert inference_backend_for_imagery(None) == PYTORCH_INFERENCE_BACKEND


def test_worker_launches_geoalert_runner_for_ortho_job(tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    (run_dir / "logs").mkdir(parents=True)
    config_path = run_dir / "pseudo.yaml"
    config_path.write_text("{}", encoding="utf-8")

    script = _worker._write_pseudo_run_script(
        SimpleNamespace(project_root=tmp_path),
        run_dir,
        config_path,
        inference_backend=GEOALERT_INFERENCE_BACKEND,
    ).read_text(encoding="utf-8")

    assert "mlsystem2.training_ui_api._geoalert_runner" in script
    assert "mlsystem2.training_ui_api._pseudo_runner" not in script


def test_native_geoalert_pipeline_contains_context_gpu_and_postprocess_bricks() -> None:
    pipeline = yaml.safe_load(
        _model_export._pipeline_yaml(
            "ortho_model",
            768,
            3,
            context=128,
            resolution_m=0.5,
            postprocess_config={
                "postprocess.mask_min_object_pixels": 32,
                "postprocess.mask_min_hole_pixels": 16,
                "postprocess.binary_closing_radius": 2,
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
    )
    bricks = pipeline["config"]["bricks"]
    segmentation = bricks[1]

    assert segmentation["bounds"] == 128
    assert segmentation["sample_size"] == [512, 512]
    assert segmentation["res"] == [0.5, 0.5]
    assert [brick["_class"] for brick in bricks[2:5]] == [
        "MaskMorphology",
        "MaskMorphology",
        "MaskMorphology",
    ]
    assert bricks[5]["_class"] == "VectorizeMasks"
    assert bricks[6]["_class"] == "UnifiedVectorProcessing"
    assert bricks[5]["output_fcs"] == ["output"]
    assert bricks[5]["mark_boundary_objects"] is True
    assert bricks[5]["boundary_tolerance_pixels"] == 1.0
    assert bricks[5]["boundary_tag"] == "_touches_raster_boundary"
    assert bricks[6]["input"] == "output"
    assert bricks[6]["output"] == "output"
    assert [brick["_class"] for brick in bricks[6]["bricks"]] == [
        "FilterSmallObjects",
        "RemoveSmallHoles",
        "FilterCompactObjects",
        "Smooth",
        "Simplify",
        "RemoveTags",
    ]
    assert bricks[6]["bricks"][0]["preserve_boundary_objects"] is True
    assert bricks[6]["bricks"][0]["min_area"] == 10000.0
    assert bricks[6]["bricks"][0]["area_tag"] == "area"
    assert bricks[6]["bricks"][1]["min_hole_area"] == 5000.0
    assert bricks[6]["bricks"][2]["preserve_boundary_objects"] is True
    assert bricks[6]["bricks"][2]["min_isoperimetric_quotient"] == 0.25
    assert bricks[6]["bricks"][2]["max_bbox_ratio"] == 3.5
    assert bricks[6]["bricks"][3]["iterations"] == 1
    assert bricks[6]["bricks"][3]["offset"] == 0.125
    assert bricks[6]["bricks"][4]["rate"] == 1.0
    assert bricks[6]["bricks"][5]["tags"] == ["_touches_raster_boundary"]
    assert _pipeline_bricks(pipeline) == [
        "SplitRaster",
        "Segmentation",
        "TritonAdapter",
        "MaskMorphology",
        "VectorizeMasks",
        "UnifiedVectorProcessing",
        "FilterSmallObjects",
        "RemoveSmallHoles",
        "FilterCompactObjects",
        "Smooth",
        "Simplify",
        "RemoveTags",
    ]
    assert "kind: KIND_GPU" in _model_export._triton_config(
        "ortho_model",
        3,
        instance_kind="KIND_GPU",
    )


def test_geoalert_effective_postprocess_keeps_smooth_settings() -> None:
    config = {
        "postprocess_profile": "none",
        "postprocess_config": {
            **INFERENCE_BASE_DEFAULT_CONFIG,
            "postprocess.smooth.enabled": True,
            "postprocess.smooth.iterations": 1,
            "postprocess.smooth.offset": 0.125,
        },
    }

    result = _effective_postprocess_config(config, 8, external=False)

    assert result["postprocess.smooth.enabled"] is True
    assert result["postprocess.smooth.iterations"] == 1
    assert result["postprocess.smooth.offset"] == 0.125


def test_native_geoalert_pipeline_rejects_smooth_offset_above_half() -> None:
    with pytest.raises(TrainingUIAPIError, match="не больше 0.5"):
        _model_export._pipeline_yaml(
            "rivers_kanopus",
            768,
            4,
            postprocess_config={
                "postprocess.smooth.enabled": True,
                "postprocess.smooth.iterations": 1,
                "postprocess.smooth.offset": 0.5001,
            },
        )


def test_base_default_pipeline_has_no_compact_filter_or_smoothing() -> None:
    effective_config = _effective_postprocess_config(
        {
            "postprocess_config": INFERENCE_BASE_DEFAULT_CONFIG,
        },
        8,
        external=False,
    )
    pipeline = yaml.safe_load(
        _model_export._pipeline_yaml(
            "lakes_kanopus",
            768,
            4,
            postprocess_config=effective_config,
        )
    )
    bricks = pipeline["config"]["bricks"]
    vectorize = next(brick for brick in bricks if brick["_class"] == "VectorizeMasks")
    vector_processing = next(
        brick for brick in bricks if brick["_class"] == "UnifiedVectorProcessing"
    )
    pipeline_bricks = _pipeline_bricks(pipeline)

    assert "mark_boundary_objects" not in vectorize
    assert [brick["_class"] for brick in vector_processing["bricks"]] == [
        "Simplify",
        "FilterSmallObjects",
        "RemoveSmallHoles",
    ]
    assert "FilterSmallObjects" in pipeline_bricks
    assert "RemoveSmallHoles" in pipeline_bricks
    assert "Simplify" in pipeline_bricks
    assert "FilterCompactObjects" not in pipeline_bricks
    assert "Smooth" not in pipeline_bricks
    assert "RemoveTags" not in pipeline_bricks


def test_lake_seed_pipeline_keeps_compact_objects_with_configured_thresholds() -> None:
    lake_template = next(
        item
        for item in initial_inference_templates()
        if item.get("dataset_key") == "Озера\\main"
    )
    pipeline = yaml.safe_load(
        _model_export._pipeline_yaml(
            "lakes_kanopus",
            768,
            4,
            postprocess_config=lake_template["default_config"],
        )
    )
    bricks = pipeline["config"]["bricks"]
    vectorize = next(brick for brick in bricks if brick["_class"] == "VectorizeMasks")
    vector_processing = next(
        brick for brick in bricks if brick["_class"] == "UnifiedVectorProcessing"
    )
    inverse_filter = next(
        brick
        for brick in vector_processing["bricks"]
        if brick["_class"] == "FilterNonCompactObjects"
    )

    assert vectorize["mark_boundary_objects"] is True
    assert vectorize["boundary_tolerance_pixels"] == 1.0
    assert inverse_filter["min_isoperimetric_quotient"] == 0.25
    assert inverse_filter["max_bbox_ratio"] == 3.5
    assert inverse_filter["preserve_boundary_objects"] is True
    assert [brick["_class"] for brick in vector_processing["bricks"]] == [
        "FilterNonCompactObjects",
        "RemoveTags",
    ]


def test_native_geoalert_pipeline_rejects_unknown_compact_filter_mode() -> None:
    with pytest.raises(TrainingUIAPIError, match="Режим compact-фильтра"):
        _model_export._pipeline_yaml(
            "lakes_kanopus",
            768,
            4,
            postprocess_config={
                "postprocess.filter_compact_objects.enabled": True,
                "postprocess.filter_compact_objects.mode": "unknown",
            },
        )


def test_test_f1_accepts_prediction_calculated_by_geoalert(tmp_path: Path) -> None:
    image_path = tmp_path / "tile.tif"
    mask_path = tmp_path / "mask.tif"
    transform = from_origin(0, 8, 1, 1)
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    expected = np.zeros((8, 8), dtype=np.uint8)
    expected[2:6, 2:6] = 1
    _write_raster(image_path, image, transform)
    _write_raster(mask_path, expected[np.newaxis, ...], transform)
    prediction_path = tmp_path / "prediction.npy"
    instances_path = tmp_path / "instances.npy"
    np.save(prediction_path, expected)
    np.save(instances_path, expected.astype(np.int32))

    report = run_test_sample_f1(
        {
            "operation": "test_sample_f1",
            "run_root": str(tmp_path / "metrics"),
            "threshold": 0.5,
            "task": "binary",
            "postprocess_profile": "none",
            "tiles": [
                {
                    "index": 1,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "precomputed_prediction_path": str(prediction_path),
                    "precomputed_instances_path": str(instances_path),
                }
            ],
        }
    )

    assert report["status"] == "ok"
    assert report["true_positive"] == 16
    assert report["false_positive"] == 0
    assert report["false_negative"] == 0


def test_geoalert_geojson_is_rasterized_for_test_metrics(tmp_path: Path) -> None:
    image_path = tmp_path / "tile.tif"
    transform = from_origin(0, 10, 1, 1)
    _write_raster(image_path, np.zeros((3, 10, 10), dtype=np.uint8), transform)
    output_path = tmp_path / "output.geojson"
    output_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": mapping(box(2, 6, 5, 9)),
                        "properties": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    prediction, instances = _rasterize_scene_prediction(
        {"task": "binary", "object_types": []},
        {"index": 1, "image_path": str(image_path)},
        {"outputs": {"output": str(output_path)}},
    )

    assert int(prediction.sum()) == 9
    assert int(np.count_nonzero(instances)) == 9
    assert int(instances.max()) == 1


def test_runtime_model_archive_rejects_path_outside_model_root(tmp_path: Path) -> None:
    archive_path = tmp_path / "model.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("other/config.pbtxt", "name: other")

    with pytest.raises(RuntimeError, match="вне корня модели"):
        _extract_model_archive(archive_path, tmp_path / "target", "expected")


def _write_raster(path: Path, values: np.ndarray, transform) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[2],
        height=values.shape[1],
        count=values.shape[0],
        dtype=values.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dataset:
        dataset.write(values)
