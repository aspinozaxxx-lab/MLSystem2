from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping
import yaml

from mlsystem2.training_ui_api import _model_export, _worker
from mlsystem2.training_ui_api._geoalert_runner import (
    _extract_model_archive,
    _pipeline_bricks,
    _rasterize_scene_prediction,
)
from mlsystem2.training_ui_api._inference_backend import (
    GEOALERT_INFERENCE_BACKEND,
    PYTORCH_INFERENCE_BACKEND,
    inference_backend_for_imagery,
)
from mlsystem2.training_ui_api._pseudo_runner import run_test_sample_f1


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
                "postprocess.min_area_m2": 20.0,
                "postprocess.min_hole_area_m2": 10.0,
                "postprocess.simplify_m": 0.5,
                "postprocess.filter_compact_objects.enabled": True,
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
    assert _pipeline_bricks(pipeline) == [
        "SplitRaster",
        "Segmentation",
        "TritonAdapter",
        "MaskMorphology",
        "VectorizeMasks",
        "UnifiedVectorProcessing",
        "Simplify",
        "FilterSmallObjects",
        "RemoveSmallHoles",
        "FilterCompactObjects",
    ]
    assert "kind: KIND_GPU" in _model_export._triton_config(
        "ortho_model",
        3,
        instance_kind="KIND_GPU",
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
