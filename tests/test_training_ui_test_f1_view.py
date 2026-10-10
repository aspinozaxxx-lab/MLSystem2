from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

from mlsystem2.training_ui_api._test_f1_artifacts import (
    save_test_f1_manifest, save_test_f1_scene, test_f1_layers as _layers,
)
from mlsystem2.training_ui_api._test_f1_viewer import _saved_scenes, store_test_f1_view
from mlsystem2.metrics.api import compute_object_f1
from mlsystem2.metrics.contracts import ObjectF1Request


def test_multiclass_layers_keep_wrong_class_as_false_negative(tmp_path: Path):
    image_path = tmp_path / "test.tif"
    transform = from_origin(100, 200, 1, 1)
    with rasterio.open(image_path, "w", driver="GTiff", width=2, height=2, count=3, dtype="uint8", crs="EPSG:3857", transform=transform) as image:
        image.write(np.ones((3, 2, 2), np.uint8))
    truth = np.array([[1, 0], [0, 0]], np.uint8)
    prediction = np.array([[2, 1], [0, 0]], np.uint8)
    scene = save_test_f1_scene(tmp_path / "view", {"object_types": []},
        {"index": 1, "image_path": str(image_path), "target_class_id": 1}, truth, prediction, {})
    path = tmp_path / "view" / "tile-1.npz"
    stat = path.stat()
    layers = _layers(str(path), stat.st_mtime_ns, stat.st_size, tuple(scene["transform"]), scene["crs"], 1)
    for layer, expected in {"tp": 0, "fp": 1, "fn": 1, "reference": 1, "predicted": 1}.items():
        shapes = [(feature["geometry"], 1) for feature in layers["features"] if feature["properties"]["test_f1_layer"] == layer]
        actual = int(rasterize(shapes, out_shape=(2, 2), transform=transform).sum()) if shapes else 0
        assert actual == expected


def test_partial_view_publication_preserves_other_class_job(tmp_path: Path):
    source = tmp_path / "run" / "scratch" / "test_f1_view"
    source.mkdir(parents=True)
    job_id, old_id = uuid.uuid4(), uuid.uuid4()
    save_test_f1_manifest(source, str(uuid.uuid4()), [{"id": "tile-1", "target_class_slug": "second"}])
    np.savez_compressed(source / "tile-1.npz", reference=np.zeros((1, 1)), predicted=np.zeros((1, 1)))
    config = SimpleNamespace(stored_files_root=tmp_path / "stored")
    job = SimpleNamespace(id=job_id, tmp_path=tmp_path / "run")
    refs = store_test_f1_view(job, config, {"first": str(old_id), "second": str(old_id)}, selected_slugs=["second"])
    assert refs == {"first": str(old_id), "second": str(job_id)}
    assert json.loads((config.stored_files_root / "test-f1" / str(job_id) / "manifest.json").read_text())["version"] == 1
    assert (config.stored_files_root / "test-f1" / str(job_id) / "tile-1.npz").is_file()


@pytest.mark.parametrize("target", [None, 7])
def test_object_layers_use_saved_matches_instead_of_pixel_intersection(tmp_path: Path, target):
    image_path = tmp_path / "image.tif"
    transform = from_origin(100, 200, 1, 1)
    with rasterio.open(image_path, "w", driver="GTiff", width=12, height=8, count=3,
                       dtype="uint8", crs="EPSG:3857", transform=transform) as image:
        image.write(np.ones((3, 8, 12), np.uint8))
    truth = np.zeros((8, 12), np.int32)
    truth[1:4, 1:4] = 4
    truth[1:4, 7:10] = 95
    prediction = np.zeros_like(truth)
    prediction[1:4, 2:5] = 80
    prediction[1:4, 10:12] = 1000
    score = compute_object_f1(ObjectF1Request(y_true_instances=truth, y_pred_instances=prediction))
    assert score.matched_pairs == [(4, 80)]
    scene = save_test_f1_scene(tmp_path / "view", {},
        {"index": 1, "image_path": str(image_path), "target_class_id": target},
        (truth > 0).astype(np.uint8), (prediction > 0).astype(np.uint8) * (target or 1), {},
        object_snapshots={target: (truth, prediction, score.matched_pairs)})
    assert scene["object_layers_available"]
    path = tmp_path / "view" / "tile-1.npz"
    stat = path.stat()
    for metric, expected in [("pixel", {"tp": 6, "fp": 9, "fn": 12}),
                              ("objects", {"tp": 12, "fp": 6, "fn": 9})]:
        layers = _layers(str(path), stat.st_mtime_ns, stat.st_size, tuple(scene["transform"]), scene["crs"], target, metric)
        for layer, count in expected.items():
            geometry = [(feature["geometry"], 1) for feature in layers["features"] if feature["properties"]["test_f1_layer"] == layer]
            assert int(rasterize(geometry, out_shape=truth.shape, transform=transform).sum()) == count


def test_legacy_object_layers_reuse_saved_inference(tmp_path: Path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, reference=np.ones((2, 2), np.uint8), predicted=np.ones((2, 2), np.uint8))
    stat = path.stat()
    layers = _layers(str(path), stat.st_mtime_ns, stat.st_size, tuple(from_origin(0, 2, 1, 1))[:6], "EPSG:3857", None, "objects")
    assert {item["properties"]["test_f1_layer"] for item in layers["features"]} == {"tp", "reference", "predicted"}


def test_view_requires_snapshots_for_every_class_after_legacy_partial_update(tmp_path: Path):
    job_id, result_id = uuid.uuid4(), uuid.uuid4()
    root = tmp_path / "test-f1" / str(job_id)
    root.mkdir(parents=True)
    scene = {"id": "tile-1", "target_class_slug": "second", "source_tile_index": 1}
    save_test_f1_manifest(root, str(result_id), [scene])
    np.savez_compressed(root / "tile-1.npz", reference=np.zeros((1, 1)), predicted=np.zeros((1, 1)))
    metric = SimpleNamespace(metrics={"viewer_jobs": {"second": str(job_id)}, "aggregation": "macro",
        "test_samples": [{"class_slug": "first", "tile_indices": [1]}, {"class_slug": "second", "tile_indices": [1]}]})
    session = SimpleNamespace(get=lambda *_: metric)
    scenes, warnings = _saved_scenes(session, SimpleNamespace(stored_files_root=tmp_path), result_id)
    assert len(scenes) == 1
    assert warnings


def test_legacy_view_preparation_and_metric_switch_never_enqueue_inference(tmp_path, monkeypatch):
    from mlsystem2.training_ui_api import _test_f1_viewer as viewer
    from mlsystem2.training_ui_api.contracts import TrainingResultTestF1Info
    from mlsystem2.training_ui_api._models import TrainingResultRow, TestSampleRow
    from shapely.geometry import box, mapping

    result_id, sample_id = uuid.uuid4(), uuid.uuid4()
    config = SimpleNamespace(stored_files_root=tmp_path)
    sample_root = tmp_path / "test-samples" / str(sample_id)
    sample_root.mkdir(parents=True)
    image_path = sample_root / "tile_001.tif"
    transform = from_origin(0, 4, 1, 1)
    with rasterio.open(image_path, "w", driver="GTiff", width=4, height=4, count=3, dtype="uint8", crs="EPSG:3857", transform=transform) as image:
        image.write(np.ones((3, 4, 4), np.uint8))
    # Два соприкасающихся эталона должны оставаться двумя объектами.
    image_path.with_suffix(".geojson").write_text(json.dumps({"type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [{"type": "Feature", "properties": {}, "geometry": mapping(box(0, 2, 2, 4))},
                     {"type": "Feature", "properties": {}, "geometry": mapping(box(2, 2, 4, 4))}]}))
    truth = np.zeros((4, 4), np.uint8)
    truth[:2] = 1
    root = tmp_path / "view"
    scene = save_test_f1_scene(root, {}, {"index": 1, "image_path": str(image_path), "test_sample_id": str(sample_id)},
                               truth, truth, {"true_positive": 8})
    scene["public_id"] = "saved_tile-1"
    result = SimpleNamespace(id=result_id, status="ok", model_name="Проверка", class_display_name="Датасет")
    sample = SimpleNamespace(content_revision=1)
    session = SimpleNamespace(scalar=lambda *_: result,
                              get=lambda model, key: result if model is TrainingResultRow else sample if model is TestSampleRow else None)
    monkeypatch.setattr(viewer, "training_result_test_f1_info", lambda *_: TrainingResultTestF1Info(status="current", pixel_f1=1, object_f1=0))
    monkeypatch.setattr(viewer, "_saved_scenes", lambda *_: ([(root, scene)], []))
    monkeypatch.setattr(viewer, "queue_training_result_test_f1", lambda *_args, **_kwargs: pytest.fail("Повторный инференс запрещён"))
    view = viewer.prepare_test_f1_view(session, config, result_id)
    assert view.status == "ready" and view.scenes[0].object_layers_available
    assert view.scenes[0].objects.true_positive == 1 and view.scenes[0].objects.false_negative == 1
    assert view.metric.object_f1 == pytest.approx(2 / 3)
    assert not view.warnings
    for mode in ("pixel", "objects", "pixel", "objects"):
        assert viewer.test_f1_scene_layers(session, config, result_id, scene["public_id"], None, mode)["features"]
    sample.content_revision = 2
    restored = viewer.test_f1_view(session, config, result_id)
    assert restored.scenes[0].objects.f1 == 1
    assert any("связным областям" in message for message in restored.warnings)
