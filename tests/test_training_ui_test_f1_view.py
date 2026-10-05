from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

from mlsystem2.training_ui_api._test_f1_artifacts import (
    save_test_f1_manifest, save_test_f1_scene, test_f1_layers as _layers,
)
from mlsystem2.training_ui_api._test_f1_viewer import _saved_scenes, store_test_f1_view


def test_multiclass_layers_keep_wrong_class_as_false_negative(tmp_path: Path):
    image_path = tmp_path / "test.tif"
    transform = from_origin(100, 200, 1, 1)
    with rasterio.open(image_path, "w", driver="GTiff", width=2, height=2, count=3, dtype="uint8", crs="EPSG:3857", transform=transform) as image:
        image.write(np.ones((3, 2, 2), np.uint8))
    truth = np.array([[1, 0], [0, 0]], np.uint8)
    prediction = np.array([[2, 1], [0, 0]], np.uint8)
    scene = save_test_f1_scene(tmp_path / "view", {"object_types": []},
        {"index": 1, "image_path": str(image_path), "target_class_id": 1}, truth, prediction, {})
    path = tmp_path / "view" / "tile-1.npz"; stat = path.stat()
    layers = _layers(str(path), stat.st_mtime_ns, stat.st_size, tuple(scene["transform"]), scene["crs"], 1)
    for layer, expected in {"tp": 0, "fp": 1, "fn": 1, "reference": 1, "predicted": 1}.items():
        shapes = [(feature["geometry"], 1) for feature in layers["features"] if feature["properties"]["test_f1_layer"] == layer]
        actual = int(rasterize(shapes, out_shape=(2, 2), transform=transform).sum()) if shapes else 0
        assert actual == expected


def test_partial_view_publication_preserves_other_class_job(tmp_path: Path):
    source = tmp_path / "run" / "scratch" / "test_f1_view"; source.mkdir(parents=True)
    job_id, old_id = uuid.uuid4(), uuid.uuid4()
    save_test_f1_manifest(source, str(uuid.uuid4()), [{"id": "tile-1", "target_class_slug": "second"}])
    np.savez_compressed(source / "tile-1.npz", reference=np.zeros((1, 1)), predicted=np.zeros((1, 1)))
    config = SimpleNamespace(stored_files_root=tmp_path / "stored")
    job = SimpleNamespace(id=job_id, tmp_path=tmp_path / "run")
    refs = store_test_f1_view(job, config, {"first": str(old_id), "second": str(old_id)}, selected_slugs=["second"])
    assert refs == {"first": str(old_id), "second": str(job_id)}
    assert json.loads((config.stored_files_root / "test-f1" / str(job_id) / "manifest.json").read_text())["version"] == 1
    assert (config.stored_files_root / "test-f1" / str(job_id) / "tile-1.npz").is_file()


def test_view_requires_snapshots_for_every_class_after_legacy_partial_update(tmp_path: Path):
    job_id, result_id = uuid.uuid4(), uuid.uuid4()
    root = tmp_path / "test-f1" / str(job_id); root.mkdir(parents=True)
    scene = {"id": "tile-1", "target_class_slug": "second", "source_tile_index": 1}
    save_test_f1_manifest(root, str(result_id), [scene])
    np.savez_compressed(root / "tile-1.npz", reference=np.zeros((1, 1)), predicted=np.zeros((1, 1)))
    metric = SimpleNamespace(metrics={"viewer_jobs": {"second": str(job_id)}, "aggregation": "macro",
        "test_samples": [{"class_slug": "first", "tile_indices": [1]}, {"class_slug": "second", "tile_indices": [1]}]})
    session = SimpleNamespace(get=lambda *_: metric)
    scenes, warnings = _saved_scenes(session, SimpleNamespace(stored_files_root=tmp_path), result_id)
    assert len(scenes) == 1
    assert warnings
