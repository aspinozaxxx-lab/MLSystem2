"""Быстрый показ сохранённых слоёв без изменения геометрий и авторизации."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from uuid import uuid4
import gzip
import json

import numpy as np
import pytest
import rasterio
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from rasterio.features import shapes
from rasterio.transform import from_origin
from rasterio.warp import transform_geom

from mlsystem2.training_ui_api import _test_f1_artifacts as artifacts
from mlsystem2.training_ui_api._routes import results as routes
from mlsystem2.training_ui_api._routes.common import RouteContext


@pytest.mark.parametrize("crs", ["EPSG:3857", "EPSG:32637"])
def test_one_gdal_environment_preserves_exact_layer_geometry(tmp_path, monkeypatch, crs):
    reference = np.zeros((12, 17), np.uint8)
    reference[1:10, 2:14] = 1
    reference[4:6, 5:8] = 0
    predicted = np.roll(reference, 2, axis=1)
    path = tmp_path / "сохранённые_маски.npz"
    np.savez_compressed(path, reference=reference, predicted=predicted)
    transform = from_origin(400000, 6500000, 0.1, 0.1)
    stat = path.stat()
    expected = []
    masks = {"tp": (reference > 0) & (predicted > 0), "fp": (reference == 0) & (predicted > 0),
             "fn": (reference > 0) & (predicted == 0), "reference": reference > 0, "predicted": predicted > 0}
    for layer, mask in masks.items():
        for geometry, _ in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
            expected.append({"type": "Feature", "properties": {"test_f1_layer": layer},
                             "geometry": transform_geom(crs, "EPSG:3857", geometry)})
    calls = []
    original = artifacts.transform_geom
    outside = rasterio.env.hasenv()

    def transform_in_shared_environment(*args, **kwargs):
        assert rasterio.env.hasenv(), "Для каждого полигона должна действовать общая среда GDAL"
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(artifacts, "transform_geom", transform_in_shared_environment)
    result = artifacts.test_f1_layers(str(path), stat.st_mtime_ns, stat.st_size, tuple(transform)[:6], crs, None)
    assert calls and result == {"type": "FeatureCollection", "features": expected}
    assert rasterio.env.hasenv() == outside


@pytest.fixture
def layer_cache(monkeypatch):
    monkeypatch.setattr(artifacts, "_LAYER_CONTENT_CACHE", OrderedDict())
    monkeypatch.setattr(artifacts, "_LAYER_CONTENT_BYTES", 0)
    calls = []

    def layers(*args):
        calls.append(args)
        return {"type": "FeatureCollection", "features": [], "пояснение": "Точные сохранённые слои"}

    monkeypatch.setattr(artifacts, "test_f1_layers", layers)
    return calls


LAYER_KEY = ("маски.npz", 1, 10, (1, 0, 0, 0, -1, 10), "EPSG:3857", None, "pixel", None, ())


@pytest.mark.parametrize("index,value", [
    (0, "другие_маски.npz"), (1, 2), (2, 11), (3, (2, 0, 0, 0, -2, 10)), (4, "EPSG:32637"),
    (5, 1), (6, "objects"), (7, ("эталон.geojson", 1, 10, "снимок.tif")), (8, ((1, "участок"),)),
])
def test_compressed_cache_obeys_every_revision_and_mode(layer_cache, index, value):
    first = artifacts.test_f1_layer_content(*LAYER_KEY)
    assert artifacts.test_f1_layer_content(*LAYER_KEY) is first
    assert len(layer_cache) == 1
    assert json.loads(gzip.decompress(first))["пояснение"] == "Точные сохранённые слои"
    changed = list(LAYER_KEY)
    changed[index] = value
    artifacts.test_f1_layer_content(*changed)
    assert len(layer_cache) == 2
    assert artifacts._LAYER_CONTENT_BYTES == sum(map(len, artifacts._LAYER_CONTENT_CACHE.values()))


def test_compressed_cache_is_bounded_in_bytes_and_keeps_recent_entries(layer_cache, monkeypatch):
    content = artifacts.test_f1_layer_content(*LAYER_KEY)
    monkeypatch.setattr(artifacts, "_LAYER_CONTENT_LIMIT", len(content) * 2)
    second, third = list(LAYER_KEY), list(LAYER_KEY)
    second[1], third[1] = 2, 3
    artifacts.test_f1_layer_content(*second)
    artifacts.test_f1_layer_content(*LAYER_KEY)
    artifacts.test_f1_layer_content(*third)
    assert tuple(second) not in artifacts._LAYER_CONTENT_CACHE
    assert len(artifacts._LAYER_CONTENT_CACHE) == 2
    assert artifacts._LAYER_CONTENT_BYTES <= artifacts._LAYER_CONTENT_LIMIT
    artifacts.test_f1_layer_content(*LAYER_KEY)
    assert len(layer_cache) == 3


def test_oversized_response_is_not_kept_in_memory(layer_cache, monkeypatch):
    monkeypatch.setattr(artifacts, "_LAYER_CONTENT_LIMIT", 1)
    for _ in range(2):
        assert artifacts.test_f1_layer_content(*LAYER_KEY)
    assert len(layer_cache) == 2
    assert not artifacts._LAYER_CONTENT_CACHE and artifacts._LAYER_CONTENT_BYTES == 0


def test_many_empty_scenes_cannot_accumulate_unbounded_cache_keys(layer_cache):
    for revision in range(130):
        key = list(LAYER_KEY)
        key[1] = revision
        artifacts.test_f1_layer_content(*key)
    assert len(artifacts._LAYER_CONTENT_CACHE) == 128
    assert artifacts._LAYER_CONTENT_BYTES == sum(map(len, artifacts._LAYER_CONTENT_CACHE.values()))


def test_concurrent_publication_does_not_double_count_cache_bytes(layer_cache, monkeypatch):
    barrier = Barrier(2)

    def simultaneous_layers(*_):
        barrier.wait(timeout=10)
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr(artifacts, "test_f1_layers", simultaneous_layers)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _: artifacts.test_f1_layer_content(*LAYER_KEY), range(2)))
    assert first is second
    assert len(artifacts._LAYER_CONTENT_CACHE) == 1
    assert artifacts._LAYER_CONTENT_BYTES == len(first)


@pytest.mark.parametrize("header,expected", [("", False), ("identity", False), ("br, gzip", True),
    ("GZip; q=0.5", True), ("gzip;q=0", False), ("gzip;q=нет", False), ("gzip;q=2", False),
    ("*;q=0.5", True), ("gzip;q=0,*;q=1", False)])
def test_gzip_negotiation_honours_disabled_encodings(header, expected):
    assert routes._accepts_gzip(header) is expected


def test_layer_http_cache_stays_private_and_authenticates_before_304(monkeypatch):
    value = {"type": "FeatureCollection", "features": [], "подпись": "Эталон и прогноз"}
    content = gzip.compress(json.dumps(value, ensure_ascii=False).encode(), mtime=0)
    result_id = uuid4()
    calls = []

    def authenticated(request: Request):
        if request.headers.get("Authorization") != "Bearer test-user":
            raise HTTPException(401, "Войдите в систему")
        return "Проверяющий"

    def layers(_, __, actual_result, scene, class_id, metric):
        calls.append((actual_result, scene, class_id, metric))
        if actual_result != result_id or scene != "сохранённый":
            raise HTTPException(404, "Снимок не принадлежит этой сети")
        return content

    monkeypatch.setattr(routes, "test_f1_scene_layers", layers)
    app = FastAPI()
    routes.register_result_routes(app, RouteContext(SimpleNamespace(), lambda: None, authenticated, authenticated))
    path = f"/api/v1/results/training/{result_id}/test-f1/layers/сохранённый"
    with TestClient(app) as client:
        client.headers["Authorization"] = "Bearer test-user"
        compressed = client.get(path, headers={"Accept-Encoding": "gzip"})
        assert compressed.status_code == 200 and compressed.json() == value
        assert compressed.headers["Content-Encoding"] == "gzip"
        assert int(compressed.headers["Content-Length"]) == len(content)
        assert compressed.headers["Cache-Control"] == "private, no-cache"
        assert compressed.headers["Vary"] == "Accept-Encoding"
        etag = compressed.headers["ETag"]
        cached = client.get(path, headers={"Accept-Encoding": "gzip", "If-None-Match": etag})
        assert cached.status_code == 304 and not cached.content
        assert cached.headers["ETag"] == etag
        assert len(calls) == 2
        identity = client.get(path, headers={"Accept-Encoding": "gzip;q=0", "If-None-Match": etag})
        assert identity.status_code == 200 and identity.json() == value
        assert "Content-Encoding" not in identity.headers
        assert identity.headers["ETag"] != etag
        assert client.get(path, headers={"Accept-Encoding": "identity", "If-None-Match": identity.headers["ETag"]}).status_code == 304
        assert client.get(path, headers={"Accept-Encoding": "gzip", "If-None-Match": "W/" + etag}).status_code == 304
        assert client.get(path.replace("сохранённый", "чужой"), headers={"If-None-Match": etag}).status_code == 404
        assert client.get(path.replace(str(result_id), str(uuid4())), headers={"If-None-Match": etag}).status_code == 404
        before = len(calls)
        del client.headers["Authorization"]
        assert client.get(path, headers={"If-None-Match": etag}).status_code == 401
        assert len(calls) == before
