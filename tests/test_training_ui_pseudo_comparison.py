from __future__ import annotations

import json
import asyncio
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.enums import ColorInterp
from rasterio.features import rasterize, shapes
from rasterio.transform import from_origin
from rasterio.warp import transform_geom
from rasterio.warp import transform_bounds
from sqlalchemy import select

from mlsystem2.training_ui_api.api import create_app
from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import create_session_factory
from mlsystem2.training_ui_api._models import JobRow, PseudoMarkupResultRow, StoredFileRow
from mlsystem2.training_ui_api._pseudo_viewer import _scene_id
from mlsystem2.training_ui_api._raster_http import raster_revision


@pytest.fixture
def comparison_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "stored"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_USER", "demo")
    credential = uuid.uuid4().hex
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_PASSWORD", credential)
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SESSION_SECRET", uuid.uuid4().hex)
    config = get_config()
    config.images_root.mkdir()
    config.stored_files_root.mkdir()
    client = TestClient(create_app())
    assert client.post("/api/v1/auth/login", json={"username": "demo", "password": credential}).status_code == 200
    yield SimpleNamespace(client=client, config=config, factory=create_session_factory(config),
                          transform=from_origin(500000, 6000000, 10, 10), crs="EPSG:32637")
    client.close()


def _image(env, name, valid):
    path = env.config.images_root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((4, *valid.shape), 90, dtype=np.uint8)
    data[3] = np.where(valid, 255, 0)
    with rasterio.open(path, "w", driver="GTiff", width=valid.shape[1], height=valid.shape[0],
                       count=4, dtype="uint8", crs=env.crs, transform=env.transform) as target:
        target.write(data)
        target.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
    return path


def _result(env, images, mask, label):
    geometries = [transform_geom(env.crs, "EPSG:4326", geometry)
                  for geometry, value in shapes(mask.astype("uint8"), mask=mask, transform=env.transform) if value]
    # Повтор одного полигона не должен удваивать площадь.
    features = [{"type": "Feature", "properties": {}, "geometry": geometry} for geometry in geometries * 2]
    path = env.config.stored_files_root / f"{uuid.uuid4()}.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    with env.factory() as session:
        file = StoredFileRow(kind="pseudo_markup_geojson", original_name=path.name, path=str(path),
                             size_bytes=path.stat().st_size, object_count=len(features))
        job = JobRow(type="inference", source="manual", status="completed", dataset_key=label,
                     dataset_name=label, model_name=label, architecture="smp_unet_resnet34", queue_position=1,
                     config={"pseudo_processed_images": [str(image) for image in images]})
        session.add_all([file, job]); session.flush()
        result = PseudoMarkupResultRow(class_key=label, dataset_key=label, source_dataset_name=label,
                                      job_id=job.id, geojson_file_id=file.id, image_count=len(images), status="ok")
        session.add(result); session.commit()
        return str(result.id), path


def _pair(env):
    valid = np.ones((8, 8), dtype=bool); valid[3, 3] = False
    first = np.zeros_like(valid); first[1:5, 1:5] = True
    second = np.zeros_like(valid); second[3:7, 3:7] = True
    image = _image(env, "общий.tif", valid)
    other = _image(env, "второй.tif", valid)
    private = _image(env, "отдельный.tif", valid)
    first_id, first_file = _result(env, [image, other, private], first, "Реки / Север")
    second_id, _ = _result(env, [image, other], second, "Здания / Проверка")
    return SimpleNamespace(first=first, second=second, valid=valid, image=image, other=other, private=private,
                           ids=[first_id, second_id], first_file=first_file)


def test_pair_counts_native_pixels_excludes_alpha_and_sums_only_shared_images(comparison_environment):
    env = comparison_environment; pair = _pair(env)
    response = env.client.post("/api/v1/results/pseudo-markup/compare/counts", json={"result_ids": pair.ids})
    assert response.status_code == 200, response.text
    values = response.json()
    expected = {"intersection": 3, "only_first": 12, "only_second": 12}
    assert values["scenes"] == {_scene_id(pair.image): expected, _scene_id(pair.other): expected}
    assert values["total"] == {key: value * 2 for key, value in expected.items()}
    assert values["result_ids"] == pair.ids
    with env.factory() as session:
        assert len(session.scalars(select(JobRow)).all()) == 2


def test_layers_match_pixel_counts_and_keep_missing_scene_out_of_pair(comparison_environment):
    env = comparison_environment; pair = _pair(env)
    response = env.client.post(f"/api/v1/results/pseudo-markup/compare/{_scene_id(pair.image)}/layers", json={"result_ids": pair.ids})
    assert response.status_code == 200, response.text
    data = response.json()
    expected = {"intersection": pair.first & pair.second & pair.valid,
                "difference": pair.first & ~pair.second & pair.valid}
    for kind, mask in expected.items():
        geometries = [(transform_geom("EPSG:3857", env.crs, feature["geometry"]), 1) for feature in data["geojson"]["features"]
                      if feature["properties"]["comparison_kind"] == kind
                      and (kind == "intersection" or feature["properties"]["comparison_result_id"] == pair.ids[0])]
        actual = rasterize(geometries, out_shape=pair.valid.shape, transform=env.transform).astype(bool)
        np.testing.assert_array_equal(actual, mask)
    assert data["counts"] == {"intersection": 3, "only_first": 12, "only_second": 12}
    private = env.client.post(f"/api/v1/results/pseudo-markup/compare/{_scene_id(pair.private)}/layers", json={"result_ids": pair.ids}).json()
    assert private["counts"] is None
    assert private["available_result_ids"] == pair.ids[:1]
    assert {feature["properties"]["comparison_kind"] for feature in private["geojson"]["features"]} == {"layer"}


def test_empty_prediction_is_a_real_zero_and_direction_follows_active_order(comparison_environment):
    env = comparison_environment; pair = _pair(env)
    empty, _ = _result(env, [pair.image], np.zeros_like(pair.first), "Пустая сеть")
    post = lambda ids: env.client.post("/api/v1/results/pseudo-markup/compare/counts", json={"result_ids": ids}).json()["total"]
    assert post([pair.ids[0], empty]) == {"intersection": 0, "only_first": 15, "only_second": 0}
    assert post([empty, pair.ids[0]]) == {"intersection": 0, "only_first": 0, "only_second": 15}


def test_three_layers_show_pairwise_overlap_and_unique_regions(comparison_environment):
    env = comparison_environment; pair = _pair(env)
    third_mask = np.zeros_like(pair.first); third_mask[0:4, 4:8] = True
    third, _ = _result(env, [pair.image], third_mask, "Третий класс")
    data = env.client.post(f"/api/v1/results/pseudo-markup/compare/{_scene_id(pair.image)}/layers", json={"result_ids": [*pair.ids, third]}).json()
    assert data["counts"] is None
    overlap = np.sum([pair.first, pair.second, third_mask], axis=0) >= 2
    geometries = [(transform_geom("EPSG:3857", env.crs, feature["geometry"]), 1) for feature in data["geojson"]["features"]
                  if feature["properties"]["comparison_kind"] == "intersection"]
    actual = rasterize(geometries, out_shape=pair.valid.shape, transform=env.transform).astype(bool)
    np.testing.assert_array_equal(actual, overlap & pair.valid)


def test_identical_names_in_different_paths_are_not_the_same_image(comparison_environment):
    env = comparison_environment; pair = _pair(env)
    copy = _image(env, "копия/общий.tif", pair.valid)
    copied, _ = _result(env, [copy], pair.first, "Другой датасет")
    response = env.client.post("/api/v1/results/pseudo-markup/compare/counts", json={"result_ids": [pair.ids[0], copied]})
    assert response.status_code == 200
    assert response.json()["scenes"] == {}
    assert response.json()["warnings"]


def test_comparison_requires_auth_ready_results_membership_and_current_revision(comparison_environment):
    env = comparison_environment; pair = _pair(env)
    body = {"result_ids": pair.ids}
    url = f"/api/v1/results/pseudo-markup/compare/{_scene_id(pair.image)}/layers"
    assert env.client.post(url, json={**body, "scene_revisions": {_scene_id(pair.image): "stale"}}).status_code == 412
    assert env.client.post("/api/v1/results/pseudo-markup/compare/foreign/layers", json=body).status_code == 400
    assert env.client.post(url, json={"result_ids": [pair.ids[0], pair.ids[0]]}).status_code == 422
    assert env.client.post("/api/v1/results/pseudo-markup/compare/counts", json={"result_ids": pair.ids[:1]}).status_code == 400
    with env.factory() as session:
        session.get(PseudoMarkupResultRow, uuid.UUID(pair.ids[1])).status = "error"
        session.commit()
    assert env.client.post(url, json=body).status_code == 400
    env.client.cookies.clear()
    assert env.client.post(url, json=body).status_code == 401
    assert env.client.post("/api/v1/results/pseudo-markup/compare/counts", json=body).status_code == 401


def test_file_revision_invalidates_comparison_cache(comparison_environment):
    env = comparison_environment; pair = _pair(env)
    body = {"result_ids": pair.ids, "scene_revisions": {_scene_id(pair.image): raster_revision(pair.image)}}
    url = "/api/v1/results/pseudo-markup/compare/counts"
    assert env.client.post(url, json=body).json()["total"]["intersection"] == 6
    pair.first_file.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    assert env.client.post(url, json=body).json()["total"] == {"intersection": 0, "only_first": 0, "only_second": 30}


def test_chunked_comparison_across_window_edges(comparison_environment):
    env = comparison_environment
    valid = np.ones((5, 1030), dtype=bool)
    first = valid.copy(); second = np.zeros_like(valid); second[:, 500:1020] = True
    image = _image(env, "длинный.tif", valid)
    ids = [_result(env, [image], mask, str(index))[0] for index, mask in enumerate((first, second))]
    response = env.client.post("/api/v1/results/pseudo-markup/compare/counts", json={"result_ids": ids})
    assert response.json()["total"] == {"intersection": 2600, "only_first": 2550, "only_second": 0}


def test_projected_image_edge_keeps_pixels_outside_straight_geographic_outline(comparison_environment):
    env = comparison_environment
    valid = np.ones((20, 3200), dtype=bool)
    mask = np.zeros_like(valid); mask[0, 1599] = True
    image = _image(env, "край_проекционного_снимка.tif", valid)
    ids = [_result(env, [image], mask, str(index))[0] for index in range(2)]
    counts = env.client.post("/api/v1/results/pseudo-markup/compare/counts", json={"result_ids": ids})
    expected = {"intersection": 1, "only_first": 0, "only_second": 0}
    assert counts.json()["total"] == expected
    layers = env.client.post(f"/api/v1/results/pseudo-markup/compare/{_scene_id(image)}/layers", json={"result_ids": ids}).json()
    assert layers["counts"] == expected
    assert any(feature["properties"]["comparison_kind"] == "intersection" for feature in layers["geojson"]["features"])


def test_incremental_counts_only_read_requested_shared_scene(comparison_environment, monkeypatch):
    from mlsystem2.training_ui_api import _pseudo_comparison as comparison
    env = comparison_environment; pair = _pair(env)
    scene_id = _scene_id(pair.image)
    original = comparison._count_scene
    read = []

    def counted(key, files, cancel=None):
        read.append(key[0])
        return original(key, files, cancel)

    monkeypatch.setattr(comparison, "_count_scene", counted)
    response = env.client.post("/api/v1/results/pseudo-markup/compare/counts", json={"result_ids": pair.ids, "scene_id": scene_id})
    assert response.status_code == 200
    assert read == [str(pair.image)]
    assert response.json()["scenes"] == {scene_id: {"intersection": 3, "only_first": 12, "only_second": 12}}
    assert response.json()["total"] == response.json()["scenes"][scene_id]
    assert env.client.post("/api/v1/results/pseudo-markup/compare/counts", json={"result_ids": pair.ids, "scene_id": _scene_id(pair.private)}).status_code == 400
    assert env.client.post("/api/v1/results/pseudo-markup/compare/counts", json={"result_ids": pair.ids, "scene_id": scene_id, "scene_revisions": {scene_id: "stale"}}).status_code == 412


def _viewport(image, width, height):
    with rasterio.open(image) as source:
        return {"bounds": transform_bounds(source.crs, "EPSG:3857", *source.bounds, densify_pts=21),
                "width": width, "height": height}


def test_viewport_preview_does_not_wait_for_native_counts_and_refines_to_exact_pixels(comparison_environment, monkeypatch):
    from mlsystem2.training_ui_api import _pseudo_comparison as comparison
    env = comparison_environment; pair = _pair(env)
    def no_full_scene(*args, **kwargs):
        raise AssertionError("Показ карты не должен считать или векторизовать весь снимок.")
    monkeypatch.setattr(comparison, "_count_scene", no_full_scene)
    monkeypatch.setattr(comparison, "_layer_scene", no_full_scene)
    scene_id = _scene_id(pair.image)
    body = {"result_ids": pair.ids, "viewport": _viewport(pair.image, 8, 8)}
    url = f"/api/v1/results/pseudo-markup/compare/{scene_id}/layers"
    response = env.client.post(url, json=body)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["counts"] is None
    for kind, result_id, expected in [("intersection", None, pair.first & pair.second & pair.valid),
                                      ("difference", pair.ids[0], pair.first & ~pair.second & pair.valid)]:
        geometries = [(transform_geom("EPSG:3857", env.crs, feature["geometry"]), 1) for feature in data["geojson"]["features"]
                      if feature["properties"]["comparison_kind"] == kind and feature["properties"]["comparison_result_id"] == result_id]
        np.testing.assert_array_equal(rasterize(geometries, out_shape=pair.valid.shape, transform=env.transform).astype(bool), expected)
    assert env.client.post(url, json={**body, "scene_revisions": {scene_id: "stale"}}).status_code == 412
    env.client.cookies.clear()
    assert env.client.post(url, json=body).status_code == 401


def test_overview_masks_are_bounded_and_empty_viewport_is_valid(comparison_environment, monkeypatch):
    from mlsystem2.training_ui_api import _pseudo_comparison as comparison
    env = comparison_environment
    valid = np.ones((1024, 1536), dtype=bool)
    mask = valid.copy(); mask[500:] = False
    image = _image(env, "обзор.tif", valid)
    ids = [_result(env, [image], mask, str(index))[0] for index in range(2)]
    original = comparison._masks
    sizes = []
    def sampled(source, trees, window, step=1):
        value = original(source, trees, window, step)
        if value is not None:
            sizes.append((value[1][0].shape, step))
        return value
    monkeypatch.setattr(comparison, "_masks", sampled)
    viewport = _viewport(image, 64, 32)
    url = f"/api/v1/results/pseudo-markup/compare/{_scene_id(image)}/layers"
    assert env.client.post(url, json={"result_ids": ids, "viewport": viewport}).status_code == 200
    assert sizes and sizes[0][1] > 1
    assert sizes[0][0][0] <= 33 and sizes[0][0][1] <= 65
    empty = {**viewport, "bounds": [value + 1000000 for value in viewport["bounds"]]}
    response = env.client.post(url, json={"result_ids": ids, "viewport": empty})
    assert response.status_code == 200 and response.json()["geojson"]["features"] == []
    assert len(sizes) == 1
    for invalid in [{**viewport, "width": 2049}, {**viewport, "bounds": [4, 1, 3, 5]}]:
        assert env.client.post(url, json={"result_ids": ids, "viewport": invalid}).status_code == 422


def test_cancelled_window_count_is_not_cached_as_partial_success(comparison_environment, monkeypatch):
    from threading import Event
    from mlsystem2.training_ui_api import _pseudo_comparison as comparison
    env = comparison_environment
    mask = np.ones((5, 1030), dtype=bool)
    image = _image(env, "отмена.tif", mask)
    ids = [_result(env, [image], mask, str(index))[0] for index in range(2)]
    cancel = Event(); original = comparison._masks; calls = []
    def interrupted(*args, **kwargs):
        calls.append(1)
        value = original(*args, **kwargs)
        cancel.set()
        return value
    monkeypatch.setattr(comparison, "_masks", interrupted)
    stat = image.stat()
    from mlsystem2.training_ui_api.contracts import PseudoMarkupComparisonRequest
    with env.factory() as session:
        entries, _ = comparison._inputs(session, env.config, PseudoMarkupComparisonRequest(result_ids=ids))
    key = (str(image), stat.st_mtime_ns, stat.st_size)
    with pytest.raises(comparison._ComparisonCancelled):
        comparison._count_scene(key, tuple(item[1] for item in entries), cancel)
    assert calls == [1]
    monkeypatch.setattr(comparison, "_masks", original)
    assert comparison._count_scene(key, tuple(item[1] for item in entries)) == {"intersection": 5150, "only_first": 0, "only_second": 0}


def test_disconnect_stops_worker_inside_request_timing_middleware(comparison_environment, monkeypatch):
    from threading import Event
    from mlsystem2.training_ui_api._pseudo_comparison import _ComparisonCancelled
    from mlsystem2.training_ui_api._routes import results as routes
    started, cancelled = Event(), Event()
    def calculation(*args, cancel):
        started.set()
        if cancel.wait(timeout=1):
            cancelled.set()
            raise _ComparisonCancelled()
        raise AssertionError("Расчёт продолжился после закрытия соединения.")
    monkeypatch.setattr(routes, "pseudo_comparison_counts", calculation)
    env = comparison_environment
    body = json.dumps({"result_ids": [str(uuid.uuid4()), str(uuid.uuid4())]}).encode()
    cookie = "; ".join(f"{key}={value}" for key, value in env.client.cookies.items()).encode()
    async def interrupted_request():
        body_sent = False
        async def receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            while not started.is_set():
                await asyncio.sleep(0.001)
            return {"type": "http.disconnect"}
        async def send(message):
            pass
        path = "/api/v1/results/pseudo-markup/compare/counts"
        await env.client.app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                              "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(),
                              "root_path": "", "query_string": b"", "server": ("testserver", 80),
                              "client": ("127.0.0.1", 10000),
                              "headers": [(b"host", b"testserver"), (b"content-type", b"application/json"),
                                          (b"cookie", cookie)]}, receive, send)
    asyncio.run(interrupted_request())
    assert cancelled.is_set()
