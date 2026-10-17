"""Проверки ручных тестовых разметок и запрета фонового пересчёта F1."""

import hashlib
import json
import uuid
from threading import Event, Thread

import pytest
import numpy as np
import rasterio
from fastapi.testclient import TestClient
from shapely.geometry import box, mapping, shape
from sqlalchemy import select
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin

from mlsystem2.training_ui_api._models import DatasetEditorDraftRow, JobRow, TestSampleRow as SampleRow, TrainingResultRow
from mlsystem2.training_ui_api._database import Base, configure_schema, create_session_factory
from mlsystem2.training_ui_api._test_sample_editor import _copy_image, sample_editor_publish
from mlsystem2.training_ui_api.contracts import ImageryType
from mlsystem2.training_ui_api._markup_lock import test_sample_storage_lock as storage_lock
from mlsystem2.training_ui_api._markup_lock import TestSampleStorageBusy as StorageBusy
from mlsystem2.training_ui_api import _worker
from mlsystem2.training_ui_api.api import create_app
from test_training_ui_markup_export import _configure_export_environment, _write_export_dataset


@pytest.fixture
def manual_client(tmp_path, monkeypatch):
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_export_dataset(config.mlmarkup_root, config.images_root)
    configure_schema(None)
    Base.metadata.create_all(create_session_factory(config).kw["bind"])
    with TestClient(create_app()) as client:
        assert client.post("/api/v1/auth/login", json={"username": "mluser", "password": "secret"}).status_code == 200
        result = client.post("/api/v1/test-samples", json={
            "dataset_key": "Вырубки\\main", "name": "Ручная проверочная выборка", "creation_mode": "manual",
        })
        assert result.status_code == 200, result.text
        sample = result.json()
        yield client, config, sample


def _add_image(client, sample):
    response = client.post(f"/api/v1/test-samples/{sample['id']}/editor/scenes", json={"image_paths": ["region_a/scene_a.tif"]})
    assert response.status_code == 200, response.text
    return response.json()["scenes"][0]


def _annotation(geometry=None):
    return {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
            "features": [{"type": "Feature", "id": "контур-1", "properties": {},
                          "geometry": mapping(geometry or box(11, 49, 13, 51))}]}


def test_empty_manual_sample_and_editor_are_available_without_jobs(manual_client):
    client, config, sample = manual_client
    assert sample["image_count"] == sample["actual_object_count"] == 0
    assert sample["tiles"] == []
    base = f"/api/v1/test-samples/{sample['id']}/editor"
    assert client.get(base).json()["datasets"][0]["scene_count"] == 0
    assert client.get(base + "/scenes").json()["scenes"] == []
    assert client.post(f"/api/v1/test-samples/classes/{sample['class_key']}/reconcile").status_code == 204
    factory = create_session_factory(config)
    with factory() as session:
        assert session.scalars(select(JobRow)).all() == []


def test_manual_annotations_draft_publish_and_archive_do_not_change_source(manual_client):
    client, config, sample = manual_client
    source = config.images_root / "kanopus/region_a/scene_a.tif"
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    scene = _add_image(client, sample)
    name = scene["annotation_name"]
    base = f"/api/v1/test-samples/{sample['id']}/editor"
    assert client.get(scene["raster_url"], headers={"Range": "bytes=0-15"}).status_code == 206
    response = client.put(base + f"/drafts/{name}", json={"base_revision": scene["revision"], "geojson": _annotation()})
    assert response.status_code == 200, response.text
    assert response.json()["positive_count"] == 1
    assert client.get(base + f"/scenes/{name}").json()["draft"]["geojson"]["features"]
    assert client.get(f"/api/v1/test-samples/{sample['id']}").json()["actual_object_count"] == 0
    publication = client.post(base + "/drafts/publish")
    assert publication.status_code == 200, publication.text
    updated = client.get(f"/api/v1/test-samples/{sample['id']}").json()
    assert updated["actual_object_count"] == 1 and updated["content_revision"] == 3
    assert client.get(updated["tiles"][0]["thumbnail_url"]).status_code == 200
    root = config.stored_files_root / "test-samples" / sample["id"]
    assert json.loads((root / name).read_text(encoding="utf-8"))["features"][0]["properties"]["_mlsystem2_role"] == "positive"
    with rasterio.open(root / "tile_001_mask.png") as raster:
        assert raster.read(1).sum() > 0
    assert client.get(updated["download_url"]).status_code == 200
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with create_session_factory(config)() as session:
        assert session.scalars(select(JobRow)).all() == []
        assert session.scalars(select(DatasetEditorDraftRow)).all() == []


def test_manual_editor_hole_survives_publication_and_becomes_background_in_mask(manual_client):
    client, config, sample = manual_client
    scene = _add_image(client, sample)
    name = scene["annotation_name"]
    base = f"/api/v1/test-samples/{sample['id']}/editor"
    outer = box(10, 30, 30, 50)
    island = box(15, 35, 25, 45)
    payload = _annotation(outer.difference(island))
    saved = client.put(base + f"/drafts/{name}", json={
        "base_revision": scene["revision"], "geojson": payload,
    })
    assert saved.status_code == 200, saved.text
    reopened = client.get(base + f"/scenes/{name}").json()
    assert shape(reopened["draft"]["geojson"]["features"][0]["geometry"]).equals(outer.difference(island))
    published = client.post(base + "/drafts/publish")
    assert published.status_code == 200, published.text
    updated = client.get(f"/api/v1/test-samples/{sample['id']}").json()
    assert updated["actual_object_count"] == 1
    root = config.stored_files_root / "test-samples" / sample["id"]
    stored = json.loads((root / name).read_text(encoding="utf-8"))["features"][0]
    assert stored["id"] == "контур-1"
    assert shape(stored["geometry"]).equals(outer.difference(island))
    with rasterio.open(root / "tile_001.tif") as image, rasterio.open(root / "tile_001_mask.png") as mask:
        values = mask.read(1)
        assert values[image.index(20, 40)] == 0
        assert values[image.index(12, 40)] == 1
        assert np.count_nonzero(values) == 300
    with create_session_factory(config)() as session:
        assert session.scalars(select(JobRow)).all() == []


def test_manual_import_and_conflicts_are_atomic(manual_client):
    client, config, sample = manual_client
    base = f"/api/v1/test-samples/{sample['id']}/editor"
    request = {"scenes": [{"annotation_name": "region_a_scene_a.geojson", "geojson": _annotation(box(0, 0, 64, 64))}]}
    refused = client.post(base + "/drafts/import", json=request)
    assert refused.status_code == 400, refused.text
    assert refused.json()["code"] == "annotation_outside_footprint"
    assert client.get(base + "/scenes").json()["scenes"] == []
    request["clip_to_footprint"] = True
    assert client.post(base + "/drafts/import", json=request).status_code == 200
    assert client.post(base + "/drafts/import", json=request).status_code == 409
    assert client.post(base + "/drafts/publish").status_code == 200
    scene = client.get(base + "/scenes").json()["scenes"][0]
    rejected = client.put(base + f"/drafts/{scene['annotation_name']}", json={"base_revision": "устаревшая", "geojson": _annotation()})
    assert rejected.status_code == 409
    with create_session_factory(config)() as session:
        assert session.scalars(select(JobRow)).all() == []


def test_manual_publication_restores_files_when_transaction_fails(manual_client):
    client, config, sample = manual_client
    scene = _add_image(client, sample)
    base = f"/api/v1/test-samples/{sample['id']}/editor"
    response = client.put(base + f"/drafts/{scene['annotation_name']}", json={"base_revision": scene["revision"], "geojson": _annotation()})
    assert response.status_code == 200, response.text
    root = config.stored_files_root / "test-samples" / sample["id"]
    before = (root / scene["annotation_name"]).read_bytes()
    with create_session_factory(config)() as session:
        with pytest.raises(RuntimeError, match="Ошибка транзакции"):
            with sample_editor_publish(session, config, uuid.UUID(sample["id"]), "mluser"):
                raise RuntimeError("Ошибка транзакции")
        session.rollback()
        assert session.get(SampleRow, uuid.UUID(sample["id"])).actual_object_count == 0
    assert (root / scene["annotation_name"]).read_bytes() == before


def test_manual_editor_rejects_foreign_paths_and_training_roles(manual_client):
    client, _, sample = manual_client
    base = f"/api/v1/test-samples/{sample['id']}/editor"
    assert client.post(base + "/scenes", json={"image_paths": ["../чужой.tif"]}).status_code == 400
    scene = _add_image(client, sample)
    payload = _annotation()
    payload["features"][0]["properties"]["_mlsystem2_role"] = "hard_negative"
    assert client.put(base + f"/drafts/{scene['annotation_name']}", json={"base_revision": scene["revision"], "geojson": payload}).status_code == 400
    assert client.post(base + "/scenes", json={"image_paths": ["region_a/scene_a.tif"]}).status_code == 409
    assert client.get(base + "/rasters/region_b_scene_b.geojson").status_code == 400


def test_existing_generated_sample_can_be_edited_and_emptied_without_inference(manual_client):
    client, config, _ = manual_client
    response = client.post("/api/v1/test-samples", json={"dataset_key": "Вырубки\\main",
        "name": "Прежняя нарезка", "tile_width": 16, "tile_height": 16, "image_count": 1, "object_count": 1})
    assert response.status_code == 200, response.text
    sample = response.json()
    base = f"/api/v1/test-samples/{sample['id']}/editor"
    scene = client.get(base + "/scenes").json()["scenes"][0]
    detail = client.get(base + f"/scenes/{scene['annotation_name']}").json()
    result = client.put(base + f"/drafts/{scene['annotation_name']}", json={
        "base_revision": scene["revision"], "geojson": detail["geojson"], "deleted": True})
    assert result.status_code == 200, result.text
    assert client.post(base + "/drafts/publish").status_code == 200
    updated = client.get(f"/api/v1/test-samples/{sample['id']}").json()
    assert updated["image_count"] == updated["actual_object_count"] == updated["territory_count"] == 0
    with create_session_factory(config)() as session:
        assert session.scalars(select(JobRow)).all() == []


def test_manual_network_recalculation_is_idempotent_and_catalog_does_not_queue(manual_client):
    client, config, sample = manual_client
    _add_image(client, sample)
    assert client.put(f"/api/v1/test-samples/{sample['id']}/primary", json={"is_primary": True}).status_code == 200
    with create_session_factory(config)() as session:
        result = TrainingResultRow(source="manual", dataset_key=sample["dataset_key"], class_key=sample["dataset_key"],
            class_display_name=sample["dataset_name"], architecture="segformer_b2", model_name="Проверяемая сеть",
            mlflow_run_id="manual-evaluation-run", status="ok")
        session.add(result)
        session.commit()
        result_id = result.id
    client.get(f"/api/v1/test-samples/cards?class_key={sample['class_key']}")
    client.get(f"/api/v1/results/datasets/{sample['dataset_key']}")
    client.post(f"/api/v1/test-samples/classes/{sample['class_key']}/reconcile")
    with create_session_factory(config)() as session:
        assert session.scalars(select(JobRow)).all() == []
    endpoint = f"/api/v1/results/training/{result_id}/test-f1"
    first = client.post(endpoint)
    assert first.status_code == 200, first.text
    second = client.post(endpoint)
    assert second.status_code == 200, second.text
    assert first.json()["job_id"] == second.json()["job_id"]
    assert first.json()["status"] == "queued"
    with create_session_factory(config)() as session:
        jobs = session.scalars(select(JobRow)).all()
        assert len(jobs) == 1 and jobs[0].source == "manual"


@pytest.mark.parametrize("imagery", [ImageryType.KANOPUS, ImageryType.ORTHO])
def test_manual_raster_preserves_nir_or_converts_alpha_to_mask(tmp_path, imagery):
    source_path, target_path = tmp_path / "исходный.tif", tmp_path / "подготовленный.tif"
    data = np.full((4, 32, 32), 100, dtype="uint8")
    data[3] = 0 if imagery == ImageryType.KANOPUS else 255
    if imagery == ImageryType.ORTHO:
        data[3, :4, :4] = 0
    with rasterio.open(source_path, "w", driver="GTiff", count=4, dtype="uint8", width=32, height=32,
                       crs="EPSG:3857", transform=from_origin(0, 32, 1, 1),
                       ALPHA="NO" if imagery == ImageryType.KANOPUS else "YES") as source:
        source.write(data)
        source.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue,
                              ColorInterp.undefined if imagery == ImageryType.KANOPUS else ColorInterp.alpha)
    _copy_image(source_path, target_path, imagery)
    with rasterio.open(target_path) as target:
        if imagery == ImageryType.KANOPUS:
            assert target.count == 4
            assert ColorInterp.alpha not in target.colorinterp
            assert not target.read(4).any()
            assert target.dataset_mask().min() == 255
        else:
            assert target.count == 3
            assert not target.dataset_mask()[:4, :4].any()
            assert target.dataset_mask()[4:, 4:].min() == 255
            np.testing.assert_array_equal(target.read(), data[:3])


def test_snapshot_does_not_wait_for_publication_while_holding_job_transaction(manual_client):
    _, config, sample = manual_client
    ready, release = Event(), Event()

    def publish():
        with storage_lock(config, sample["id"], write=True):
            ready.set()
            release.wait(timeout=5)

    writer = Thread(target=publish)
    writer.start()
    try:
        assert ready.wait(timeout=5)
        with pytest.raises(BlockingIOError, match="Публикуется"):
            with storage_lock(config, sample["id"], wait=False):
                pytest.fail("Чтение прошло сквозь публикацию")
    finally:
        release.set()
        writer.join(timeout=5)
    assert not writer.is_alive()


def test_busy_publication_defers_dispatch_without_gpu_failure_or_new_job(manual_client, monkeypatch):
    _, config, sample = manual_client

    def busy(*_args):
        raise StorageBusy("Публикуется разметка")

    monkeypatch.setattr(_worker, "_build_test_sample_f1_config", busy)
    with create_session_factory(config)() as session:
        job = JobRow(type="inference", source="manual", status="queued", queue_position=1,
                     dataset_key=sample["dataset_key"], dataset_name=sample["dataset_name"],
                     model_name="Проверяемая сеть", architecture="segformer_b2", tile_size=512,
                     config={"operation": "test_sample_f1"})
        session.add(job)
        session.commit()
        job_id = job.id
        _worker._start_inference_job(session, job, config, popen_factory=lambda *_a, **_k: pytest.fail("GPU не должен запускаться"))
        session.commit()
        assert session.get(JobRow, job_id).status == "queued"
        assert session.get(JobRow, job_id).tmp_path is None
        assert len(session.scalars(select(JobRow)).all()) == 1
