"""Области, версии класса, совместимость ручных наборов и точность вырезки."""

from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import uuid
from zipfile import ZipFile

import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from shapely.geometry import box, mapping, Polygon, shape
from sqlalchemy import select

from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import create_session_factory
from mlsystem2.training_ui_api._models import AnnotationVersionRow, TestSampleRow as SampleRow
from mlsystem2.training_ui_api._test_region_crop import crop_test_region
from mlsystem2.training_ui_api.contracts import ImageryType
from test_training_ui_dataset_editor import editor_environment, _git


@pytest.fixture
def region_sample(editor_environment):
    env = editor_environment
    client = env.client
    for image_path in get_config().images_root.rglob("*.tif*"):
        with rasterio.open(image_path) as image:
            profile = {**image.profile, "count": 4, "dtype": "uint8", "nodata": 0, "ALPHA": "NO"}
            valid = image.dataset_mask() > 0
        with rasterio.open(image_path, "w", **profile) as image:
            image.write(np.broadcast_to(valid.astype(np.uint8) * 30, (4, profile["height"], profile["width"])))
            image.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.undefined)
    # Вторая версия того же снимка и ещё один снимок только из другого датасета.
    other = env.editor_root / "Реки/empty"
    other.mkdir(exist_ok=True)
    payload = json.loads((env.editor_dataset / env.live_annotation.name).read_text(encoding="utf-8"))
    payload["features"] = payload["features"][:1]
    (other / env.live_annotation.name).write_text(json.dumps(payload), encoding="utf-8")
    (other / "batch_SCN02.geojson").write_text(json.dumps(payload), encoding="utf-8")
    _git(env.editor_root, "add", ".")
    _git(env.editor_root, "commit", "-m", "Дополнительные источники тестовых областей")
    created = client.post("/api/v1/test-samples", json={"dataset_key": env.dataset_key,
        "name": "Выборка с областями", "creation_mode": "manual"})
    assert created.status_code == 200, created.text
    sample = created.json()
    base = f"/api/v1/test-samples/{sample['id']}/editor"
    response = client.get(base + "/scenes?include_class_sources=true")
    assert response.status_code == 200, response.text
    scenes = response.json()["scenes"]
    assert len(scenes) == 2
    scene = next(s for s in scenes if s["image_name"].endswith("SCN01.part.tif"))
    return client, get_config(), sample, base, scene


def _region(identifier="область-1", geometry=None):
    return {"type": "Feature", "id": identifier, "properties": {"_mlsystem2_role": "annotation_zone"},
            "geometry": mapping(geometry if geometry is not None else box(-252, -252, 260, 260))}


def _save(client, base, scene, *, geometry=None, version_id=None):
    detail = client.get(base + "/scenes/" + scene["annotation_name"]).json()
    catalog = detail["annotation_versions"]
    selected_id = version_id or catalog["selected_id"]
    version = next(v for v in catalog["versions"] if v["id"] == selected_id)
    payload = deepcopy(detail["geojson"])
    payload["features"] = [f for f in payload["features"] if f["properties"].get("_mlsystem2_role") != "annotation_zone"] + [_region(geometry=geometry)]
    request = {"base_revision": detail["scene"]["revision"], "geojson": payload,
               "annotation_versions": {"catalog_revision": catalog["revision"], "selected_id": selected_id,
                                       "selected_revision": version["revision"]}}
    saved = client.put(base + "/drafts/" + scene["annotation_name"], json=request)
    assert saved.status_code == 200, saved.text
    return saved.json(), request


def test_class_catalog_versions_and_mixed_publication(region_sample):
    client, config, sample, base, scene = region_sample
    assert scene["test_source"] and scene["annotation_version_count"] == 2
    assert client.get(base + "/scenes").json()["scenes"] == []
    detail = client.get(base + "/scenes/" + scene["annotation_name"]).json()
    assert detail["raster_grid"] == [1, 0, 0, 0, -1, 8]
    assert {v["usages"][0]["dataset_name"] for v in detail["annotation_versions"]["versions"]} == {"test", "empty"}
    standalone = client.post(base + "/scenes", json={"image_paths": ["batch/SCN03.tiff"]})
    assert standalone.status_code == 200, standalone.text
    saved, _ = _save(client, base, scene)
    assert saved["annotation_zone_count"] == 1
    assert client.get(f"/api/v1/test-samples/{sample['id']}").json()["image_count"] == 1
    response = client.post(base + "/drafts/publish")
    assert response.status_code == 200, response.text
    after = client.get(f"/api/v1/test-samples/{sample['id']}").json()
    assert after["image_count"] == 2
    assert after["has_region_sources"] is True
    assert len(response.json()["scenes"]) == 2
    root = config.stored_files_root / "test-samples" / sample["id"]
    with rasterio.open(root / "tile_002.tif") as image:
        assert image.width == image.height == 512
        assert image.count == 4 and image.dataset_mask()[0, 0] == 0
        assert image.dataset_mask().sum() > 0
    polygons = json.loads((root / "tile_002.geojson").read_text(encoding="utf-8"))["features"]
    assert all(f["properties"]["_mlsystem2_role"] == "positive" and shape(f["geometry"]).is_valid for f in polygons)
    downloaded = client.get(after["download_url"])
    assert downloaded.status_code == 200, downloaded.text
    with ZipFile(BytesIO(downloaded.content)) as archive:
        assert len([n for n in archive.namelist() if n.endswith(".tif")]) == 2
    reopened = client.get(base + "/scenes/" + scene["annotation_name"]).json()
    assert len([f for f in reopened["geojson"]["features"] if f["properties"]["_mlsystem2_role"] == "annotation_zone"]) == 1
    # Тот же исходник разрешён также как отдельный импортированный тестовый снимок.
    added = client.post(base + "/scenes", json={"image_paths": ["Olskij/SCN01.part.tif"]})
    assert added.status_code == 200, added.text
    assert client.get(f"/api/v1/test-samples/{sample['id']}").json()["image_count"] == 3


def test_snapshot_refresh_and_region_removal(region_sample):
    client, config, sample, base, scene = region_sample
    saved, _ = _save(client, base, scene)
    selected_id = saved["annotation_versions"]["selected_id"]
    assert client.post(base + "/drafts/publish").status_code == 200
    root = config.stored_files_root / "test-samples" / sample["id"]
    before = (root / "tile_001.geojson").read_bytes()
    with create_session_factory(config)() as session:
        row = session.get(AnnotationVersionRow, uuid.UUID(selected_id))
        row.geojson = {**row.geojson, "features": []}
        row.revision += 1
        session.commit()
    detail = client.get(base + "/scenes/" + scene["annotation_name"]).json()
    assert detail["scene"]["annotation_version_revision"] == 1
    assert any(f["properties"]["_mlsystem2_role"] == "positive" for f in detail["geojson"]["features"])
    assert (root / "tile_001.geojson").read_bytes() == before
    # Правка области сохраняет старую выбранную редакцию, даже после изменения источника.
    request = {"base_revision": detail["scene"]["revision"], "geojson": detail["geojson"],
               "annotation_versions": {"catalog_revision": detail["annotation_versions"]["revision"],
                   "selected_id": selected_id, "selected_revision": 1}}
    response = client.put(base + "/drafts/" + scene["annotation_name"], json=request)
    assert response.status_code == 200, response.text
    assert response.json()["positive_count"] > 0
    # Явное обновление берёт только серверную новую версию, не присланные клиентом объекты.
    request["annotation_versions"]["selected_revision"] = 2
    response = client.put(base + "/drafts/" + scene["annotation_name"], json=request)
    assert response.status_code == 200, response.text
    assert response.json()["positive_count"] == 0
    assert client.post(base + "/drafts/publish").status_code == 200
    assert json.loads((root / "tile_001.geojson").read_bytes())["features"] == []
    detail = client.get(base + "/scenes/" + scene["annotation_name"]).json()
    request.update(base_revision=detail["scene"]["revision"], geojson={**detail["geojson"], "features": []})
    assert client.put(base + "/drafts/" + scene["annotation_name"], json=request).status_code == 200
    assert client.post(base + "/drafts/publish").status_code == 200
    assert client.get(f"/api/v1/test-samples/{sample['id']}").json()["image_count"] == 0
    assert not (root / "tile_001.tif").exists()
    with create_session_factory(config)() as session:
        assert session.get(SampleRow, uuid.UUID(sample["id"])).sources == []


def test_failed_publication_and_later_success_preserve_hardlinked_f1_snapshot(region_sample, tmp_path):
    client, config, sample, base, scene = region_sample
    _save(client, base, scene)
    assert client.post(base + "/drafts/publish").status_code == 200
    root = config.stored_files_root / "test-samples" / sample["id"]
    snapshot = tmp_path / "снимок-расчёта-f1"
    snapshot.mkdir()
    before = {}
    for name in ("tile_001.tif", "tile_001.geojson", "tile_001_mask.png"):
        before[name] = (root / name).read_bytes()
        os.link(root / name, snapshot / name)
    detail = client.get(base + "/scenes/" + scene["annotation_name"]).json()
    payload = deepcopy(detail["geojson"])
    payload["features"] = [f for f in payload["features"] if f["properties"].get("_mlsystem2_role") != "annotation_zone"]
    payload["features"] += [_region(geometry=box(1, 1, 4, 4)), _region("вне-снимка", box(100, 100, 105, 105))]
    request = {"base_revision": detail["scene"]["revision"], "geojson": payload, "annotation_versions": {
        "catalog_revision": detail["annotation_versions"]["revision"],
        "selected_id": detail["annotation_versions"]["selected_id"], "selected_revision": 1}}
    assert client.put(base + "/drafts/" + scene["annotation_name"], json=request).status_code == 200
    failed = client.post(base + "/drafts/publish")
    assert failed.status_code == 400, failed.text
    for name, content in before.items():
        assert (root / name).read_bytes() == content
        assert (snapshot / name).read_bytes() == content
    assert client.get(base + "/scenes/" + scene["annotation_name"]).json()["draft"]["annotation_zone_count"] == 2
    request["geojson"]["features"].pop()
    assert client.put(base + "/drafts/" + scene["annotation_name"], json=request).status_code == 200
    assert client.post(base + "/drafts/publish").status_code == 200
    assert (root / "tile_001.tif").read_bytes() != before["tile_001.tif"]
    for name, content in before.items():
        assert (snapshot / name).read_bytes() == content
    # Старый экран не может затереть опубликованные области.
    assert client.put(base + "/drafts/" + scene["annotation_name"], json=request).status_code == 409


def test_version_selection_without_regions_does_not_include_image(region_sample):
    client, config, sample, base, scene = region_sample
    detail = client.get(base + "/scenes/" + scene["annotation_name"]).json()
    request = {"base_revision": "new", "geojson": detail["geojson"], "annotation_versions": {
        "catalog_revision": detail["annotation_versions"]["revision"],
        "selected_id": detail["annotation_versions"]["versions"][1]["id"], "selected_revision": 1}}
    assert client.put(base + "/drafts/" + scene["annotation_name"], json=request).status_code == 200
    assert client.post(base + "/drafts/publish").status_code == 200
    result = client.get(f"/api/v1/test-samples/{sample['id']}").json()
    assert result["image_count"] == 0 and result["content_revision"] == sample["content_revision"]
    request["annotation_versions"]["selected_id"] = str(uuid.uuid4())
    assert client.put(base + "/drafts/" + scene["annotation_name"], json=request).status_code == 409


def test_snapshot_rechecks_revision_when_source_changes_between_reads(region_sample, monkeypatch):
    from mlsystem2.training_ui_api import _test_sample_sources as sources
    from mlsystem2.training_ui_api._test_sample_editor import _dataset
    from mlsystem2.training_ui_api._dataset_editor import DatasetEditorConflict
    from mlsystem2.training_ui_api.contracts import DatasetEditorSaveDraftRequest

    client, config, sample, base, scene = region_sample
    detail = client.get(base + "/scenes/" + scene["annotation_name"]).json()
    catalog = detail["annotation_versions"]
    request = DatasetEditorSaveDraftRequest.model_validate({"base_revision": "new", "geojson": detail["geojson"],
        "annotation_versions": {"catalog_revision": catalog["revision"], "selected_id": catalog["selected_id"], "selected_revision": 1}})
    original = sources.source_version

    def changed_before_snapshot(session, *args):
        version = session.get(AnnotationVersionRow, uuid.UUID(catalog["selected_id"]))
        version.revision += 1
        version.geojson = {**version.geojson, "features": []}
        session.flush()
        return original(session, *args)

    monkeypatch.setattr(sources, "source_version", changed_before_snapshot)
    with create_session_factory(config)() as session:
        row = session.get(SampleRow, uuid.UUID(sample["id"]))
        with pytest.raises(DatasetEditorConflict, match="уже изменилась"):
            sources.prepare_source_draft(session, config, row, _dataset(session, config, row), scene["annotation_name"], request, None)


@pytest.mark.parametrize("channels,alpha,imagery", [(4, False, ImageryType.KANOPUS), (3, False, ImageryType.ORTHO), (4, True, ImageryType.ORTHO)])
def test_crop_preserves_pixels_grid_holes_and_nodata(tmp_path, channels, alpha, imagery):
    source, target = tmp_path / "исходный.tif", tmp_path / "вырезка.tif"
    data = np.arange(1, channels * 12 * 12 + 1, dtype=np.uint16).reshape(channels, 12, 12).astype(np.uint8)
    data[:, 0, :] = 0
    if alpha:
        data[3] = 255
        data[3, :2] = 0
    with rasterio.open(source, "w", driver="GTiff", width=12, height=12, count=channels,
                       dtype="uint8", crs="EPSG:3857", transform=from_origin(100, 200, .3, .3), nodata=0, ALPHA="YES" if alpha else "NO") as image:
        image.write(data)
        image.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue) + ((ColorInterp.alpha if alpha else ColorInterp.undefined,) if channels == 4 else ())
    region = Polygon([(99.7, 200.3), (103.9, 200.3), (103.9, 196.1), (99.7, 200.3)], holes=[[(102.4, 199.4), (103, 199.4), (103, 198.8), (102.4, 198.8), (102.4, 199.4)]])
    payload = {"type": "FeatureCollection", "features": [{"type": "Feature", "id": "полигон",
        "properties": {"_mlsystem2_role": "positive"}, "geometry": mapping(box(99, 195, 105, 202))}]}
    result = crop_test_region(source, target, mapping(region), payload, imagery)
    with rasterio.open(target) as image:
        assert (image.width, image.height) == (14, 14)
        assert image.count == (4 if imagery == ImageryType.KANOPUS else 3)
        assert image.transform.a == .3 and image.transform.e == -.3
        valid = image.dataset_mask() > 0
        assert valid.any() and not valid[0].any()
        assert not image.read()[:, ~valid].any()
        inner = valid[1:13, 1:13]
        assert np.array_equal(image.read()[:, 1:13, 1:13][:, inner], data[:image.count, :, :][:, inner])
    geometry = shape(result["features"][0]["geometry"])
    assert geometry.is_valid and geometry.difference(region).area < 1e-10
    assert not geometry.intersects(box(102.5, 199, 102.9, 199.3))
