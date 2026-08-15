from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np
import pytest
import rasterio
import yaml
from fastapi.testclient import TestClient
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.geometry import mapping
from shapely.ops import transform as transform_geometry
from pyproj import Transformer
from sqlalchemy import select

from mlsystem2.training_ui_api.api import create_app
from mlsystem2.training_ui_api import _worker
from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import create_session_factory
from mlsystem2.training_ui_api._models import (
    DatasetClassRow,
    DatasetEditorDraftRow,
    DatasetRow,
    JobRow,
    ManagedDatasetSceneRow,
    PseudoMarkupResultRow,
    StoredFileRow,
    TrainingResultRow,
)
from mlsystem2.training_ui_api._dataset_editor import _footprint_covers_geometry
from mlsystem2.training_ui_api._managed_migration import _git_geojson_payloads


@dataclass(frozen=True)
class _EditorEnvironment:
    client: TestClient
    dataset_key: str
    empty_dataset_key: str
    live_annotation: Path
    editor_root: Path
    editor_dataset: Path
    database_path: Path
    release_marker: Path


@pytest.fixture
def editor_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    live_root = tmp_path / "live"
    source_relative = Path("Реки") / "test"
    live_dataset = live_root / source_relative
    live_dataset.mkdir(parents=True)
    (live_root / "Реки" / "empty").mkdir()
    initial_payload = _annotation_payload(
        [
            _feature(1, "positive", [[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]]),
            _feature(
                2,
                "hard_negative",
                [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]],
                properties={"source": "legacy"},
            ),
            _feature(
                3,
                "positive",
                [[6.25, 0.25], [7.75, 0.25], [7.75, 1.75], [6.25, 1.75], [6.25, 0.25]],
            ),
        ]
    )
    live_annotation = live_dataset / "Olskij_SCN01.part.geojson"
    live_annotation.write_text(
        json.dumps(initial_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    images_root = tmp_path / "images"
    olskij = images_root / "kanopus" / "Olskij"
    batch = images_root / "kanopus" / "batch"
    olskij.mkdir(parents=True)
    batch.mkdir(parents=True)
    scene_image = olskij / "SCN01.part.tif"
    _write_raster(scene_image, value=11, nodata_corner=True)
    _write_raster(batch / "SCN02.tif", value=22)
    _write_raster(batch / "SCN03.tiff", value=33)

    seed = tmp_path / "seed"
    seed_dataset = seed / source_relative
    seed_dataset.mkdir(parents=True)
    shutil.copy2(live_annotation, seed_dataset / live_annotation.name)
    _git(seed, "init", "-b", "main")
    _git(seed, "config", "user.name", "Тест")
    _git(seed, "config", "user.email", "test@example.invalid")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "Начальная разметка")
    origin = tmp_path / "origin.git"
    editor_root = tmp_path / "editor"
    subprocess.run(
        ["git", "clone", "--bare", str(seed), str(origin)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "clone", str(origin), str(editor_root)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    _git(editor_root, "config", "user.name", "Тест")
    _git(editor_root, "config", "user.email", "test@example.invalid")

    release_marker = tmp_path / "release-marker"
    database_path = tmp_path / "ui.db"
    monkeypatch.setenv(
        "MLSYSTEM2_TRAINING_UI_DATABASE_URL",
        f"sqlite:///{database_path}",
    )
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(live_root))
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_EDITOR_ROOT", str(editor_root))
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_RELEASE_MARKER", str(release_marker))
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_EDITOR_BRANCH", "main")
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(images_root))
    monkeypatch.setenv(
        "MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT",
        str(tmp_path / "stored"),
    )
    monkeypatch.setenv(
        "MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT",
        str(tmp_path / "scratch"),
    )
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_USER", "editor")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_PASSWORD", "secret")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SESSION_SECRET", "test-secret")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    with TestClient(create_app()) as client:
        login = client.post(
            "/api/v1/auth/login",
            json={"username": "editor", "password": "secret"},
        )
        assert login.status_code == 200
        datasets = client.get("/api/v1/dataset-editor/datasets")
        assert datasets.status_code == 200
        payload = datasets.json()["datasets"]
        assert {item["dataset_name"] for item in payload} == {"empty", "test"}
        dataset = next(item for item in payload if item["dataset_name"] == "test")
        empty_dataset = next(item for item in payload if item["dataset_name"] == "empty")
        yield _EditorEnvironment(
            client=client,
            dataset_key=dataset["key"],
            empty_dataset_key=empty_dataset["key"],
            live_annotation=live_annotation,
            editor_root=editor_root,
            editor_dataset=editor_root / source_relative,
            database_path=database_path,
            release_marker=release_marker,
        )


def test_dataset_editor_requires_auth_and_lists_counts_and_raster_ranges(
    editor_environment: _EditorEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = editor_environment
    dataset_path = quote(env.dataset_key, safe="")
    scenes_url = f"/api/v1/dataset-editor/datasets/{dataset_path}/scenes"

    unauthenticated = TestClient(create_app())
    try:
        assert unauthenticated.get(scenes_url).status_code == 401
    finally:
        unauthenticated.close()

    response = env.client.get(scenes_url)
    assert response.status_code == 200
    scenes = response.json()["scenes"]
    assert len(scenes) == 1
    assert scenes[0]["total_count"] == 3
    assert scenes[0]["positive_count"] == 2
    assert scenes[0]["hard_negative_count"] == 1
    assert len(scenes[0]["revision"]) == 40
    empty_path = quote(env.empty_dataset_key, safe="")
    empty_scenes = env.client.get(f"/api/v1/dataset-editor/datasets/{empty_path}/scenes")
    assert empty_scenes.status_code == 200
    assert empty_scenes.json()["scenes"] == []

    full = env.client.get(scenes[0]["raster_url"])
    partial = env.client.get(
        scenes[0]["raster_url"],
        headers={"Range": "bytes=0-31"},
    )
    invalid = env.client.get(
        scenes[0]["raster_url"],
        headers={"Range": "bytes=999999999-"},
    )
    assert full.status_code == 200
    assert full.headers["accept-ranges"] == "bytes"
    assert partial.status_code == 206
    assert partial.content == full.content[:32]
    assert partial.headers["content-range"].startswith("bytes 0-31/")
    assert invalid.status_code == 416


def test_dataset_editor_save_checks_revision_geometry_and_publication(
    editor_environment: _EditorEnvironment,
) -> None:
    env = editor_environment
    live_before = env.live_annotation.read_bytes()
    dataset_path = quote(env.dataset_key, safe="")
    scenes_url = f"/api/v1/dataset-editor/datasets/{dataset_path}/scenes"
    scene = env.client.get(scenes_url).json()["scenes"][0]
    annotation_path = quote(scene["annotation_name"], safe="")
    detail_url = f"{scenes_url}/{annotation_path}"
    detail = env.client.get(detail_url).json()
    assert detail["valid_data_footprint"]["type"] == "Polygon"
    assert len(detail["geojson"]["features"]) == 2
    updated = detail["geojson"]
    updated["features"][0]["geometry"] = {
        "type": "Polygon",
        "coordinates": [[[1, 1], [3.5, 1], [3.5, 3.5], [1, 3.5], [1, 1]]],
    }
    for feature in updated["features"]:
        feature.setdefault("properties", {}).setdefault(
            "_mlsystem2_role",
            "positive",
        )

    saved = env.client.put(
        detail_url,
        json={"revision": scene["revision"], "geojson": updated},
    )
    assert saved.status_code == 200
    commit = saved.json()["commit"]
    assert saved.json()["publication_status"] == "publishing"
    assert env.live_annotation.read_bytes() == live_before
    assert _git(env.editor_root, "show", "-s", "--format=%B", commit).stdout.endswith(
        "MLSystem2-User: editor\n\n"
    )
    assert _git(env.editor_root, "show", "-s", "--format=%an", commit).stdout.strip() == "editor"

    stale = env.client.put(
        detail_url,
        json={"revision": scene["revision"], "geojson": updated},
    )
    assert stale.status_code == 409

    current = env.client.get(detail_url).json()
    out_of_bounds = current["geojson"]
    out_of_bounds["features"][0]["geometry"] = {
        "type": "Polygon",
        "coordinates": [[[7, 7], [9, 7], [9, 9], [7, 9], [7, 7]]],
    }
    rejected = env.client.put(
        detail_url,
        json={
            "revision": current["scene"]["revision"],
            "geojson": out_of_bounds,
        },
    )
    assert rejected.status_code == 400
    assert "footprint" in rejected.json()["detail"]

    inside_nodata = current["geojson"]
    inside_nodata["features"][0]["geometry"] = {
        "type": "Polygon",
        "coordinates": [[[6.25, 0.25], [7.75, 0.25], [7.75, 1.75], [6.25, 1.75], [6.25, 0.25]]],
    }
    rejected_nodata = env.client.put(
        detail_url,
        json={
            "revision": current["scene"]["revision"],
            "geojson": inside_nodata,
        },
    )
    assert rejected_nodata.status_code == 400
    assert "footprint" in rejected_nodata.json()["detail"]

    publishing = env.client.get(f"/api/v1/dataset-editor/publication/{commit}")
    assert publishing.json()["status"] == "publishing"
    env.release_marker.write_text(commit + "\n", encoding="utf-8")
    published = env.client.get(f"/api/v1/dataset-editor/publication/{commit}")
    assert published.json()["status"] == "published"


def test_dataset_editor_persists_discards_and_publishes_server_drafts(
    editor_environment: _EditorEnvironment,
) -> None:
    env = editor_environment
    dataset_path = quote(env.dataset_key, safe="")
    scenes_url = f"/api/v1/dataset-editor/datasets/{dataset_path}/scenes"
    scene = env.client.get(scenes_url).json()["scenes"][0]
    annotation_path = quote(scene["annotation_name"], safe="")
    detail_url = f"{scenes_url}/{annotation_path}"
    draft_url = f"/api/v1/dataset-editor/datasets/{dataset_path}/drafts/{annotation_path}"
    live_before = env.live_annotation.read_bytes()
    head_before = _git(env.editor_root, "rev-parse", "HEAD").stdout.strip()
    payload = deepcopy(env.client.get(detail_url).json()["geojson"])
    payload["features"] = payload["features"][:1]

    saved = env.client.put(
        draft_url,
        json={"base_revision": scene["revision"], "geojson": payload},
    )
    assert saved.status_code == 200
    assert saved.json()["total_count"] == 1
    assert env.live_annotation.read_bytes() == live_before
    assert _git(env.editor_root, "rev-parse", "HEAD").stdout.strip() == head_before
    listed = env.client.get(scenes_url).json()["scenes"][0]
    assert listed["draft"]["total_count"] == 1
    reopened = env.client.get(detail_url).json()
    assert reopened["draft"]["geojson"] == saved.json()["geojson"]
    with create_session_factory(get_config())() as session:
        assert session.scalar(select(DatasetEditorDraftRow)) is not None
        assert not [
            row
            for row in session.scalars(select(JobRow)).all()
            if (row.config or {}).get("operation") == "dataset_editor_scene_pseudo"
        ]

    discarded = env.client.delete(draft_url)
    assert discarded.status_code == 200
    assert discarded.json()["deleted_count"] == 1
    assert env.client.get(detail_url).json()["draft"] is None

    assert (
        env.client.put(
            draft_url,
            json={"base_revision": scene["revision"], "geojson": payload},
        ).status_code
        == 200
    )
    published = env.client.post(f"/api/v1/dataset-editor/datasets/{dataset_path}/drafts/publish")
    assert published.status_code == 200
    assert published.json()["commit"] != head_before
    assert env.client.get(detail_url).json()["draft"] is None
    with create_session_factory(get_config())() as session:
        assert session.scalar(select(DatasetEditorDraftRow)) is None


def test_dataset_editor_publishes_multiple_scenes_atomically(
    editor_environment: _EditorEnvironment,
) -> None:
    env = editor_environment
    dataset_path = quote(env.dataset_key, safe="")
    scenes_url = f"/api/v1/dataset-editor/datasets/{dataset_path}/scenes"
    added = env.client.post(scenes_url, json={"image_paths": ["batch/SCN02.tif"]})
    assert added.status_code == 200

    scenes = env.client.get(scenes_url).json()["scenes"]
    first = next(item for item in scenes if item["annotation_name"] == "Olskij_SCN01.part.geojson")
    second = next(item for item in scenes if item["annotation_name"] == "batch_SCN02.geojson")

    def detail(scene: dict[str, object]) -> dict[str, object]:
        annotation = quote(str(scene["annotation_name"]), safe="")
        return env.client.get(f"{scenes_url}/{annotation}").json()

    first_detail = detail(first)
    second_detail = detail(second)
    first_geojson = deepcopy(first_detail["geojson"])
    first_geojson["features"][0]["geometry"] = {
        "type": "Polygon",
        "coordinates": [[[1, 1], [3.5, 1], [3.5, 3.5], [1, 3.5], [1, 1]]],
    }
    second_geojson = deepcopy(second_detail["geojson"])
    second_geojson["features"] = [
        _feature(3, "positive", [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]),
        _feature(4, "hard_negative", [[4, 4], [5, 4], [5, 5], [4, 5], [4, 4]]),
    ]
    request = {
        "scenes": [
            {
                "annotation_name": first["annotation_name"],
                "revision": first["revision"],
                "geojson": first_geojson,
            },
            {
                "annotation_name": second["annotation_name"],
                "revision": second["revision"],
                "geojson": second_geojson,
            },
        ]
    }
    commits_before = int(_git(env.editor_root, "rev-list", "--count", "HEAD").stdout)
    published = env.client.put(scenes_url, json=request)
    assert published.status_code == 200
    assert int(_git(env.editor_root, "rev-list", "--count", "HEAD").stdout) == (commits_before + 1)
    assert [item["annotation_name"] for item in published.json()["scenes"]] == [
        first["annotation_name"],
        second["annotation_name"],
    ]
    assert published.json()["scenes"][1]["total_count"] == 2

    current_first = detail(published.json()["scenes"][0])
    current_second = detail(published.json()["scenes"][1])
    first_path = env.editor_root / "Реки" / "test" / str(first["annotation_name"])
    second_path = env.editor_root / "Реки" / "test" / str(second["annotation_name"])
    files_before = (first_path.read_bytes(), second_path.read_bytes())
    commits_before = int(_git(env.editor_root, "rev-list", "--count", "HEAD").stdout)

    stale_request = deepcopy(request)
    stale_request["scenes"][1]["revision"] = current_second["scene"]["revision"]
    stale_request["scenes"][1]["geojson"] = deepcopy(current_second["geojson"])
    stale_request["scenes"][1]["geojson"]["features"][0]["geometry"] = {
        "type": "Polygon",
        "coordinates": [[[7, 7], [9, 7], [9, 9], [7, 9], [7, 7]]],
    }
    stale = env.client.put(scenes_url, json=stale_request)
    assert stale.status_code == 409
    assert (first_path.read_bytes(), second_path.read_bytes()) == files_before
    assert int(_git(env.editor_root, "rev-list", "--count", "HEAD").stdout) == commits_before

    invalid_first = deepcopy(current_first["geojson"])
    invalid_second = deepcopy(current_second["geojson"])
    invalid_second["features"][0]["geometry"] = {
        "type": "Polygon",
        "coordinates": [[[7, 7], [9, 7], [9, 9], [7, 9], [7, 7]]],
    }
    invalid = env.client.put(
        scenes_url,
        json={
            "scenes": [
                {
                    "annotation_name": first["annotation_name"],
                    "revision": current_first["scene"]["revision"],
                    "geojson": invalid_first,
                },
                {
                    "annotation_name": second["annotation_name"],
                    "revision": current_second["scene"]["revision"],
                    "geojson": invalid_second,
                },
            ]
        },
    )
    assert invalid.status_code == 400
    assert (first_path.read_bytes(), second_path.read_bytes()) == files_before
    assert int(_git(env.editor_root, "rev-list", "--count", "HEAD").stdout) == commits_before

    assert env.client.put(scenes_url, json={"scenes": []}).status_code == 422
    duplicate = {"scenes": [request["scenes"][0], request["scenes"][0]]}
    assert env.client.put(scenes_url, json=duplicate).status_code == 422
    openapi = env.client.get("/openapi.json").json()
    assert "put" in openapi["paths"]["/api/v1/dataset-editor/datasets/{dataset_key}/scenes"]
    assert "DatasetEditorPublishRequest" in openapi["components"]["schemas"]


def test_dataset_editor_adds_folder_atomically_and_deletes_one_scene(
    editor_environment: _EditorEnvironment,
) -> None:
    env = editor_environment
    dataset_path = quote(env.dataset_key, safe="")
    scenes_url = f"/api/v1/dataset-editor/datasets/{dataset_path}/scenes"
    live_before = env.live_annotation.read_bytes()
    commits_before = int(_git(env.editor_root, "rev-list", "--count", "HEAD").stdout)

    browser = env.client.get(
        f"/api/v1/dataset-editor/datasets/{dataset_path}/rasters",
        params={"folder": "batch"},
    )
    assert browser.status_code == 200
    assert [item["name"] for item in browser.json()["rasters"]] == [
        "SCN02.tif",
        "SCN03.tiff",
    ]
    added = env.client.post(scenes_url, json={"folder_path": "batch"})
    assert added.status_code == 200
    assert len(added.json()["scenes"]) == 2
    assert int(_git(env.editor_root, "rev-list", "--count", "HEAD").stdout) == (commits_before + 1)
    footprint_path = env.editor_dataset / "batch_SCN02_footprint.geojson"
    assert footprint_path.is_file()
    footprint = json.loads(footprint_path.read_text(encoding="utf-8"))
    assert len(footprint["features"]) == 1
    assert footprint["features"][0]["properties"] == {
        "_mlsystem2_type": "valid_data_footprint"
    }
    assert env.live_annotation.read_bytes() == live_before
    assert not (env.live_annotation.parent / "batch_SCN02.geojson").exists()
    already_added = env.client.post(scenes_url, json={"folder_path": "batch"})
    assert already_added.status_code == 400
    assert "уже добавлены" in already_added.json()["detail"]
    duplicate_single = env.client.post(
        scenes_url,
        json={"image_paths": ["batch/SCN02.tif"]},
    )
    assert duplicate_single.status_code == 409
    _write_raster(env.editor_root.parent / "images" / "kanopus" / "batch" / "SCN04.tif", value=44)
    mixed_folder = env.client.post(scenes_url, json={"folder_path": "batch"})
    assert mixed_folder.status_code == 200
    assert [item["annotation_name"] for item in mixed_folder.json()["scenes"]] == [
        "batch_SCN04.geojson"
    ]

    scene = next(
        item
        for item in env.client.get(scenes_url).json()["scenes"]
        if item["annotation_name"] == "batch_SCN02.geojson"
    )
    head_after_add = _git(env.editor_root, "rev-parse", "HEAD").stdout.strip()
    deleted = env.client.request(
        "DELETE",
        f"{scenes_url}/{quote(scene['annotation_name'], safe='')}",
        json={"revision": scene["revision"]},
    )
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert _git(env.editor_root, "rev-parse", "HEAD").stdout.strip() == head_after_add
    remaining = env.client.get(scenes_url).json()["scenes"]
    assert {item["annotation_name"] for item in remaining} == {
        "Olskij_SCN01.part.geojson",
        "batch_SCN02.geojson",
        "batch_SCN03.geojson",
        "batch_SCN04.geojson",
    }
    pending = next(item for item in remaining if item["annotation_name"] == "batch_SCN02.geojson")
    assert pending["draft"]["deleted"] is True

    draft_url = (
        f"/api/v1/dataset-editor/datasets/{dataset_path}/drafts/"
        f"{quote(scene['annotation_name'], safe='')}"
    )
    assert env.client.delete(draft_url).json()["deleted_count"] == 1
    restored = next(
        item
        for item in env.client.get(scenes_url).json()["scenes"]
        if item["annotation_name"] == "batch_SCN02.geojson"
    )
    assert restored["draft"] is None

    assert (
        env.client.request(
            "DELETE",
            f"{scenes_url}/{quote(scene['annotation_name'], safe='')}",
            json={"revision": scene["revision"]},
        ).status_code
        == 200
    )
    published = env.client.post(f"/api/v1/dataset-editor/datasets/{dataset_path}/drafts/publish")
    assert published.status_code == 200
    remaining = env.client.get(scenes_url).json()["scenes"]
    assert {item["annotation_name"] for item in remaining} == {
        "Olskij_SCN01.part.geojson",
        "batch_SCN03.geojson",
        "batch_SCN04.geojson",
    }
    assert not (env.editor_dataset / "batch_SCN02.geojson").exists()
    assert not footprint_path.exists()


def test_dataset_editor_deletes_dataset_folder_but_keeps_database_history(
    editor_environment: _EditorEnvironment,
) -> None:
    env = editor_environment
    live_before = env.live_annotation.read_bytes()
    response = env.client.delete(
        f"/api/v1/dataset-editor/datasets/{quote(env.dataset_key, safe='')}"
    )

    assert response.status_code == 200
    commit = response.json()["commit"]
    assert response.json()["publication_status"] == "publishing"
    assert not env.editor_dataset.exists()
    assert env.live_annotation.read_bytes() == live_before
    assert _git(env.editor_root, "show", "-s", "--format=%B", commit).stdout.endswith(
        "MLSystem2-User: editor\n\n"
    )
    assert _git(env.editor_root, "show", "-s", "--format=%an", commit).stdout.strip() == "editor"

    catalog = env.client.get("/api/v1/dataset-catalog")
    assert catalog.status_code == 200
    active_keys = {
        dataset["key"]
        for class_info in catalog.json()["classes"]
        for dataset in class_info["datasets"]
    }
    assert env.dataset_key not in active_keys
    assert (
        env.client.get(
            f"/api/v1/dataset-editor/datasets/{quote(env.dataset_key, safe='')}/scenes"
        ).status_code
        == 400
    )

    with sqlite3.connect(env.database_path) as connection:
        source_path, deleted_at = connection.execute(
            "SELECT source_path, deleted_at FROM datasets WHERE key = ?",
            (env.dataset_key,),
        ).fetchone()
    assert source_path
    assert deleted_at
    assert (
        "delete"
        in env.client.get("/openapi.json").json()["paths"][
            "/api/v1/dataset-editor/datasets/{dataset_key}"
        ]
    )


def test_dataset_editor_returns_service_unavailable_for_missing_clone(
    editor_environment: _EditorEnvironment,
) -> None:
    env = editor_environment
    git_dir = env.editor_root / ".git"
    hidden_git_dir = env.editor_root / ".git.disabled"
    git_dir.rename(hidden_git_dir)
    try:
        response = env.client.get("/api/v1/dataset-editor/datasets")
    finally:
        hidden_git_dir.rename(git_dir)

    assert response.status_code == 503
    assert "Editor-клон" in response.json()["detail"]


def test_dataset_editor_footprint_allows_only_numerical_boundary_sliver() -> None:
    footprint = box(0, 0, 10, 10)

    assert _footprint_covers_geometry(footprint, box(-1e-13, 1, 2, 2))
    assert not _footprint_covers_geometry(footprint, box(-0.01, 1, 2, 2))


def test_dataset_editor_returns_primary_network_pseudo_fragment(
    editor_environment: _EditorEnvironment,
) -> None:
    env = editor_environment
    result_id = _create_primary_training_result(env)
    stored_root = get_config().stored_files_root
    stored_root.mkdir(parents=True, exist_ok=True)
    scenes_path = stored_root / "covered-scenes.txt"
    scenes_path.write_text("Olskij/SCN01.part\n", encoding="utf-8")
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    inside = transform_geometry(to_wgs84.transform, box(1, 1, 3, 3))
    outside = transform_geometry(to_wgs84.transform, box(20, 20, 21, 21))
    pseudo_path = stored_root / "full-pseudo.geojson"
    pseudo_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"confidence": 0.9},
                        "geometry": mapping(inside),
                    },
                    {
                        "type": "Feature",
                        "properties": {"confidence": 0.8},
                        "geometry": mapping(outside),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    with create_session_factory(get_config())() as session:
        scenes_file = StoredFileRow(
            kind="scenes_txt",
            original_name=scenes_path.name,
            content_type="text/plain",
            path=str(scenes_path),
            size_bytes=scenes_path.stat().st_size,
        )
        pseudo_file = StoredFileRow(
            kind="pseudo_markup_geojson",
            original_name=pseudo_path.name,
            content_type="application/geo+json",
            path=str(pseudo_path),
            size_bytes=pseudo_path.stat().st_size,
            object_count=2,
        )
        session.add_all([scenes_file, pseudo_file])
        session.flush()
        session.add(
            PseudoMarkupResultRow(
                dataset_key="другой-датасет-того-же-класса",
                training_result_id=result_id,
                class_key=env.dataset_key,
                source_dataset_name="Реки / test",
                image_count=1,
                scenes_file_id=scenes_file.id,
                geojson_file_id=pseudo_file.id,
                status="ok",
            )
        )
        session.commit()

    scene = env.client.get(
        f"/api/v1/dataset-editor/datasets/{quote(env.dataset_key, safe='')}/scenes"
    ).json()["scenes"][0]
    endpoint = (
        "/api/v1/dataset-editor/datasets/"
        f"{quote(env.dataset_key, safe='')}/scenes/"
        f"{quote(scene['annotation_name'], safe='')}/pseudo-markup"
    )
    read_only = env.client.get(endpoint)
    assert read_only.status_code == 200
    assert read_only.json()["status"] == "ready"
    assert read_only.json()["source"] == "dataset"
    response = env.client.post(endpoint)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["source"] == "dataset"
    assert payload["training_result_id"] == str(result_id)
    assert payload["object_count"] == 1
    assert payload["geojson"]["crs"]["properties"]["name"] == "EPSG:4326"
    with create_session_factory(get_config())() as session:
        assert not any(
            (row.config or {}).get("operation") == "dataset_editor_scene_pseudo"
            for row in session.scalars(select(JobRow)).all()
        )


def test_dataset_editor_reuses_latest_dataset_pseudo_without_explicit_primary(
    editor_environment: _EditorEnvironment,
) -> None:
    env = editor_environment
    local_result_id = _create_primary_training_result(env)
    stored_root = get_config().stored_files_root
    stored_root.mkdir(parents=True, exist_ok=True)
    scenes_path = stored_root / "covered-scenes-without-primary.txt"
    scenes_path.write_text("Olskij/SCN01.part\n", encoding="utf-8")
    pseudo_path = stored_root / "full-pseudo-without-primary.geojson"
    pseudo_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": []}),
        encoding="utf-8",
    )
    with create_session_factory(get_config())() as session:
        dataset = session.scalar(select(DatasetRow).where(DatasetRow.key == env.dataset_key))
        other_dataset = session.scalar(
            select(DatasetRow).where(DatasetRow.key == env.empty_dataset_key)
        )
        assert dataset is not None and other_dataset is not None
        class_row = session.get(DatasetClassRow, dataset.class_id)
        assert class_row is not None
        class_row.primary_training_result_id = None
        newer_class_result = TrainingResultRow(
            dataset_key=other_dataset.key,
            dataset_version="newer-other-version",
            class_key=other_dataset.key,
            class_display_name=class_row.name,
            architecture="smp_segformer_b2",
            model_name="Более новая сеть другого датасета",
            quality_metric="pixel",
            task="binary",
            class_schema=[],
            status="ok",
            trained_at=datetime.now(timezone.utc) + timedelta(minutes=1),
            mlflow_run_id="run-newer-other-dataset",
        )
        scenes_file = StoredFileRow(
            kind="scenes_txt",
            original_name=scenes_path.name,
            content_type="text/plain",
            path=str(scenes_path),
            size_bytes=scenes_path.stat().st_size,
        )
        pseudo_file = StoredFileRow(
            kind="pseudo_markup_geojson",
            original_name=pseudo_path.name,
            content_type="application/geo+json",
            path=str(pseudo_path),
            size_bytes=pseudo_path.stat().st_size,
            object_count=0,
        )
        session.add_all([newer_class_result, scenes_file, pseudo_file])
        session.flush()
        session.add(
            PseudoMarkupResultRow(
                dataset_key=dataset.key,
                training_result_id=local_result_id,
                class_key=dataset.key,
                source_dataset_name="Реки / test",
                image_count=1,
                scenes_file_id=scenes_file.id,
                geojson_file_id=pseudo_file.id,
                status="ok",
            )
        )
        session.commit()

    scenes_response = env.client.get(
        f"/api/v1/dataset-editor/datasets/{quote(env.dataset_key, safe='')}/scenes"
    )
    assert scenes_response.status_code == 200
    scenes_payload = scenes_response.json()
    assert scenes_payload["dataset"]["primary_training_result_id"] == str(local_result_id)
    scene = scenes_payload["scenes"][0]
    endpoint = (
        "/api/v1/dataset-editor/datasets/"
        f"{quote(env.dataset_key, safe='')}/scenes/"
        f"{quote(scene['annotation_name'], safe='')}/pseudo-markup"
    )
    read_only = env.client.get(endpoint)
    assert read_only.status_code == 200
    assert read_only.json()["status"] == "ready"
    assert read_only.json()["training_result_id"] == str(local_result_id)
    response = env.client.post(endpoint)
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["source"] == "dataset"
    assert payload["training_result_id"] == str(local_result_id)
    with create_session_factory(get_config())() as session:
        assert not any(
            (row.config or {}).get("operation") == "dataset_editor_scene_pseudo"
            for row in session.scalars(select(JobRow)).all()
        )


def test_managed_dataset_publication_writes_new_object_to_selected_source(
    editor_environment: _EditorEnvironment,
) -> None:
    env = editor_environment
    annotation_name = env.live_annotation.name
    second_relative = Path("Озера") / "main"
    second_payload = _annotation_payload(
        [_feature(20, "positive", [[4, 4], [5, 4], [5, 5], [4, 5], [4, 4]])]
    )
    live_root = env.live_annotation.parents[2]
    live_second = live_root / second_relative
    editor_second = env.editor_root / second_relative
    live_second.mkdir(parents=True)
    editor_second.mkdir(parents=True)
    for target in (live_second / annotation_name, editor_second / annotation_name):
        target.write_text(json.dumps(second_payload, ensure_ascii=False), encoding="utf-8")
    _git(env.editor_root, "add", second_relative.as_posix())
    _git(env.editor_root, "commit", "-m", "Добавить второй исходный датасет")
    _git(env.editor_root, "push", "origin", "HEAD:main")

    catalog_response = env.client.post("/api/v1/dataset-catalog/sync")
    assert catalog_response.status_code == 200
    catalog = catalog_response.json()
    rivers = next(item for item in catalog["classes"] if item["name"] == "Реки")
    lakes = next(item for item in catalog["classes"] if item["name"] == "Озера")
    target_class_response = env.client.post(
        "/api/v1/dataset-classes",
        json={"name": "Реки и озера", "imagery_type": "kanopus"},
    )
    assert target_class_response.status_code == 200
    target_class = next(
        item for item in target_class_response.json()["classes"] if item["name"] == "Реки и озера"
    )
    composed_response = env.client.post(
        "/api/v1/managed-datasets/compose",
        json={
            "class_key": target_class["key"],
            "name": "main",
            "sources": [
                {
                    "dataset_key": next(
                        item["key"]
                        for item in rivers["datasets"]
                        if item["dataset_name"] == "test"
                    ),
                    "priority": 100,
                },
                {"dataset_key": lakes["datasets"][0]["key"], "priority": 0},
            ],
        },
    )
    assert composed_response.status_code == 200, composed_response.text
    managed = next(
        item
        for item in composed_response.json()["classes"]
        if item["key"] == target_class["key"]
    )["datasets"][0]
    assert managed["managed"] is True

    scenes_response = env.client.get(
        f"/api/v1/dataset-editor/datasets/{quote(managed['key'], safe='')}/scenes"
    )
    assert scenes_response.status_code == 200, scenes_response.text
    scene = scenes_response.json()["scenes"][0]
    detail_response = env.client.get(
        "/api/v1/dataset-editor/datasets/"
        f"{quote(managed['key'], safe='')}/scenes/{quote(annotation_name, safe='')}"
    )
    assert detail_response.status_code == 200
    payload = detail_response.json()["geojson"]
    lake_type = next(
        item
        for item in scenes_response.json()["dataset"]["object_types"]
        if item["name"] == "Озера"
    )
    payload["features"].append(
        {
            "type": "Feature",
            "id": "new-lake",
            "properties": {
                "_mlsystem2_role": "positive",
                "_mlsystem2_class": lake_type["slug"],
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0.5, 4], [1.5, 4], [1.5, 5], [0.5, 5], [0.5, 4]]],
            },
        }
    )
    publish = env.client.put(
        f"/api/v1/dataset-editor/datasets/{quote(managed['key'], safe='')}/scenes",
        json={
            "scenes": [
                {
                    "annotation_name": annotation_name,
                    "revision": scene["revision"],
                    "geojson": payload,
                }
            ]
        },
    )
    assert publish.status_code == 200, publish.text
    saved = json.loads((editor_second / annotation_name).read_text(encoding="utf-8"))
    assert sum(
        feature["properties"]["_mlsystem2_role"] == "positive"
        for feature in saved["features"]
    ) == 2
    assert any(feature.get("id") == "new-lake" for feature in saved["features"])
    assert not (env.editor_root / "Реки и озера" / "main").exists()

    added_image = env.editor_root.parent / "images" / "kanopus" / "batch" / "MANAGED04.tif"
    _write_raster(added_image, value=44)
    added = env.client.post(
        f"/api/v1/dataset-editor/datasets/{quote(managed['key'], safe='')}/scenes",
        json={"image_paths": ["batch/MANAGED04.tif"]},
    )
    assert added.status_code == 200, added.text
    added_scene = added.json()["scenes"][0]
    added_annotation = added_scene["annotation_name"]
    river_target = env.editor_dataset / added_annotation
    lake_target = editor_second / added_annotation
    assert not river_target.exists()
    assert not lake_target.exists()
    with create_session_factory(get_config())() as session:
        explicit = session.scalar(
            select(ManagedDatasetSceneRow).where(
                ManagedDatasetSceneRow.annotation_name == added_annotation
            )
        )
        assert explicit is not None
        assert explicit.image_relative_path == "batch/MANAGED04.tif"

    added_detail_url = (
        "/api/v1/dataset-editor/datasets/"
        f"{quote(managed['key'], safe='')}/scenes/{quote(added_annotation, safe='')}"
    )
    added_detail = env.client.get(added_detail_url)
    assert added_detail.status_code == 200, added_detail.text
    added_payload = added_detail.json()["geojson"]
    added_payload["features"] = [
        {
            "type": "Feature",
            "id": "managed-lake",
            "properties": {
                "_mlsystem2_role": "positive",
                "_mlsystem2_class": lake_type["slug"],
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]],
            },
        },
        {
            "type": "Feature",
            "id": "managed-hard-negative",
            "properties": {"_mlsystem2_role": "hard_negative"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[4, 4], [5, 4], [5, 5], [4, 5], [4, 4]]],
            },
        },
    ]
    published_added = env.client.put(
        f"/api/v1/dataset-editor/datasets/{quote(managed['key'], safe='')}/scenes",
        json={
            "scenes": [
                {
                    "annotation_name": added_annotation,
                    "revision": added_scene["revision"],
                    "geojson": added_payload,
                }
            ]
        },
    )
    assert published_added.status_code == 200, published_added.text
    river_features = json.loads(river_target.read_text(encoding="utf-8"))["features"]
    lake_features = json.loads(lake_target.read_text(encoding="utf-8"))["features"]
    assert [item["properties"]["_mlsystem2_role"] for item in river_features] == [
        "hard_negative"
    ]
    assert {item["properties"]["_mlsystem2_role"] for item in lake_features} == {
        "positive",
        "hard_negative",
    }
    refreshed_added = env.client.get(added_detail_url).json()
    assert refreshed_added["scene"]["positive_count"] == 1
    assert refreshed_added["scene"]["hard_negative_count"] == 1

    marked_deleted = env.client.request(
        "DELETE",
        added_detail_url,
        json={"revision": refreshed_added["scene"]["revision"]},
    )
    assert marked_deleted.status_code == 200
    assert marked_deleted.json()["deleted"] is True
    published_deletion = env.client.post(
        f"/api/v1/dataset-editor/datasets/{quote(managed['key'], safe='')}/drafts/publish"
    )
    assert published_deletion.status_code == 200, published_deletion.text
    assert not river_target.exists()
    assert not lake_target.exists()
    with create_session_factory(get_config())() as session:
        assert session.scalar(
            select(ManagedDatasetSceneRow).where(
                ManagedDatasetSceneRow.annotation_name == added_annotation
            )
        ) is None

    empty_image = added_image.with_name("MANAGED05.tif")
    _write_raster(empty_image, value=55)
    empty_added = env.client.post(
        f"/api/v1/dataset-editor/datasets/{quote(managed['key'], safe='')}/scenes",
        json={"image_paths": ["batch/MANAGED05.tif"]},
    ).json()["scenes"][0]
    empty_url = (
        "/api/v1/dataset-editor/datasets/"
        f"{quote(managed['key'], safe='')}/scenes/{quote(empty_added['annotation_name'], safe='')}"
    )
    assert env.client.request(
        "DELETE",
        empty_url,
        json={"revision": empty_added["revision"]},
    ).status_code == 200
    empty_deleted = env.client.post(
        f"/api/v1/dataset-editor/datasets/{quote(managed['key'], safe='')}/drafts/publish"
    )
    assert empty_deleted.status_code == 200, empty_deleted.text
    assert env.client.get(empty_url).status_code == 400


def test_managed_migration_reads_cyrillic_git_paths(
    editor_environment: _EditorEnvironment,
) -> None:
    env = editor_environment
    initial_commit = _git(
        env.editor_root,
        "rev-list",
        "--max-parents=0",
        "HEAD",
    ).stdout.strip()

    payloads = _git_geojson_payloads(
        get_config(),
        initial_commit,
        "Реки/test",
    )

    assert set(payloads) == {env.live_annotation.name}
    assert len(payloads[env.live_annotation.name]["features"]) == 3


def test_dataset_editor_queues_one_urgent_scene_inference(
    editor_environment: _EditorEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = editor_environment
    result_id = _create_primary_training_result(env)
    with create_session_factory(get_config())() as session:
        result = session.get(TrainingResultRow, result_id)
        assert result is not None
        selected = SimpleNamespace(
            result=result,
            dataset_name="Реки / test",
            input_channels=4,
            checkpoint=SimpleNamespace(
                artifact_uri="s3://artifacts/best.pt",
                artifact_path="checkpoints/best.pt",
                threshold=0.7,
                f1_score=0.8,
                epoch=12,
            ),
            external_model=None,
            inference_template_id=None,
            inference_template_config={},
        )
    monkeypatch.setattr(
        "mlsystem2.training_ui_api._dataset_editor._select_model",
        lambda *_args, **_kwargs: selected,
    )
    scene = env.client.get(
        f"/api/v1/dataset-editor/datasets/{quote(env.dataset_key, safe='')}/scenes"
    ).json()["scenes"][0]
    endpoint = (
        "/api/v1/dataset-editor/datasets/"
        f"{quote(env.dataset_key, safe='')}/scenes/"
        f"{quote(scene['annotation_name'], safe='')}/pseudo-markup"
    )
    first = env.client.post(endpoint)
    second = env.client.post(endpoint)
    assert first.status_code == 200
    assert first.json()["status"] == "queued"
    assert second.json()["job_id"] == first.json()["job_id"]
    with create_session_factory(get_config())() as session:
        rows = session.scalars(select(JobRow)).all()
        editor_jobs = [
            row
            for row in rows
            if (row.config or {}).get("operation") == "dataset_editor_scene_pseudo"
        ]
        assert len(editor_jobs) == 1
        assert editor_jobs[0].config["priority"] == "urgent"
        assert editor_jobs[0].config["editor_pseudo"]["image_relative"] == "Olskij/SCN01.part"
        job_id = editor_jobs[0].id
        editor_jobs[0].status = "failed"
        editor_jobs[0].error = "тестовая ошибка"
        session.commit()

    failed = env.client.post(endpoint)
    assert failed.json()["status"] == "failed"
    assert failed.json()["can_retry"] is True
    retried = env.client.post(f"{endpoint}?retry=true")
    assert retried.json()["status"] == "queued"
    assert retried.json()["job_id"] == str(job_id)

    with create_session_factory(get_config())() as session:
        _worker.dispatch_inference_queue_once(
            session,
            get_config(),
            popen_factory=lambda *_args, **_kwargs: SimpleNamespace(pid=4321),
        )
        session.flush()
        job = session.get(JobRow, job_id)
        assert job is not None and job.tmp_path is not None
        run_dir = Path(job.tmp_path)
        runner_config = yaml.safe_load((run_dir / "pseudo_config.yaml").read_text(encoding="utf-8"))
        assert runner_config["scenes_file"].endswith("editor_scene.txt")
        assert runner_config["inference_backend"] == "pytorch_one_off"
        assert runner_config["class_key"] == env.dataset_key
        assert runner_config["checkpoint_uri"] == "s3://artifacts/best.pt"
        assert Path(runner_config["images_root"]).name == "kanopus"
        output = run_dir / "scratch" / "pseudo_markup.geojson"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
        (run_dir / "scratch" / "report.json").write_text(
            '{"status":"ok","processed":1,"feature_count":0}',
            encoding="utf-8",
        )
        (run_dir / "exit_code").write_text("0\n", encoding="utf-8")
        _worker.dispatch_inference_queue_once(session, get_config())
        session.commit()
        completed = session.get(JobRow, job.id)
        assert completed is not None
        assert completed.status == "completed"
        assert completed.config["editor_pseudo"]["result_file_id"]

    ready = env.client.get(endpoint)
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["source"] == "scene"
    repeated_ready = env.client.post(endpoint)
    assert repeated_ready.json()["status"] == "ready"
    assert repeated_ready.json()["job_id"] == str(job_id)
    lightweight = env.client.get(f"/api/v1/dataset-editor/pseudo-markup/{job_id}")
    assert lightweight.status_code == 200
    assert lightweight.json()["status"] == "ready"
    assert lightweight.json()["job_id"] == str(job_id)


def _create_primary_training_result(env: _EditorEnvironment):
    with create_session_factory(get_config())() as session:
        dataset = session.scalar(select(DatasetRow).where(DatasetRow.key == env.dataset_key))
        assert dataset is not None
        class_row = session.get(DatasetClassRow, dataset.class_id)
        assert class_row is not None
        result = TrainingResultRow(
            dataset_key=dataset.key,
            dataset_version="test-version",
            class_key=dataset.key,
            class_display_name=class_row.name,
            architecture="smp_segformer_b2",
            model_name="SegFormer B2",
            quality_metric="pixel",
            task="binary",
            class_schema=[],
            status="ok",
            trained_at=datetime.now(timezone.utc),
            mlflow_run_id="run-primary",
        )
        session.add(result)
        session.flush()
        class_row.primary_training_result_id = result.id
        session.commit()
        return result.id


def _annotation_payload(features: list[dict[str, object]]) -> dict[str, object]:
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": features,
    }


def _feature(
    feature_id: int,
    role: str,
    ring: list[list[float]],
    *,
    properties: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "type": "Feature",
        "id": feature_id,
        "properties": {
            **(properties or {}),
            "_mlsystem2_role": role,
        },
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def _write_raster(path: Path, *, value: int, nodata_corner: bool = False) -> None:
    data = np.full((1, 8, 8), value, dtype=np.uint16)
    if nodata_corner:
        data[:, 6:, 6:] = 0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=1,
        dtype="uint16",
        crs="EPSG:3857",
        transform=from_origin(0, 8, 1, 1),
        nodata=0,
    ) as dataset:
        dataset.write(data)


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
