from __future__ import annotations

from copy import deepcopy
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin
from shapely.geometry import box

from mlsystem2.training_ui_api.api import create_app
from mlsystem2.training_ui_api._dataset_editor import _footprint_covers_geometry


@dataclass(frozen=True)
class _EditorEnvironment:
    client: TestClient
    dataset_key: str
    empty_dataset_key: str
    live_annotation: Path
    editor_root: Path
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
    live_annotation = live_dataset / "Olskij_SCN01.geojson"
    live_annotation.write_text(
        json.dumps(initial_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    images_root = tmp_path / "images"
    olskij = images_root / "kanopus" / "Olskij"
    batch = images_root / "kanopus" / "batch"
    olskij.mkdir(parents=True)
    batch.mkdir(parents=True)
    scene_image = olskij / "SCN01.tif"
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

    release_marker = tmp_path / "release-marker"
    monkeypatch.setenv(
        "MLSYSTEM2_TRAINING_UI_DATABASE_URL",
        f"sqlite:///{tmp_path / 'ui.db'}",
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
        empty_dataset = next(
            item for item in payload if item["dataset_name"] == "empty"
        )
        yield _EditorEnvironment(
            client=client,
            dataset_key=dataset["key"],
            empty_dataset_key=empty_dataset["key"],
            live_annotation=live_annotation,
            editor_root=editor_root,
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
    empty_scenes = env.client.get(
        f"/api/v1/dataset-editor/datasets/{empty_path}/scenes"
    )
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
        "coordinates": [
            [[6.25, 0.25], [7.75, 0.25], [7.75, 1.75], [6.25, 1.75], [6.25, 0.25]]
        ],
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


def test_dataset_editor_publishes_multiple_scenes_atomically(
    editor_environment: _EditorEnvironment,
) -> None:
    env = editor_environment
    dataset_path = quote(env.dataset_key, safe="")
    scenes_url = f"/api/v1/dataset-editor/datasets/{dataset_path}/scenes"
    added = env.client.post(scenes_url, json={"image_paths": ["batch/SCN02.tif"]})
    assert added.status_code == 200

    scenes = env.client.get(scenes_url).json()["scenes"]
    first = next(item for item in scenes if item["annotation_name"] == "Olskij_SCN01.geojson")
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
    assert int(_git(env.editor_root, "rev-list", "--count", "HEAD").stdout) == (
        commits_before + 1
    )
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
    assert "put" in openapi["paths"][
        "/api/v1/dataset-editor/datasets/{dataset_key}/scenes"
    ]
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
    assert int(_git(env.editor_root, "rev-list", "--count", "HEAD").stdout) == (
        commits_before + 1
    )
    assert env.live_annotation.read_bytes() == live_before
    assert not (env.live_annotation.parent / "batch_SCN02.geojson").exists()

    scene = next(
        item
        for item in env.client.get(scenes_url).json()["scenes"]
        if item["annotation_name"] == "batch_SCN02.geojson"
    )
    deleted = env.client.request(
        "DELETE",
        f"{scenes_url}/{quote(scene['annotation_name'], safe='')}",
        json={"revision": scene["revision"]},
    )
    assert deleted.status_code == 200
    remaining = env.client.get(scenes_url).json()["scenes"]
    assert {item["annotation_name"] for item in remaining} == {
        "Olskij_SCN01.geojson",
        "batch_SCN03.geojson",
    }


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
