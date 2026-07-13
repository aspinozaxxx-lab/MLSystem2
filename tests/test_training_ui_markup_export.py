from __future__ import annotations

import json
import os
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from pyproj import CRS
from rasterio.enums import ColorInterp
from rasterio.features import rasterize
from rasterio.shutil import copy as raster_copy
from rasterio.transform import from_origin
from shapely.geometry import box, shape

from mlsystem2.training_ui_api import _markup_export
from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api.api import create_app
from mlsystem2.training_ui_api.contracts import MarkupExportRequest, TrainingUIAPIError


pytestmark = pytest.mark.filterwarnings(
    "ignore:Dataset has no geotransform, gcps, or rpcs.*:rasterio.errors.NotGeoreferencedWarning"
)


def test_markup_export_builds_black_free_georeferenced_archive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_export_dataset(config.mlmarkup_root, config.images_root)

    info = _markup_export.build_markup_export(
        MarkupExportRequest(
            dataset_key="Вырубки\\main",
            tile_width=16,
            tile_height=16,
            image_count=2,
            object_count=4,
        ),
        config,
    )

    assert info.actual_object_count == 4
    assert info.image_count == 2
    assert info.territory_count == 2
    assert info.warnings == []
    assert {item.territory for item in info.tiles} == {
        "kanopus/region_a",
        "kanopus/region_b",
    }
    artifact = _markup_export.load_markup_export(info.id, config)
    with zipfile.ZipFile(artifact.archive_path) as archive:
        names = archive.namelist()
    assert len(names) == 8
    assert sum(name.endswith(".tif") for name in names) == 2
    assert sum(name.endswith(".geojson") for name in names) == 2
    assert sum(name.endswith("_mask.png") for name in names) == 2
    assert sum(name.endswith("_overlay.png") for name in names) == 2

    output_root = config.scratch_root / _markup_export.EXPORT_ROOT_NAME / str(info.id)
    output_tiffs = sorted(output_root.glob("*.tif"))
    footprints = []
    for path in output_tiffs:
        with rasterio.open(path) as dataset:
            assert dataset.width == 16
            assert dataset.height == 16
            assert dataset.count == 4
            assert dataset.dtypes == ("uint8",) * 4
            assert dataset.nodata == 0
            assert dataset.crs.to_epsg() == 3857
            assert dataset.tags()["TEST_TAG"] == "source"
            assert dataset.tags(1)["BAND_TAG"] == "first"
            assert dataset.descriptions[0] == "первый канал"
            assert dataset.scales == (1.0, 2.0, 3.0, 4.0)
            assert dataset.offsets == (0.0, 1.0, 2.0, 3.0)
            assert dataset.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"
            assert dataset.compression.name.lower() == "deflate"
            assert dataset.interleaving.name.lower() == "pixel"
            assert bool(np.all(dataset.dataset_mask() != 0))
            assert not bool(np.any(np.all(dataset.read() == 0, axis=0)))
            assert dataset.bounds.left % 100 >= 8
            assert dataset.bounds.right % 100 <= 56
            footprints.append(box(*dataset.bounds))
            tile_bounds = box(*dataset.bounds)
            tile_transform = dataset.transform
        geojson = json.loads(path.with_suffix(".geojson").read_text(encoding="utf-8"))
        geometries = [shape(feature["geometry"]) for feature in geojson["features"]]
        assert all(geometry.difference(tile_bounds).area < 1e-9 for geometry in geometries)
        expected_mask = rasterize(
            [(geometry, 255) for geometry in geometries],
            out_shape=(16, 16),
            transform=tile_transform,
            fill=0,
            dtype="uint8",
        )
        with rasterio.open(output_root / f"{path.stem}_mask.png") as mask_dataset:
            assert np.array_equal(mask_dataset.read(1), expected_mask)
    assert not footprints[0].intersects(footprints[1])

    for path in output_root.glob("*.geojson"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["crs"]["properties"]["name"] == "EPSG:3857"
        assert payload["features"]
        assert all("kind" in feature["properties"] for feature in payload["features"])
        assert all("id" in feature for feature in payload["features"])
    for path in output_root.glob("*_mask.png"):
        with rasterio.open(path) as dataset:
            values = set(np.unique(dataset.read(1)).tolist())
            assert dataset.count == 1
            assert dataset.width == 16
            assert dataset.height == 16
            assert values <= {0, 255}
            assert 255 in values
    for path in output_root.glob("*_overlay.png"):
        with rasterio.open(path) as dataset:
            assert dataset.count == 3
            assert dataset.width == 16
            assert dataset.height == 16


def test_markup_export_reports_nearest_object_count(tmp_path: Path, monkeypatch) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_export_dataset(config.mlmarkup_root, config.images_root)

    info = _markup_export.build_markup_export(
        MarkupExportRequest(
            dataset_key="Вырубки\\main",
            tile_width=16,
            tile_height=16,
            image_count=2,
            object_count=5,
        ),
        config,
    )

    assert info.actual_object_count == 4
    assert any("запрошено 5, сформировано 4" in warning for warning in info.warnings)


def test_markup_export_rejects_empty_positive_markup(tmp_path: Path, monkeypatch) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    dataset_root = config.mlmarkup_root / "Пустая разметка" / "main"
    dataset_root.mkdir(parents=True)
    (dataset_root / "scenes.txt").write_text("region_empty_markup\n", encoding="utf-8")
    (dataset_root / "empty.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
                "features": [],
            }
        ),
        encoding="utf-8",
    )
    _write_cog(
        config.images_root / "kanopus" / "region_empty_markup" / "empty.tif",
        left=0,
        top=64,
        valid_slice=(slice(0, 64), slice(0, 64)),
    )

    with pytest.raises(TrainingUIAPIError, match="не содержит валидных"):
        _markup_export.build_markup_export(
            MarkupExportRequest(
                dataset_key="Пустая разметка\\main",
                tile_width=16,
                tile_height=16,
                image_count=1,
                object_count=1,
            ),
            config,
        )


def test_markup_export_cleanup_removes_expired_and_startup_incomplete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    export_root = config.scratch_root / _markup_export.EXPORT_ROOT_NAME
    stale = export_root / ".building-stale"
    fresh = export_root / ".building-fresh"
    stale.mkdir(parents=True)
    fresh.mkdir()
    now = datetime.now(tz=timezone.utc)
    old_timestamp = (now - timedelta(hours=2)).timestamp()
    os.utime(stale, (old_timestamp, old_timestamp))

    _markup_export.cleanup_expired_markup_exports(config, now=now)

    assert not stale.exists()
    assert fresh.exists()
    _markup_export.cleanup_expired_markup_exports(
        config,
        now=now,
        remove_incomplete=True,
    )
    assert not fresh.exists()


def test_markup_export_rejects_tiles_with_only_black_data(tmp_path: Path, monkeypatch) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    dataset_root = config.mlmarkup_root / "Пусто" / "main"
    dataset_root.mkdir(parents=True)
    (dataset_root / "scenes.txt").write_text("region_empty\n", encoding="utf-8")
    _write_geojson(
        dataset_root / "empty.geojson",
        [(1, box(10, 40, 14, 44), "empty")],
    )
    image_path = config.images_root / "kanopus" / "region_empty" / "empty.tif"
    _write_cog(image_path, left=0, top=64, valid_slice=None, nodata=None)

    with pytest.raises(TrainingUIAPIError, match="полностью валидных тайлов"):
        _markup_export.build_markup_export(
            MarkupExportRequest(
                dataset_key="Пусто\\main",
                tile_width=16,
                tile_height=16,
                image_count=1,
                object_count=1,
            ),
            config,
        )


def test_markup_export_shifts_window_away_from_black_edge(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    dataset_root = config.mlmarkup_root / "Сдвиг" / "main"
    dataset_root.mkdir(parents=True)
    (dataset_root / "scenes.txt").write_text("region_shift\n", encoding="utf-8")
    _write_geojson(
        dataset_root / "shift.geojson",
        [(1, box(22, 40, 24, 42), "shift")],
    )
    _write_cog(
        config.images_root / "kanopus" / "region_shift" / "shift.tif",
        left=0,
        top=64,
        valid_slice=(slice(0, 64), slice(0, 64)),
        black_slice=(slice(0, 64), slice(0, 19)),
        nodata=None,
    )

    info = _markup_export.build_markup_export(
        MarkupExportRequest(
            dataset_key="Сдвиг\\main",
            tile_width=16,
            tile_height=16,
            image_count=1,
            object_count=1,
        ),
        config,
    )

    output_root = config.scratch_root / _markup_export.EXPORT_ROOT_NAME / str(info.id)
    with rasterio.open(next(output_root.glob("*.tif"))) as dataset:
        assert dataset.bounds.left == 19
        assert not bool(np.any(np.all(dataset.read() == 0, axis=0)))


def test_candidate_selection_allows_only_touching_as_fallback() -> None:
    crs = CRS.from_epsg(3857)
    candidates = [
        _markup_export._Candidate(
            source_path=Path(f"scene-{index}.tif"),
            source_name=f"region-{index}/scene-{index}.tif",
            territory=f"region-{index}",
            column=index * 16,
            row=0,
            raster_crs=crs,
            raster_footprint=box(index * 16, 0, (index + 1) * 16, 16),
            annotation_footprint=box(index * 16, 0, (index + 1) * 16, 16),
            feature_positions=(index,),
        )
        for index in range(2)
    ]
    request = MarkupExportRequest(
        dataset_key="test\\main",
        tile_width=16,
        tile_height=16,
        image_count=2,
        object_count=2,
    )

    assert _markup_export._select_candidates(candidates, request, allow_touching=False) is None
    assert _markup_export._select_candidates(candidates, request, allow_touching=True) == [0, 1]


def test_candidate_selection_prioritizes_territories_then_sources_deterministically() -> None:
    crs = CRS.from_epsg(3857)
    candidates = [
        _selection_candidate(crs, index=0, territory="a", source="a/one.tif", count=5),
        _selection_candidate(crs, index=1, territory="a", source="a/one.tif", count=5),
        _selection_candidate(crs, index=2, territory="a", source="a/two.tif", count=1),
        _selection_candidate(crs, index=3, territory="b", source="b/one.tif", count=1),
    ]
    request = MarkupExportRequest(
        dataset_key="test\\main",
        tile_width=16,
        tile_height=16,
        image_count=2,
        object_count=10,
    )

    first = _markup_export._select_candidates(candidates, request, allow_touching=False)
    second = _markup_export._select_candidates(candidates, request, allow_touching=False)

    assert first == second
    assert first is not None
    selected = [candidates[index] for index in first]
    assert {item.territory for item in selected} == {"a", "b"}
    assert len({item.source_name for item in selected}) == 2

    same_territory = candidates[:3]
    source_first = _markup_export._select_candidates(
        same_territory,
        request,
        allow_touching=False,
    )
    assert source_first is not None
    assert 2 in source_first
    assert sum(same_territory[index].object_count for index in source_first) == 6


def test_markup_export_http_flow_and_expiry(tmp_path: Path, monkeypatch) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_export_dataset(config.mlmarkup_root, config.images_root)

    with TestClient(create_app()) as client:
        openapi = client.get("/openapi.json").json()
        assert "MarkupExportRequest" in openapi["components"]["schemas"]
        preview_contract = openapi["paths"][
            "/api/v1/markup-export/{export_id}/tiles/{tile_index}/preview"
        ]["get"]["responses"]["200"]["content"]
        download_contract = openapi["paths"][
            "/api/v1/markup-export/{export_id}/download"
        ]["get"]["responses"]["200"]["content"]
        assert set(preview_contract) == {"image/png"}
        assert set(download_contract) == {"application/zip"}
        assert client.post("/api/v1/markup-export", json={}).status_code == 401
        login = client.post(
            "/api/v1/auth/login",
            json={"username": "mluser", "password": "secret"},
        )
        assert login.status_code == 200
        assert client.post(
            "/api/v1/markup-export",
            json={"dataset_key": "Вырубки\\main", "image_count": 0},
        ).status_code == 422
        assert client.post(
            "/api/v1/markup-export",
            json={"dataset_key": "Неизвестно\\main"},
        ).status_code == 400
        assert client.get(
            "/api/v1/markup-export/00000000-0000-0000-0000-000000000000/download"
        ).status_code == 404
        response = client.post(
            "/api/v1/markup-export",
            json={
                "dataset_key": "Вырубки\\main",
                "tile_width": 16,
                "tile_height": 16,
                "image_count": 2,
                "object_count": 4,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        preview = client.get(payload["tiles"][0]["preview_url"])
        assert preview.status_code == 200
        assert preview.headers["content-type"] == "image/png"
        download = client.get(payload["download_url"])
        assert download.status_code == 200
        assert download.headers["content-type"] == "application/zip"
        assert "deforestation_test_markup.zip" in download.headers["content-disposition"]
        assert client.get(
            f"/api/v1/markup-export/{payload['id']}/tiles/99/preview"
        ).status_code == 404

        manifest_path = (
            config.scratch_root
            / _markup_export.EXPORT_ROOT_NAME
            / payload["id"]
            / _markup_export.MANIFEST_NAME
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["info"]["expires_at"] = "2000-01-01T00:00:00Z"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        assert client.get(payload["download_url"]).status_code == 404


def _configure_export_environment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(
        "MLSYSTEM2_TRAINING_UI_DATABASE_URL",
        f"sqlite:///{tmp_path / 'ui.db'}",
    )
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_USER", "mluser")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_PASSWORD", "secret")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(tmp_path / "MLMarkup"))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "prepared_images"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    return get_config()


def _write_export_dataset(mlmarkup_root: Path, images_root: Path) -> None:
    dataset_root = mlmarkup_root / "Вырубки" / "main"
    dataset_root.mkdir(parents=True)
    (dataset_root / "scenes.txt").write_text("region_a\nregion_b\n", encoding="utf-8")
    _write_geojson(
        dataset_root / "deforestation.geojson",
        [
            (1, box(11, 49, 13, 51), "a-1"),
            (2, box(17, 49, 19, 51), "a-2"),
            (3, box(111, 49, 113, 51), "b-1"),
            (4, box(117, 49, 119, 51), "b-2"),
        ],
    )
    _write_cog(
        images_root / "kanopus" / "region_a" / "scene_a.tif",
        left=0,
        top=64,
        valid_slice=(slice(8, 56), slice(8, 56)),
    )
    _write_cog(
        images_root / "kanopus" / "region_b" / "scene_b.tif",
        left=100,
        top=64,
        valid_slice=(slice(8, 56), slice(8, 56)),
    )


def _write_geojson(path: Path, features: list[tuple[int, object, str]]) -> None:
    from shapely.geometry import mapping

    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [
            {
                "type": "Feature",
                "id": feature_id,
                "properties": {"kind": kind},
                "geometry": mapping(geometry),
            }
            for feature_id, geometry, kind in features
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_cog(
    path: Path,
    *,
    left: float,
    top: float,
    valid_slice: tuple[slice, slice] | None,
    black_slice: tuple[slice, slice] | None = None,
    nodata: float | None = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_path = path.with_suffix(".source.tif")
    data = np.zeros((4, 64, 64), dtype=np.uint8)
    if valid_slice is not None:
        data[:, valid_slice[0], valid_slice[1]] = np.asarray([40, 80, 120, 160])[:, None, None]
    if black_slice is not None:
        data[:, black_slice[0], black_slice[1]] = 0
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=4,
        dtype="uint8",
        nodata=nodata,
        crs="EPSG:3857",
        transform=from_origin(left, top, 1, 1),
        tiled=True,
        blockxsize=16,
        blockysize=16,
        compress="deflate",
        interleave="pixel",
    ) as dataset:
        dataset.write(data)
        dataset.colorinterp = (ColorInterp.undefined,) * 4
        dataset.update_tags(TEST_TAG="source")
        dataset.update_tags(1, BAND_TAG="first")
        dataset.set_band_description(1, "первый канал")
        dataset.scales = (1.0, 2.0, 3.0, 4.0)
        dataset.offsets = (0.0, 1.0, 2.0, 3.0)
    raster_copy(
        source_path,
        path,
        driver="COG",
        BLOCKSIZE=128,
        COMPRESS="DEFLATE",
        INTERLEAVE="PIXEL",
        OVERVIEWS="AUTO",
        RESAMPLING="NEAREST",
    )
    source_path.unlink()


def _selection_candidate(
    crs: CRS,
    *,
    index: int,
    territory: str,
    source: str,
    count: int,
) -> _markup_export._Candidate:
    left = index * 32
    footprint = box(left, 0, left + 16, 16)
    feature_start = index * 100
    return _markup_export._Candidate(
        source_path=Path(source),
        source_name=source,
        territory=territory,
        column=left,
        row=0,
        raster_crs=crs,
        raster_footprint=footprint,
        annotation_footprint=footprint,
        feature_positions=tuple(range(feature_start, feature_start + count)),
    )
