from __future__ import annotations

import json
import os
import threading
import time
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from PIL import Image
from pyproj import CRS, Transformer
from rasterio.enums import ColorInterp
from rasterio.features import rasterize
from rasterio.shutil import copy as raster_copy
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import transform as transform_geometry
from sqlalchemy import select

from mlsystem2.dataset_preparing.api import prepare_dataset, resolve_scene_images
from mlsystem2.dataset_preparing.contracts import (
    DatasetPreparationRequest,
    SceneImageResolutionRequest,
)
from mlsystem2.training_ui_api import _markup_export, _service, _test_samples, _worker
from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import Base, configure_schema, create_session_factory
from mlsystem2.training_ui_api._dataset_catalog import dataset_class_row
from mlsystem2.training_ui_api._models import (
    DatasetRow,
    JobRow,
    PseudoMarkupResultRow,
    StoredFileRow,
    TestSampleRow as _TestSampleRow,
    TestSampleTileRow as _TestSampleTileRow,
    TrainingResultRow,
    TrainingResultTestMetricRow,
)
from mlsystem2.training_ui_api._raster_valid_data import (
    clip_geometries_to_valid_data,
)
from mlsystem2.training_ui_api._test_samples import (
    _object_counts,
    build_test_sample_download,
    build_test_samples_download,
    cleanup_test_sample_storage,
    create_test_sample,
    evaluate_test_sample_by_id,
    evaluate_test_sample_preview,
    evaluate_test_samples_for_pseudo_markup,
    mark_test_samples_stale_for_pseudo_markup,
    optimize_test_sample,
    optimize_test_sample_preview,
    queue_test_sample_evaluation,
    queue_training_result_test_f1,
    reconcile_test_sample_evaluations,
    reconcile_training_result_test_f1,
    test_sample_detail as _test_sample_detail,
    training_result_test_f1_info,
    update_test_sample_primary,
    update_test_sample_tile,
    update_test_sample,
)
from mlsystem2.training_ui_api._worker import _finish_test_sample_f1_job
from mlsystem2.training_ui_api.api import create_app
from mlsystem2.training_ui_api.contracts import (
    ImageryType,
    JobSource,
    MarkupExportRequest,
    StoredFileKind,
    TestSampleCreate as _TestSampleCreate,
    TestSampleEvaluationPreviewRequest as _TestSampleEvaluationPreviewRequest,
    TestSampleBatchCreate as _TestSampleBatchCreate,
    TestSampleOptimizeRequest as _TestSampleOptimizeRequest,
    TestSamplePrimaryUpdate as _TestSamplePrimaryUpdate,
    TestSampleTileUpdate as _TestSampleTileUpdate,
    TestSampleUpdate as _TestSampleUpdate,
    TrainingUIAPIError,
)


pytestmark = pytest.mark.filterwarnings(
    "ignore:Dataset has no geotransform, gcps, or rpcs.*:rasterio.errors.NotGeoreferencedWarning"
)

_DOWNLOADED_TILE_SUFFIXES = {
    ".tif",
    ".geojson",
    "_mask.png",
    "_rgb.jpg",
    "_rgb_markup.jpg",
    "_nrg.jpg",
    "_nrg_markup.jpg",
    "_ngb.jpg",
    "_ngb_markup.jpg",
}


def _downloaded_tile_names(base_name: str, *, folder: str | None = None) -> set[str]:
    prefix = f"{folder}/" if folder else ""
    return {f"{prefix}{base_name}{suffix}" for suffix in _DOWNLOADED_TILE_SUFFIXES}


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
    assert artifact.archive_filename == "вырубки_test_markup.zip"
    with zipfile.ZipFile(artifact.archive_path) as archive:
        names = archive.namelist()
    assert set(names) == {
        f"tile_{index:03d}{suffix}"
        for index in range(1, 3)
        for suffix in (".tif", ".geojson", "_mask.png", "_preview.png")
    }

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
        assert {
            feature["geometry"]["type"] for feature in payload["features"]
        } == {"MultiPolygon"}
    for path in output_root.glob("*_mask.png"):
        with rasterio.open(path) as dataset:
            values = set(np.unique(dataset.read(1)).tolist())
            assert dataset.count == 1
            assert dataset.width == 16
            assert dataset.height == 16
            assert values <= {0, 255}
            assert 255 in values
    for path in output_root.glob("*_preview.png"):
        with rasterio.open(path) as dataset:
            assert dataset.count == 3
            assert dataset.width == 16
            assert dataset.height == 16


def test_markup_export_uses_only_positive_features_from_per_image_dataset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    dataset_root = config.mlmarkup_root / "Реки" / "test"
    dataset_root.mkdir(parents=True)
    first_image = config.images_root / "kanopus" / "region_a" / "scene_a.tif"
    second_image = config.images_root / "kanopus" / "region_b" / "scene_b.tif"
    _write_cog(
        first_image,
        left=0,
        top=64,
        valid_slice=(slice(8, 56), slice(8, 56)),
    )
    _write_cog(
        second_image,
        left=100,
        top=64,
        valid_slice=(slice(8, 56), slice(8, 56)),
    )
    _write_per_image_geojson(
        dataset_root / "region_a_scene_a.geojson",
        [
            (1, box(11, 49, 13, 51), "positive"),
            (2, box(17, 49, 19, 51), "hard_negative"),
        ],
    )
    _write_per_image_geojson(
        dataset_root / "region_b_scene_b.geojson",
        [(3, box(111, 49, 113, 51), "positive")],
    )

    info = _markup_export.build_markup_export(
        MarkupExportRequest(
            dataset_key="Реки\\test",
            tile_width=16,
            tile_height=16,
            image_count=2,
            object_count=2,
        ),
        config,
    )

    assert info.actual_object_count == 2
    assert info.image_count == 2
    output_root = config.scratch_root / _markup_export.EXPORT_ROOT_NAME / str(info.id)
    features = [
        feature
        for path in output_root.glob("tile_*.geojson")
        for feature in json.loads(path.read_text(encoding="utf-8"))["features"]
    ]
    assert {feature["id"] for feature in features} == {1, 3}
    assert all(
        feature["properties"]["_mlsystem2_role"] == "positive"
        for feature in features
    )


def test_markup_export_preview_uses_two_pixel_yellow_contour() -> None:
    image = np.full((3, 9, 9), 100, dtype=np.uint8)
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[1:8, 1:8] = 255

    preview = _markup_export._overlay_image(image, mask)
    yellow = np.all(preview == np.asarray([255, 255, 0]), axis=2)

    assert int(yellow.sum()) == 40
    assert yellow[1, 1]
    assert yellow[2, 2]
    assert not yellow[3, 3]
    assert not bool(np.any(np.all(preview == np.asarray([255, 0, 0]), axis=2)))


def test_test_sample_jpeg_previews_keep_dimensions_channels_and_markup() -> None:
    size = 128
    rows, columns = np.mgrid[:size, :size]
    image = np.stack(
        [
            columns * 2 + 1,
            rows * 2 + 1,
            np.clip(columns + rows + 1, 1, 255),
            np.clip(255 - columns + rows // 2, 1, 255),
        ],
        axis=0,
    ).astype(np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[24:104, 24:104] = 255

    previews = _test_samples._test_sample_jpeg_previews(
        image,
        mask,
        tile_name="tile001",
    )

    assert set(previews) == {
        "rgb",
        "rgb_markup",
        "nrg",
        "nrg_markup",
        "ngb",
        "ngb_markup",
    }
    stretched = [_markup_export._stretch_channel(channel) for channel in image]
    expected_channels = {
        "rgb": (0, 1, 2),
        "nrg": (3, 0, 1),
        "ngb": (3, 1, 2),
    }
    decoded: dict[str, np.ndarray] = {}
    for name, payload in previews.items():
        assert len(payload) <= 300 * 1024
        with Image.open(BytesIO(payload)) as preview:
            assert preview.format == "JPEG"
            assert preview.size == (size, size)
            assert preview.info.get("progressive") == 1
            decoded[name] = np.asarray(preview.convert("RGB"))

    for name, indices in expected_channels.items():
        expected = np.stack([stretched[index] for index in indices], axis=2)
        mean_error = np.abs(decoded[name].astype(int) - expected.astype(int)).mean()
        assert mean_error < 8.0

    edge = _markup_export._mask_edge(mask)
    yellow = np.asarray([255, 255, 0])
    edge_error = np.abs(decoded["rgb_markup"][edge].astype(int) - yellow).mean()
    assert edge_error < 55.0
    unchanged_error = np.abs(
        decoded["rgb_markup"][:12, :12].astype(int)
        - decoded["rgb"][:12, :12].astype(int)
    ).mean()
    assert unchanged_error < 4.0


def test_test_sample_jpeg_encoder_enforces_hard_size_limit(monkeypatch) -> None:
    random_image = np.random.default_rng(7).integers(
        0,
        256,
        size=(1024, 1024, 3),
        dtype=np.uint8,
    )
    payload = _test_samples._encode_test_sample_jpeg(
        random_image,
        tile_name="tile001",
        preview_name="rgb",
    )
    assert len(payload) <= 300 * 1024

    monkeypatch.setattr(_test_samples, "_JPEG_PREVIEW_MAX_BYTES", 1)
    with pytest.raises(TrainingUIAPIError, match="не помещается в 300 КБ"):
        _test_samples._encode_test_sample_jpeg(
            random_image[:16, :16],
            tile_name="tile001",
            preview_name="rgb",
        )


def test_test_sample_jpeg_previews_for_rgb_have_no_nir_compositions() -> None:
    previews = _test_samples._test_sample_jpeg_previews(
        np.zeros((3, 16, 16), dtype=np.uint8),
        np.zeros((16, 16), dtype=np.uint8),
        tile_name="tile001",
    )

    assert set(previews) == {"rgb", "rgb_markup"}


def test_test_sample_jpeg_previews_reject_unsupported_channel_count() -> None:
    with pytest.raises(TrainingUIAPIError, match="найдено каналов: 2"):
        _test_samples._test_sample_jpeg_previews(
            np.zeros((2, 16, 16), dtype=np.uint8),
            np.zeros((16, 16), dtype=np.uint8),
            tile_name="tile001",
        )


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


def test_markup_export_counts_long_object_once_per_selected_tile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    dataset_root = config.mlmarkup_root / "Реки" / "main"
    dataset_root.mkdir(parents=True)
    (dataset_root / "scenes.txt").write_text("region_river\n", encoding="utf-8")
    _write_geojson(
        dataset_root / "rivers.geojson",
        [(1, box(2, 24, 62, 40), "river")],
    )
    _write_cog(
        config.images_root / "kanopus" / "region_river" / "river.tif",
        left=0,
        top=64,
        valid_slice=(slice(0, 64), slice(0, 64)),
    )

    info = _markup_export.build_markup_export(
        MarkupExportRequest(
            dataset_key="Реки\\main",
            tile_width=16,
            tile_height=16,
            image_count=2,
            object_count=2,
        ),
        config,
    )

    assert info.actual_object_count == 2
    assert [tile.object_count for tile in info.tiles] == [1, 1]
    output_root = config.scratch_root / _markup_export.EXPORT_ROOT_NAME / str(info.id)
    footprints = []
    for tif_path in sorted(output_root.glob("*.tif")):
        with rasterio.open(tif_path) as raster:
            footprints.append(box(*raster.bounds))
        payload = json.loads(tif_path.with_suffix(".geojson").read_text(encoding="utf-8"))
        assert [feature["id"] for feature in payload["features"]] == [1]
    assert not footprints[0].intersects(footprints[1])


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


def test_candidate_selection_counts_one_long_object_in_separate_tiles() -> None:
    crs = CRS.from_epsg(3857)
    candidates = [
        _markup_export._Candidate(
            source_path=Path("river.tif"),
            source_name="region/river.tif",
            territory="region",
            column=index * 32,
            row=0,
            raster_crs=crs,
            raster_footprint=box(index * 32, 0, index * 32 + 16, 16),
            annotation_footprint=box(index * 32, 0, index * 32 + 16, 16),
            feature_positions=(0,),
        )
        for index in range(2)
    ]
    request = MarkupExportRequest(
        dataset_key="Реки\\main",
        tile_width=16,
        tile_height=16,
        image_count=2,
        object_count=2,
    )

    selected = _markup_export._select_candidates(
        candidates,
        request,
        allow_touching=False,
    )

    assert selected == [0, 1]
    assert sum(candidates[index].object_count for index in selected) == 2


def test_candidate_pool_accepts_final_subset_in_image_count_range() -> None:
    crs = CRS.from_epsg(3857)
    candidates = [
        _selection_candidate(
            crs,
            index=index,
            territory=f"region-{index}",
            source=f"region-{index}/scene.tif",
            count=1,
        )
        for index in range(7)
    ]
    selected = _markup_export._select_candidate_pool(
        candidates,
        min_final_image_count=5,
        max_final_image_count=10,
        min_final_object_count=5,
        max_pool_count=7,
        target_pool_object_count=15,
        allow_touching=False,
    )
    impossible = _markup_export._select_candidate_pool(
        candidates,
        min_final_image_count=8,
        max_final_image_count=10,
        min_final_object_count=5,
        max_pool_count=7,
        target_pool_object_count=15,
        allow_touching=False,
    )

    assert selected == list(range(7))
    assert impossible is None


def test_milp_uses_one_minute_default_time_limit(monkeypatch) -> None:
    captured_options: dict[str, object] = {}

    def fake_milp(*args, **kwargs):
        del args
        captured_options.update(kwargs["options"])
        return SimpleNamespace(status=0, success=True, x=np.ones(1, dtype=float))

    monkeypatch.setattr(_markup_export, "milp", fake_milp)

    result = _markup_export._run_milp(
        np.zeros(1, dtype=float),
        integrality=np.ones(1, dtype=int),
        bounds=_markup_export.Bounds(np.zeros(1), np.ones(1)),
        constraints=[],
    )

    assert result is not None
    assert result.tolist() == [1.0]
    assert captured_options == {"presolve": True, "time_limit": 60.0}


def test_milp_accepts_only_valid_incumbent_on_time_limit(monkeypatch) -> None:
    constraint = _markup_export._single_constraint(
        2,
        [(0, 1.0), (1, 1.0)],
        minimum=1.0,
        maximum=1.0,
    )

    monkeypatch.setattr(
        _markup_export,
        "milp",
        lambda *args, **kwargs: SimpleNamespace(
            status=1,
            success=False,
            x=np.asarray([1.0, 0.0]),
        ),
    )
    result = _markup_export._run_milp(
        np.zeros(2, dtype=float),
        integrality=np.ones(2, dtype=int),
        bounds=_markup_export.Bounds(np.zeros(2), np.ones(2)),
        constraints=[constraint],
        accept_feasible_on_time_limit=True,
    )

    assert result is not None
    assert result.tolist() == [1.0, 0.0]

    monkeypatch.setattr(
        _markup_export,
        "milp",
        lambda *args, **kwargs: SimpleNamespace(
            status=1,
            success=False,
            x=np.asarray([1.0, 1.0]),
        ),
    )
    with pytest.raises(_markup_export._MilpTimeLimitError):
        _markup_export._run_milp(
            np.zeros(2, dtype=float),
            integrality=np.ones(2, dtype=int),
            bounds=_markup_export.Bounds(np.zeros(2), np.ones(2)),
            constraints=[constraint],
            accept_feasible_on_time_limit=True,
        )


def test_candidate_pool_uses_greedy_seed_without_large_milp(monkeypatch) -> None:
    crs = CRS.from_epsg(3857)
    candidates = [
        _selection_candidate(
            crs,
            index=index,
            territory=f"region-{index}",
            source=f"region-{index}/scene.tif",
            count=1,
        )
        for index in range(7)
    ]
    monkeypatch.setattr(
        _markup_export,
        "_run_milp",
        lambda *args, **kwargs: pytest.fail("MILP не должен запускаться"),
    )
    selected = _markup_export._select_candidate_pool(
        candidates,
        min_final_image_count=5,
        max_final_image_count=7,
        min_final_object_count=5,
        max_pool_count=7,
        target_pool_object_count=15,
        allow_touching=False,
    )

    assert selected == list(range(7))


def test_candidate_pool_greedy_seed_avoids_blocking_dense_tile(monkeypatch) -> None:
    crs = CRS.from_epsg(3857)
    candidates = [
        _selection_candidate_with_footprint(
            crs,
            index=0,
            count=100,
            footprint=box(0, 0, 40, 10),
        ),
        _selection_candidate_with_footprint(
            crs,
            index=1,
            count=60,
            footprint=box(0, 0, 10, 10),
        ),
        _selection_candidate_with_footprint(
            crs,
            index=2,
            count=60,
            footprint=box(30, 0, 40, 10),
        ),
    ]
    monkeypatch.setattr(
        _markup_export,
        "_run_milp",
        lambda *args, **kwargs: pytest.fail("MILP не должен запускаться"),
    )

    selected = _markup_export._select_candidate_pool(
        candidates,
        min_final_image_count=2,
        max_final_image_count=2,
        min_final_object_count=120,
        max_pool_count=2,
        target_pool_object_count=360,
        allow_touching=True,
    )

    assert selected == [1, 2]


def test_candidate_pool_uses_small_milp_when_greedy_seed_is_missing(
    monkeypatch,
) -> None:
    crs = CRS.from_epsg(3857)
    candidates = [
        _selection_candidate(
            crs,
            index=index,
            territory=f"region-{index}",
            source=f"region-{index}/scene.tif",
            count=1,
        )
        for index in range(7)
    ]
    monkeypatch.setattr(_markup_export, "_greedy_final_subset", lambda *args, **kwargs: None)
    selected = _markup_export._select_candidate_pool(
        candidates,
        min_final_image_count=5,
        max_final_image_count=7,
        min_final_object_count=5,
        max_pool_count=7,
        target_pool_object_count=15,
        allow_touching=False,
    )

    assert selected == list(range(7))


def test_direct_candidate_selection_still_reports_milp_timeout(monkeypatch) -> None:
    crs = CRS.from_epsg(3857)
    candidates = [
        _selection_candidate(
            crs,
            index=index,
            territory=f"region-{index}",
            source=f"region-{index}/scene.tif",
            count=1,
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
    monkeypatch.setattr(
        _markup_export,
        "_run_milp",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            _markup_export._MilpTimeLimitError("Диагностический тайм-аут.")
        ),
    )

    with pytest.raises(_markup_export._MilpTimeLimitError):
        _markup_export._select_candidates(
            candidates,
            request,
            allow_touching=False,
        )


def test_achievable_object_maximum_respects_tile_conflicts() -> None:
    crs = CRS.from_epsg(3857)
    candidates = [
        _selection_candidate_with_footprint(
            crs,
            index=0,
            count=100,
            footprint=box(0, 0, 40, 10),
        ),
        _selection_candidate_with_footprint(
            crs,
            index=1,
            count=60,
            footprint=box(0, 0, 10, 10),
        ),
        _selection_candidate_with_footprint(
            crs,
            index=2,
            count=60,
            footprint=box(30, 0, 40, 10),
        ),
    ]

    maximum = _markup_export._maximum_achievable_object_count(
        candidates,
        max_image_count=2,
        allow_touching=True,
    )

    assert maximum == 120
    assert sum(sorted((item.object_count for item in candidates), reverse=True)[:2]) == 160


def test_impossible_pool_error_reports_achievable_object_maximum(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_export_dataset(config.mlmarkup_root, config.images_root)

    with pytest.raises(
        TrainingUIAPIError,
        match=r"достижимый максимум объектов при не более чем 2 тайлах — 4",
    ):
        _markup_export.generate_markup_pool_files(
            dataset_key="Вырубки\\main",
            tile_size=16,
            min_final_image_count=1,
            max_final_image_count=2,
            min_object_count=1000,
            config=config,
            output_root=tmp_path / "pool",
        )


def test_projection_repairs_production_polygon_before_intersection() -> None:
    coordinates = [
        (37.493821659955344, 47.06105450614723),
        (37.493832865467674, 47.061050913957054),
        (37.493829569728746, 47.061047770790445),
        (37.49384670757115, 47.06103340202644),
        (37.49385066245785, 47.061019931306674),
        (37.49384077524108, 47.0610064605835),
        (37.493826273989825, 47.06100825668012),
        (37.493822978250904, 47.06100107229328),
        (37.49381111359079, 47.06100152131748),
        (37.493811772738574, 47.0609907447355),
        (37.49380056722624, 47.06099119375978),
        (37.493798589782884, 47.060984458395),
        (37.4937893617139, 47.060981764248844),
        (37.4937306975611, 47.06100601155933),
        (37.493729379265524, 47.061005113510994),
        (37.49373135670888, 47.06100241936588),
        (37.49372344693547, 47.06100331741427),
        (37.493722787787675, 47.06101005277668),
        (37.49370301335415, 47.061019482282624),
        (37.49370433164972, 47.061020380330724),
        (37.49370762738864, 47.06101903325857),
        (37.4936964218763, 47.061031156906715),
        (37.49370564994529, 47.06103834128951),
        (37.493700376763016, 47.06104283152825),
        (37.49370828653642, 47.06104328055211),
        (37.49370630909307, 47.06104866883806),
        (37.49372344693547, 47.06105809833718),
        (37.49371883290098, 47.06106977295282),
        (37.49372740182217, 47.06107291611813),
        (37.4937260835266, 47.06108234561295),
        (37.49373794818672, 47.06107965147175),
        (37.49373728903894, 47.06108414170701),
        (37.49374124392564, 47.061080549518834),
        (37.493743221368995, 47.06109357119986),
        (37.49376826898481, 47.06109267315299),
        (37.49377090557594, 47.06110030655086),
        (37.49376563239367, 47.06108503975402),
        (37.49378606597498, 47.0610792024482),
        (37.493791339157255, 47.0610715690473),
        (37.4937854068272, 47.06106887490555),
        (37.49380122637402, 47.06106573174001),
        (37.49380386296516, 47.061072467094526),
        (37.493822978250904, 47.061063037597954),
        (37.493821659955344, 47.06105450614723),
    ]
    source = Polygon(coordinates)
    source_crs = CRS.from_epsg(4326)
    target_crs = CRS.from_epsg(3857)
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    uncorrected = transform_geometry(transformer.transform, source)

    assert source.is_valid
    assert not uncorrected.is_valid
    repaired = _markup_export._transform_between_crs(
        source,
        source_crs,
        target_crs,
    )
    assert repaired.is_valid
    assert repaired.geom_type == "MultiPolygon"
    clipped = _test_samples._geometries_for_tile(
        [source],
        source_crs,
        target_crs,
        box(*uncorrected.bounds),
    )
    assert len(clipped) == 1
    assert clipped[0].is_valid
    assert clipped[0].area > 0.0


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


def test_scene_list_export_finds_recursive_unicode_scenes_and_excludes_touching(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_cog(
        config.images_root / "kanopus" / "регион_а" / "Яблоня.tif",
        left=0,
        top=64,
        valid_slice=(slice(0, 64), slice(0, 64)),
    )
    _write_cog(
        config.images_root / "kanopus" / "регион_б" / "Берёза.TIFF",
        left=100,
        top=64,
        valid_slice=(slice(0, 64), slice(0, 64)),
    )
    _write_cog(
        config.images_root / "kanopus" / "регион_в" / "Только_касание.tif",
        left=200,
        top=64,
        valid_slice=(slice(0, 64), slice(0, 64)),
    )
    _write_cog(
        config.images_root / "orto" / "регион" / "Ортофото.tif",
        left=0,
        top=64,
        valid_slice=(slice(0, 64), slice(0, 64)),
    )
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    geojson = _geojson_bytes(
        [
            transform_geometry(to_wgs84.transform, box(10, 10, 20, 20)),
            transform_geometry(to_wgs84.transform, box(110, 10, 120, 20)),
        ],
        crs=None,
    )

    artifact = _markup_export.build_scene_list_export(
        imagery_type=ImageryType.KANOPUS,
        geojson_filename=r"C:\fakepath\Разметка рек.geojson",
        geojson_bytes=geojson,
        config=config,
    )

    assert artifact.filename == "Разметка рек.txt"
    assert artifact.scene_count == 2
    assert artifact.content.decode("utf-8") == (
        "регион_а/Яблоня\n"
        "регион_б/Берёза\n"
    )
    assert artifact.footprints_filename == "Разметка рек_футпринты.geojson"
    footprints = json.loads(artifact.footprints_content)
    assert footprints["type"] == "FeatureCollection"
    assert [
        feature["properties"]["scene_id"] for feature in footprints["features"]
    ] == ["регион_а/Яблоня", "регион_б/Берёза"]
    assert all(
        feature["properties"]["imagery_type"] == "kanopus"
        for feature in footprints["features"]
    )
    expected_first = transform_geometry(to_wgs84.transform, box(0, 0, 64, 64))
    assert shape(footprints["features"][0]["geometry"]).bounds == pytest.approx(
        expected_first.bounds
    )
    with zipfile.ZipFile(BytesIO(artifact.archive_content)) as archive:
        assert archive.namelist() == [
            "Разметка рек.txt",
            "Разметка рек_футпринты.geojson",
        ]
        assert archive.read("Разметка рек.txt") == artifact.content
        assert (
            archive.read("Разметка рек_футпринты.geojson")
            == artifact.footprints_content
        )

    touching_artifact = _markup_export.build_scene_list_export(
        imagery_type=ImageryType.KANOPUS,
        geojson_filename="Касание.geojson",
        geojson_bytes=_geojson_bytes([box(190, 10, 200, 20)]),
        config=config,
    )
    assert touching_artifact.content == b""


def test_scene_list_export_ignores_objects_only_in_nodata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_cog(
        config.images_root / "kanopus" / "регион" / "только_nodata.tif",
        left=0,
        top=64,
        valid_slice=(slice(0, 32), slice(0, 32)),
    )
    _write_cog(
        config.images_root / "kanopus" / "регион" / "валидное_пересечение.tif",
        left=100,
        top=64,
        valid_slice=(slice(0, 32), slice(0, 32)),
    )

    artifact = _markup_export.build_scene_list_export(
        imagery_type=ImageryType.KANOPUS,
        geojson_filename="реки.geojson",
        geojson_bytes=_geojson_bytes(
            [
                box(40, 40, 50, 50),
                box(110, 40, 120, 50),
            ]
        ),
        config=config,
    )

    assert artifact.scene_count == 1
    assert artifact.content.decode("utf-8") == "регион/валидное_пересечение\n"


def test_clip_geometries_to_valid_data_uses_native_pixel_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "точная_маска.tif"
    _write_cog(
        image_path,
        left=0,
        top=64,
        valid_slice=(slice(16, 48), slice(8, 40)),
    )

    with rasterio.open(image_path) as dataset:
        clipped = clip_geometries_to_valid_data(dataset, (box(0, 0, 64, 64),))

    assert len(clipped) == 1
    assert clipped[0].equals(box(8, 16, 40, 48))


def test_scene_list_export_returns_empty_txt_without_matches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_cog(
        config.images_root / "orto" / "область" / "Орто сцена.tif",
        left=0,
        top=64,
        valid_slice=(slice(0, 64), slice(0, 64)),
    )

    artifact = _markup_export.build_scene_list_export(
        imagery_type=ImageryType.ORTHO,
        geojson_filename="Пустая выборка.geojson",
        geojson_bytes=_geojson_bytes([box(500, 500, 510, 510)]),
        config=config,
    )

    assert artifact.filename == "Пустая выборка.txt"
    assert artifact.scene_count == 0
    assert artifact.content == b""
    assert json.loads(artifact.footprints_content)["features"] == []


def test_scene_list_export_disambiguates_duplicate_stems_with_relative_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_cog(
        config.images_root / "kanopus" / "первая" / "Одинаковая.tif",
        left=0,
        top=64,
        valid_slice=(slice(0, 64), slice(0, 64)),
    )
    _write_cog(
        config.images_root / "kanopus" / "вторая" / "одинаковая.tiff",
        left=100,
        top=64,
        valid_slice=(slice(0, 64), slice(0, 64)),
    )

    artifact = _markup_export.build_scene_list_export(
        imagery_type=ImageryType.KANOPUS,
        geojson_filename="разметка.geojson",
        geojson_bytes=_geojson_bytes(
            [box(10, 10, 20, 20), box(110, 10, 120, 20)]
        ),
        config=config,
    )

    assert artifact.content.decode("utf-8") == (
        "вторая/одинаковая\n"
        "первая/Одинаковая\n"
    )


def test_scene_list_export_is_accepted_by_training_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    images_root = config.images_root / "kanopus"
    duplicate_name = "Канопус PMS.SCN02.tif"
    _write_cog(
        images_root / "Качугский" / duplicate_name,
        left=100,
        top=64,
        valid_slice=(slice(0, 64), slice(0, 64)),
    )
    selected = images_root / "Ольхонский район" / duplicate_name
    _write_cog(
        selected,
        left=0,
        top=64,
        valid_slice=(slice(0, 64), slice(0, 64)),
    )
    geojson = _geojson_bytes([box(10, 10, 20, 20)])
    artifact = _markup_export.build_scene_list_export(
        imagery_type=ImageryType.KANOPUS,
        geojson_filename="Ветровая эрозия.geojson",
        geojson_bytes=geojson,
        config=config,
    )
    scenes_file = tmp_path / artifact.filename
    scenes_file.write_bytes(artifact.content)
    annotation_file = tmp_path / "Ветровая эрозия.geojson"
    annotation_file.write_bytes(geojson)

    resolution = resolve_scene_images(
        SceneImageResolutionRequest(
            images_dir=str(images_root),
            scenes_file=str(scenes_file),
        )
    )
    prepared = prepare_dataset(
        DatasetPreparationRequest(
            images_dir=str(images_root),
            scenes_file=str(scenes_file),
            annotation_file=str(annotation_file),
            val_fraction=0.5,
            expected_band_count=4,
            expected_dtype="uint8",
        )
    )

    assert artifact.content.decode("utf-8") == (
        "Ольхонский район/Канопус PMS.SCN02\n"
    )
    assert [Path(item.image_path) for item in resolution.images] == [selected]
    assert prepared.report.status == "ok"
    assert prepared.dataset is not None
    assert prepared.report.scenes[0].image_path == selected.resolve().as_posix()


def test_scene_list_export_rejects_duplicate_relative_path_without_extension(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_cog(
        config.images_root / "kanopus" / "регион" / "Одинаковая.tif",
        left=0,
        top=64,
        valid_slice=(slice(0, 64), slice(0, 64)),
    )
    _write_cog(
        config.images_root / "kanopus" / "регион" / "Одинаковая.tiff",
        left=100,
        top=64,
        valid_slice=(slice(0, 64), slice(0, 64)),
    )

    with pytest.raises(TrainingUIAPIError, match="совпадают относительные пути"):
        _markup_export.build_scene_list_export(
            imagery_type=ImageryType.KANOPUS,
            geojson_filename="разметка.geojson",
            geojson_bytes=_geojson_bytes([box(10, 10, 20, 20)]),
            config=config,
        )


def test_scene_list_export_validates_input_root_and_rasters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    valid_geojson = _geojson_bytes([box(10, 10, 20, 20)])

    with pytest.raises(TrainingUIAPIError, match="Папка снимков"):
        _markup_export.build_scene_list_export(
            imagery_type=ImageryType.KANOPUS,
            geojson_filename="разметка.geojson",
            geojson_bytes=valid_geojson,
            config=config,
        )

    images_root = config.images_root / "kanopus"
    images_root.mkdir(parents=True)
    damaged_path = images_root / "повреждённый.tif"
    damaged_path.write_bytes(b"not-a-raster")
    with pytest.raises(TrainingUIAPIError, match="Не удалось прочитать снимок"):
        _markup_export.build_scene_list_export(
            imagery_type=ImageryType.KANOPUS,
            geojson_filename="разметка.geojson",
            geojson_bytes=valid_geojson,
            config=config,
        )

    damaged_path.unlink()
    without_crs_path = images_root / "без_crs.tif"
    with rasterio.open(
        without_crs_path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=1,
        dtype="uint8",
        transform=from_origin(0, 8, 1, 1),
    ) as dataset:
        dataset.write(np.ones((1, 8, 8), dtype=np.uint8))
    with pytest.raises(TrainingUIAPIError, match="отсутствует CRS"):
        _markup_export.build_scene_list_export(
            imagery_type=ImageryType.KANOPUS,
            geojson_filename="разметка.geojson",
            geojson_bytes=valid_geojson,
            config=config,
        )

    with pytest.raises(TrainingUIAPIError, match="загруженный GeoJSON"):
        _markup_export.build_scene_list_export(
            imagery_type=ImageryType.KANOPUS,
            geojson_filename="разметка.geojson",
            geojson_bytes=b"not-json",
            config=config,
        )
    with pytest.raises(TrainingUIAPIError, match="расширением .geojson"):
        _markup_export.build_scene_list_export(
            imagery_type=ImageryType.KANOPUS,
            geojson_filename="разметка.json",
            geojson_bytes=valid_geojson,
            config=config,
        )


def test_scene_list_export_http_downloads_unicode_filename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_cog(
        config.images_root / "kanopus" / "регион" / "Сцена один.tif",
        left=0,
        top=64,
        valid_slice=(slice(0, 64), slice(0, 64)),
    )
    matching_geojson = _geojson_bytes([box(10, 10, 20, 20)])
    empty_geojson = _geojson_bytes([box(500, 500, 510, 510)])

    with TestClient(create_app()) as client:
        assert client.post(
            "/api/v1/scene-list-export",
            data={"imagery_type": "kanopus"},
            files={"geojson": ("Разметка рек.geojson", matching_geojson)},
        ).status_code == 401
        _login(client)
        openapi = client.get("/openapi.json").json()
        response_content = openapi["paths"]["/api/v1/scene-list-export"]["post"][
            "responses"
        ]["200"]["content"]
        assert "text/plain" in response_content
        assert "application/zip" in response_content

        response = client.post(
            "/api/v1/scene-list-export",
            data={"imagery_type": "kanopus"},
            files={
                "geojson": (
                    "Разметка рек.geojson",
                    matching_geojson,
                    "application/geo+json",
                )
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert response.content.decode("utf-8") == "регион/Сцена один\n"
        assert "Разметка рек.txt" in unquote(response.headers["content-disposition"])

        archive_response = client.post(
            "/api/v1/scene-list-export",
            data={"imagery_type": "kanopus", "include_footprints": "true"},
            files={
                "geojson": (
                    "Разметка рек.geojson",
                    matching_geojson,
                    "application/geo+json",
                )
            },
        )
        assert archive_response.status_code == 200
        assert archive_response.headers["content-type"] == "application/zip"
        assert "Разметка рек.zip" in unquote(
            archive_response.headers["content-disposition"]
        )
        with zipfile.ZipFile(BytesIO(archive_response.content)) as archive:
            assert archive.namelist() == [
                "Разметка рек.txt",
                "Разметка рек_футпринты.geojson",
            ]
            assert archive.read("Разметка рек.txt").decode("utf-8") == (
                "регион/Сцена один\n"
            )
            footprint_payload = json.loads(
                archive.read("Разметка рек_футпринты.geojson")
            )
            assert footprint_payload["features"][0]["properties"] == {
                "scene_id": "регион/Сцена один",
                "relative_path": "регион/Сцена один.tif",
                "filename": "Сцена один.tif",
                "imagery_type": "kanopus",
            }

        empty_response = client.post(
            "/api/v1/scene-list-export",
            data={"imagery_type": "kanopus"},
            files={"geojson": ("Пусто.geojson", empty_geojson)},
        )
        assert empty_response.status_code == 200
        assert empty_response.content == b""
        assert "Пусто.txt" in unquote(empty_response.headers["content-disposition"])

        invalid_response = client.post(
            "/api/v1/scene-list-export",
            data={"imagery_type": "kanopus"},
            files={"geojson": ("ошибка.geojson", b"not-json")},
        )
        assert invalid_response.status_code == 400


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
        content_disposition = unquote(download.headers["content-disposition"])
        assert "вырубки_test_markup.zip" in content_disposition.casefold()
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


def test_persistent_test_sample_http_catalog_editor_and_delete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_export_dataset(config.mlmarkup_root, config.images_root)

    with TestClient(create_app()) as client:
        assert client.get("/api/v1/test-samples").status_code == 401
        assert client.post("/api/v1/test-samples", json={}).status_code == 401
        assert client.post(
            "/api/v1/test-samples/00000000-0000-0000-0000-000000000000/optimize",
            json={
                "min_tile_count": 1,
                "max_tile_count": 1,
                "min_object_count": 1,
                "metric": "objects",
            },
        ).status_code == 401
        _login(client)
        openapi = client.get("/openapi.json").json()
        assert "TestSampleDetail" in openapi["components"]["schemas"]
        assert "f1_score" in openapi["components"]["schemas"]["TestSampleTileInfo"][
            "properties"
        ]
        assert "TestSampleDraftPreview" in openapi["components"]["schemas"]
        assert "/api/v1/test-samples/reconcile" in openapi["paths"]
        assert "/api/v1/test-samples/{sample_id}/evaluate" in openapi["paths"]
        assert "/api/v1/test-samples/{sample_id}/pseudo-markup" in openapi["paths"]
        assert "/api/v1/test-samples/{sample_id}/evaluate-preview" in openapi["paths"]
        assert "/api/v1/test-samples/{sample_id}/optimize" in openapi["paths"]
        assert "/api/v1/test-samples/{sample_id}/optimize-preview" in openapi["paths"]
        assert "post" in openapi["paths"][
            "/api/v1/test-samples/{sample_id}/download"
        ]
        assert "post" in openapi["paths"]["/api/v1/test-samples/download"]
        assert "/api/v1/test-samples/primary/download" not in openapi["paths"]
        assert "TestSampleBulkDownloadRequest" in openapi["components"]["schemas"]
        assert "не более одной разметки" in (
            openapi["components"]["schemas"]["TestSampleBulkDownloadRequest"][
                "properties"
            ]["sample_ids"]["description"]
        )
        assert (
            openapi["components"]["schemas"]["TestSampleCreate"]["properties"][
                "tile_width"
            ]["default"]
            == 1536
        )
        response = client.post(
            "/api/v1/test-samples",
            json={
                "name": "Контрольная выборка",
                "dataset_key": "Вырубки\\main",
                "tile_width": 16,
                "tile_height": 16,
                "image_count": 2,
                "object_count": 4,
            },
        )
        assert response.status_code == 200
        sample = response.json()
        sample_id = sample["id"]
        assert sample["evaluation"]["status"] == "unavailable"
        assert sample["enabled_image_count"] == 2
        assert [tile["enabled"] for tile in sample["tiles"]] == [True, True]
        assert [tile["f1_score"] for tile in sample["tiles"]] == [None, None]
        draft_preview = client.post(
            f"/api/v1/test-samples/{sample_id}/evaluate-preview",
            json={"enabled_tile_indices": [2]},
        )
        assert draft_preview.status_code == 200
        assert draft_preview.json()["enabled_tile_indices"] == [2]
        assert draft_preview.json()["enabled_image_count"] == 1
        assert client.get(f"/api/v1/test-samples/{sample_id}").json()[
            "enabled_image_count"
        ] == 2
        unavailable_optimization = client.post(
            f"/api/v1/test-samples/{sample_id}/optimize",
            json={
                "min_tile_count": 1,
                "max_tile_count": 2,
                "min_object_count": 1,
                "metric": "objects",
            },
        )
        assert unavailable_optimization.status_code == 400
        unchanged = client.get(f"/api/v1/test-samples/{sample_id}").json()
        assert [tile["enabled"] for tile in unchanged["tiles"]] == [True, True]

        catalog = client.get("/api/v1/test-samples").json()
        assert catalog["classes"][0]["name"] == "Вырубки"
        assert catalog["classes"][0]["datasets"] == []
        assert catalog["classes"][0]["samples"][0]["name"] == "Контрольная выборка"
        assert catalog["classes"][0]["samples"][0]["source_dataset_name"] == (
            "Вырубки\\main"
        )
        reconciled = client.post("/api/v1/test-samples/reconcile")
        assert reconciled.status_code == 200
        assert reconciled.json()["classes"][0]["samples"][0][
            "evaluation"
        ]["status"] == "unavailable"

        preview = client.get(sample["tiles"][0]["preview_url"])
        assert preview.status_code == 200
        assert preview.headers["content-type"] == "image/png"
        renamed = client.patch(
            f"/api/v1/test-samples/{sample_id}",
            json={"name": "Переименованная выборка"},
        ).json()
        assert renamed["name"] == "Переименованная выборка"
        toggled = client.patch(
            f"/api/v1/test-samples/{sample_id}/tiles/1",
            json={"enabled": False},
        ).json()
        assert toggled["enabled_image_count"] == 1
        assert toggled["tiles"][0]["enabled"] is False
        assert client.patch(
            f"/api/v1/test-samples/{sample_id}/tiles/99",
            json={"enabled": False},
        ).status_code == 404

        archive_response = client.get(toggled["download_url"])
        assert archive_response.status_code == 200
        with zipfile.ZipFile(BytesIO(archive_response.content)) as archive:
            names = set(archive.namelist())
            assert names == _downloaded_tile_names("tile001")
            for name in names:
                if name.endswith(".jpg"):
                    assert archive.getinfo(name).file_size <= 300 * 1024
        draft_archive_response = client.post(
            toggled["download_url"],
            json={"enabled_tile_indices": [1, 2]},
        )
        assert draft_archive_response.status_code == 200
        with zipfile.ZipFile(BytesIO(draft_archive_response.content)) as archive:
            assert set(archive.namelist()) == (
                _downloaded_tile_names("tile001")
                | _downloaded_tile_names("tile002")
            )
        draft_without_previews = client.post(
            toggled["download_url"],
            json={
                "enabled_tile_indices": [1, 2],
                "include_previews": False,
            },
        )
        assert draft_without_previews.status_code == 200
        with zipfile.ZipFile(BytesIO(draft_without_previews.content)) as archive:
            assert set(archive.namelist()) == {
                "tile001.tif",
                "tile001.geojson",
                "tile002.tif",
                "tile002.geojson",
            }
        assert client.post(
            "/api/v1/test-samples/download",
            json={"sample_ids": []},
        ).status_code == 422
        assert client.post(
            "/api/v1/test-samples/download",
            json={"sample_ids": [sample_id, sample_id]},
        ).status_code == 422
        assert client.post(
            "/api/v1/test-samples/download",
            json={"sample_ids": [str(uuid.uuid4())]},
        ).status_code == 404
        second_sample_response = client.post(
            "/api/v1/test-samples",
            json={
                "name": "Вторая выборка",
                "dataset_key": "Вырубки\\main",
                "tile_width": 16,
                "tile_height": 16,
                "image_count": 1,
                "object_count": 1,
            },
        )
        assert second_sample_response.status_code == 200
        second_sample_id = second_sample_response.json()["id"]
        assert client.put(
            f"/api/v1/test-samples/{sample_id}/primary",
            json={"is_primary": True},
        ).json()["is_primary"] is True
        assert client.put(
            f"/api/v1/test-samples/{second_sample_id}/primary",
            json={"is_primary": True},
        ).json()["is_primary"] is True
        assert client.get(f"/api/v1/test-samples/{sample_id}").json()["is_primary"] is False
        duplicate_class_response = client.post(
            "/api/v1/test-samples/download",
            json={"sample_ids": [sample_id, second_sample_id]},
        )
        assert duplicate_class_response.status_code == 400
        assert "не более одной разметки" in duplicate_class_response.json()["detail"]
        assert client.delete(
            f"/api/v1/test-samples/{second_sample_id}"
        ).status_code == 204
        bulk_without_previews = client.post(
            "/api/v1/test-samples/download",
            json={
                "sample_ids": [sample_id],
                "include_previews": False,
            },
        )
        assert bulk_without_previews.status_code == 200
        with zipfile.ZipFile(BytesIO(bulk_without_previews.content)) as archive:
            assert set(archive.namelist()) == {
                "Вырубки_main/tile001.tif",
                "Вырубки_main/tile001.geojson",
            }
            assert all(
                info.compress_type == zipfile.ZIP_STORED
                for info in archive.infolist()
            )
        persisted_after_download = client.get(
            f"/api/v1/test-samples/{sample_id}"
        ).json()
        assert persisted_after_download["enabled_image_count"] == 1
        assert [tile["enabled"] for tile in persisted_after_download["tiles"]] == [
            False,
            True,
        ]
        assert client.post(
            toggled["download_url"],
            json={"enabled_tile_indices": [1, 1]},
        ).status_code == 400
        disabled = client.patch(
            f"/api/v1/test-samples/{sample_id}/tiles/2",
            json={"enabled": False},
        ).json()
        assert disabled["enabled_image_count"] == 0
        assert client.get(disabled["download_url"]).status_code == 400
        unavailable = client.post(
            f"/api/v1/test-samples/{sample_id}/evaluate"
        ).json()
        assert unavailable["evaluation"]["status"] == "unavailable"
        client.patch(
            f"/api/v1/test-samples/{sample_id}/tiles/2",
            json={"enabled": True},
        ).raise_for_status()

    with TestClient(create_app()) as client:
        _login(client)
        persisted = client.get(f"/api/v1/test-samples/{sample_id}")
        assert persisted.status_code == 200
        assert persisted.json()["name"] == "Переименованная выборка"
        assert persisted.json()["tiles"][0]["enabled"] is False
        sample_root = (
            config.stored_files_root
            / "test-samples"
            / sample_id
        )
        assert sample_root.is_dir()
        assert client.delete(f"/api/v1/test-samples/{sample_id}").status_code == 204
        assert not sample_root.exists()
        assert client.get(f"/api/v1/test-samples/{sample_id}").status_code == 404


def test_test_sample_batch_latest_preserves_next_form_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_export_dataset(config.mlmarkup_root, config.images_root)

    with TestClient(create_app()) as client:
        _login(client)
        created = client.post(
            "/api/v1/test-sample-batches",
            json={
                "tile_size": 2048,
                "min_image_count": 5,
                "image_count": 7,
                "items": [
                    {
                        "dataset_key": "Вырубки\\main",
                        "min_object_count": 41,
                    }
                ],
            },
        )
        assert created.status_code == 200

        latest = client.get("/api/v1/test-sample-batches/latest")

        assert latest.status_code == 200
        payload = latest.json()
        assert payload["id"] == created.json()["id"]
        assert payload["tile_size"] == 2048
        assert payload["min_image_count"] == 5
        assert payload["image_count"] == 7
        assert payload["items"][0]["min_object_count"] == 41
        assert payload["items"][0]["metric"] == "pixel"


def test_test_sample_batch_request_keeps_exact_legacy_image_count() -> None:
    request = _TestSampleBatchCreate(
        image_count=3,
        items=[{"dataset_key": "Вырубки\\main"}],
    )

    assert request.min_image_count == 3
    assert request.items[0].metric == "pixel"

    with pytest.raises(ValueError, match="Минимальное число снимков"):
        _TestSampleBatchCreate(
            min_image_count=4,
            image_count=3,
            items=[{"dataset_key": "Вырубки\\main"}],
        )


@pytest.mark.parametrize("tile_size", [2560, 3072, 3584])
def test_test_sample_batch_request_accepts_large_tile_sizes(tile_size: int) -> None:
    request = _TestSampleBatchCreate(
        tile_size=tile_size,
        image_count=3,
        items=[{"dataset_key": "Вырубки\\main"}],
    )

    assert request.tile_size == tile_size


def test_test_sample_batch_request_rejects_unlisted_tile_size() -> None:
    with pytest.raises(ValueError):
        _TestSampleBatchCreate(
            tile_size=3000,
            image_count=3,
            items=[{"dataset_key": "Вырубки\\main"}],
        )


def test_persistent_test_sample_metrics_and_stale_revision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_export_dataset(config.mlmarkup_root, config.images_root)
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        sample = create_test_sample(
            session,
            _TestSampleCreate(
                name="Метрики",
                dataset_key="Вырубки\\main",
                tile_width=16,
                tile_height=16,
                image_count=2,
                object_count=4,
            ),
            config,
        )
        session.commit()
        sample_id = sample.id

    prediction_path = tmp_path / "prediction.geojson"
    _write_prediction_from_sample(config, sample_id, prediction_path)
    with session_factory() as session:
        training = TrainingResultRow(
            dataset_key="Вырубки\\main",
            class_key="Вырубки\\main",
            class_display_name="Вырубки\\main",
            architecture="segformer_b2",
            model_name="segformer b2",
            status="ok",
        )
        stored = StoredFileRow(
            kind=StoredFileKind.PSEUDO_MARKUP_GEOJSON.value,
            original_name="prediction.geojson",
            content_type="application/geo+json",
            path=str(prediction_path),
            size_bytes=prediction_path.stat().st_size,
        )
        session.add_all([training, stored])
        session.flush()
        mismatched = PseudoMarkupResultRow(
            dataset_key="Вырубки\\other",
            training_result_id=training.id,
            class_key="Вырубки\\main",
            source_dataset_name="Вырубки\\other",
            geojson_file_id=stored.id,
            status="ok",
        )
        mismatched.geojson_file = stored
        mismatched.training_result = training
        session.add(mismatched)
        session.flush()
        evaluate_test_samples_for_pseudo_markup(session, mismatched, config)
        assert _test_sample_detail(session, sample_id).evaluation.status == "unavailable"

        pseudo = PseudoMarkupResultRow(
            dataset_key="Вырубки\\main",
            training_result_id=training.id,
            class_key="Вырубки\\main",
            source_dataset_name="Вырубки\\main",
            geojson_file_id=stored.id,
            status="ok",
        )
        pseudo.geojson_file = stored
        pseudo.training_result = training
        session.add(pseudo)
        session.flush()

        evaluate_test_samples_for_pseudo_markup(session, pseudo, config)
        detail = _test_sample_detail(session, sample_id)
        assert detail.evaluation.status == "unavailable"
        assert detail.evaluation.pixel is None
        assert detail.evaluation.objects is None
        assert detail.evaluation.pseudo_markup_result_id == pseudo.id
        assert all(tile.f1_score == pytest.approx(1.0) for tile in detail.tiles)

        full_preview = evaluate_test_sample_preview(
            session,
            sample_id,
            _TestSampleEvaluationPreviewRequest(enabled_tile_indices=[1, 2]),
            config,
        )
        assert full_preview.evaluation.pixel is not None
        assert full_preview.evaluation.objects is not None
        assert full_preview.evaluation.pixel.f1 == pytest.approx(1.0)
        assert full_preview.evaluation.objects.f1 == pytest.approx(1.0)
        expected_pixels = 0
        source_root = config.stored_files_root / "test-samples" / str(sample_id)
        for mask_path in source_root.glob("tile_*_mask.png"):
            with rasterio.open(mask_path) as mask_dataset:
                expected_pixels += int((mask_dataset.read(1) > 0).sum())
        assert full_preview.evaluation.pixel.true_positive == expected_pixels
        assert full_preview.evaluation.pixel.false_positive == 0
        assert full_preview.evaluation.pixel.false_negative == 0
        assert full_preview.evaluation.objects.true_positive == detail.actual_object_count
        assert full_preview.evaluation.objects.false_positive == 0
        assert full_preview.evaluation.objects.false_negative == 0

        sample_row = session.get(_TestSampleRow, sample_id)
        assert sample_row is not None
        for tile in sample_row.tiles:
            tile.pixel_f1 = None
            tile.object_f1 = None
        session.flush()
        backfilled = _test_sample_detail(session, sample_id, config)
        assert all(tile.f1_score == pytest.approx(1.0) for tile in backfilled.tiles)

        first_tile = sample_row.tiles[0]
        first_tile.pixel_f1 = 0.25
        first_tile.object_f1 = 0.75
        sample_row.quality_metric = "objects"
        assert _test_sample_detail(session, sample_id).tiles[0].f1_score == pytest.approx(
            0.75
        )
        sample_row.quality_metric = "pixel"
        assert _test_sample_detail(session, sample_id).tiles[0].f1_score == pytest.approx(
            0.25
        )
        evaluate_test_samples_for_pseudo_markup(session, pseudo, config)

        saved_revision = sample_row.content_revision
        saved_evaluated_revision = sample_row.evaluated_revision
        saved_enabled = [tile.tile_index for tile in sample_row.tiles if tile.enabled]
        saved_tile_f1 = [
            (tile.pixel_f1, tile.object_f1)
            for tile in sample_row.tiles
        ]
        evaluation_preview = evaluate_test_sample_preview(
            session,
            sample_id,
            _TestSampleEvaluationPreviewRequest(enabled_tile_indices=[2]),
            config,
        )
        assert evaluation_preview.enabled_tile_indices == [2]
        assert evaluation_preview.enabled_image_count == 1
        assert evaluation_preview.evaluation.pixel is not None
        assert evaluation_preview.evaluation.pixel.f1 == pytest.approx(1.0)
        optimization_preview = optimize_test_sample_preview(
            session,
            sample_id,
            _TestSampleOptimizeRequest(
                min_tile_count=1,
                max_tile_count=1,
                min_object_count=1,
                metric="pixel",
            ),
            config,
        )
        assert len(optimization_preview.enabled_tile_indices) == 1
        session.flush()
        assert sample_row.content_revision == saved_revision
        assert sample_row.evaluated_revision == saved_evaluated_revision
        assert [tile.tile_index for tile in sample_row.tiles if tile.enabled] == saved_enabled
        assert [
            (tile.pixel_f1, tile.object_f1)
            for tile in sample_row.tiles
        ] == saved_tile_f1

        update_test_sample_tile(
            session,
            sample_id,
            1,
            _TestSampleTileUpdate(enabled=False),
        )
        optimized = optimize_test_sample(
            session,
            sample_id,
            _TestSampleOptimizeRequest(
                min_tile_count=2,
                max_tile_count=2,
                min_object_count=4,
                metric="objects",
            ),
            config,
        )
        assert optimized.enabled_image_count == 2
        assert all(tile.enabled for tile in optimized.tiles)
        assert optimized.evaluation.status == "unavailable"
        assert optimized.evaluation.objects is None

        detail = update_test_sample_tile(
            session,
            sample_id,
            1,
            _TestSampleTileUpdate(enabled=False),
        )
        assert detail.evaluation.status == "unavailable"
        assert detail.evaluation.pixel is None

        detail = evaluate_test_sample_by_id(session, sample_id, config)
        assert detail.evaluation.status == "unavailable"
        assert detail.evaluation.pixel is None
        assert detail.tiles[0].enabled is False
        assert all(tile.f1_score == pytest.approx(1.0) for tile in detail.tiles)

        artifact = build_test_sample_download(session, sample_id, config)
        try:
            with zipfile.ZipFile(artifact.path) as archive:
                assert set(archive.namelist()) == _downloaded_tile_names("tile001")
        finally:
            artifact.cleanup()
        artifact = build_test_sample_download(
            session,
            sample_id,
            config,
            include_previews=False,
        )
        try:
            with zipfile.ZipFile(artifact.path) as archive:
                assert set(archive.namelist()) == {
                    "tile001.tif",
                    "tile001.geojson",
                }
        finally:
            artifact.cleanup()

        mark_test_samples_stale_for_pseudo_markup(session, pseudo.id)
        detail = _test_sample_detail(session, sample_id)
        assert detail.evaluation.status == "unavailable"
        assert detail.evaluation.pseudo_markup_result_id is None


def test_object_f1_matching_uses_inclusive_half_iou() -> None:
    truth = [box(0, 0, 2, 2)]

    matched = _object_counts(truth, [box(0, 0, 1, 2)], 0.5)
    missed = _object_counts(truth, [box(0, 0, 0.99, 2)], 0.5)

    assert matched == _test_samples_metric_counts(1, 0, 0)
    assert missed == _test_samples_metric_counts(0, 1, 1)


def test_saved_test_samples_are_evaluated_by_current_primary_network(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_export_dataset(config.mlmarkup_root, config.images_root)
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        first = create_test_sample(
            session,
            _TestSampleCreate(
                name="Основная",
                dataset_key="Вырубки\\main",
                tile_width=16,
                tile_height=16,
                image_count=2,
                object_count=4,
            ),
            config,
        )
        second = create_test_sample(
            session,
            _TestSampleCreate(
                name="Другой датасет класса",
                dataset_key="Вырубки\\main",
                tile_width=16,
                tile_height=16,
                image_count=2,
                object_count=4,
            ),
            config,
        )
        second_row = session.get(_TestSampleRow, second.id)
        assert second_row is not None
        second_row.dataset_key = "Вырубки\\strict"
        second_row.dataset_name = "Вырубки\\strict"
        second_row.dataset_short_name = "strict"

        first_result = TrainingResultRow(
            source="manual",
            dataset_key="Вырубки\\main",
            class_key="Вырубки\\main",
            class_display_name="Вырубки\\main",
            architecture="segformer_b2",
            model_name="Основная сеть 1",
            mlflow_run_id="run-primary-1",
            status="ok",
        )
        session.add(first_result)
        session.flush()
        class_row = dataset_class_row(session, first.class_key)
        assert class_row is not None
        class_row.primary_training_result_id = first_result.id
        session.flush()

        pseudo_before = _test_sample_detail(session, first.id)
        assert pseudo_before.pseudo_markup.status == "unavailable"
        assert pseudo_before.pseudo_markup.can_create is True
        assert pseudo_before.pseudo_markup.training_result_id == first_result.id
        monkeypatch.setattr(_service, "_best_training_checkpoint", lambda *_args: None)
        pseudo_job = _service.ensure_test_sample_pseudo_markup_job(
            session,
            first.id,
            config,
        )
        same_pseudo_job = _service.ensure_test_sample_pseudo_markup_job(
            session,
            first.id,
            config,
        )
        assert same_pseudo_job.id == pseudo_job.id
        pseudo_pending = _test_sample_detail(session, first.id)
        assert pseudo_pending.pseudo_markup.status == "queued"
        assert pseudo_pending.pseudo_markup.job_id == pseudo_job.id

        assert reconcile_test_sample_evaluations(session, config) == 2
        assert reconcile_test_sample_evaluations(session, config) == 0
        direct_jobs = [
            job
            for job in session.scalars(select(JobRow)).all()
            if (job.config or {}).get("metric_target") == "test_sample"
        ]
        assert len(direct_jobs) == 2
        assert all(job.source == "automation" for job in direct_jobs)
        assert {job.dataset_key for job in direct_jobs} == {
            "Вырубки\\main",
            "Вырубки\\strict",
        }
        strict_job = next(job for job in direct_jobs if job.dataset_key == "Вырубки\\strict")
        monkeypatch.setattr(
            _worker,
            "_best_training_checkpoint",
            lambda _config, _run_id: SimpleNamespace(
                artifact_uri="file:///checkpoint.pt",
                artifact_path="checkpoints/best.pt",
                f1_score=0.8,
                epoch=20,
                threshold=0.5,
            ),
        )
        worker_config = _worker._build_test_sample_f1_config(
            session,
            strict_job,
            config,
            tmp_path / "strict-worker-config",
        )
        assert worker_config["metric_target"] == "test_sample"
        assert worker_config["test_sample_id"] == str(second.id)
        assert [tile["index"] for tile in worker_config["tiles"]] == [1, 2]
        queued = _test_sample_detail(session, first.id)
        assert queued.evaluation.status == "queued"
        assert queued.evaluation.training_result_id is None
        assert queued.evaluation.target_training_result_id == first_result.id
        assert queued.evaluation.target_model_name == "Основная сеть 1"
        assert queued.evaluation.target_training_dataset_key == "Вырубки\\main"
        assert queued.evaluation.target_training_dataset_name == "Вырубки\\main"
        assert queued.pseudo_markup.status == "queued"
        assert queued.pseudo_markup.training_result_id == first_result.id

        first_job = next(
            job
            for job in direct_jobs
            if (job.config or {}).get("test_sample_id") == str(first.id)
        )
        run_root = tmp_path / "direct-evaluation-1"
        (run_root / "scratch").mkdir(parents=True)
        (run_root / "scratch" / "report.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "processed": 2,
                    "threshold": 0.61,
                    "true_positive": 90,
                    "false_positive": 10,
                    "false_negative": 20,
                    "object_true_positive": 7,
                    "object_false_positive": 2,
                    "object_false_negative": 1,
                    "metrics": {"task": "binary"},
                }
            ),
            encoding="utf-8",
        )
        first_job.tmp_path = str(run_root)
        first_job.status = "running"
        _finish_test_sample_f1_job(
            session,
            first_job,
            config,
            succeeded=True,
        )
        current = _test_sample_detail(session, first.id)
        assert current.evaluation.status == "current"
        assert current.evaluation.training_result_id == first_result.id
        assert current.evaluation.model_name == "Основная сеть 1"
        assert current.evaluation.training_dataset_key == "Вырубки\\main"
        assert current.evaluation.training_dataset_name == "Вырубки\\main"
        assert current.evaluation.threshold == pytest.approx(0.61)
        assert current.evaluation.pixel is not None
        assert current.evaluation.pixel.true_positive == 90
        assert current.evaluation.objects is not None
        assert current.evaluation.objects.true_positive == 7

        second_result = TrainingResultRow(
            source="manual",
            dataset_key="Вырубки\\main",
            class_key="Вырубки\\main",
            class_display_name="Вырубки\\main",
            architecture="segformer_b2",
            model_name="Основная сеть 2",
            mlflow_run_id="run-primary-2",
            status="ok",
        )
        session.add(second_result)
        session.flush()
        class_row.primary_training_result_id = second_result.id
        session.flush()
        assert reconcile_test_sample_evaluations(
            session,
            config,
            class_keys={first.class_key},
        ) == 2
        pending = _test_sample_detail(session, first.id)
        assert pending.evaluation.status == "queued"
        assert pending.evaluation.training_result_id == first_result.id
        assert pending.evaluation.model_name == "Основная сеть 1"
        assert pending.evaluation.target_training_result_id == second_result.id
        assert pending.evaluation.target_model_name == "Основная сеть 2"

        failed_job = session.get(JobRow, pending.evaluation.job_id)
        assert failed_job is not None
        failed_root = tmp_path / "direct-evaluation-failed"
        (failed_root / "scratch").mkdir(parents=True)
        failed_job.tmp_path = str(failed_root)
        failed_job.status = "running"
        _finish_test_sample_f1_job(
            session,
            failed_job,
            config,
            succeeded=False,
        )
        failed = _test_sample_detail(session, first.id)
        assert failed.evaluation.status == "error"
        assert failed.evaluation.pixel is not None
        assert reconcile_test_sample_evaluations(
            session,
            config,
            sample_ids={first.id},
        ) == 0
        first_row = session.get(_TestSampleRow, first.id)
        assert first_row is not None
        assert queue_test_sample_evaluation(
            session,
            first_row,
            config,
            source=JobSource.MANUAL,
            force=True,
        )
        retry = _test_sample_detail(session, first.id)
        assert retry.evaluation.status == "queued"
        retry_job = session.get(JobRow, retry.evaluation.job_id)
        assert retry_job is not None and retry_job.source == "manual"
        first_row.content_revision += 1
        race_root = tmp_path / "direct-evaluation-race"
        (race_root / "scratch").mkdir(parents=True)
        (race_root / "scratch" / "report.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "processed": 2,
                    "true_positive": 999,
                    "false_positive": 0,
                    "false_negative": 0,
                    "object_true_positive": 999,
                    "object_false_positive": 0,
                    "object_false_negative": 0,
                }
            ),
            encoding="utf-8",
        )
        retry_job.tmp_path = str(race_root)
        retry_job.status = "running"
        _finish_test_sample_f1_job(
            session,
            retry_job,
            config,
            succeeded=True,
        )
        raced = _test_sample_detail(session, first.id)
        assert raced.evaluation.status == "queued"
        assert raced.evaluation.pixel is not None
        assert raced.evaluation.pixel.true_positive == 90
        assert raced.evaluation.job_id != retry_job.id


def test_direct_test_sample_evaluation_requires_primary_and_compatible_schema(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_export_dataset(config.mlmarkup_root, config.images_root)
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        detail = create_test_sample(
            session,
            _TestSampleCreate(
                dataset_key="Вырубки\\main",
                tile_width=16,
                tile_height=16,
                image_count=1,
                object_count=1,
            ),
            config,
        )
        sample = session.get(_TestSampleRow, detail.id)
        assert sample is not None
        assert reconcile_test_sample_evaluations(session, config) == 0
        assert _test_sample_detail(session, detail.id).evaluation.status == "unavailable"

        result = TrainingResultRow(
            source="manual",
            dataset_key="Вырубки\\main",
            class_key="Вырубки\\main",
            class_display_name="Вырубки\\main",
            architecture="segformer_b2",
            model_name="Основная сеть",
            mlflow_run_id="run-primary",
            task="binary",
            status="ok",
        )
        session.add(result)
        session.flush()
        class_row = dataset_class_row(session, detail.class_key)
        assert class_row is not None
        class_row.primary_training_result_id = result.id
        sample.task = "multiclass"
        sample.class_schema = [
            {"id": 1, "slug": "first", "name": "Первый", "color": "#F59E0B"},
            {"id": 2, "slug": "second", "name": "Второй", "color": "#8B5CF6"},
        ]
        session.flush()

        assert reconcile_test_sample_evaluations(session, config) == 0
        incompatible_task = _test_sample_detail(session, detail.id)
        assert incompatible_task.evaluation.status == "error"
        assert "Тип задачи" in (incompatible_task.evaluation.error or "")

        result.task = "multiclass"
        result.class_schema = [{"id": 1, "slug": "first"}]
        session.flush()
        assert reconcile_test_sample_evaluations(session, config) == 0
        incompatible_schema = _test_sample_detail(session, detail.id)
        assert incompatible_schema.evaluation.status == "error"
        assert "Схема типов" in (incompatible_schema.evaluation.error or "")

        result.class_schema = list(sample.class_schema)
        session.flush()
        assert reconcile_test_sample_evaluations(session, config) == 1
        compatible = _test_sample_detail(session, detail.id)
        assert compatible.evaluation.status == "queued"


def test_primary_sample_queues_network_f1_and_stales_it_after_tile_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_export_dataset(config.mlmarkup_root, config.images_root)
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        sample = create_test_sample(
            session,
            _TestSampleCreate(
                dataset_key="Вырубки\\main",
                tile_width=16,
                tile_height=16,
                image_count=2,
                object_count=4,
            ),
            config,
        )
        update_test_sample_primary(
            session,
            sample.id,
            _TestSamplePrimaryUpdate(is_primary=True),
        )
        training = TrainingResultRow(
            source="manual",
            dataset_key="Вырубки\\main",
            class_key="Вырубки\\main",
            class_display_name="Вырубки\\main",
            architecture="segformer_b2",
            model_name="segformer b2",
            mlflow_run_id="run-123",
            status="ok",
        )
        session.add(training)
        pseudo = PseudoMarkupResultRow(
            source="manual",
            dataset_key="Вырубки\\main",
            class_key="Вырубки\\main",
            source_dataset_name="Вырубки\\main",
            image_count=63,
            status="ok",
        )
        session.add(pseudo)
        session.flush()
        sample_row = session.get(_TestSampleRow, sample.id)
        assert sample_row is not None
        sample_row.evaluation_pseudo_result_id = pseudo.id
        session.flush()

        assert queue_training_result_test_f1(session, training, config) is True
        metric = session.get(TrainingResultTestMetricRow, training.id)
        assert metric is not None
        assert metric.status == "queued"
        assert metric.sample_id == sample.id
        assert metric.sample_revision == 1
        job = session.get(JobRow, metric.job_id)
        assert job is not None
        assert job.config["operation"] == "test_sample_f1"
        assert job.config["metric_target"] == "training_result"
        assert job.config["test_sample_tile_indices"] == [1, 2]
        assert job.config["postprocess_profile"] == "strong"
        assert job.config["test_f1_evaluator_version"] == 3

        current_hash = metric.inference_config_hash
        metric.status = "current"
        metric.f1 = 0.5
        metric.inference_config_hash = "legacy-evaluator-hash"
        info = training_result_test_f1_info(session, training, config)
        assert info is not None
        assert info.status == "stale"
        metric.status = "queued"
        metric.f1 = None
        metric.inference_config_hash = current_hash

        update_test_sample_tile(
            session,
            sample.id,
            1,
            _TestSampleTileUpdate(enabled=False),
        )

        assert job.status == "cancelled"
        assert metric.job_id is None
        assert metric.status == "unavailable"
        assert "изменён" in (metric.error or "")


def test_class_primary_sample_evaluates_network_from_another_dataset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_export_dataset(config.mlmarkup_root, config.images_root)
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        sample = create_test_sample(
            session,
            _TestSampleCreate(
                dataset_key="Вырубки\\main",
                tile_width=16,
                tile_height=16,
                image_count=1,
                object_count=1,
            ),
            config,
        )
        update_test_sample_primary(
            session,
            sample.id,
            _TestSamplePrimaryUpdate(is_primary=True),
        )
        class_row = dataset_class_row(session, sample.class_key)
        assert class_row is not None
        new_dataset = DatasetRow(
            key=str(uuid.uuid4()),
            class_id=class_row.id,
            name="main_new",
            source_type="mlmarkup",
            source_path="Вырубки/main_new",
            legacy_version=False,
        )
        session.add(new_dataset)
        session.flush()
        training = TrainingResultRow(
            source="manual",
            dataset_key=new_dataset.key,
            class_key=new_dataset.key,
            class_display_name="Вырубки\\main_new",
            architecture="segformer_b2",
            model_name="segformer b2",
            mlflow_run_id="run-main-new",
            status="ok",
        )
        session.add(training)
        session.flush()

        assert queue_training_result_test_f1(session, training, config) is True
        metric = session.get(TrainingResultTestMetricRow, training.id)
        assert metric is not None
        job = session.get(JobRow, metric.job_id)
        assert job is not None
        assert job.dataset_key == new_dataset.key
        assert job.config["test_sample_id"] == str(sample.id)

        monkeypatch.setattr(
            _worker,
            "_best_training_checkpoint",
            lambda _config, _run_id: SimpleNamespace(
                artifact_uri="file:///checkpoint.pt",
                artifact_path="checkpoints/best.pt",
                f1_score=0.8,
                epoch=20,
                threshold=0.5,
            ),
        )
        run_root = tmp_path / "main-new-network-test-f1"
        payload = _worker._build_test_sample_f1_config(
            session,
            job,
            config,
            run_root,
        )
        assert payload["test_sample_id"] == str(sample.id)
        assert payload["training_result_id"] == str(training.id)

        (run_root / "scratch").mkdir(parents=True)
        (run_root / "scratch" / "report.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "processed": 1,
                    "threshold": 0.5,
                    "true_positive": 8,
                    "false_positive": 1,
                    "false_negative": 1,
                    "object_true_positive": 1,
                    "object_false_positive": 0,
                    "object_false_negative": 0,
                }
            ),
            encoding="utf-8",
        )
        job.tmp_path = str(run_root)
        job.status = "running"
        _finish_test_sample_f1_job(
            session,
            job,
            config,
            succeeded=True,
        )

        assert job.status == "completed"
        assert metric.status == "current"
        assert metric.f1 == pytest.approx(8 / 9)
        assert metric.object_f1 == pytest.approx(1.0)


def test_primary_sample_is_unique_and_selected_bulk_zip_uses_enabled_tiles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_export_dataset(config.mlmarkup_root, config.images_root)
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        samples = [
            create_test_sample(
                session,
                _TestSampleCreate(
                    name=f"Выборка {index}",
                    dataset_key="Вырубки\\main",
                    tile_width=16,
                    tile_height=16,
                    image_count=2,
                    object_count=4,
                ),
                config,
            )
            for index in (1, 2)
        ]
        update_test_sample_primary(
            session,
            samples[0].id,
            _TestSamplePrimaryUpdate(is_primary=True),
        )
        update_test_sample_primary(
            session,
            samples[1].id,
            _TestSamplePrimaryUpdate(is_primary=True),
        )
        update_test_sample_tile(
            session,
            samples[1].id,
            1,
            _TestSampleTileUpdate(enabled=False),
        )

        assert _test_sample_detail(session, samples[0].id).is_primary is False
        assert _test_sample_detail(session, samples[1].id).is_primary is True

        first_row = session.get(_TestSampleRow, samples[0].id)
        assert first_row is not None
        first_row.dataset_key = "Пожары\\main"
        first_row.dataset_name = "Пожары\\main"
        first_row.class_key = "Пожары"
        first_row.class_name = "Пожары"
        first_row.dataset_short_name = "main"
        session.flush()
        update_test_sample_primary(
            session,
            samples[0].id,
            _TestSamplePrimaryUpdate(is_primary=True),
        )

        artifact = build_test_samples_download(
            session,
            [sample.id for sample in reversed(samples)],
            config,
        )
        try:
            with zipfile.ZipFile(artifact.path) as archive:
                names = set(archive.namelist())
                assert all(
                    info.compress_type == zipfile.ZIP_STORED
                    for info in archive.infolist()
                )
            assert names == (
                _downloaded_tile_names(
                    "tile001",
                    folder="Вырубки_main",
                )
                | _downloaded_tile_names(
                    "tile001",
                    folder="Пожары_main",
                )
                | _downloaded_tile_names(
                    "tile002",
                    folder="Пожары_main",
                )
            )
        finally:
            artifact.cleanup()


def test_bulk_download_rejects_duplicate_datasets_and_normalized_folder_collisions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_export_dataset(config.mlmarkup_root, config.images_root)
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        samples = [
            create_test_sample(
                session,
                _TestSampleCreate(
                    name=f"Выборка {index}",
                    dataset_key="Вырубки\\main",
                    tile_width=16,
                    tile_height=16,
                    image_count=1,
                    object_count=1,
                ),
                config,
            )
            for index in (1, 2)
        ]
        sample_ids = [sample.id for sample in samples]
        with pytest.raises(
            TrainingUIAPIError,
            match="не более одной разметки каждого класса",
        ):
            build_test_samples_download(session, sample_ids, config)

        rows = [session.get(_TestSampleRow, sample_id) for sample_id in sample_ids]
        assert all(row is not None for row in rows)
        first_row, second_row = rows
        assert first_row is not None
        assert second_row is not None
        first_row.dataset_name = "Вырубки\\main/test"
        first_row.dataset_short_name = "main/test"
        second_row.dataset_key = "Вырубки\\main test"
        second_row.dataset_name = "Вырубки\\main test"
        second_row.dataset_short_name = "main test"
        second_row.class_key = "другой-класс"
        session.flush()

        with pytest.raises(
            TrainingUIAPIError,
            match="одинаковое имя папки архива «Вырубки_main_test»",
        ):
            build_test_samples_download(session, sample_ids, config)

    download_root = config.scratch_root / _test_samples.TEST_SAMPLE_DOWNLOAD_ROOT_NAME
    assert not download_root.exists()


def test_bulk_download_uses_eight_workers_and_cleans_partial_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_export_dataset(config.mlmarkup_root, config.images_root)
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        samples = [
            create_test_sample(
                session,
                _TestSampleCreate(
                    name="Одинаковое имя",
                    dataset_key="Вырубки\\main",
                    tile_width=16,
                    tile_height=16,
                    image_count=1,
                    object_count=1,
                ),
                config,
            )
            for _ in range(9)
        ]
        for index, sample in enumerate(samples, start=1):
            row = session.get(_TestSampleRow, sample.id)
            assert row is not None
            row.dataset_key = f"Вырубки\\set-{index:02d}"
            row.dataset_name = f"Вырубки\\set-{index:02d}"
            row.dataset_short_name = f"set-{index:02d}"
            row.class_key = f"класс-{index:02d}"
        session.flush()
        original_prepare = _test_samples._prepare_test_sample_download
        lock = threading.Lock()
        active = 0
        maximum_active = 0
        call_count = 0

        def tracked_prepare(
            descriptor,
            staging_root,
            *,
            include_previews,
        ):
            nonlocal active, maximum_active, call_count
            with lock:
                active += 1
                call_count += 1
                maximum_active = max(maximum_active, active)
            try:
                time.sleep(0.05)
                return original_prepare(
                    descriptor,
                    staging_root,
                    include_previews=include_previews,
                )
            finally:
                with lock:
                    active -= 1

        monkeypatch.setattr(
            _test_samples,
            "_prepare_test_sample_download",
            tracked_prepare,
        )
        artifact = build_test_samples_download(
            session,
            [sample.id for sample in samples],
            config,
            include_previews=False,
        )
        try:
            with zipfile.ZipFile(artifact.path) as archive:
                folders = {name.split("/", maxsplit=1)[0] for name in archive.namelist()}
                assert len(folders) == 9
                assert len(archive.namelist()) == 18
                assert all(
                    info.compress_type == zipfile.ZIP_STORED
                    for info in archive.infolist()
                )
        finally:
            artifact.cleanup()

        assert call_count == 9
        assert maximum_active == 8

        broken_root = (
            config.stored_files_root
            / "test-samples"
            / str(samples[0].id)
        )
        (broken_root / "tile_001.geojson").unlink()
        with pytest.raises(TrainingUIAPIError, match="Файл тестового тайла не найден"):
            build_test_samples_download(
                session,
                [sample.id for sample in samples],
                config,
                include_previews=False,
            )

    download_root = config.scratch_root / _test_samples.TEST_SAMPLE_DOWNLOAD_ROOT_NAME
    assert not list(download_root.glob("*.zip"))
    assert not list(download_root.glob(".building-*"))


def test_test_sample_download_removes_partial_archive_when_jpeg_cannot_fit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_export_dataset(config.mlmarkup_root, config.images_root)
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        sample = create_test_sample(
            session,
            _TestSampleCreate(
                dataset_key="Вырубки\\main",
                tile_width=16,
                tile_height=16,
                image_count=1,
                object_count=1,
            ),
            config,
        )
        monkeypatch.setattr(_test_samples, "_JPEG_PREVIEW_MAX_BYTES", 1)
        with pytest.raises(TrainingUIAPIError, match="не помещается в 300 КБ"):
            build_test_sample_download(session, sample.id, config)

    download_root = config.scratch_root / _test_samples.TEST_SAMPLE_DOWNLOAD_ROOT_NAME
    assert not list(download_root.glob("*.zip"))


def test_atomic_test_markup_save_requeues_all_networks_and_reconciliation_is_idempotent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_export_dataset(config.mlmarkup_root, config.images_root)
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        first = create_test_sample(
            session,
            _TestSampleCreate(
                name="Первая",
                dataset_key="Вырубки\\main",
                tile_width=16,
                tile_height=16,
                image_count=2,
                object_count=4,
            ),
            config,
        )
        second = create_test_sample(
            session,
            _TestSampleCreate(
                name="Вторая",
                dataset_key="Вырубки\\main",
                tile_width=16,
                tile_height=16,
                image_count=2,
                object_count=4,
            ),
            config,
        )
        update_test_sample_primary(
            session,
            first.id,
            _TestSamplePrimaryUpdate(is_primary=True),
        )
        results = [
            TrainingResultRow(
                source="manual",
                dataset_key="Вырубки\\main",
                class_key="Вырубки\\main",
                class_display_name="Вырубки\\main",
                architecture=architecture,
                model_name=architecture,
                mlflow_run_id=f"run-{index}",
                status="ok",
            )
            for index, architecture in enumerate(("segformer_b2", "segformer_b3"), start=1)
        ]
        session.add_all(results)
        session.flush()

        updated = update_test_sample(
            session,
            first.id,
            _TestSampleUpdate(
                name="Первая сохранённая",
                is_primary=True,
                enabled_tile_indices=[2],
            ),
            config,
        )
        assert updated.name == "Первая сохранённая"
        assert updated.is_primary is True
        assert [tile.index for tile in updated.tiles if tile.enabled] == [2]
        first_row = session.get(_TestSampleRow, first.id)
        assert first_row is not None
        assert first_row.content_revision == 2
        metrics = [session.get(TrainingResultTestMetricRow, result.id) for result in results]
        assert all(metric is not None and metric.status == "queued" for metric in metrics)
        queued_jobs = session.scalars(
            select(JobRow).where(JobRow.status == "queued")
        ).all()
        assert len(queued_jobs) == 2
        assert all(job.config["test_sample_tile_indices"] == [2] for job in queued_jobs)

        update_test_sample(
            session,
            first.id,
            _TestSampleUpdate(
                name="Первая сохранённая",
                is_primary=True,
                enabled_tile_indices=[2],
            ),
            config,
        )
        assert len(session.scalars(select(JobRow).where(JobRow.status == "queued")).all()) == 2
        assert reconcile_training_result_test_f1(session, config) == 0

        for metric in metrics:
            assert metric is not None and metric.job_id is not None
            job = session.get(JobRow, metric.job_id)
            assert job is not None
            job.status = "failed"
            metric.status = "error"
            metric.error = "Диагностическая ошибка"
        session.flush()
        assert reconcile_training_result_test_f1(session, config) == 0

        switched = update_test_sample(
            session,
            second.id,
            _TestSampleUpdate(
                name="Вторая",
                is_primary=True,
                enabled_tile_indices=[1, 2],
            ),
            config,
        )
        assert switched.is_primary is True
        assert _test_sample_detail(session, first.id).is_primary is False
        for result in results:
            metric = session.get(TrainingResultTestMetricRow, result.id)
            assert metric is not None
            assert metric.sample_id == second.id
            assert metric.status == "queued"
        assert reconcile_training_result_test_f1(session, config) == 0

        with pytest.raises(TrainingUIAPIError, match="не должны повторяться"):
            update_test_sample(
                session,
                second.id,
                _TestSampleUpdate(enabled_tile_indices=[1, 1]),
                config,
            )


def test_app_startup_recovers_missing_test_markup_f1(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    _write_export_dataset(config.mlmarkup_root, config.images_root)
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        sample = create_test_sample(
            session,
            _TestSampleCreate(
                dataset_key="Вырубки\\main",
                tile_width=16,
                tile_height=16,
                image_count=1,
                object_count=1,
            ),
            config,
        )
        update_test_sample_primary(
            session,
            sample.id,
            _TestSamplePrimaryUpdate(is_primary=True),
        )
        result = TrainingResultRow(
            source="manual",
            dataset_key="Вырубки\\main",
            class_key="Вырубки\\main",
            class_display_name="Вырубки\\main",
            architecture="segformer_b2",
            model_name="segformer b2",
            mlflow_run_id="startup-run",
            status="ok",
        )
        session.add(result)
        session.commit()
        result_id = result.id

    with TestClient(create_app()):
        pass

    with session_factory() as session:
        metric = session.get(TrainingResultTestMetricRow, result_id)
        assert metric is not None
        assert metric.status == "queued"
        assert metric.job_id is not None


def test_test_sample_optimizer_uses_all_tiles_and_resolves_equal_f1_by_territory() -> None:
    from mlsystem2.training_ui_api._test_samples import _select_optimized_tile_indices

    tiles = [
        _TestSampleTileRow(
            tile_index=1,
            source_name="a/one.tif",
            territory="a",
            object_count=5,
            enabled=True,
        ),
        _TestSampleTileRow(
            tile_index=2,
            source_name="a/two.tif",
            territory="a",
            object_count=10,
            enabled=False,
        ),
        _TestSampleTileRow(
            tile_index=3,
            source_name="b/one.tif",
            territory="b",
            object_count=2,
            enabled=False,
        ),
        _TestSampleTileRow(
            tile_index=4,
            source_name="c/one.tif",
            territory="c",
            object_count=1,
            enabled=True,
        ),
    ]
    perfect = _test_samples_metric_counts(5, 0, 0)
    metrics = {tile.tile_index: (perfect, perfect) for tile in tiles}
    request = _TestSampleOptimizeRequest(
        min_tile_count=2,
        max_tile_count=2,
        min_object_count=1,
        metric="objects",
    )

    first = _select_optimized_tile_indices(tiles, metrics, request)
    second = _select_optimized_tile_indices(tiles, metrics, request)

    assert first == second == [2, 3]


def test_test_sample_optimizer_prioritizes_aggregate_f1_before_diversity() -> None:
    from mlsystem2.training_ui_api._test_samples import _select_optimized_tile_indices

    tiles = [
        _TestSampleTileRow(
            tile_index=index,
            source_name=f"source-{index}.tif",
            territory="a" if index <= 2 else chr(96 + index),
            object_count=1,
            enabled=index % 2 == 0,
        )
        for index in range(1, 5)
    ]
    perfect = _test_samples_metric_counts(10, 0, 0)
    weak = _test_samples_metric_counts(5, 5, 5)
    metrics = {
        1: (perfect, perfect),
        2: (perfect, perfect),
        3: (weak, weak),
        4: (weak, weak),
    }

    selected = _select_optimized_tile_indices(
        tiles,
        metrics,
        _TestSampleOptimizeRequest(
            min_tile_count=2,
            max_tile_count=2,
            min_object_count=2,
            metric="pixel",
        ),
    )

    assert selected == [1, 2]


def test_test_sample_optimizer_selects_best_count_inside_range() -> None:
    from mlsystem2.training_ui_api._test_samples import _select_optimized_tile_indices

    tiles = [
        _TestSampleTileRow(
            tile_index=index,
            source_name=f"source-{index}.tif",
            territory=f"territory-{index}",
            object_count=1,
            enabled=True,
        )
        for index in range(1, 4)
    ]
    perfect = _test_samples_metric_counts(10, 0, 0)
    false_positive = _test_samples_metric_counts(0, 10, 0)
    metrics = {
        1: (perfect, perfect),
        2: (false_positive, false_positive),
        3: (false_positive, false_positive),
    }

    selected = _select_optimized_tile_indices(
        tiles,
        metrics,
        _TestSampleOptimizeRequest(
            min_tile_count=1,
            max_tile_count=3,
            min_object_count=1,
            metric="pixel",
        ),
    )

    assert selected == [1]


def test_test_sample_optimizer_uses_requested_metric() -> None:
    from mlsystem2.training_ui_api._test_samples import _select_optimized_tile_indices

    tiles = [
        _TestSampleTileRow(
            tile_index=index,
            source_name=f"source-{index}.tif",
            territory=f"territory-{index}",
            object_count=1,
            enabled=True,
        )
        for index in (1, 2)
    ]
    perfect = _test_samples_metric_counts(10, 0, 0)
    weak = _test_samples_metric_counts(5, 5, 5)
    metrics = {
        1: (perfect, weak),
        2: (weak, perfect),
    }

    def selected(metric: str) -> list[int]:
        return _select_optimized_tile_indices(
            tiles,
            metrics,
            _TestSampleOptimizeRequest(
                min_tile_count=1,
                max_tile_count=1,
                min_object_count=1,
                metric=metric,
            ),
        )

    assert selected("pixel") == [1]
    assert selected("objects") == [2]


def test_test_sample_cleanup_keeps_ready_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _configure_export_environment(tmp_path, monkeypatch)
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    root = config.stored_files_root / "test-samples"
    ready = root / "00000000-0000-0000-0000-000000000001"
    building = root / ".building-00000000-0000-0000-0000-000000000002"
    deleting = root / ".deleting-00000000-0000-0000-0000-000000000003"
    download = config.scratch_root / "test-sample-downloads" / "unfinished.zip"
    ready.mkdir(parents=True)
    building.mkdir()
    deleting.mkdir()
    download.parent.mkdir(parents=True)
    download.write_bytes(b"unfinished")

    with session_factory() as session:
        cleanup_test_sample_storage(session, config)

    assert ready.is_dir()
    assert not building.exists()
    assert not deleting.exists()
    assert not download.exists()


def _test_samples_metric_counts(true_positive: int, false_positive: int, false_negative: int):
    from mlsystem2.training_ui_api._test_samples import _MetricCounts

    return _MetricCounts(true_positive, false_positive, false_negative)


def _login(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "mluser", "password": "secret"},
    )
    assert response.status_code == 200


def _write_prediction_from_sample(config, sample_id, output_path: Path) -> None:
    source_root = config.stored_files_root / "test-samples" / str(sample_id)
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    features = []
    for geojson_path in sorted(source_root.glob("tile_*.geojson")):
        payload = json.loads(geojson_path.read_text(encoding="utf-8"))
        for feature in payload["features"]:
            geometry = transform_geometry(transformer.transform, shape(feature["geometry"]))
            features.append(
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": mapping(geometry),
                }
            )
    output_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )


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
    monkeypatch.setenv(
        "MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT",
        str(tmp_path / "stored_files"),
    )
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


def _write_per_image_geojson(
    path: Path,
    features: list[tuple[int, object, str]],
) -> None:
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [
            {
                "type": "Feature",
                "id": feature_id,
                "properties": {"_mlsystem2_role": role},
                "geometry": mapping(geometry),
            }
            for feature_id, geometry, role in features
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _geojson_bytes(geometries: list[object], *, crs: str | None = "EPSG:3857") -> bytes:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": mapping(geometry),
            }
            for geometry in geometries
        ],
    }
    if crs is not None:
        payload["crs"] = {"type": "name", "properties": {"name": crs}}
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


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


def _selection_candidate_with_footprint(
    crs: CRS,
    *,
    index: int,
    count: int,
    footprint,
) -> _markup_export._Candidate:
    feature_start = index * 100
    return _markup_export._Candidate(
        source_path=Path(f"scene-{index}.tif"),
        source_name=f"region-{index}/scene-{index}.tif",
        territory=f"region-{index}",
        column=index,
        row=0,
        raster_crs=crs,
        raster_footprint=footprint,
        annotation_footprint=footprint,
        feature_positions=tuple(range(feature_start, feature_start + count)),
    )
