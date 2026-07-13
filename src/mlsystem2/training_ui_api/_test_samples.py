"""Постоянное хранение и оценка тестовых выборок."""

from __future__ import annotations

import json
import re
import shutil
import uuid
import warnings
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS as PyprojCRS
from pyproj import Transformer
from rasterio.errors import NotGeoreferencedWarning
from rasterio.features import rasterize
from scipy.optimize import Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union
from shapely.strtree import STRtree
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ._config import TrainingUIAPIConfig
from ._markup_export import _run_milp, _single_constraint, generate_markup_files
from ._models import PseudoMarkupResultRow, TestSampleRow, TestSampleTileRow
from .contracts import (
    MarkupExportRequest,
    TestSampleCatalogResponse,
    TestSampleClassGroup,
    TestSampleCreate,
    TestSampleDetail,
    TestSampleEvaluationInfo,
    TestSampleMetric,
    TestSampleOptimizeRequest,
    TestSampleSummary,
    TestSampleTileInfo,
    TestSampleTileUpdate,
    TestSampleUpdate,
    TestSampleVariantGroup,
    TrainingUIAPIError,
)


TEST_SAMPLE_ROOT_NAME = "test-samples"
TEST_SAMPLE_DOWNLOAD_ROOT_NAME = "test-sample-downloads"
OBJECT_IOU_THRESHOLD = 0.5
_TILE_SUFFIXES = (".tif", ".geojson", "_mask.png", "_preview.png")


class TestSampleUnavailable(FileNotFoundError):
    """Тестовая выборка или её постоянный файл не найдены."""


@dataclass(frozen=True)
class TestSampleDownloadArtifact:
    path: Path
    filename: str

    def cleanup(self) -> None:
        self.path.unlink(missing_ok=True)


@dataclass(frozen=True)
class _MetricCounts:
    true_positive: int
    false_positive: int
    false_negative: int

    def info(self) -> TestSampleMetric:
        precision_denominator = self.true_positive + self.false_positive
        recall_denominator = self.true_positive + self.false_negative
        precision = (
            self.true_positive / precision_denominator if precision_denominator else 0.0
        )
        recall = self.true_positive / recall_denominator if recall_denominator else 0.0
        f1_denominator = precision + recall
        f1 = 2.0 * precision * recall / f1_denominator if f1_denominator else 0.0
        return TestSampleMetric(
            precision=precision,
            recall=recall,
            f1=f1,
            true_positive=self.true_positive,
            false_positive=self.false_positive,
            false_negative=self.false_negative,
        )

    def __add__(self, other: "_MetricCounts") -> "_MetricCounts":
        return _MetricCounts(
            true_positive=self.true_positive + other.true_positive,
            false_positive=self.false_positive + other.false_positive,
            false_negative=self.false_negative + other.false_negative,
        )


@dataclass(frozen=True)
class _GeometrySet:
    crs: PyprojCRS
    geometries: tuple[BaseGeometry, ...]
    tree: STRtree


def create_test_sample(
    session: Session,
    request: TestSampleCreate,
    config: TrainingUIAPIConfig,
) -> TestSampleDetail:
    sample_id = uuid.uuid4()
    root = _test_sample_root(config)
    root.mkdir(parents=True, exist_ok=True)
    building_root = root / f".building-{sample_id}"
    final_root = root / str(sample_id)
    building_root.mkdir(parents=False, exist_ok=False)
    try:
        generated = generate_markup_files(
            MarkupExportRequest(
                dataset_key=request.dataset_key,
                tile_width=request.tile_width,
                tile_height=request.tile_height,
                image_count=request.image_count,
                object_count=request.object_count,
            ),
            config,
            building_root,
        )
        building_root.replace(final_root)
        row = TestSampleRow(
            id=sample_id,
            name=_sample_name(request.name, generated.dataset_name),
            dataset_key=generated.dataset_key,
            dataset_name=generated.dataset_name,
            dataset_version=generated.dataset_version,
            class_key=generated.class_key,
            class_name=generated.class_name,
            variant_key=generated.variant_key,
            variant_name=generated.variant_name,
            tile_width=generated.tile_width,
            tile_height=generated.tile_height,
            image_count=len(generated.tiles),
            requested_object_count=generated.requested_object_count,
            actual_object_count=generated.actual_object_count,
            territory_count=generated.territory_count,
            warnings=list(generated.warnings),
            content_revision=1,
            metric_status="unavailable",
            object_iou_threshold=OBJECT_IOU_THRESHOLD,
        )
        row.tiles = [
            TestSampleTileRow(
                tile_index=tile.index,
                source_name=tile.source_name,
                territory=tile.territory,
                object_count=tile.object_count,
                enabled=True,
            )
            for tile in generated.tiles
        ]
        session.add(row)
        session.flush()
        evaluate_test_sample(session, row, config)
        session.flush()
        return _detail(row)
    except Exception:
        shutil.rmtree(building_root, ignore_errors=True)
        shutil.rmtree(final_root, ignore_errors=True)
        raise


def test_sample_catalog(session: Session) -> TestSampleCatalogResponse:
    rows = session.scalars(
        select(TestSampleRow)
        .options(selectinload(TestSampleRow.tiles))
        .order_by(TestSampleRow.created_at.desc(), TestSampleRow.id.desc())
    ).all()
    grouped: dict[tuple[str, str], dict[tuple[str, str], list[TestSampleSummary]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        grouped[(row.class_key, row.class_name)][
            (row.variant_key, row.variant_name)
        ].append(_summary(row))
    classes = [
        TestSampleClassGroup(
            key=class_key,
            name=class_name,
            variants=[
                TestSampleVariantGroup(key=variant_key, name=variant_name, samples=samples)
                for (variant_key, variant_name), samples in sorted(
                    variants.items(), key=lambda item: item[0][1].casefold()
                )
            ],
        )
        for (class_key, class_name), variants in sorted(
            grouped.items(), key=lambda item: item[0][1].casefold()
        )
    ]
    return TestSampleCatalogResponse(classes=classes)


def test_sample_detail(session: Session, sample_id: uuid.UUID) -> TestSampleDetail:
    return _detail(_sample_row(session, sample_id))


def update_test_sample(
    session: Session,
    sample_id: uuid.UUID,
    request: TestSampleUpdate,
) -> TestSampleDetail:
    row = _sample_row(session, sample_id)
    name = request.name.strip()
    if not name:
        raise TrainingUIAPIError("Название тестовой выборки не может быть пустым.")
    row.name = name
    row.updated_at = _utc_now()
    session.flush()
    return _detail(row)


def update_test_sample_tile(
    session: Session,
    sample_id: uuid.UUID,
    tile_index: int,
    request: TestSampleTileUpdate,
) -> TestSampleDetail:
    row = _sample_row(session, sample_id)
    tile = next((item for item in row.tiles if item.tile_index == tile_index), None)
    if tile is None:
        raise TestSampleUnavailable(str(tile_index))
    if tile.enabled != request.enabled:
        tile.enabled = request.enabled
        tile.updated_at = _utc_now()
        row.content_revision += 1
        row.metric_status = "stale" if _has_metrics(row) else "unavailable"
        row.evaluation_error = None
        row.updated_at = _utc_now()
        session.flush()
    return _detail(row)


def delete_test_sample(
    session: Session,
    sample_id: uuid.UUID,
    config: TrainingUIAPIConfig,
) -> None:
    row = _sample_row(session, sample_id)
    sample_root = _sample_root(config, row.id)
    deleting_root = sample_root.with_name(f".deleting-{row.id}")
    if deleting_root.exists():
        shutil.rmtree(deleting_root, ignore_errors=True)
    if sample_root.exists():
        sample_root.replace(deleting_root)
    try:
        session.delete(row)
        session.flush()
    except Exception:
        if deleting_root.exists() and not sample_root.exists():
            deleting_root.replace(sample_root)
        raise
    shutil.rmtree(deleting_root, ignore_errors=True)


def evaluate_test_sample_by_id(
    session: Session,
    sample_id: uuid.UUID,
    config: TrainingUIAPIConfig,
) -> TestSampleDetail:
    row = _sample_row(session, sample_id)
    evaluate_test_sample(session, row, config)
    session.flush()
    return _detail(row)


def optimize_test_sample(
    session: Session,
    sample_id: uuid.UUID,
    request: TestSampleOptimizeRequest,
    config: TrainingUIAPIConfig,
) -> TestSampleDetail:
    row = _sample_row(session, sample_id)
    _validate_optimization_request(row, request)
    source = _latest_pseudo_markup(session, row.dataset_key)
    if source is None or source.geojson_file is None:
        raise TrainingUIAPIError(
            "Нет успешной разметки для этого подкласса и варианта датасета; "
            "оптимизация состава недоступна."
        )
    source_path = Path(source.geojson_file.path)
    if not source_path.is_file():
        raise TrainingUIAPIError(
            "Файл последней разметки не найден на сервере; состав выборки не изменён."
        )

    tile_metrics = _calculate_tile_metrics(row, source_path, config)
    selected_indices = _select_optimized_tile_indices(
        row.tiles,
        tile_metrics,
        request,
    )
    selected = set(selected_indices)
    pixel_counts = _sum_tile_metrics(tile_metrics, selected, metric_index=0)
    object_counts = _sum_tile_metrics(tile_metrics, selected, metric_index=1)

    now = _utc_now()
    for tile in row.tiles:
        enabled = tile.tile_index in selected
        if tile.enabled != enabled:
            tile.enabled = enabled
            tile.updated_at = now
    row.content_revision += 1
    row.updated_at = now
    _apply_evaluation(row, source, pixel_counts, object_counts)
    session.flush()
    return _detail(row)


def evaluate_test_sample(
    session: Session,
    row: TestSampleRow,
    config: TrainingUIAPIConfig,
    *,
    pseudo_result: PseudoMarkupResultRow | None = None,
) -> None:
    if not any(tile.enabled for tile in row.tiles):
        _evaluation_unavailable(row, "В тестовой выборке нет включённых тайлов.")
        return
    source = pseudo_result or _latest_pseudo_markup(session, row.dataset_key)
    if source is None or source.geojson_file is None:
        _evaluation_unavailable(
            row,
            "Нет успешной разметки для этого подкласса и варианта датасета.",
        )
        return
    source_path = Path(source.geojson_file.path)
    if not source_path.is_file():
        _evaluation_error(row, "Файл последней разметки не найден на сервере.")
        return
    try:
        pixel_counts, object_counts = _calculate_metrics(row, source_path, config)
    except Exception as exc:  # noqa: BLE001
        _evaluation_error(row, f"Не удалось рассчитать F1: {exc}")
        return
    _apply_evaluation(row, source, pixel_counts, object_counts)


def _apply_evaluation(
    row: TestSampleRow,
    source: PseudoMarkupResultRow,
    pixel_counts: _MetricCounts,
    object_counts: _MetricCounts,
) -> None:
    pixel = pixel_counts.info()
    objects = object_counts.info()
    row.pixel_precision = pixel.precision
    row.pixel_recall = pixel.recall
    row.pixel_f1 = pixel.f1
    row.pixel_true_positive = pixel.true_positive
    row.pixel_false_positive = pixel.false_positive
    row.pixel_false_negative = pixel.false_negative
    row.object_precision = objects.precision
    row.object_recall = objects.recall
    row.object_f1 = objects.f1
    row.object_true_positive = objects.true_positive
    row.object_false_positive = objects.false_positive
    row.object_false_negative = objects.false_negative
    row.evaluation_pseudo_result_id = source.id
    row.evaluation_model_name = (
        source.training_result.model_name
        if source.training_result is not None
        else "псевдоразметка"
    )
    row.evaluation_markup_created_at = source.updated_at or source.created_at
    row.evaluated_revision = row.content_revision
    row.metric_status = "current"
    row.evaluated_at = _utc_now()
    row.evaluation_error = None
    row.updated_at = _utc_now()


def _validate_optimization_request(
    row: TestSampleRow,
    request: TestSampleOptimizeRequest,
) -> None:
    if request.min_tile_count > request.max_tile_count:
        raise TrainingUIAPIError(
            "Минимальное число тайлов не может быть больше максимального."
        )
    if request.max_tile_count > len(row.tiles):
        raise TrainingUIAPIError(
            "Максимальное число тайлов превышает число тайлов в выборке."
        )
    maximum_objects = sum(
        sorted((tile.object_count for tile in row.tiles), reverse=True)[
            : request.max_tile_count
        ]
    )
    if request.min_object_count > maximum_objects:
        raise TrainingUIAPIError(
            "Минимальное число объектов недостижимо при заданном максимуме тайлов."
        )


def _select_optimized_tile_indices(
    tiles: list[TestSampleTileRow],
    tile_metrics: dict[int, tuple[_MetricCounts, _MetricCounts]],
    request: TestSampleOptimizeRequest,
) -> list[int]:
    ordered_tiles = sorted(tiles, key=lambda tile: tile.tile_index)
    tile_count = len(ordered_tiles)
    metric_index = 0 if request.metric == "pixel" else 1
    counts = [tile_metrics[tile.tile_index][metric_index] for tile in ordered_tiles]
    numerators = np.asarray(
        [2 * item.true_positive for item in counts],
        dtype=float,
    )
    denominators = np.asarray(
        [
            2 * item.true_positive + item.false_positive + item.false_negative
            for item in counts
        ],
        dtype=float,
    )
    object_counts = np.asarray(
        [tile.object_count for tile in ordered_tiles],
        dtype=float,
    )
    territories = sorted({tile.territory for tile in ordered_tiles}, key=str.casefold)
    sources = sorted({tile.source_name for tile in ordered_tiles}, key=str.casefold)
    territory_offset = tile_count
    source_offset = territory_offset + len(territories)
    variable_count = source_offset + len(sources)

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(
        coefficients: list[tuple[int, float]],
        *,
        minimum: float = -np.inf,
        maximum: float = np.inf,
    ) -> None:
        row_index = len(lower)
        lower.append(minimum)
        upper.append(maximum)
        for column, value in coefficients:
            rows.append(row_index)
            columns.append(column)
            values.append(float(value))

    add_constraint(
        [(index, 1.0) for index in range(tile_count)],
        minimum=request.min_tile_count,
        maximum=request.max_tile_count,
    )
    add_constraint(
        [(index, object_counts[index]) for index in range(tile_count)],
        minimum=request.min_object_count,
    )

    for territory_index, territory in enumerate(territories):
        selected_indices = [
            index
            for index, tile in enumerate(ordered_tiles)
            if tile.territory == territory
        ]
        variable_index = territory_offset + territory_index
        add_constraint(
            [(variable_index, 1.0)]
            + [(index, -1.0) for index in selected_indices],
            maximum=0.0,
        )
        add_constraint(
            [(index, 1.0) for index in selected_indices]
            + [(variable_index, -float(request.max_tile_count))],
            maximum=0.0,
        )

    for source_index, source in enumerate(sources):
        selected_indices = [
            index
            for index, tile in enumerate(ordered_tiles)
            if tile.source_name == source
        ]
        variable_index = source_offset + source_index
        add_constraint(
            [(variable_index, 1.0)]
            + [(index, -1.0) for index in selected_indices],
            maximum=0.0,
        )
        add_constraint(
            [(index, 1.0) for index in selected_indices]
            + [(variable_index, -float(request.max_tile_count))],
            maximum=0.0,
        )

    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(len(lower), variable_count),
        dtype=float,
    ).tocsr()
    constraints: list[LinearConstraint] = [
        LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))
    ]
    bounds = Bounds(
        np.zeros(variable_count, dtype=float),
        np.ones(variable_count, dtype=float),
    )
    integrality = np.ones(variable_count, dtype=int)

    numerator_objective = np.zeros(variable_count, dtype=float)
    numerator_objective[:tile_count] = -numerators
    numerator_result = _run_milp(
        numerator_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        allow_infeasible=True,
    )
    if numerator_result is None:
        raise TrainingUIAPIError(
            "Невозможно подобрать состав выборки с заданными ограничениями."
        )

    optimal_result = numerator_result
    optimal_numerator = _selected_sum(optimal_result, numerators, tile_count)
    optimal_denominator = _selected_sum(optimal_result, denominators, tile_count)
    if optimal_numerator > 0:
        for _ in range(64):
            ratio = optimal_numerator / optimal_denominator
            ratio_objective = np.zeros(variable_count, dtype=float)
            ratio_objective[:tile_count] = ratio * denominators - numerators
            next_result = _run_milp(
                ratio_objective,
                integrality=integrality,
                bounds=bounds,
                constraints=constraints,
            )
            next_numerator = _selected_sum(next_result, numerators, tile_count)
            next_denominator = _selected_sum(next_result, denominators, tile_count)
            improvement = (
                next_numerator * optimal_denominator
                - optimal_numerator * next_denominator
            )
            if improvement <= 0:
                break
            optimal_result = next_result
            optimal_numerator = next_numerator
            optimal_denominator = next_denominator
        else:
            raise TrainingUIAPIError(
                "Оптимизатор F1 не сошёлся за допустимое число итераций."
            )

        divisor = gcd(optimal_numerator, optimal_denominator)
        reduced_numerator = optimal_numerator // divisor
        reduced_denominator = optimal_denominator // divisor
        ratio_coefficients = [
            (
                index,
                reduced_denominator * int(numerators[index])
                - reduced_numerator * int(denominators[index]),
            )
            for index in range(tile_count)
        ]
        constraints.append(
            _single_constraint(
                variable_count,
                ratio_coefficients,
                minimum=0.0,
                maximum=0.0,
            )
        )

    territory_objective = np.zeros(variable_count, dtype=float)
    territory_objective[territory_offset:source_offset] = -1.0
    territory_result = _run_milp(
        territory_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
    )
    territory_optimum = int(
        round(float(np.sum(territory_result[territory_offset:source_offset])))
    )
    constraints.append(
        _single_constraint(
            variable_count,
            [
                (index, 1.0)
                for index in range(territory_offset, source_offset)
            ],
            minimum=territory_optimum,
            maximum=territory_optimum,
        )
    )

    object_objective = np.zeros(variable_count, dtype=float)
    object_objective[:tile_count] = -object_counts
    object_result = _run_milp(
        object_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
    )
    object_optimum = _selected_sum(object_result, object_counts, tile_count)
    constraints.append(
        _single_constraint(
            variable_count,
            [(index, object_counts[index]) for index in range(tile_count)],
            minimum=object_optimum,
            maximum=object_optimum,
        )
    )

    source_objective = np.zeros(variable_count, dtype=float)
    source_objective[source_offset:] = -1.0
    source_result = _run_milp(
        source_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
    )
    source_optimum = int(round(float(np.sum(source_result[source_offset:]))))
    constraints.append(
        _single_constraint(
            variable_count,
            [(index, 1.0) for index in range(source_offset, variable_count)],
            minimum=source_optimum,
            maximum=source_optimum,
        )
    )

    tile_count_objective = np.zeros(variable_count, dtype=float)
    tile_count_objective[:tile_count] = 1.0
    tile_count_result = _run_milp(
        tile_count_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
    )
    selected_count = int(
        round(float(np.sum(tile_count_result[:tile_count])))
    )
    constraints.append(
        _single_constraint(
            variable_count,
            [(index, 1.0) for index in range(tile_count)],
            minimum=selected_count,
            maximum=selected_count,
        )
    )

    stable_objective = np.zeros(variable_count, dtype=float)
    stable_objective[:tile_count] = np.arange(1, tile_count + 1, dtype=float)
    stable_result = _run_milp(
        stable_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
    )
    selected = [
        ordered_tiles[index].tile_index
        for index in range(tile_count)
        if stable_result[index] > 0.5
    ]
    if not request.min_tile_count <= len(selected) <= request.max_tile_count:
        raise TrainingUIAPIError("Оптимизатор вернул некорректное число тайлов.")
    return selected


def _selected_sum(result: np.ndarray, values: np.ndarray, tile_count: int) -> int:
    return int(
        round(
            sum(
                float(values[index])
                for index in range(tile_count)
                if result[index] > 0.5
            )
        )
    )


def _sum_tile_metrics(
    tile_metrics: dict[int, tuple[_MetricCounts, _MetricCounts]],
    tile_indices: set[int],
    *,
    metric_index: int,
) -> _MetricCounts:
    total = _MetricCounts(0, 0, 0)
    for tile_index in sorted(tile_indices):
        total += tile_metrics[tile_index][metric_index]
    return total


def evaluate_test_samples_for_pseudo_markup(
    session: Session,
    pseudo_result: PseudoMarkupResultRow,
    config: TrainingUIAPIConfig,
) -> None:
    if (
        pseudo_result.status != "ok"
        or pseudo_result.geojson_file is None
        or not pseudo_result.class_key
        or pseudo_result.dataset_key != pseudo_result.class_key
    ):
        return
    latest = _latest_pseudo_markup(session, pseudo_result.class_key)
    if latest is None or latest.id != pseudo_result.id:
        return
    rows = session.scalars(
        select(TestSampleRow)
        .where(TestSampleRow.dataset_key == pseudo_result.class_key)
        .options(selectinload(TestSampleRow.tiles))
        .order_by(TestSampleRow.created_at)
    ).all()
    for row in rows:
        evaluate_test_sample(session, row, config, pseudo_result=pseudo_result)
    session.flush()


def mark_test_samples_stale_for_pseudo_markup(
    session: Session,
    pseudo_result_id: uuid.UUID,
) -> None:
    rows = session.scalars(
        select(TestSampleRow).where(
            TestSampleRow.evaluation_pseudo_result_id == pseudo_result_id
        )
    ).all()
    for row in rows:
        row.metric_status = "stale" if _has_metrics(row) else "unavailable"
        row.evaluation_pseudo_result_id = None
        row.evaluation_error = "Использованная разметка удалена; требуется пересчёт."
        row.updated_at = _utc_now()
    session.flush()


def test_sample_preview_path(
    session: Session,
    sample_id: uuid.UUID,
    tile_index: int,
    config: TrainingUIAPIConfig,
) -> Path:
    row = _sample_row(session, sample_id)
    if not any(tile.tile_index == tile_index for tile in row.tiles):
        raise TestSampleUnavailable(str(tile_index))
    path = _sample_root(config, row.id) / f"tile_{tile_index:03d}_preview.png"
    if not path.is_file():
        raise TestSampleUnavailable(str(path))
    return path


def build_test_sample_download(
    session: Session,
    sample_id: uuid.UUID,
    config: TrainingUIAPIConfig,
) -> TestSampleDownloadArtifact:
    row = _sample_row(session, sample_id)
    enabled = [tile for tile in row.tiles if tile.enabled]
    if not enabled:
        raise TrainingUIAPIError("В тестовой выборке нет включённых тайлов.")
    source_root = _sample_root(config, row.id)
    download_root = Path(config.scratch_root) / TEST_SAMPLE_DOWNLOAD_ROOT_NAME
    download_root.mkdir(parents=True, exist_ok=True)
    archive_path = download_root / f"{row.id}-{uuid.uuid4()}.zip"
    try:
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for tile in enabled:
                base_name = f"tile_{tile.tile_index:03d}"
                for suffix in _TILE_SUFFIXES:
                    path = source_root / f"{base_name}{suffix}"
                    if not path.is_file():
                        raise TrainingUIAPIError(
                            f"Файл тестового тайла не найден: {path.name}"
                        )
                    archive.write(path, path.name)
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return TestSampleDownloadArtifact(
        path=archive_path,
        filename=f"{_safe_name(row.class_name.casefold(), 'markup')}_test_markup.zip",
    )


def cleanup_test_sample_storage(
    session: Session,
    config: TrainingUIAPIConfig,
) -> None:
    del session
    shutil.rmtree(
        Path(config.scratch_root) / TEST_SAMPLE_DOWNLOAD_ROOT_NAME,
        ignore_errors=True,
    )
    root = _test_sample_root(config)
    if not root.is_dir():
        return
    for child in root.iterdir():
        if child.name.startswith((".building-", ".deleting-")):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)


def _calculate_metrics(
    row: TestSampleRow,
    prediction_path: Path,
    config: TrainingUIAPIConfig,
) -> tuple[_MetricCounts, _MetricCounts]:
    enabled = {tile.tile_index for tile in row.tiles if tile.enabled}
    tile_metrics = _calculate_tile_metrics(
        row,
        prediction_path,
        config,
        tile_indices=enabled,
    )
    return (
        _sum_tile_metrics(tile_metrics, enabled, metric_index=0),
        _sum_tile_metrics(tile_metrics, enabled, metric_index=1),
    )


def _calculate_tile_metrics(
    row: TestSampleRow,
    prediction_path: Path,
    config: TrainingUIAPIConfig,
    *,
    tile_indices: set[int] | None = None,
) -> dict[int, tuple[_MetricCounts, _MetricCounts]]:
    predictions = _load_geometries(prediction_path, default_crs="EPSG:4326")
    result: dict[int, tuple[_MetricCounts, _MetricCounts]] = {}
    sample_root = _sample_root(config, row.id)
    for tile in row.tiles:
        if tile_indices is not None and tile.tile_index not in tile_indices:
            continue
        base_name = f"tile_{tile.tile_index:03d}"
        tif_path = sample_root / f"{base_name}.tif"
        mask_path = sample_root / f"{base_name}_mask.png"
        geojson_path = sample_root / f"{base_name}.geojson"
        if not tif_path.is_file() or not mask_path.is_file() or not geojson_path.is_file():
            raise TrainingUIAPIError(f"Неполный набор файлов тайла {base_name}.")
        with rasterio.open(tif_path) as raster:
            if raster.crs is None:
                raise TrainingUIAPIError(f"У тайла {base_name} отсутствует CRS.")
            raster_crs = PyprojCRS.from_user_input(raster.crs)
            tile_footprint = box(*raster.bounds)
            predicted_geometries = _prediction_geometries_for_tile(
                predictions,
                raster_crs,
                tile_footprint,
            )
            predicted_mask = (
                rasterize(
                    [(geometry, 1) for geometry in predicted_geometries],
                    out_shape=(raster.height, raster.width),
                    transform=raster.transform,
                    fill=0,
                    dtype="uint8",
                    all_touched=False,
                )
                if predicted_geometries
                else np.zeros((raster.height, raster.width), dtype=np.uint8)
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(mask_path) as mask_dataset:
                true_mask = mask_dataset.read(1) > 0
        if true_mask.shape != predicted_mask.shape:
            raise TrainingUIAPIError(f"Размер маски не совпадает с TIFF для {base_name}.")
        pixel_counts = _pixel_counts(true_mask, predicted_mask > 0)
        ground_truth = _load_geometries(geojson_path, default_crs=str(raster_crs))
        ground_truth_geometries = _geometries_for_tile(
            ground_truth.geometries,
            ground_truth.crs,
            raster_crs,
            tile_footprint,
        )
        result[tile.tile_index] = (
            pixel_counts,
            _object_counts(
                ground_truth_geometries,
                predicted_geometries,
                row.object_iou_threshold,
            ),
        )
    return result


def _prediction_geometries_for_tile(
    predictions: _GeometrySet,
    raster_crs: PyprojCRS,
    tile_footprint: BaseGeometry,
) -> list[BaseGeometry]:
    query_footprint = _transform_geometry(tile_footprint, raster_crs, predictions.crs)
    indices = predictions.tree.query(query_footprint, predicate="intersects")
    selected = [predictions.geometries[int(index)] for index in indices]
    return _geometries_for_tile(
        selected,
        predictions.crs,
        raster_crs,
        tile_footprint,
    )


def _geometries_for_tile(
    geometries: tuple[BaseGeometry, ...] | list[BaseGeometry],
    source_crs: PyprojCRS,
    target_crs: PyprojCRS,
    tile_footprint: BaseGeometry,
) -> list[BaseGeometry]:
    result: list[BaseGeometry] = []
    for geometry in geometries:
        transformed = _transform_geometry(geometry, source_crs, target_crs)
        clipped = _polygonal_geometry(transformed.intersection(tile_footprint))
        if clipped is not None and not clipped.is_empty and clipped.area > 0.0:
            result.append(clipped)
    return result


def _pixel_counts(true_mask: np.ndarray, predicted_mask: np.ndarray) -> _MetricCounts:
    return _MetricCounts(
        true_positive=int(np.logical_and(true_mask, predicted_mask).sum()),
        false_positive=int(np.logical_and(~true_mask, predicted_mask).sum()),
        false_negative=int(np.logical_and(true_mask, ~predicted_mask).sum()),
    )


def _object_counts(
    ground_truth: list[BaseGeometry],
    predicted: list[BaseGeometry],
    threshold: float,
) -> _MetricCounts:
    adjacency: list[list[int]] = []
    for truth in ground_truth:
        matches: list[tuple[float, int]] = []
        for index, prediction in enumerate(predicted):
            intersection_area = truth.intersection(prediction).area
            if intersection_area <= 0.0:
                continue
            union_area = truth.union(prediction).area
            iou = intersection_area / union_area if union_area > 0.0 else 0.0
            if iou >= threshold:
                matches.append((iou, index))
        matches.sort(key=lambda item: (-item[0], item[1]))
        adjacency.append([index for _, index in matches])

    assigned_truth_by_prediction: dict[int, int] = {}

    def assign(truth_index: int, visited: set[int]) -> bool:
        for prediction_index in adjacency[truth_index]:
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            previous = assigned_truth_by_prediction.get(prediction_index)
            if previous is None or assign(previous, visited):
                assigned_truth_by_prediction[prediction_index] = truth_index
                return True
        return False

    true_positive = sum(assign(index, set()) for index in range(len(ground_truth)))
    return _MetricCounts(
        true_positive=true_positive,
        false_positive=len(predicted) - true_positive,
        false_negative=len(ground_truth) - true_positive,
    )


def _load_geometries(path: Path, *, default_crs: str) -> _GeometrySet:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise TrainingUIAPIError(f"Не удалось прочитать GeoJSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise TrainingUIAPIError("GeoJSON должен быть FeatureCollection.")
    raw_features = payload.get("features")
    if not isinstance(raw_features, list):
        raise TrainingUIAPIError("В GeoJSON отсутствует список features.")
    geometries: list[BaseGeometry] = []
    for feature in raw_features:
        if not isinstance(feature, dict) or not feature.get("geometry"):
            continue
        try:
            geometry = _polygonal_geometry(shape(feature["geometry"]))
        except Exception:  # noqa: BLE001
            geometry = None
        if geometry is not None and not geometry.is_empty and geometry.area > 0.0:
            geometries.append(geometry)
    crs = _payload_crs(payload, default_crs)
    geometry_tuple = tuple(geometries)
    return _GeometrySet(crs=crs, geometries=geometry_tuple, tree=STRtree(geometry_tuple))


def _payload_crs(payload: dict[str, Any], default: str) -> PyprojCRS:
    raw_crs = payload.get("crs")
    if raw_crs is None:
        return PyprojCRS.from_user_input(default)
    value: Any = raw_crs
    if isinstance(raw_crs, dict):
        properties = raw_crs.get("properties")
        if isinstance(properties, dict):
            value = properties.get("name") or properties.get("href")
        if not value:
            value = raw_crs.get("name")
    try:
        return PyprojCRS.from_user_input(value)
    except Exception as exc:  # noqa: BLE001
        raise TrainingUIAPIError(f"Не удалось определить CRS GeoJSON: {value}") from exc


def _transform_geometry(
    geometry: BaseGeometry,
    source_crs: PyprojCRS,
    target_crs: PyprojCRS,
) -> BaseGeometry:
    if source_crs == target_crs:
        return geometry
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return transform_geometry(transformer.transform, geometry)


def _polygonal_geometry(geometry: BaseGeometry) -> Polygon | MultiPolygon | None:
    if geometry.is_empty:
        return None
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygons = [
            part
            for part in geometry.geoms
            if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty
        ]
        if not polygons:
            return None
        merged = unary_union(polygons)
        return merged if isinstance(merged, (Polygon, MultiPolygon)) else None
    return None


def _latest_pseudo_markup(
    session: Session,
    dataset_key: str,
) -> PseudoMarkupResultRow | None:
    return session.scalar(
        select(PseudoMarkupResultRow)
        .where(
            PseudoMarkupResultRow.status == "ok",
            PseudoMarkupResultRow.class_key == dataset_key,
            PseudoMarkupResultRow.dataset_key == dataset_key,
            PseudoMarkupResultRow.geojson_file_id.is_not(None),
        )
        .options(
            selectinload(PseudoMarkupResultRow.geojson_file),
            selectinload(PseudoMarkupResultRow.training_result),
        )
        .order_by(
            PseudoMarkupResultRow.updated_at.desc(),
            PseudoMarkupResultRow.created_at.desc(),
            PseudoMarkupResultRow.id.desc(),
        )
        .limit(1)
    )


def _sample_row(session: Session, sample_id: uuid.UUID) -> TestSampleRow:
    row = session.scalar(
        select(TestSampleRow)
        .where(TestSampleRow.id == sample_id)
        .options(selectinload(TestSampleRow.tiles))
    )
    if row is None:
        raise TestSampleUnavailable(str(sample_id))
    return row


def _summary(row: TestSampleRow) -> TestSampleSummary:
    enabled = [tile for tile in row.tiles if tile.enabled]
    return TestSampleSummary(
        id=row.id,
        name=row.name,
        dataset_key=row.dataset_key,
        dataset_name=row.dataset_name,
        dataset_version=row.dataset_version,
        class_key=row.class_key,
        class_name=row.class_name,
        variant_key=row.variant_key,
        variant_name=row.variant_name,
        image_count=row.image_count,
        enabled_image_count=len(enabled),
        actual_object_count=row.actual_object_count,
        enabled_object_count=sum(tile.object_count for tile in enabled),
        evaluation=_evaluation_info(row),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _detail(row: TestSampleRow) -> TestSampleDetail:
    summary = _summary(row)
    return TestSampleDetail(
        **summary.model_dump(),
        tile_width=row.tile_width,
        tile_height=row.tile_height,
        requested_object_count=row.requested_object_count,
        territory_count=row.territory_count,
        warnings=list(row.warnings or []),
        download_url=f"/api/v1/test-samples/{row.id}/download",
        tiles=[
            TestSampleTileInfo(
                index=tile.tile_index,
                source_name=tile.source_name,
                territory=tile.territory,
                object_count=tile.object_count,
                enabled=tile.enabled,
                preview_url=(
                    f"/api/v1/test-samples/{row.id}/tiles/{tile.tile_index}/preview"
                ),
            )
            for tile in row.tiles
        ],
    )


def _evaluation_info(row: TestSampleRow) -> TestSampleEvaluationInfo:
    status = row.metric_status
    if _has_metrics(row) and row.evaluated_revision != row.content_revision:
        status = "stale"
    return TestSampleEvaluationInfo(
        status=status,
        pixel=_metric_info(
            row.pixel_precision,
            row.pixel_recall,
            row.pixel_f1,
            row.pixel_true_positive,
            row.pixel_false_positive,
            row.pixel_false_negative,
        ),
        objects=_metric_info(
            row.object_precision,
            row.object_recall,
            row.object_f1,
            row.object_true_positive,
            row.object_false_positive,
            row.object_false_negative,
        ),
        object_iou_threshold=row.object_iou_threshold,
        pseudo_markup_result_id=row.evaluation_pseudo_result_id,
        model_name=row.evaluation_model_name,
        markup_created_at=row.evaluation_markup_created_at,
        evaluated_at=row.evaluated_at,
        error=row.evaluation_error,
    )


def _metric_info(
    precision: float | None,
    recall: float | None,
    f1: float | None,
    true_positive: int | None,
    false_positive: int | None,
    false_negative: int | None,
) -> TestSampleMetric | None:
    values = (
        precision,
        recall,
        f1,
        true_positive,
        false_positive,
        false_negative,
    )
    if any(value is None for value in values):
        return None
    return TestSampleMetric(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        true_positive=int(true_positive),
        false_positive=int(false_positive),
        false_negative=int(false_negative),
    )


def _evaluation_unavailable(row: TestSampleRow, message: str) -> None:
    row.metric_status = "stale" if _has_metrics(row) else "unavailable"
    row.evaluation_error = message
    row.updated_at = _utc_now()


def _evaluation_error(row: TestSampleRow, message: str) -> None:
    row.metric_status = "stale" if _has_metrics(row) else "error"
    row.evaluation_error = message
    row.updated_at = _utc_now()


def _has_metrics(row: TestSampleRow) -> bool:
    return row.pixel_f1 is not None and row.object_f1 is not None


def _sample_name(value: str, dataset_name: str) -> str:
    normalized = value.strip()
    if normalized:
        return normalized
    created = datetime.now().astimezone().strftime("%d.%m.%Y %H:%M")
    return f"{dataset_name} — {created}"


def _safe_name(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._-")
    return (normalized or fallback)[:120]


def _test_sample_root(config: TrainingUIAPIConfig) -> Path:
    return Path(config.stored_files_root) / TEST_SAMPLE_ROOT_NAME


def _sample_root(config: TrainingUIAPIConfig, sample_id: uuid.UUID) -> Path:
    root = _test_sample_root(config).resolve()
    path = (root / str(sample_id)).resolve()
    path.relative_to(root)
    return path


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


__all__ = [
    "TestSampleDownloadArtifact",
    "TestSampleUnavailable",
    "build_test_sample_download",
    "cleanup_test_sample_storage",
    "create_test_sample",
    "delete_test_sample",
    "evaluate_test_sample_by_id",
    "evaluate_test_samples_for_pseudo_markup",
    "mark_test_samples_stale_for_pseudo_markup",
    "optimize_test_sample",
    "test_sample_catalog",
    "test_sample_detail",
    "test_sample_preview_path",
    "update_test_sample",
    "update_test_sample_tile",
]
