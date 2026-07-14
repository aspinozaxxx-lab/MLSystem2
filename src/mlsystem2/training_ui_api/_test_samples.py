"""Постоянное хранение и оценка тестовых выборок."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
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
from sqlalchemy.orm import Session, selectinload, sessionmaker

from ._config import TrainingUIAPIConfig
from ._dataset_catalog import find_managed_dataset
from ._markup_export import (
    _run_milp,
    _single_constraint,
    generate_markup_files,
    generate_markup_pool_files,
)
from ._models import (
    InferenceTemplateRow,
    JobRow,
    PseudoMarkupResultRow,
    TestSampleBatchItemRow,
    TestSampleBatchRow,
    TestSampleRow,
    TestSampleTileRow,
    TrainingResultRow,
    TrainingResultTestMetricRow,
)
from ._processes import terminate_job_process
from ._pseudo_runner import postprocess_profile_name
from ._queueing import next_queue_position
from ._templates import sanitize_inference_template_config
from .contracts import (
    JobSource,
    JobStatus,
    JobType,
    MarkupExportRequest,
    RuntimeProgress,
    TestSampleBatchCreate,
    TestSampleBatchInfo,
    TestSampleBatchItemInfo,
    TestSampleCatalogResponse,
    TestSampleClassGroup,
    TestSampleCreate,
    TestSampleDetail,
    TestSampleEvaluationInfo,
    TestSampleMetric,
    TestSampleOptimizeRequest,
    TestSamplePrimaryUpdate,
    TestSampleSummary,
    TestSampleTileInfo,
    TestSampleTileUpdate,
    TestSampleUpdate,
    TestSampleVariantGroup,
    TrainingResultTestF1Info,
    TrainingUIAPIError,
)


TEST_SAMPLE_ROOT_NAME = "test-samples"
TEST_SAMPLE_DOWNLOAD_ROOT_NAME = "test-sample-downloads"
TEST_SAMPLE_F1_OPERATION = "test_sample_f1"
TEST_SAMPLE_F1_EVALUATOR_VERSION = 3
OBJECT_IOU_THRESHOLD = 0.5
_TILE_SUFFIXES = (".tif", ".geojson", "_mask.png", "_preview.png")
_BATCH_ACTIVE_STATUSES = ("queued", "running")
_BATCH_FINISHED_ITEM_STATUSES = ("ok", "error")
LOGGER = logging.getLogger(__name__)


class TestSampleUnavailable(FileNotFoundError):
    """Тестовая выборка или её постоянный файл не найдены."""


class TestSampleBatchUnavailable(FileNotFoundError):
    """Групповой запуск тестовых выборок не найден."""


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
        dataset = find_managed_dataset(session, config, request.dataset_key)
        if dataset is None or dataset.is_custom:
            raise TrainingUIAPIError(f"Подкласс не найден: {request.dataset_key}")
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
            dataset=dataset,
        )
        building_root.replace(final_root)
        row = _new_test_sample_row(
            sample_id,
            request.name,
            generated,
            quality_metric=dataset.quality_metric,
        )
        session.add(row)
        session.flush()
        evaluate_test_sample(session, row, config)
        session.flush()
        return _detail(row)
    except Exception:
        shutil.rmtree(building_root, ignore_errors=True)
        shutil.rmtree(final_root, ignore_errors=True)
        raise


def _new_test_sample_row(
    sample_id,
    requested_name: str,
    generated,
    *,
    quality_metric: str = "pixel",
) -> TestSampleRow:
    row = TestSampleRow(
        id=sample_id,
        name=_sample_name(requested_name, generated.dataset_name),
        dataset_key=generated.dataset_key,
        dataset_name=generated.dataset_name,
        dataset_version=generated.dataset_version,
        class_key=generated.class_key,
        class_name=generated.class_name,
        variant_key=generated.variant_key,
        variant_name=generated.variant_name,
        quality_metric=quality_metric,
        tile_width=generated.tile_width,
        tile_height=generated.tile_height,
        image_count=len(generated.tiles),
        requested_object_count=generated.requested_object_count,
        actual_object_count=generated.actual_object_count,
        territory_count=generated.territory_count,
        is_primary=False,
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
    return row


def create_test_sample_batch(
    session: Session,
    request: TestSampleBatchCreate,
    config: TrainingUIAPIConfig,
) -> TestSampleBatchInfo:
    active = session.scalar(
        select(TestSampleBatchRow.id).where(TestSampleBatchRow.active_slot == 1).limit(1)
    )
    if active is not None:
        raise TrainingUIAPIError(
            "Групповое создание тестовых выборок уже выполняется. Дождитесь его завершения."
        )
    keys = [item.dataset_key for item in request.items]
    if len(keys) != len(set(keys)):
        raise TrainingUIAPIError("Один подкласс нельзя добавить в групповой запуск дважды.")

    batch = TestSampleBatchRow(
        status="queued",
        active_slot=1,
        tile_size=request.tile_size,
        min_image_count=request.min_image_count,
        image_count=request.image_count,
    )
    rows: list[TestSampleBatchItemRow] = []
    for position, item in enumerate(request.items, start=1):
        dataset = find_managed_dataset(session, config, item.dataset_key)
        if dataset is None or dataset.is_custom:
            raise TrainingUIAPIError(f"Подкласс не найден: {item.dataset_key}")
        if dataset.diagnostics:
            raise TrainingUIAPIError(f"{dataset.name}: {'; '.join(dataset.diagnostics)}")
        if not dataset.scenes_file or not dataset.annotation_file:
            raise TrainingUIAPIError(
                f"{dataset.name}: нужны TXT со сценами и один positive GeoJSON."
            )
        class_name = dataset.class_name or dataset.name.split("\\", maxsplit=1)[0]
        variant_name = dataset.variant_name or dataset.variant_key or "main"
        rows.append(
            TestSampleBatchItemRow(
                position=position,
                dataset_key=dataset.key,
                dataset_name=dataset.name,
                dataset_version=dataset.version,
                class_key=dataset.class_key or class_name,
                class_name=class_name,
                variant_key=dataset.variant_key or variant_name,
                variant_name=variant_name,
                min_object_count=item.min_object_count,
                metric=dataset.quality_metric,
                status="queued",
            )
        )
    batch.items = rows
    session.add(batch)
    session.flush()
    return _batch_info(batch)


def latest_test_sample_batch(session: Session) -> TestSampleBatchInfo:
    row = session.scalar(
        select(TestSampleBatchRow)
        .options(
            selectinload(TestSampleBatchRow.items).selectinload(
                TestSampleBatchItemRow.sample
            )
        )
        .order_by(TestSampleBatchRow.created_at.desc(), TestSampleBatchRow.id.desc())
        .limit(1)
    )
    if row is None:
        raise TestSampleBatchUnavailable()
    return _batch_info(row)


def test_sample_batch_detail(
    session: Session,
    batch_id: uuid.UUID,
) -> TestSampleBatchInfo:
    row = session.scalar(
        select(TestSampleBatchRow)
        .where(TestSampleBatchRow.id == batch_id)
        .options(
            selectinload(TestSampleBatchRow.items).selectinload(
                TestSampleBatchItemRow.sample
            )
        )
    )
    if row is None:
        raise TestSampleBatchUnavailable(str(batch_id))
    return _batch_info(row)


def recover_test_sample_batches(session: Session) -> None:
    rows = session.scalars(
        select(TestSampleBatchRow)
        .where(TestSampleBatchRow.active_slot == 1)
        .options(selectinload(TestSampleBatchRow.items))
    ).all()
    for row in rows:
        for item in row.items:
            if item.status == "running":
                item.status = "queued"
                item.started_at = None
                item.error = None
        if all(item.status in _BATCH_FINISHED_ITEM_STATUSES for item in row.items):
            _finish_batch(row)
        else:
            row.status = "queued"
            row.updated_at = _utc_now()
    session.flush()


async def run_test_sample_batch_worker(
    session_factory: sessionmaker[Session],
    config: TrainingUIAPIConfig,
) -> None:
    interval = max(1, config.worker_interval_seconds)
    LOGGER.info("Исполнитель групповых тестовых выборок запущен")
    while True:
        try:
            await asyncio.to_thread(
                process_test_sample_batch_once,
                session_factory,
                config,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Ошибка шага группового создания тестовых выборок")
        await asyncio.sleep(interval)


def process_test_sample_batch_once(
    session_factory: sessionmaker[Session],
    config: TrainingUIAPIConfig,
) -> None:
    item_id: uuid.UUID | None = None
    with session_factory() as session:
        batch = session.scalar(
            select(TestSampleBatchRow)
            .where(TestSampleBatchRow.active_slot == 1)
            .options(selectinload(TestSampleBatchRow.items))
            .order_by(TestSampleBatchRow.created_at, TestSampleBatchRow.id)
            .limit(1)
        )
        if batch is None:
            return
        item = next((row for row in batch.items if row.status == "queued"), None)
        if item is None:
            if all(row.status in _BATCH_FINISHED_ITEM_STATUSES for row in batch.items):
                _finish_batch(batch)
                session.commit()
            return
        now = _utc_now()
        batch.status = "running"
        batch.started_at = batch.started_at or now
        batch.updated_at = now
        item.status = "running"
        item.started_at = now
        item.finished_at = None
        item.error = None
        item.updated_at = now
        item_id = item.id
        session.commit()

    assert item_id is not None
    try:
        with session_factory() as session:
            item = session.get(TestSampleBatchItemRow, item_id)
            if item is None:
                return
            batch = session.get(TestSampleBatchRow, item.batch_id)
            if batch is None:
                return
            detail = _create_grouped_test_sample(session, batch, item, config)
            item.sample_id = detail.id
            item.pool_tile_count = detail.image_count
            item.pool_object_count = detail.actual_object_count
            item.status = "ok"
            item.error = None
            item.finished_at = _utc_now()
            item.updated_at = item.finished_at
            _finish_batch_if_complete(session, batch.id)
            session.commit()
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Не удалось создать тестовую выборку для строки %s", item_id)
        with session_factory() as session:
            item = session.get(TestSampleBatchItemRow, item_id)
            if item is None:
                return
            item.status = "error"
            item.error = str(exc) or exc.__class__.__name__
            item.finished_at = _utc_now()
            item.updated_at = item.finished_at
            _finish_batch_if_complete(session, item.batch_id)
            session.commit()


def _create_grouped_test_sample(
    session: Session,
    batch: TestSampleBatchRow,
    item: TestSampleBatchItemRow,
    config: TrainingUIAPIConfig,
) -> TestSampleDetail:
    source = _latest_pseudo_markup(session, item.dataset_key)
    if source is None or source.geojson_file is None:
        raise TrainingUIAPIError(
            "Нет успешной псевдоразметки точного подкласса для оптимизации состава."
        )
    source_path = Path(source.geojson_file.path)
    if not source_path.is_file():
        raise TrainingUIAPIError("Файл последней псевдоразметки не найден на сервере.")

    sample_id = uuid.uuid4()
    root = _test_sample_root(config)
    root.mkdir(parents=True, exist_ok=True)
    building_root = root / f".building-{sample_id}"
    final_root = root / str(sample_id)
    building_root.mkdir(parents=False, exist_ok=False)
    try:
        dataset = find_managed_dataset(session, config, item.dataset_key)
        if dataset is None or dataset.is_custom:
            raise TrainingUIAPIError(f"Подкласс не найден: {item.dataset_key}")
        generated = generate_markup_pool_files(
            dataset_key=item.dataset_key,
            tile_size=batch.tile_size,
            min_final_image_count=batch.min_image_count,
            max_final_image_count=batch.image_count,
            min_object_count=item.min_object_count,
            config=config,
            output_root=building_root,
            dataset=dataset,
        )
        building_root.replace(final_root)
        row = _new_test_sample_row(
            sample_id,
            "",
            generated,
            quality_metric=dataset.quality_metric,
        )
        session.add(row)
        session.flush()
        _optimize_test_sample_row(
            row,
            TestSampleOptimizeRequest(
                min_tile_count=batch.min_image_count,
                max_tile_count=min(batch.image_count, row.image_count),
                min_object_count=item.min_object_count,
                metric=item.metric,
            ),
            config,
            source,
            source_path,
        )
        session.flush()
        return _detail(row)
    except Exception:
        shutil.rmtree(building_root, ignore_errors=True)
        shutil.rmtree(final_root, ignore_errors=True)
        raise


def _finish_batch_if_complete(session: Session, batch_id: uuid.UUID) -> None:
    batch = session.scalar(
        select(TestSampleBatchRow)
        .where(TestSampleBatchRow.id == batch_id)
        .options(selectinload(TestSampleBatchRow.items))
    )
    if batch is not None and all(
        item.status in _BATCH_FINISHED_ITEM_STATUSES for item in batch.items
    ):
        _finish_batch(batch)


def _finish_batch(batch: TestSampleBatchRow) -> None:
    successful = sum(item.status == "ok" for item in batch.items)
    if successful == len(batch.items):
        batch.status = "ok"
    elif successful:
        batch.status = "partial"
    else:
        batch.status = "error"
    batch.active_slot = None
    batch.finished_at = _utc_now()
    batch.updated_at = batch.finished_at


def _batch_info(row: TestSampleBatchRow) -> TestSampleBatchInfo:
    finished = sum(item.status in _BATCH_FINISHED_ITEM_STATUSES for item in row.items)
    end = row.finished_at or _utc_now()
    start = row.started_at or row.created_at
    elapsed = max(0, int((_aware_datetime(end) - _aware_datetime(start)).total_seconds()))
    return TestSampleBatchInfo(
        id=row.id,
        status=row.status,
        tile_size=row.tile_size,
        min_image_count=row.min_image_count,
        image_count=row.image_count,
        completed_count=finished,
        total_count=len(row.items),
        elapsed_seconds=elapsed,
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        items=[
            TestSampleBatchItemInfo(
                id=item.id,
                position=item.position,
                dataset_key=item.dataset_key,
                dataset_name=item.dataset_name,
                dataset_version=item.dataset_version,
                class_key=item.class_key,
                class_name=item.class_name,
                variant_key=item.variant_key,
                variant_name=item.variant_name,
                min_object_count=item.min_object_count,
                metric=item.metric,
                status=item.status,
                pool_tile_count=item.pool_tile_count,
                pool_object_count=item.pool_object_count,
                sample_id=item.sample_id,
                sample_name=item.sample.name if item.sample is not None else None,
                error=item.error,
                started_at=item.started_at,
                finished_at=item.finished_at,
            )
            for item in row.items
        ],
    )


def _aware_datetime(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


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


def update_test_sample_primary(
    session: Session,
    sample_id: uuid.UUID,
    request: TestSamplePrimaryUpdate,
) -> TestSampleDetail:
    row = _sample_row(session, sample_id)
    if row.is_primary == request.is_primary:
        return _detail(row)
    if request.is_primary:
        current_rows = session.scalars(
            select(TestSampleRow).where(
                TestSampleRow.dataset_key == row.dataset_key,
                TestSampleRow.is_primary.is_(True),
                TestSampleRow.id != row.id,
            )
        ).all()
        for current in current_rows:
            current.is_primary = False
            current.updated_at = _utc_now()
        if current_rows:
            session.flush()
        row.is_primary = True
    else:
        row.is_primary = False
    row.updated_at = _utc_now()
    _mark_training_test_metrics_stale(
        session,
        row.dataset_key,
        "Основная тестовая выборка изменена; требуется пересчёт.",
    )
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
        if row.is_primary:
            _mark_training_test_metrics_stale(
                session,
                row.dataset_key,
                "Состав основной тестовой выборки изменён; требуется пересчёт.",
            )
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
        if row.is_primary:
            _mark_training_test_metrics_stale(
                session,
                row.dataset_key,
                "Основная тестовая выборка удалена.",
                unavailable=True,
            )
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

    previous_revision = row.content_revision
    _optimize_test_sample_row(row, request, config, source, source_path)
    if row.is_primary and row.content_revision != previous_revision:
        _mark_training_test_metrics_stale(
            session,
            row.dataset_key,
            "Состав основной тестовой выборки оптимизирован; требуется пересчёт.",
        )
    session.flush()
    return _detail(row)


def _optimize_test_sample_row(
    row: TestSampleRow,
    request: TestSampleOptimizeRequest,
    config: TrainingUIAPIConfig,
    source: PseudoMarkupResultRow,
    source_path: Path | None = None,
) -> None:
    _validate_optimization_request(row, request)
    request = request.model_copy(update={"metric": row.quality_metric})
    prediction_path = source_path or (
        Path(source.geojson_file.path) if source.geojson_file is not None else None
    )
    if prediction_path is None or not prediction_path.is_file():
        raise TrainingUIAPIError(
            "Файл разметки для оптимизации не найден на сервере."
        )
    tile_metrics = _calculate_tile_metrics(row, prediction_path, config)
    selected_indices = _select_optimized_tile_indices(row.tiles, tile_metrics, request)
    selected = set(selected_indices)
    pixel_counts = _sum_tile_metrics(tile_metrics, selected, metric_index=0)
    object_counts = _sum_tile_metrics(tile_metrics, selected, metric_index=1)

    now = _utc_now()
    changed = False
    for tile in row.tiles:
        enabled = tile.tile_index in selected
        if tile.enabled != enabled:
            tile.enabled = enabled
            tile.updated_at = now
            changed = True
    if changed:
        row.content_revision += 1
    row.updated_at = now
    _apply_evaluation(row, source, pixel_counts, object_counts)


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


def build_primary_test_samples_download(
    session: Session,
    config: TrainingUIAPIConfig,
) -> TestSampleDownloadArtifact:
    rows = session.scalars(
        select(TestSampleRow)
        .where(TestSampleRow.is_primary.is_(True))
        .options(selectinload(TestSampleRow.tiles))
        .order_by(
            TestSampleRow.class_name,
            TestSampleRow.variant_name,
            TestSampleRow.id,
        )
    ).all()
    if not rows:
        raise TrainingUIAPIError("Основные тестовые выборки не назначены.")
    empty = [row.name for row in rows if not any(tile.enabled for tile in row.tiles)]
    if empty:
        raise TrainingUIAPIError(
            "В основных тестовых выборках нет включённых тайлов: " + ", ".join(empty)
        )

    download_root = Path(config.scratch_root) / TEST_SAMPLE_DOWNLOAD_ROOT_NAME
    download_root.mkdir(parents=True, exist_ok=True)
    archive_path = download_root / f"primary-{uuid.uuid4()}.zip"
    used_folders: set[str] = set()
    try:
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for row in rows:
                folder = _safe_name(
                    f"{row.class_name}_{row.variant_name}",
                    "test_sample",
                )
                if folder in used_folders:
                    folder = f"{folder}_{str(row.id)[:8]}"
                used_folders.add(folder)
                source_root = _sample_root(config, row.id)
                for tile in row.tiles:
                    if not tile.enabled:
                        continue
                    base_name = f"tile_{tile.tile_index:03d}"
                    for suffix in _TILE_SUFFIXES:
                        path = source_root / f"{base_name}{suffix}"
                        if not path.is_file():
                            raise TrainingUIAPIError(
                                f"Файл основной тестовой выборки не найден: {path.name}"
                            )
                        archive.write(path, f"{folder}/{path.name}")
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return TestSampleDownloadArtifact(
        path=archive_path,
        filename="основные_тестовые_выборки.zip",
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


def _mark_training_test_metrics_stale(
    session: Session,
    dataset_key: str,
    reason: str,
    *,
    unavailable: bool = False,
) -> None:
    rows = session.scalars(
        select(TrainingResultTestMetricRow)
        .join(
            TrainingResultRow,
            TrainingResultRow.id == TrainingResultTestMetricRow.training_result_id,
        )
        .where(TrainingResultRow.class_key == dataset_key)
    ).all()
    for row in rows:
        _cancel_test_metric_job(session, row)
        if unavailable and row.f1 is None:
            row.status = "unavailable"
        else:
            row.status = "stale" if row.f1 is not None else "unavailable"
        row.error = reason
        row.job_id = None
        row.updated_at = _utc_now()


def queue_class_test_f1(
    session: Session,
    dataset_key: str,
    config: TrainingUIAPIConfig,
) -> int:
    """Ставит в общую inference-очередь недостающие оценки успешных сетей."""

    sample = _primary_sample(session, dataset_key)
    if sample is None:
        raise TrainingUIAPIError("Для подкласса не назначена основная тестовая выборка.")
    if not any(tile.enabled for tile in sample.tiles):
        raise TrainingUIAPIError("В основной тестовой выборке нет включённых тайлов.")
    results = session.scalars(
        select(TrainingResultRow)
        .where(
            TrainingResultRow.class_key == dataset_key,
            TrainingResultRow.status == "ok",
        )
        .order_by(TrainingResultRow.created_at.desc(), TrainingResultRow.id.desc())
    ).all()
    created = 0
    for result in results:
        if queue_training_result_test_f1(
            session,
            result,
            config,
            source=JobSource.MANUAL,
        ):
            created += 1
    session.flush()
    return created


def queue_training_result_test_f1(
    session: Session,
    result: TrainingResultRow,
    config: TrainingUIAPIConfig,
    *,
    source: JobSource | None = None,
) -> bool:
    """Создаёт задание F1 для одной сети, если её оценка неактуальна."""

    if result.status != "ok":
        return False
    sample = _primary_sample(session, result.class_key)
    if sample is None or not any(tile.enabled for tile in sample.tiles):
        return False
    postprocess_profile = _test_f1_postprocess_profile_name(session, sample, config)
    template, template_config, config_hash = _effective_inference_template(
        session,
        result.architecture,
        result.class_key,
        postprocess_profile,
    )
    metric = session.get(TrainingResultTestMetricRow, result.id)
    if metric is not None and _metric_matches(
        metric,
        sample,
        template,
        config_hash,
    ):
        if metric.status == "current":
            return False
        if metric.status in {"queued", "running"} and _metric_job_is_active(session, metric):
            return False

    if not result.mlflow_run_id:
        if metric is None:
            metric = TrainingResultTestMetricRow(training_result_id=result.id)
            session.add(metric)
        _cancel_test_metric_job(session, metric)
        metric.sample_id = sample.id
        metric.sample_revision = sample.content_revision
        metric.status = "error"
        metric.job_id = None
        metric.inference_template_id = template.id if template is not None else None
        metric.inference_template_version = template.version if template is not None else None
        metric.inference_config_hash = config_hash
        metric.error = "У результата обучения отсутствует MLflow run id с checkpoint."
        metric.updated_at = _utc_now()
        return False

    if metric is None:
        metric = TrainingResultTestMetricRow(training_result_id=result.id)
        session.add(metric)
        session.flush()
    else:
        _cancel_test_metric_job(session, metric)

    job_source = source or JobSource(result.source)
    job = JobRow(
        type=JobType.INFERENCE.value,
        source=job_source.value,
        status=JobStatus.QUEUED.value,
        queue_position=next_queue_position(session, JobType.INFERENCE, job_source),
        dataset_key=result.class_key,
        dataset_version=result.dataset_version,
        dataset_name=sample.dataset_name,
        training_dataset_name=result.class_display_name,
        inference_dataset_name=sample.name,
        model_name=result.model_name,
        architecture=result.architecture,
        tile_size=sample.tile_width,
        config={
            "operation": TEST_SAMPLE_F1_OPERATION,
            "class_key": result.class_key,
            "training_result_id": str(result.id),
            "test_sample_id": str(sample.id),
            "test_sample_revision": sample.content_revision,
            "test_sample_tile_indices": [
                tile.tile_index for tile in sample.tiles if tile.enabled
            ],
            "inference_template_id": str(template.id) if template is not None else None,
            "inference_template_version": template.version if template is not None else None,
            "inference_template_config": template_config,
            "inference_config_hash": config_hash,
            "postprocess_profile": postprocess_profile,
            "test_f1_evaluator_version": TEST_SAMPLE_F1_EVALUATOR_VERSION,
            "mlflow_run_id": result.mlflow_run_id,
        },
    )
    session.add(job)
    session.flush()
    metric.sample_id = sample.id
    metric.sample_revision = sample.content_revision
    metric.status = "queued"
    metric.job_id = job.id
    metric.inference_template_id = template.id if template is not None else None
    metric.inference_template_version = template.version if template is not None else None
    metric.inference_config_hash = config_hash
    metric.error = None
    metric.updated_at = _utc_now()
    session.flush()
    return True


def training_result_test_f1_info(
    session: Session,
    result: TrainingResultRow,
    config: TrainingUIAPIConfig,
) -> TrainingResultTestF1Info | None:
    sample = _primary_sample(session, result.class_key)
    if sample is None:
        return None
    metric = session.get(TrainingResultTestMetricRow, result.id)
    if metric is None:
        return TrainingResultTestF1Info(
            status="unavailable",
            quality_metric=result.quality_metric,
            sample_id=sample.id,
            sample_name=sample.name,
            sample_revision=sample.content_revision,
            error="Для сети ещё не рассчитан F1 на основной тестовой выборке.",
        )
    postprocess_profile = _test_f1_postprocess_profile_name(session, sample, config)
    template, _, config_hash = _effective_inference_template(
        session,
        result.architecture,
        result.class_key,
        postprocess_profile,
    )
    status = metric.status
    if not _metric_matches(metric, sample, template, config_hash):
        status = "stale" if metric.f1 is not None else "unavailable"
    elif status in {"queued", "running"} and not _metric_job_is_active(session, metric):
        status = "stale" if metric.f1 is not None else "error"
    if status not in {"current", "stale", "queued", "running", "error", "unavailable"}:
        status = "stale" if metric.f1 is not None else "unavailable"
    job = session.get(JobRow, metric.job_id) if metric.job_id is not None else None
    use_objects = result.quality_metric == "objects"
    return TrainingResultTestF1Info(
        status=status,
        precision=metric.object_precision if use_objects else metric.precision,
        recall=metric.object_recall if use_objects else metric.recall,
        f1=metric.object_f1 if use_objects else metric.f1,
        true_positive=metric.object_true_positive if use_objects else metric.true_positive,
        false_positive=metric.object_false_positive if use_objects else metric.false_positive,
        false_negative=metric.object_false_negative if use_objects else metric.false_negative,
        quality_metric=result.quality_metric,
        pixel_precision=metric.precision,
        pixel_recall=metric.recall,
        pixel_f1=metric.f1,
        pixel_true_positive=metric.true_positive,
        pixel_false_positive=metric.false_positive,
        pixel_false_negative=metric.false_negative,
        object_precision=metric.object_precision,
        object_recall=metric.object_recall,
        object_f1=metric.object_f1,
        object_true_positive=metric.object_true_positive,
        object_false_positive=metric.object_false_positive,
        object_false_negative=metric.object_false_negative,
        sample_id=metric.sample_id or sample.id,
        sample_name=metric.sample.name if metric.sample is not None else sample.name,
        sample_revision=metric.sample_revision,
        job_id=metric.job_id,
        evaluated_at=metric.evaluated_at,
        error=metric.error,
        progress=_test_f1_progress(job),
    )


def primary_test_sample(
    session: Session,
    dataset_key: str,
) -> TestSampleRow | None:
    return _primary_sample(session, dataset_key)


def _primary_sample(session: Session, dataset_key: str) -> TestSampleRow | None:
    return session.scalar(
        select(TestSampleRow)
        .where(
            TestSampleRow.dataset_key == dataset_key,
            TestSampleRow.is_primary.is_(True),
        )
        .options(selectinload(TestSampleRow.tiles))
        .limit(1)
    )


def _effective_inference_template(
    session: Session,
    architecture: str,
    dataset_key: str,
    postprocess_profile: str,
) -> tuple[InferenceTemplateRow | None, dict[str, Any], str]:
    template = session.scalar(
        select(InferenceTemplateRow).where(
            InferenceTemplateRow.architecture == architecture,
            InferenceTemplateRow.dataset_key == dataset_key,
        )
    )
    if template is None or not template.is_active:
        template = session.scalar(
            select(InferenceTemplateRow).where(
                InferenceTemplateRow.architecture == architecture,
                InferenceTemplateRow.dataset_key.is_(None),
            )
        )
    template_config = (
        sanitize_inference_template_config(template.default_config)
        if template is not None
        else {}
    )
    serialized = json.dumps(
        {
            "evaluator_version": TEST_SAMPLE_F1_EVALUATOR_VERSION,
            "postprocess_profile": postprocess_profile,
            "template_config": template_config,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return template, template_config, hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _test_f1_postprocess_profile_name(
    session: Session,
    sample: TestSampleRow,
    config: TrainingUIAPIConfig,
) -> str:
    source = (
        session.get(PseudoMarkupResultRow, sample.evaluation_pseudo_result_id)
        if sample.evaluation_pseudo_result_id is not None
        else None
    )
    if source is None or source.image_count is None:
        source = _latest_pseudo_markup(session, sample.dataset_key)
    image_count = source.image_count if source is not None else None
    if image_count is None:
        dataset = find_managed_dataset(session, config, sample.dataset_key)
        image_count = dataset.image_count if dataset is not None else None
    if image_count is None:
        image_count = len({tile.source_name for tile in sample.tiles})
    return postprocess_profile_name(max(0, int(image_count)))


def _metric_matches(
    metric: TrainingResultTestMetricRow,
    sample: TestSampleRow,
    template: InferenceTemplateRow | None,
    config_hash: str,
) -> bool:
    return (
        metric.sample_id == sample.id
        and metric.sample_revision == sample.content_revision
        and metric.inference_template_id == (template.id if template is not None else None)
        and metric.inference_template_version
        == (template.version if template is not None else None)
        and metric.inference_config_hash == config_hash
    )


def _metric_job_is_active(
    session: Session,
    metric: TrainingResultTestMetricRow,
) -> bool:
    if metric.job_id is None:
        return False
    job = session.get(JobRow, metric.job_id)
    return bool(job and job.status in {JobStatus.QUEUED.value, JobStatus.RUNNING.value})


def _cancel_test_metric_job(
    session: Session,
    metric: TrainingResultTestMetricRow,
) -> None:
    if metric.job_id is None:
        return
    job = session.get(JobRow, metric.job_id)
    if job is None or (job.config or {}).get("operation") != TEST_SAMPLE_F1_OPERATION:
        return
    if job.status == JobStatus.RUNNING.value:
        terminate_job_process(job)
        if job.tmp_path:
            shutil.rmtree(job.tmp_path, ignore_errors=True)
            job.tmp_path = None
    if job.status in {JobStatus.QUEUED.value, JobStatus.RUNNING.value}:
        job.status = JobStatus.CANCELLED.value
        job.finished_at = _utc_now()
        job.process_pid = None
    metric.job_id = None


def _test_f1_progress(job: JobRow | None) -> RuntimeProgress | None:
    if job is None or job.status != JobStatus.RUNNING.value or job.tmp_path is None:
        return None
    path = Path(job.tmp_path) / "scratch" / "progress.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    current = payload.get("current")
    total = payload.get("total")
    try:
        parsed_current = int(current) if current is not None else None
        parsed_total = int(total) if total is not None else None
    except (TypeError, ValueError):
        return None
    return RuntimeProgress(
        current=parsed_current,
        total=parsed_total,
        elapsed_minutes=(
            max(0, int((_utc_now() - _aware_datetime(job.started_at)).total_seconds() // 60))
            if job.started_at is not None
            else None
        ),
    )


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
        quality_metric=row.quality_metric,
        image_count=row.image_count,
        enabled_image_count=len(enabled),
        actual_object_count=row.actual_object_count,
        enabled_object_count=sum(tile.object_count for tile in enabled),
        is_primary=row.is_primary,
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
    "TestSampleBatchUnavailable",
    "TestSampleDownloadArtifact",
    "TestSampleUnavailable",
    "build_primary_test_samples_download",
    "build_test_sample_download",
    "cleanup_test_sample_storage",
    "create_test_sample",
    "create_test_sample_batch",
    "delete_test_sample",
    "evaluate_test_sample_by_id",
    "evaluate_test_samples_for_pseudo_markup",
    "latest_test_sample_batch",
    "mark_test_samples_stale_for_pseudo_markup",
    "optimize_test_sample",
    "primary_test_sample",
    "process_test_sample_batch_once",
    "queue_class_test_f1",
    "queue_training_result_test_f1",
    "recover_test_sample_batches",
    "run_test_sample_batch_worker",
    "test_sample_batch_detail",
    "test_sample_catalog",
    "test_sample_detail",
    "test_sample_preview_path",
    "training_result_test_f1_info",
    "update_test_sample",
    "update_test_sample_primary",
    "update_test_sample_tile",
]
