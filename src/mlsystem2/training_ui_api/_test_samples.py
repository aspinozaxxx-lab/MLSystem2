"""Постоянное хранение и оценка тестовых разметок."""

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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from math import gcd
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import rasterio
from PIL import Image
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
from shapely.validation import make_valid
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from ._config import TrainingUIAPIConfig
from ._dataset_catalog import (
    dataset_class_row,
    find_managed_dataset,
    primary_training_result,
)
from ._markup_export import (
    _mask_edge,
    _run_milp,
    _single_constraint,
    _stretch_channel,
    generate_markup_files,
    generate_markup_pool_files,
)
from ._models import (
    DatasetRow,
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
    TestSampleDraftPreview,
    TestSampleEvaluationPreviewRequest,
    TestSampleEvaluationInfo,
    TestSampleMetric,
    TestSampleOptimizeRequest,
    TestSamplePrimaryUpdate,
    TestSamplePseudoMarkupInfo,
    TestSampleSummary,
    TestSampleTileInfo,
    TestSampleTileUpdate,
    TestSampleUpdate,
    TrainingResultTestF1Info,
    TrainingUIAPIError,
)


TEST_SAMPLE_ROOT_NAME = "test-samples"
TEST_SAMPLE_DOWNLOAD_ROOT_NAME = "test-sample-downloads"
TEST_SAMPLE_F1_OPERATION = "test_sample_f1"
TEST_SAMPLE_F1_EVALUATOR_VERSION = 3
TEST_SAMPLE_EVALUATION_TARGET = "test_sample"
TRAINING_RESULT_TEST_METRIC_TARGET = "training_result"
OBJECT_IOU_THRESHOLD = 0.5
_DOWNLOAD_BASE_TILE_SUFFIXES = (".tif", ".geojson")
_DOWNLOAD_PREVIEW_SOURCE_SUFFIXES = ("_mask.png",)
_BULK_DOWNLOAD_MAX_WORKERS = 8
_JPEG_PREVIEW_CHANNELS = {
    "rgb": (0, 1, 2),
    "nrg": (3, 0, 1),
    "ngb": (3, 1, 2),
}
_JPEG_PREVIEW_MAX_BYTES = 300 * 1024
_JPEG_QUALITY_MIN = 1
_JPEG_QUALITY_MAX = 95
_BATCH_ACTIVE_STATUSES = ("queued", "running")
_BATCH_FINISHED_ITEM_STATUSES = ("ok", "error")
LOGGER = logging.getLogger(__name__)


class TestSampleUnavailable(FileNotFoundError):
    """Тестовая разметка или её постоянный файл не найдены."""


class TestSampleBatchUnavailable(FileNotFoundError):
    """Групповой запуск тестовых разметок не найден."""


@dataclass(frozen=True)
class TestSampleDownloadArtifact:
    path: Path
    filename: str

    def cleanup(self) -> None:
        self.path.unlink(missing_ok=True)


@dataclass(frozen=True)
class _TestSampleDownloadDescriptor:
    sample_id: uuid.UUID
    folder: str
    source_root: Path
    tile_indices: tuple[int, ...]


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
    class_slugs: tuple[str | None, ...]
    tree: STRtree


_TileMetric = tuple[_MetricCounts, _MetricCounts, dict[str, Any]]


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
            raise TrainingUIAPIError(f"Датасет не найден: {request.dataset_key}")
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
        _normalize_test_sample_class(session, row)
        session.add(row)
        session.flush()
        evaluate_test_sample(session, row, config)
        queue_test_sample_evaluation(
            session,
            row,
            config,
            source=JobSource.AUTOMATION,
        )
        session.flush()
        return _detail(session, row)
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
        dataset_short_name=generated.dataset_short_name,
        quality_metric=quality_metric,
        task=generated.task,
        class_schema=list(generated.class_schema),
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
            class_object_counts=dict(tile.class_object_counts),
            enabled=True,
        )
        for tile in generated.tiles
    ]
    return row


def _normalize_test_sample_class(session: Session, row: TestSampleRow) -> None:
    class_row = dataset_class_row(session, row.dataset_key)
    if class_row is None:
        return
    row.class_key = class_row.key
    row.class_name = class_row.name


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
            "Групповое создание тестовых разметок уже выполняется. Дождитесь его завершения."
        )
    keys = [item.dataset_key for item in request.items]
    if len(keys) != len(set(keys)):
        raise TrainingUIAPIError("Один датасет нельзя добавить в групповой запуск дважды.")

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
            raise TrainingUIAPIError(f"Датасет не найден: {item.dataset_key}")
        if dataset.diagnostics:
            raise TrainingUIAPIError(f"{dataset.name}: {'; '.join(dataset.diagnostics)}")
        if not (
            (dataset.scenes_file and dataset.annotation_file)
            or (dataset.annotations_dir and (dataset.image_count or 0) > 0)
        ):
            raise TrainingUIAPIError(
                f"{dataset.name}: датасет не готов к формированию тестовой разметки."
            )
        class_name = dataset.class_name or dataset.name.split("\\", maxsplit=1)[0]
        dataset_name = dataset.dataset_name or dataset.name
        rows.append(
            TestSampleBatchItemRow(
                position=position,
                dataset_key=dataset.key,
                dataset_name=dataset.name,
                dataset_version=dataset.version,
                class_key=dataset.class_key or class_name,
                class_name=class_name,
                dataset_short_name=dataset_name,
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
    LOGGER.info("Исполнитель групповых тестовых разметок запущен")
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
            LOGGER.exception("Ошибка шага группового создания тестовых разметок")
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
        LOGGER.exception("Не удалось создать тестовую разметку для строки %s", item_id)
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
    source = _latest_pseudo_markup(
        session,
        item.dataset_key,
        class_key=item.class_key,
    )
    if source is None or source.geojson_file is None:
        raise TrainingUIAPIError(
            "Нет успешной псевдоразметки выбранного датасета для оптимизации состава."
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
            raise TrainingUIAPIError(f"Датасет не найден: {item.dataset_key}")
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
        _normalize_test_sample_class(session, row)
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
        queue_test_sample_evaluation(
            session,
            row,
            config,
            source=JobSource.AUTOMATION,
        )
        session.flush()
        return _detail(session, row)
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
    grouped: dict[tuple[str, str], list[TestSampleSummary]] = defaultdict(list)
    for row in rows:
        grouped[(row.class_key, row.class_name)].append(_summary(session, row))
    classes = [
        TestSampleClassGroup(
            key=class_key,
            name=class_name,
            samples=samples,
        )
        for (class_key, class_name), samples in sorted(
            grouped.items(), key=lambda item: item[0][1].casefold()
        )
    ]
    return TestSampleCatalogResponse(classes=classes)


def test_sample_detail(
    session: Session,
    sample_id: uuid.UUID,
    config: TrainingUIAPIConfig | None = None,
) -> TestSampleDetail:
    row = _sample_row(session, sample_id)
    if config is not None:
        _backfill_test_sample_tile_f1(session, row, config)
    return _detail(session, row)


def _backfill_test_sample_tile_f1(
    session: Session,
    row: TestSampleRow,
    config: TrainingUIAPIConfig,
) -> None:
    if all(
        tile.pixel_f1 is not None and tile.object_f1 is not None
        for tile in row.tiles
    ):
        return
    if (
        row.evaluation_pseudo_result_id is None
    ):
        return
    source = session.get(PseudoMarkupResultRow, row.evaluation_pseudo_result_id)
    if source is None or source.status != "ok" or source.geojson_file is None:
        return
    source_path = Path(source.geojson_file.path)
    if not source_path.is_file():
        return
    try:
        tile_metrics = _calculate_tile_metrics(row, source_path, config)
    except Exception:  # noqa: BLE001
        return
    _apply_test_sample_tile_f1(row, tile_metrics)
    session.flush()


def update_test_sample(
    session: Session,
    sample_id: uuid.UUID,
    request: TestSampleUpdate,
    config: TrainingUIAPIConfig,
) -> TestSampleDetail:
    row = _sample_row(session, sample_id)
    was_primary = row.is_primary
    content_changed = False
    primary_changed = False

    if request.name is not None:
        name = request.name.strip()
        if not name:
            raise TrainingUIAPIError("Название тестовой разметки не может быть пустым.")
        if row.name != name:
            row.name = name
            row.updated_at = _utc_now()

    if request.enabled_tile_indices is not None:
        enabled_indices = _validated_tile_indices(row, request.enabled_tile_indices)
        now = _utc_now()
        for tile in row.tiles:
            enabled = tile.tile_index in enabled_indices
            if tile.enabled != enabled:
                tile.enabled = enabled
                tile.updated_at = now
                content_changed = True
        if content_changed:
            row.content_revision += 1
            row.updated_at = now
            _mark_test_sample_evaluation_stale(
                session,
                row,
                "Состав тестовой разметки изменён; требуется пересчёт основной сетью.",
            )

    if request.is_primary is not None:
        primary_changed = _set_test_sample_primary(session, row, request.is_primary)

    if primary_changed or (content_changed and (was_primary or row.is_primary)):
        _refresh_training_metrics_after_primary_change(
            session,
            row.class_key,
            config,
            reason=(
                "Основная тестовая разметка изменена; требуется пересчёт."
                if primary_changed
                else "Состав основной тестовой разметки изменён; требуется пересчёт."
            ),
        )
    elif request.enabled_tile_indices is not None and row.is_primary:
        reconcile_training_result_test_f1(
            session,
            config,
            dataset_keys=_class_scope_keys(session, row.class_key),
        )
    if content_changed:
        queue_test_sample_evaluation(
            session,
            row,
            config,
            source=JobSource.AUTOMATION,
        )
    session.flush()
    return _detail(session, row)


def update_test_sample_primary(
    session: Session,
    sample_id: uuid.UUID,
    request: TestSamplePrimaryUpdate,
    config: TrainingUIAPIConfig | None = None,
) -> TestSampleDetail:
    row = _sample_row(session, sample_id)
    if not _set_test_sample_primary(session, row, request.is_primary):
        return _detail(session, row)
    _refresh_training_metrics_after_primary_change(
        session,
        row.class_key,
        config,
        reason="Основная тестовая разметка изменена; требуется пересчёт.",
    )
    session.flush()
    return _detail(session, row)


def _set_test_sample_primary(
    session: Session,
    row: TestSampleRow,
    is_primary: bool,
) -> bool:
    if row.is_primary == is_primary:
        return False
    if is_primary:
        current_rows = session.scalars(
            select(TestSampleRow).where(
                TestSampleRow.class_key.in_(_class_scope_keys(session, row.class_key)),
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
    return True


def _validated_tile_indices(row: TestSampleRow, values: list[int]) -> set[int]:
    selected = set(values)
    if len(selected) != len(values):
        raise TrainingUIAPIError("Индексы тайлов тестовой разметки не должны повторяться.")
    available = {tile.tile_index for tile in row.tiles}
    unknown = sorted(selected - available)
    if unknown:
        rendered = ", ".join(str(index) for index in unknown)
        raise TrainingUIAPIError(f"Тайлы тестовой разметки не найдены: {rendered}.")
    return selected


def update_test_sample_tile(
    session: Session,
    sample_id: uuid.UUID,
    tile_index: int,
    request: TestSampleTileUpdate,
    config: TrainingUIAPIConfig | None = None,
) -> TestSampleDetail:
    row = _sample_row(session, sample_id)
    tile = next((item for item in row.tiles if item.tile_index == tile_index), None)
    if tile is None:
        raise TestSampleUnavailable(str(tile_index))
    if tile.enabled != request.enabled:
        tile.enabled = request.enabled
        tile.updated_at = _utc_now()
        row.content_revision += 1
        _mark_test_sample_evaluation_stale(
            session,
            row,
            "Состав тестовой разметки изменён; требуется пересчёт основной сетью.",
        )
        row.updated_at = _utc_now()
        if row.is_primary:
            _refresh_training_metrics_after_primary_change(
                session,
                row.class_key,
                config,
                reason="Состав основной тестовой разметки изменён; требуется пересчёт.",
            )
        if config is not None:
            queue_test_sample_evaluation(
                session,
                row,
                config,
                source=JobSource.AUTOMATION,
            )
        session.flush()
    return _detail(session, row)


def delete_test_sample(
    session: Session,
    sample_id: uuid.UUID,
    config: TrainingUIAPIConfig,
) -> None:
    row = _sample_row(session, sample_id)
    _cancel_test_sample_evaluation_job(session, row)
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
                row.class_key,
                "Основная тестовая разметка удалена.",
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
    queue_test_sample_evaluation(
        session,
        row,
        config,
        source=JobSource.MANUAL,
        force=True,
    )
    session.flush()
    return _detail(session, row)


def evaluate_test_sample_preview(
    session: Session,
    sample_id: uuid.UUID,
    request: TestSampleEvaluationPreviewRequest,
    config: TrainingUIAPIConfig,
) -> TestSampleDraftPreview:
    row = _sample_row(session, sample_id)
    selected = _validated_tile_indices(row, request.enabled_tile_indices)
    evaluation = _preview_evaluation(session, row, selected, config)
    return _draft_preview(row, selected, evaluation)


def optimize_test_sample_preview(
    session: Session,
    sample_id: uuid.UUID,
    request: TestSampleOptimizeRequest,
    config: TrainingUIAPIConfig,
) -> TestSampleDraftPreview:
    row = _sample_row(session, sample_id)
    _validate_optimization_request(row, request)
    source = _latest_pseudo_markup(session, row.dataset_key, class_key=row.class_key)
    if source is None or source.geojson_file is None:
        raise TrainingUIAPIError(
            "Нет успешной разметки для этого датасета; "
            "оптимизация состава недоступна."
        )
    source_path = Path(source.geojson_file.path)
    if not source_path.is_file():
        raise TrainingUIAPIError(
            "Файл последней разметки не найден на сервере; состав не изменён."
        )
    selected, pixel_counts, object_counts = _optimized_selection_and_metrics(
        row,
        request,
        config,
        source_path,
    )
    evaluation = _preview_evaluation_info(source, pixel_counts, object_counts)
    return _draft_preview(row, selected, evaluation)


def _draft_preview(
    row: TestSampleRow,
    selected: set[int],
    evaluation: TestSampleEvaluationInfo,
) -> TestSampleDraftPreview:
    enabled_tiles = [tile for tile in row.tiles if tile.tile_index in selected]
    return TestSampleDraftPreview(
        enabled_tile_indices=sorted(selected),
        enabled_image_count=len(enabled_tiles),
        enabled_object_count=sum(tile.object_count for tile in enabled_tiles),
        evaluation=evaluation,
    )


def _preview_evaluation(
    session: Session,
    row: TestSampleRow,
    selected: set[int],
    config: TrainingUIAPIConfig,
) -> TestSampleEvaluationInfo:
    if not selected:
        return TestSampleEvaluationInfo(
            status="unavailable",
            error="В тестовой разметке нет включённых тайлов.",
        )
    source = _latest_pseudo_markup(session, row.dataset_key, class_key=row.class_key)
    if source is None or source.geojson_file is None:
        return TestSampleEvaluationInfo(
            status="unavailable",
            error="Нет успешной разметки для этого датасета.",
        )
    source_path = Path(source.geojson_file.path)
    if not source_path.is_file():
        return TestSampleEvaluationInfo(
            status="error",
            error="Файл последней разметки не найден на сервере.",
        )
    try:
        pixel_counts, object_counts = _calculate_metrics(
            row,
            source_path,
            config,
            tile_indices=selected,
        )
    except Exception as exc:  # noqa: BLE001
        return TestSampleEvaluationInfo(
            status="error",
            error=f"Не удалось рассчитать F1: {exc}",
        )
    return _preview_evaluation_info(source, pixel_counts, object_counts)


def _preview_evaluation_info(
    source: PseudoMarkupResultRow,
    pixel_counts: _MetricCounts,
    object_counts: _MetricCounts,
) -> TestSampleEvaluationInfo:
    return TestSampleEvaluationInfo(
        status="current",
        pixel=pixel_counts.info(),
        objects=object_counts.info(),
        object_iou_threshold=OBJECT_IOU_THRESHOLD,
        pseudo_markup_result_id=source.id,
        model_name=(
            source.training_result.model_name
            if source.training_result is not None
            else "псевдоразметка"
        ),
        training_result_id=source.training_result_id,
        training_dataset_key=(
            source.training_result.dataset_key or source.training_result.class_key
            if source.training_result is not None
            else None
        ),
        training_dataset_name=(
            source.training_result.class_display_name
            if source.training_result is not None
            else None
        ),
        markup_created_at=source.updated_at or source.created_at,
        evaluated_at=_utc_now(),
    )


def optimize_test_sample(
    session: Session,
    sample_id: uuid.UUID,
    request: TestSampleOptimizeRequest,
    config: TrainingUIAPIConfig,
) -> TestSampleDetail:
    row = _sample_row(session, sample_id)
    _validate_optimization_request(row, request)
    source = _latest_pseudo_markup(session, row.dataset_key, class_key=row.class_key)
    if source is None or source.geojson_file is None:
        raise TrainingUIAPIError(
            "Нет успешной разметки для этого датасета; "
            "оптимизация состава недоступна."
        )
    source_path = Path(source.geojson_file.path)
    if not source_path.is_file():
        raise TrainingUIAPIError(
            "Файл последней разметки не найден на сервере; состав тестовой разметки не изменён."
        )

    previous_revision = row.content_revision
    _optimize_test_sample_row(row, request, config, source, source_path)
    if row.content_revision != previous_revision:
        _mark_test_sample_evaluation_stale(
            session,
            row,
            "Состав тестовой разметки оптимизирован; требуется пересчёт основной сетью.",
        )
        if row.is_primary:
            _refresh_training_metrics_after_primary_change(
                session,
                row.class_key,
                config,
                reason="Состав основной тестовой разметки оптимизирован; требуется пересчёт.",
            )
        queue_test_sample_evaluation(
            session,
            row,
            config,
            source=JobSource.AUTOMATION,
        )
    session.flush()
    return _detail(session, row)


def _optimize_test_sample_row(
    row: TestSampleRow,
    request: TestSampleOptimizeRequest,
    config: TrainingUIAPIConfig,
    source: PseudoMarkupResultRow,
    source_path: Path | None = None,
) -> None:
    prediction_path = source_path or (
        Path(source.geojson_file.path) if source.geojson_file is not None else None
    )
    if prediction_path is None or not prediction_path.is_file():
        raise TrainingUIAPIError(
            "Файл разметки для оптимизации не найден на сервере."
        )
    selected, pixel_counts, object_counts = _optimized_selection_and_metrics(
        row,
        request,
        config,
        prediction_path,
        persist_tile_f1=True,
    )

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
    row.evaluation_pseudo_result_id = source.id
    row.evaluation_markup_created_at = source.updated_at or source.created_at


def _optimized_selection_and_metrics(
    row: TestSampleRow,
    request: TestSampleOptimizeRequest,
    config: TrainingUIAPIConfig,
    prediction_path: Path,
    *,
    persist_tile_f1: bool = False,
) -> tuple[set[int], _MetricCounts, _MetricCounts]:
    _validate_optimization_request(row, request)
    effective_request = request.model_copy(update={"metric": row.quality_metric})
    tile_metrics = _calculate_tile_metrics(row, prediction_path, config)
    if persist_tile_f1:
        _apply_test_sample_tile_f1(row, tile_metrics)
    selected = set(
        _select_optimized_tile_indices(row.tiles, tile_metrics, effective_request)
    )
    return (
        selected,
        _sum_tile_metrics(tile_metrics, selected, metric_index=0),
        _sum_tile_metrics(tile_metrics, selected, metric_index=1),
    )


def evaluate_test_sample(
    session: Session,
    row: TestSampleRow,
    config: TrainingUIAPIConfig,
    *,
    pseudo_result: PseudoMarkupResultRow | None = None,
) -> None:
    """Обновить поснимочный кэш оптимизатора по псевдоразметке."""

    source = pseudo_result or _latest_pseudo_markup(
        session,
        row.dataset_key,
        class_key=row.class_key,
    )
    if source is None or source.geojson_file is None:
        _clear_test_sample_tile_f1(row)
        row.evaluation_pseudo_result_id = None
        row.evaluation_markup_created_at = None
        return
    source_path = Path(source.geojson_file.path)
    if not source_path.is_file():
        _clear_test_sample_tile_f1(row)
        row.evaluation_pseudo_result_id = None
        row.evaluation_markup_created_at = None
        return
    try:
        tile_metrics = _calculate_tile_metrics(row, source_path, config)
    except Exception:  # noqa: BLE001
        _clear_test_sample_tile_f1(row)
        row.evaluation_pseudo_result_id = None
        row.evaluation_markup_created_at = None
        return
    _apply_test_sample_tile_f1(row, tile_metrics)
    row.evaluation_pseudo_result_id = source.id
    row.evaluation_markup_created_at = source.updated_at or source.created_at
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
            "Максимальное число тайлов превышает число тайлов в тестовой разметке."
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
    tile_metrics: dict[int, _TileMetric],
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
            "Невозможно подобрать состав тестовой разметки с заданными ограничениями."
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
    tile_metrics: dict[int, _TileMetric],
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
        or not pseudo_result.dataset_key
    ):
        return
    rows = session.scalars(
        select(TestSampleRow)
        .where(TestSampleRow.dataset_key == pseudo_result.dataset_key)
        .options(selectinload(TestSampleRow.tiles))
        .order_by(TestSampleRow.created_at)
    ).all()
    for row in rows:
        latest = _latest_pseudo_markup(
            session,
            row.dataset_key,
            class_key=row.class_key,
        )
        if latest is None or latest.id != pseudo_result.id:
            continue
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
        _clear_test_sample_tile_f1(row)
        row.evaluation_pseudo_result_id = None
        row.evaluation_markup_created_at = None
        if (
            row.evaluation_training_result_id is None
            and row.evaluation_job_id is None
            and _has_metrics(row)
        ):
            row.metric_status = "stale" if _has_metrics(row) else "unavailable"
            row.evaluation_error = (
                "Устаревшая оценка была получена по удалённой псевдоразметке; "
                "требуется прямой пересчёт основной сетью."
            )
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
    *,
    enabled_tile_indices: list[int] | None = None,
    include_previews: bool = True,
) -> TestSampleDownloadArtifact:
    row = _sample_row(session, sample_id)
    selected = (
        _validated_tile_indices(row, enabled_tile_indices)
        if enabled_tile_indices is not None
        else {tile.tile_index for tile in row.tiles if tile.enabled}
    )
    enabled = sorted(
        (tile for tile in row.tiles if tile.tile_index in selected),
        key=lambda tile: tile.tile_index,
    )
    if not enabled:
        raise TrainingUIAPIError("В тестовой разметке нет включённых тайлов.")
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
            for download_index, tile in enumerate(enabled, start=1):
                _write_test_sample_tile_to_archive(
                    archive,
                    source_root,
                    tile.tile_index,
                    download_index,
                    include_previews=include_previews,
                )
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return TestSampleDownloadArtifact(
        path=archive_path,
        filename=f"{_safe_name(row.class_name.casefold(), 'markup')}_test_markup.zip",
    )


def build_test_samples_download(
    session: Session,
    sample_ids: list[uuid.UUID],
    config: TrainingUIAPIConfig,
    *,
    include_previews: bool = True,
) -> TestSampleDownloadArtifact:
    if not sample_ids:
        raise TrainingUIAPIError("Выберите хотя бы одну тестовую разметку.")
    if len(set(sample_ids)) != len(sample_ids):
        raise TrainingUIAPIError(
            "Идентификаторы тестовых разметок не должны повторяться."
        )
    rows = session.scalars(
        select(TestSampleRow)
        .where(TestSampleRow.id.in_(sample_ids))
        .options(selectinload(TestSampleRow.tiles))
    ).all()
    rows_by_id = {row.id: row for row in rows}
    missing = [sample_id for sample_id in sample_ids if sample_id not in rows_by_id]
    if missing:
        raise TestSampleUnavailable(", ".join(str(sample_id) for sample_id in missing))
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            row.class_name.casefold(),
            row.dataset_short_name.casefold(),
            str(row.id),
        ),
    )
    rows_by_class: dict[str, list[TestSampleRow]] = defaultdict(list)
    for row in ordered_rows:
        class_row = dataset_class_row(session, row.class_key)
        rows_by_class[class_row.key if class_row is not None else row.class_key].append(row)
    duplicate_classes = sorted(
        {
            class_rows[0].class_name
            for class_rows in rows_by_class.values()
            if len(class_rows) > 1
        },
        key=str.casefold,
    )
    if duplicate_classes:
        raise TrainingUIAPIError(
            "Для группового скачивания можно выбрать не более одной разметки "
            "каждого класса. Повторяются классы: "
            + ", ".join(duplicate_classes)
        )
    empty = [row.name for row in rows if not any(tile.enabled for tile in row.tiles)]
    if empty:
        raise TrainingUIAPIError(
            "В выбранных тестовых разметках нет включённых тайлов: " + ", ".join(empty)
        )

    descriptors = _test_sample_download_descriptors(
        ordered_rows,
        config,
    )
    download_root = Path(config.scratch_root) / TEST_SAMPLE_DOWNLOAD_ROOT_NAME
    download_root.mkdir(parents=True, exist_ok=True)
    archive_path = download_root / f"selected-{uuid.uuid4()}.zip"
    try:
        with TemporaryDirectory(
            dir=download_root,
            prefix=".building-",
        ) as temporary_directory:
            staging_root = Path(temporary_directory)
            worker_count = min(_BULK_DOWNLOAD_MAX_WORKERS, len(descriptors))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        _prepare_test_sample_download,
                        descriptor,
                        staging_root,
                        include_previews=include_previews,
                    )
                    for descriptor in descriptors
                ]
                try:
                    prepared_roots = [future.result() for future in futures]
                except Exception:
                    for future in futures:
                        future.cancel()
                    raise
            with zipfile.ZipFile(
                archive_path,
                mode="w",
                compression=zipfile.ZIP_STORED,
            ) as archive:
                for descriptor, prepared_root in zip(
                    descriptors,
                    prepared_roots,
                    strict=True,
                ):
                    for path in sorted(
                        (item for item in prepared_root.rglob("*") if item.is_file()),
                        key=lambda item: item.relative_to(prepared_root).as_posix(),
                    ):
                        relative = path.relative_to(prepared_root).as_posix()
                        archive.write(
                            path,
                            f"{descriptor.folder}/{relative}",
                            compress_type=zipfile.ZIP_STORED,
                        )
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return TestSampleDownloadArtifact(
        path=archive_path,
        filename="тестовые_разметки.zip",
    )


def _test_sample_download_descriptors(
    rows: list[TestSampleRow],
    config: TrainingUIAPIConfig,
) -> list[_TestSampleDownloadDescriptor]:
    used_folders: dict[str, tuple[str, str]] = {}
    result: list[_TestSampleDownloadDescriptor] = []
    for row in rows:
        folder = _safe_name(
            f"{row.class_name}_{row.dataset_short_name}",
            "test_sample",
        )
        collision = used_folders.get(folder.casefold())
        if collision is not None:
            previous_dataset_key, previous_dataset_name = collision
            raise TrainingUIAPIError(
                "После нормализации разные датасеты получают одинаковое имя "
                f"папки архива «{folder}»: "
                f"{previous_dataset_name} ({previous_dataset_key}) и "
                f"{row.dataset_name} ({row.dataset_key})."
            )
        used_folders[folder.casefold()] = (row.dataset_key, row.dataset_name)
        enabled = tuple(
            tile.tile_index
            for tile in sorted(row.tiles, key=lambda tile: tile.tile_index)
            if tile.enabled
        )
        result.append(
            _TestSampleDownloadDescriptor(
                sample_id=row.id,
                folder=folder,
                source_root=_sample_root(config, row.id),
                tile_indices=enabled,
            )
        )
    return result


def _prepare_test_sample_download(
    descriptor: _TestSampleDownloadDescriptor,
    staging_root: Path,
    *,
    include_previews: bool,
) -> Path:
    output_root = staging_root / str(descriptor.sample_id)
    output_root.mkdir(parents=True, exist_ok=False)
    for download_index, tile_index in enumerate(descriptor.tile_indices, start=1):
        entries = _test_sample_tile_download_entries(
            descriptor.source_root,
            tile_index,
            download_index,
            include_previews=include_previews,
        )
        for archive_name, payload in entries:
            destination = output_root / archive_name
            if isinstance(payload, Path):
                shutil.copy2(payload, destination)
            else:
                destination.write_bytes(payload)
    return output_root


def _test_sample_tile_download_entries(
    source_root: Path,
    tile_index: int,
    download_index: int,
    *,
    include_previews: bool,
    folder: str | None = None,
) -> list[tuple[str, Path | bytes]]:
    stored_base_name = f"tile_{tile_index:03d}"
    archive_base_name = f"tile{download_index:03d}"
    archive_prefix = f"{folder}/" if folder else ""
    suffixes = list(_DOWNLOAD_BASE_TILE_SUFFIXES)
    if include_previews:
        suffixes.extend(_DOWNLOAD_PREVIEW_SOURCE_SUFFIXES)
    paths = {
        suffix: source_root / f"{stored_base_name}{suffix}"
        for suffix in suffixes
    }
    entries: list[tuple[str, Path | bytes]] = []
    for suffix, path in paths.items():
        if not path.is_file():
            raise TrainingUIAPIError(f"Файл тестового тайла не найден: {path.name}")
        entries.append((f"{archive_prefix}{archive_base_name}{suffix}", path))
    if not include_previews:
        return entries

    tif_path = paths[".tif"]
    mask_path = paths["_mask.png"]
    with rasterio.open(tif_path) as dataset:
        image = dataset.read()
    with rasterio.open(mask_path) as dataset:
        mask = dataset.read(1)
    previews = _test_sample_jpeg_previews(
        image,
        mask,
        tile_name=archive_base_name,
    )
    entries.extend(
        (
            f"{archive_prefix}{archive_base_name}_{suffix}.jpg",
            preview,
        )
        for suffix, preview in previews.items()
    )
    return entries


def _write_test_sample_tile_to_archive(
    archive: zipfile.ZipFile,
    source_root: Path,
    tile_index: int,
    download_index: int,
    *,
    include_previews: bool,
    folder: str | None = None,
) -> None:
    for archive_name, payload in _test_sample_tile_download_entries(
        source_root,
        tile_index,
        download_index,
        include_previews=include_previews,
        folder=folder,
    ):
        if isinstance(payload, Path):
            archive.write(payload, archive_name)
        else:
            archive.writestr(archive_name, payload)


def _test_sample_jpeg_previews(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    tile_name: str,
) -> dict[str, bytes]:
    channel_count = image.shape[0] if image.ndim == 3 else 0
    if channel_count == 3:
        preview_channels = {"rgb": (0, 1, 2)}
    elif channel_count == 4:
        preview_channels = _JPEG_PREVIEW_CHANNELS
    else:
        raise TrainingUIAPIError(
            f"Для превью {tile_name} нужен RGB TIFF с 3 каналами либо "
            f"RGB+NIR TIFF с 4 каналами; найдено каналов: {channel_count}."
        )
    if mask.ndim != 2 or image.shape[1:] != mask.shape:
        raise TrainingUIAPIError(
            f"Размер маски {tile_name} не совпадает с размером TIFF."
        )

    stretched = tuple(_stretch_channel(image[index]) for index in range(channel_count))
    edge = _mask_edge(mask)
    result: dict[str, bytes] = {}
    for name, channel_indices in preview_channels.items():
        preview = np.stack(
            [stretched[index] for index in channel_indices],
            axis=2,
        )
        result[name] = _encode_test_sample_jpeg(
            preview,
            tile_name=tile_name,
            preview_name=name,
        )
        marked = preview.copy()
        marked[edge] = np.asarray([255, 255, 0], dtype=np.uint8)
        marked_name = f"{name}_markup"
        result[marked_name] = _encode_test_sample_jpeg(
            marked,
            tile_name=tile_name,
            preview_name=marked_name,
        )
    return result


def _encode_test_sample_jpeg(
    image: np.ndarray,
    *,
    tile_name: str,
    preview_name: str,
) -> bytes:
    source = Image.fromarray(np.ascontiguousarray(image))
    minimum = _JPEG_QUALITY_MIN
    maximum = _JPEG_QUALITY_MAX
    best: bytes | None = None
    while minimum <= maximum:
        quality = (minimum + maximum) // 2
        stream = BytesIO()
        source.save(
            stream,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
            subsampling=2,
        )
        payload = stream.getvalue()
        if len(payload) <= _JPEG_PREVIEW_MAX_BYTES:
            best = payload
            minimum = quality + 1
        else:
            maximum = quality - 1
    if best is None:
        raise TrainingUIAPIError(
            f"Превью {tile_name}_{preview_name}.jpg не помещается в 300 КБ "
            "даже при минимальном качестве JPEG."
        )
    return best


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
    *,
    tile_indices: set[int] | None = None,
) -> tuple[_MetricCounts, _MetricCounts]:
    enabled = (
        tile_indices
        if tile_indices is not None
        else {tile.tile_index for tile in row.tiles if tile.enabled}
    )
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


def _apply_test_sample_tile_f1(
    row: TestSampleRow,
    tile_metrics: dict[int, _TileMetric],
) -> None:
    for tile in row.tiles:
        metrics = tile_metrics.get(tile.tile_index)
        if metrics is None:
            tile.pixel_f1 = None
            tile.object_f1 = None
            tile.evaluation_metrics = {}
            continue
        tile.pixel_f1 = metrics[0].info().f1
        tile.object_f1 = metrics[1].info().f1
        tile.evaluation_metrics = dict(metrics[2])


def _clear_test_sample_tile_f1(row: TestSampleRow) -> None:
    for tile in row.tiles:
        tile.pixel_f1 = None
        tile.object_f1 = None
        tile.evaluation_metrics = {}


def _calculate_tile_metrics(
    row: TestSampleRow,
    prediction_path: Path,
    config: TrainingUIAPIConfig,
    *,
    tile_indices: set[int] | None = None,
) -> dict[int, _TileMetric]:
    predictions = _load_geometries(prediction_path, default_crs="EPSG:4326")
    result: dict[int, _TileMetric] = {}
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
            raster_transform = raster.transform
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
                true_labels = mask_dataset.read(1).astype(np.uint8, copy=False)
        if true_labels.shape != predicted_mask.shape:
            raise TrainingUIAPIError(f"Размер маски не совпадает с TIFF для {base_name}.")
        true_mask = true_labels > 0
        pixel_counts = _pixel_counts(true_mask, predicted_mask > 0)
        ground_truth = _load_geometries(geojson_path, default_crs=str(raster_crs))
        ground_truth_geometries = _geometries_for_tile(
            ground_truth.geometries,
            ground_truth.crs,
            raster_crs,
            tile_footprint,
        )
        foreground_objects = _object_counts(
            ground_truth_geometries,
            predicted_geometries,
            row.object_iou_threshold,
        )
        structured: dict[str, Any] = {}
        if row.task == "multiclass":
            class_ids = {int(item["id"]) for item in (row.class_schema or [])}
            unknown = set(np.unique(true_labels).tolist()) - {0, *class_ids}
            if unknown:
                raise TrainingUIAPIError(
                    f"Маска {base_name} содержит неизвестные class ID: "
                    + ", ".join(str(value) for value in sorted(unknown))
                )
            pixel_by_class: dict[str, _MetricCounts] = {}
            objects_by_class: dict[str, _MetricCounts] = {}
            for item in row.class_schema or []:
                slug = str(item["slug"])
                class_id = int(item["id"])
                predicted_class_geometries = _prediction_geometries_for_tile(
                    predictions,
                    raster_crs,
                    tile_footprint,
                    class_slug=slug,
                )
                predicted_class_mask = (
                    rasterize(
                        [(geometry, 1) for geometry in predicted_class_geometries],
                        out_shape=true_labels.shape,
                        transform=raster_transform,
                        fill=0,
                        dtype="uint8",
                        all_touched=False,
                    )
                    if predicted_class_geometries
                    else np.zeros(true_labels.shape, dtype=np.uint8)
                )
                pixel_by_class[slug] = _pixel_counts(
                    true_labels == class_id,
                    predicted_class_mask > 0,
                )
                ground_truth_class_geometries = _geometries_for_tile(
                    [
                        geometry
                        for geometry, geometry_slug in zip(
                            ground_truth.geometries,
                            ground_truth.class_slugs,
                            strict=True,
                        )
                        if geometry_slug == slug
                    ],
                    ground_truth.crs,
                    raster_crs,
                    tile_footprint,
                )
                objects_by_class[slug] = _object_counts(
                    ground_truth_class_geometries,
                    predicted_class_geometries,
                    row.object_iou_threshold,
                )
            structured = _structured_test_metrics(
                list(row.class_schema or []),
                pixel_by_class,
                objects_by_class,
                foreground_pixel=pixel_counts,
                foreground_objects=foreground_objects,
            )
        result[tile.tile_index] = (pixel_counts, foreground_objects, structured)
    return result


def _prediction_geometries_for_tile(
    predictions: _GeometrySet,
    raster_crs: PyprojCRS,
    tile_footprint: BaseGeometry,
    *,
    class_slug: str | None = None,
) -> list[BaseGeometry]:
    query_footprint = _transform_geometry(tile_footprint, raster_crs, predictions.crs)
    indices = predictions.tree.query(query_footprint, predicate="intersects")
    selected = [
        predictions.geometries[int(index)]
        for index in indices
        if class_slug is None or predictions.class_slugs[int(index)] == class_slug
    ]
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
        if transformed.is_empty:
            continue
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


def _metric_payload(counts: _MetricCounts) -> dict[str, Any]:
    payload = counts.info().model_dump(mode="json")
    denominator = counts.true_positive + counts.false_positive + counts.false_negative
    payload["iou"] = counts.true_positive / denominator if denominator else 0.0
    return payload


def _structured_test_metrics(
    class_schema: list[dict[str, Any]],
    pixel_by_class: dict[str, _MetricCounts],
    objects_by_class: dict[str, _MetricCounts],
    *,
    foreground_pixel: _MetricCounts,
    foreground_objects: _MetricCounts,
) -> dict[str, Any]:
    warnings_output: list[str] = []

    def build(counts_by_slug: dict[str, _MetricCounts]) -> dict[str, Any]:
        per_class: dict[str, dict[str, Any]] = {}
        micro = _MetricCounts(0, 0, 0)
        for item in class_schema:
            slug = str(item["slug"])
            counts = counts_by_slug.get(slug, _MetricCounts(0, 0, 0))
            metric = _metric_payload(counts)
            metric.update(
                {
                    "id": int(item["id"]),
                    "slug": slug,
                    "name": str(item["name"]),
                    "color": str(item["color"]),
                }
            )
            per_class[slug] = metric
            micro += counts
            if counts.true_positive + counts.false_negative == 0:
                warnings_output.append(f"В тестовой разметке отсутствует тип {item['name']}.")
        macro = {
            key: (
                sum(float(metric[key]) for metric in per_class.values()) / len(per_class)
                if per_class
                else 0.0
            )
            for key in ("precision", "recall", "f1", "iou")
        }
        return {"per_class": per_class, "macro": macro, "micro": _metric_payload(micro)}

    pixel = build(pixel_by_class)
    objects = build(objects_by_class)
    pixel["foreground"] = _metric_payload(foreground_pixel)
    objects["foreground"] = _metric_payload(foreground_objects)
    return {
        "pixel": pixel,
        "objects": objects,
        "object_iou_threshold": OBJECT_IOU_THRESHOLD,
        "warnings": list(dict.fromkeys(warnings_output)),
    }


def _aggregate_tile_structured_metrics(
    tile_metrics: dict[int, _TileMetric],
    tile_indices: set[int],
    class_schema: list[dict[str, Any]],
) -> dict[str, Any]:
    if not class_schema:
        return {}
    pixel_by_class = {str(item["slug"]): _MetricCounts(0, 0, 0) for item in class_schema}
    objects_by_class = {str(item["slug"]): _MetricCounts(0, 0, 0) for item in class_schema}
    foreground_pixel = _MetricCounts(0, 0, 0)
    foreground_objects = _MetricCounts(0, 0, 0)
    for tile_index in tile_indices:
        tile = tile_metrics.get(tile_index)
        if tile is None:
            continue
        foreground_pixel += tile[0]
        foreground_objects += tile[1]
        structured = tile[2]
        for section, target in (("pixel", pixel_by_class), ("objects", objects_by_class)):
            per_class = dict((structured.get(section) or {}).get("per_class") or {})
            for slug in target:
                raw = dict(per_class.get(slug) or {})
                target[slug] += _MetricCounts(
                    int(raw.get("true_positive") or 0),
                    int(raw.get("false_positive") or 0),
                    int(raw.get("false_negative") or 0),
                )
    return _structured_test_metrics(
        class_schema,
        pixel_by_class,
        objects_by_class,
        foreground_pixel=foreground_pixel,
        foreground_objects=foreground_objects,
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
    class_slugs: list[str | None] = []
    for feature in raw_features:
        if not isinstance(feature, dict) or not feature.get("geometry"):
            continue
        try:
            geometry = _polygonal_geometry(shape(feature["geometry"]))
        except Exception:  # noqa: BLE001
            geometry = None
        if geometry is not None and not geometry.is_empty and geometry.area > 0.0:
            geometries.append(geometry)
            properties = feature.get("properties")
            properties = properties if isinstance(properties, dict) else {}
            raw_slug = (
                properties.get("_mlsystem2_class")
                or properties.get("object_type_slug")
                or properties.get("slug")
            )
            class_slugs.append(str(raw_slug) if isinstance(raw_slug, str) and raw_slug else None)
    crs = _payload_crs(payload, default_crs)
    geometry_tuple = tuple(geometries)
    return _GeometrySet(
        crs=crs,
        geometries=geometry_tuple,
        class_slugs=tuple(class_slugs),
        tree=STRtree(geometry_tuple),
    )


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
    transformed = geometry
    if source_crs != target_crs:
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        transformed = transform_geometry(transformer.transform, geometry)
    return _polygonal_geometry(transformed) or GeometryCollection()


def _polygonal_geometry(geometry: BaseGeometry) -> Polygon | MultiPolygon | None:
    if geometry.is_empty:
        return None
    repaired = make_valid(geometry) if not geometry.is_valid else geometry
    if isinstance(repaired, Polygon):
        return repaired
    if isinstance(repaired, MultiPolygon):
        return repaired
    if isinstance(repaired, GeometryCollection):
        polygons: list[Polygon] = []
        for part in repaired.geoms:
            normalized = _polygonal_geometry(part)
            if isinstance(normalized, Polygon):
                polygons.append(normalized)
            elif isinstance(normalized, MultiPolygon):
                polygons.extend(normalized.geoms)
        if not polygons:
            return None
        merged = unary_union(polygons)
        if not merged.is_valid:
            merged = make_valid(merged)
        return merged if isinstance(merged, (Polygon, MultiPolygon)) else None
    return None


def _latest_pseudo_markup(
    session: Session,
    dataset_key: str,
    *,
    class_key: str | None = None,
) -> PseudoMarkupResultRow | None:
    primary = current_primary_training_result(session, class_key or dataset_key)
    conditions = [
            PseudoMarkupResultRow.status == "ok",
            PseudoMarkupResultRow.dataset_key == dataset_key,
            PseudoMarkupResultRow.geojson_file_id.is_not(None),
    ]
    if primary is not None:
        conditions.append(PseudoMarkupResultRow.training_result_id == primary.id)
    return session.scalar(
        select(PseudoMarkupResultRow)
        .where(*conditions)
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


def _class_scope_keys(session: Session, class_or_dataset_key: str) -> set[str]:
    """Вернуть канонический ключ класса и ключи всех его датасетов."""

    class_row = dataset_class_row(session, class_or_dataset_key)
    if class_row is None:
        return {class_or_dataset_key}
    return {
        class_row.key,
        *session.scalars(
            select(DatasetRow.key).where(DatasetRow.class_id == class_row.id)
        ).all(),
    }


def _refresh_training_metrics_after_primary_change(
    session: Session,
    dataset_key: str,
    config: TrainingUIAPIConfig | None,
    *,
    reason: str,
) -> None:
    session.flush()
    primary = _primary_sample(session, dataset_key)
    usable = primary is not None and any(tile.enabled for tile in primary.tiles)
    _mark_training_test_metrics_stale(
        session,
        dataset_key,
        reason,
        unavailable=not usable,
    )
    if config is not None and usable:
        queue_class_test_f1(session, dataset_key, config)


def _mark_training_test_metrics_stale(
    session: Session,
    dataset_key: str,
    reason: str,
    *,
    unavailable: bool = False,
) -> None:
    class_keys = _class_scope_keys(session, dataset_key)
    rows = session.scalars(
        select(TrainingResultTestMetricRow)
        .join(
            TrainingResultRow,
            TrainingResultRow.id == TrainingResultTestMetricRow.training_result_id,
        )
        .where(
            TrainingResultRow.class_key.in_(class_keys)
            | TrainingResultRow.dataset_key.in_(class_keys)
        )
    ).all()
    for row in rows:
        _cancel_test_metric_job(session, row)
        if unavailable:
            row.status = "unavailable"
        else:
            row.status = "stale" if row.f1 is not None else "unavailable"
        row.error = reason
        row.job_id = None
        row.updated_at = _utc_now()


def reconcile_test_sample_evaluations(
    session: Session,
    config: TrainingUIAPIConfig,
    *,
    class_keys: set[str] | None = None,
    sample_ids: set[uuid.UUID] | None = None,
) -> int:
    """Поставить отсутствующие и устаревшие прямые оценки в общую очередь."""

    conditions = []
    if class_keys is not None:
        if not class_keys:
            return 0
        sample_class_keys = set(class_keys)
        class_ids: set[uuid.UUID] = set()
        for class_key in class_keys:
            class_row = dataset_class_row(session, class_key)
            if class_row is None:
                continue
            class_ids.add(class_row.id)
            sample_class_keys.add(class_row.key)
        if class_ids:
            sample_class_keys.update(
                session.scalars(
                    select(DatasetRow.key).where(DatasetRow.class_id.in_(class_ids))
                ).all()
            )
        conditions.append(TestSampleRow.class_key.in_(sample_class_keys))
    if sample_ids is not None:
        if not sample_ids:
            return 0
        conditions.append(TestSampleRow.id.in_(sample_ids))
    rows = session.scalars(
        select(TestSampleRow)
        .where(*conditions)
        .options(selectinload(TestSampleRow.tiles))
        .order_by(TestSampleRow.created_at, TestSampleRow.id)
    ).all()
    created = 0
    for row in rows:
        if queue_test_sample_evaluation(
            session,
            row,
            config,
            source=JobSource.AUTOMATION,
        ):
            created += 1
    session.flush()
    return created


def queue_test_sample_evaluation(
    session: Session,
    sample: TestSampleRow,
    config: TrainingUIAPIConfig,
    *,
    source: JobSource,
    force: bool = False,
) -> bool:
    """Поставить прямую оценку одной разметки текущей основной сетью класса."""

    enabled_indices = sorted(tile.tile_index for tile in sample.tiles if tile.enabled)
    if not enabled_indices:
        _cancel_test_sample_evaluation_job(session, sample)
        _set_test_sample_evaluation_unavailable(
            sample,
            "В тестовой разметке нет включённых тайлов.",
        )
        return False

    result = current_primary_training_result(session, sample.class_key)
    if result is None or result.status != "ok":
        _cancel_test_sample_evaluation_job(session, sample)
        _set_test_sample_evaluation_unavailable(
            sample,
            "Для класса нет успешной сети.",
        )
        return False

    compatibility_error = test_sample_model_compatibility_error(
        session,
        sample,
        result,
    )
    if compatibility_error is not None:
        _cancel_test_sample_evaluation_job(session, sample)
        sample.metric_status = "error"
        sample.evaluation_error = compatibility_error
        sample.updated_at = _utc_now()
        return False

    postprocess_profile = _test_f1_postprocess_profile_name(session, sample, config)
    template, template_config, config_hash = _effective_inference_template(
        session,
        result.architecture,
        result.class_key,
        postprocess_profile,
    )
    active_job = (
        session.get(JobRow, sample.evaluation_job_id)
        if sample.evaluation_job_id is not None
        else None
    )
    if not force and _saved_test_sample_evaluation_matches(
        sample,
        result,
        template,
        config_hash,
    ) and sample.metric_status == "current":
        return False
    if not force and active_job is not None and _test_sample_evaluation_job_matches(
        active_job,
        sample,
        result,
        template,
        config_hash,
        enabled_indices,
    ):
        if active_job.status in {JobStatus.QUEUED.value, JobStatus.RUNNING.value}:
            sample.metric_status = active_job.status
            return False
        if active_job.status == JobStatus.FAILED.value and sample.metric_status == "error":
            return False

    _cancel_test_sample_evaluation_job(session, sample)
    if not result.mlflow_run_id:
        sample.metric_status = "error"
        sample.evaluation_error = (
            "У основной сети отсутствует MLflow run id с checkpoint."
        )
        sample.updated_at = _utc_now()
        return False

    job = JobRow(
        type=JobType.INFERENCE.value,
        source=source.value,
        status=JobStatus.QUEUED.value,
        queue_position=next_queue_position(session, JobType.INFERENCE, source),
        dataset_key=sample.dataset_key,
        dataset_version=sample.dataset_version,
        dataset_name=sample.dataset_name,
        training_dataset_name=result.class_display_name,
        inference_dataset_name=sample.name,
        model_name=result.model_name,
        architecture=result.architecture,
        tile_size=sample.tile_width,
        config={
            "operation": TEST_SAMPLE_F1_OPERATION,
            "metric_target": TEST_SAMPLE_EVALUATION_TARGET,
            "class_key": sample.class_key,
            "model_dataset_key": result.class_key,
            "training_result_id": str(result.id),
            "test_sample_id": str(sample.id),
            "test_sample_revision": sample.content_revision,
            "test_sample_tile_indices": enabled_indices,
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
    sample.metric_status = "queued"
    sample.evaluation_job_id = job.id
    sample.evaluation_job = job
    sample.evaluation_error = None
    sample.updated_at = _utc_now()
    session.flush()
    return True


def current_primary_training_result(
    session: Session,
    class_key: str,
) -> TrainingResultRow | None:
    """Вернуть эффективную успешную сеть класса без её неявного назначения."""

    return primary_training_result(session, class_key)


def test_sample_model_compatibility_error(
    session: Session,
    sample: TestSampleRow,
    result: TrainingResultRow,
) -> str | None:
    sample_class = dataset_class_row(session, sample.class_key)
    result_class = dataset_class_row(
        session,
        result.dataset_key or result.class_key,
    )
    if (
        sample_class is None
        or result_class is None
        or sample_class.id != result_class.id
    ):
        return "Основная сеть относится к другому классу датасетов."
    if sample.task != result.task:
        return (
            "Тип задачи основной сети не совпадает с типом тестовой разметки: "
            f"{result.task} вместо {sample.task}."
        )
    if sample.task == "multiclass" and _class_schema_signature(
        sample.class_schema
    ) != _class_schema_signature(result.class_schema):
        return "Схема типов основной сети не совпадает со схемой тестовой разметки."
    return None


def _class_schema_signature(values: list[dict[str, Any]] | None) -> tuple[tuple[int, str], ...]:
    try:
        return tuple(
            sorted((int(item["id"]), str(item["slug"])) for item in (values or []))
        )
    except (KeyError, TypeError, ValueError):
        return ()


def _saved_test_sample_evaluation_matches(
    sample: TestSampleRow,
    result: TrainingResultRow,
    template: InferenceTemplateRow | None,
    config_hash: str,
) -> bool:
    return (
        sample.evaluation_training_result_id == result.id
        and sample.evaluated_revision == sample.content_revision
        and sample.evaluation_inference_template_id
        == (template.id if template is not None else None)
        and sample.evaluation_inference_template_version
        == (template.version if template is not None else None)
        and sample.evaluation_inference_config_hash == config_hash
        and sample.evaluation_evaluator_version == TEST_SAMPLE_F1_EVALUATOR_VERSION
    )


def _test_sample_evaluation_job_matches(
    job: JobRow,
    sample: TestSampleRow,
    result: TrainingResultRow,
    template: InferenceTemplateRow | None,
    config_hash: str,
    enabled_indices: list[int],
) -> bool:
    state = job.config or {}
    return (
        state.get("operation") == TEST_SAMPLE_F1_OPERATION
        and state.get("metric_target") == TEST_SAMPLE_EVALUATION_TARGET
        and state.get("training_result_id") == str(result.id)
        and state.get("test_sample_id") == str(sample.id)
        and int(state.get("test_sample_revision") or 0) == sample.content_revision
        and [int(value) for value in state.get("test_sample_tile_indices") or []]
        == enabled_indices
        and state.get("inference_template_id")
        == (str(template.id) if template is not None else None)
        and state.get("inference_template_version")
        == (template.version if template is not None else None)
        and state.get("inference_config_hash") == config_hash
        and state.get("test_f1_evaluator_version")
        == TEST_SAMPLE_F1_EVALUATOR_VERSION
    )


def _cancel_test_sample_evaluation_job(
    session: Session,
    sample: TestSampleRow,
) -> None:
    if sample.evaluation_job_id is None:
        return
    job = session.get(JobRow, sample.evaluation_job_id)
    if job is not None and (
        (job.config or {}).get("operation") == TEST_SAMPLE_F1_OPERATION
        and (job.config or {}).get("metric_target") == TEST_SAMPLE_EVALUATION_TARGET
    ):
        if job.status == JobStatus.RUNNING.value:
            terminate_job_process(job)
            if job.tmp_path:
                shutil.rmtree(job.tmp_path, ignore_errors=True)
                job.tmp_path = None
        if job.status in {JobStatus.QUEUED.value, JobStatus.RUNNING.value}:
            job.status = JobStatus.CANCELLED.value
            job.finished_at = _utc_now()
            job.process_pid = None
    sample.evaluation_job_id = None
    sample.evaluation_job = None


def _mark_test_sample_evaluation_stale(
    session: Session,
    sample: TestSampleRow,
    reason: str,
) -> None:
    _cancel_test_sample_evaluation_job(session, sample)
    sample.metric_status = "stale" if _has_metrics(sample) else "unavailable"
    sample.evaluation_error = reason
    sample.updated_at = _utc_now()


def _set_test_sample_evaluation_unavailable(
    sample: TestSampleRow,
    reason: str,
) -> None:
    sample.metric_status = "stale" if _has_metrics(sample) else "unavailable"
    sample.evaluation_error = reason
    sample.updated_at = _utc_now()


def queue_class_test_f1(
    session: Session,
    dataset_key: str,
    config: TrainingUIAPIConfig,
) -> int:
    """Ставит в общую inference-очередь недостающие оценки успешных сетей."""

    sample = _primary_sample(session, dataset_key)
    if sample is None:
        raise TrainingUIAPIError("Для датасета не назначена основная тестовая разметка.")
    if not any(tile.enabled for tile in sample.tiles):
        raise TrainingUIAPIError("В основной тестовой разметке нет включённых тайлов.")
    class_keys = _class_scope_keys(session, dataset_key)
    results = session.scalars(
        select(TrainingResultRow)
        .where(
            (
                TrainingResultRow.class_key.in_(class_keys)
                | TrainingResultRow.dataset_key.in_(class_keys)
            ),
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


def reconcile_training_result_test_f1(
    session: Session,
    config: TrainingUIAPIConfig,
    *,
    dataset_keys: set[str] | None = None,
) -> int:
    """Восстанавливает отсутствующие и устаревшие оценки без повтора текущих ошибок."""

    conditions = [TrainingResultRow.status == "ok"]
    if dataset_keys is not None:
        if not dataset_keys:
            return 0
        conditions.append(
            TrainingResultRow.class_key.in_(dataset_keys)
            | TrainingResultRow.dataset_key.in_(dataset_keys)
        )
    results = session.scalars(
        select(TrainingResultRow)
        .where(*conditions)
        .order_by(TrainingResultRow.created_at.desc(), TrainingResultRow.id.desc())
    ).all()
    created = 0
    for result in results:
        sample = _primary_sample(session, result.class_key)
        if sample is None or not any(tile.enabled for tile in sample.tiles):
            continue
        metric = session.get(TrainingResultTestMetricRow, result.id)
        if not _test_metric_needs_reconciliation(
            session,
            result,
            sample,
            metric,
            config,
        ):
            continue
        if queue_training_result_test_f1(
            session,
            result,
            config,
            source=JobSource(result.source),
        ):
            created += 1
    session.flush()
    return created


def _test_metric_needs_reconciliation(
    session: Session,
    result: TrainingResultRow,
    sample: TestSampleRow,
    metric: TrainingResultTestMetricRow | None,
    config: TrainingUIAPIConfig,
) -> bool:
    if metric is None:
        return True
    postprocess_profile = _test_f1_postprocess_profile_name(session, sample, config)
    template, _, config_hash = _effective_inference_template(
        session,
        result.architecture,
        result.class_key,
        postprocess_profile,
    )
    if not _metric_matches(metric, sample, template, config_hash):
        return True
    if metric.status == "current" or _metric_job_is_active(session, metric):
        return False
    if metric.status in {"queued", "running"}:
        return True
    return metric.job_id is None and metric.status in {"stale", "unavailable"}


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
            "metric_target": TRAINING_RESULT_TEST_METRIC_TARGET,
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
            error="Для сети ещё не рассчитан F1 на основной тестовой разметке.",
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
        metrics=dict(metric.metrics or {}),
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
    class_keys = _class_scope_keys(session, dataset_key)
    return session.scalar(
        select(TestSampleRow)
        .where(
            TestSampleRow.class_key.in_(class_keys),
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
        source = _latest_pseudo_markup(
            session,
            sample.dataset_key,
            class_key=sample.class_key,
        )
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


def test_sample_pseudo_markup_info(
    session: Session,
    row: TestSampleRow,
) -> TestSamplePseudoMarkupInfo:
    """Описать кэш псевдоразметки исходного датасета для оптимизатора."""

    target = current_primary_training_result(session, row.class_key)
    if target is None:
        return TestSamplePseudoMarkupInfo(
            status="unavailable",
            error="Для класса нет успешной сети.",
        )
    common = {
        "training_result_id": target.id,
        "model_name": target.model_name,
        "training_dataset_key": target.dataset_key or target.class_key,
        "training_dataset_name": target.class_display_name,
    }
    ready = _latest_pseudo_markup(
        session,
        row.dataset_key,
        class_key=row.class_key,
    )
    if (
        ready is not None
        and ready.training_result_id == target.id
        and ready.geojson_file is not None
        and Path(ready.geojson_file.path).is_file()
    ):
        return TestSamplePseudoMarkupInfo(
            status="ready",
            result_id=ready.id,
            job_id=ready.job_id,
            can_create=False,
            **common,
        )
    candidate = session.scalar(
        select(PseudoMarkupResultRow)
        .where(
            PseudoMarkupResultRow.dataset_key == row.dataset_key,
            PseudoMarkupResultRow.training_result_id == target.id,
        )
        .order_by(
            PseudoMarkupResultRow.updated_at.desc(),
            PseudoMarkupResultRow.created_at.desc(),
            PseudoMarkupResultRow.id.desc(),
        )
        .limit(1)
    )
    job = (
        session.get(JobRow, candidate.job_id)
        if candidate is not None and candidate.job_id is not None
        else None
    )
    if job is not None and job.status in {JobStatus.QUEUED.value, JobStatus.RUNNING.value}:
        return TestSamplePseudoMarkupInfo(
            status=job.status,
            result_id=candidate.id if candidate is not None else None,
            job_id=job.id,
            can_create=False,
            **common,
        )
    if candidate is not None and candidate.status in {"error", "failed"}:
        return TestSamplePseudoMarkupInfo(
            status="error",
            result_id=candidate.id,
            job_id=candidate.job_id,
            can_create=True,
            error=(job.error if job is not None else None) or "Псевдоразметка завершилась с ошибкой.",
            **common,
        )
    return TestSamplePseudoMarkupInfo(
        status="unavailable",
        can_create=True,
        error="Для исходного датасета нет псевдоразметки текущей основной сети.",
        **common,
    )


def _summary(session: Session, row: TestSampleRow) -> TestSampleSummary:
    enabled = [tile for tile in row.tiles if tile.enabled]
    return TestSampleSummary(
        id=row.id,
        name=row.name,
        dataset_key=row.dataset_key,
        dataset_name=row.dataset_name,
        dataset_version=row.dataset_version,
        source_dataset_key=row.dataset_key,
        source_dataset_name=row.dataset_name,
        source_dataset_version=row.dataset_version,
        class_key=row.class_key,
        class_name=row.class_name,
        task=row.task,
        class_schema=list(row.class_schema or []),
        class_object_counts={
            str(item["slug"]): sum(
                int((tile.class_object_counts or {}).get(str(item["slug"]), 0))
                for tile in enabled
            )
            for item in (row.class_schema or [])
        },
        quality_metric=row.quality_metric,
        image_count=row.image_count,
        enabled_image_count=len(enabled),
        actual_object_count=row.actual_object_count,
        enabled_object_count=sum(tile.object_count for tile in enabled),
        is_primary=row.is_primary,
        evaluation=_evaluation_info(session, row),
        pseudo_markup=test_sample_pseudo_markup_info(session, row),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _detail(session: Session, row: TestSampleRow) -> TestSampleDetail:
    summary = _summary(session, row)
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
                class_object_counts=dict(tile.class_object_counts or {}),
                evaluation_metrics=dict(tile.evaluation_metrics or {}),
                f1_score=(
                    tile.object_f1
                    if row.quality_metric == "objects"
                    else tile.pixel_f1
                ),
                enabled=tile.enabled,
                preview_url=(
                    f"/api/v1/test-samples/{row.id}/tiles/{tile.tile_index}/preview"
                ),
            )
            for tile in row.tiles
        ],
    )


def _evaluation_info(session: Session, row: TestSampleRow) -> TestSampleEvaluationInfo:
    status = row.metric_status
    if (
        status not in {"queued", "running"}
        and _has_metrics(row)
        and row.evaluated_revision != row.content_revision
    ):
        status = "stale"
    job = row.evaluation_job
    if status in {"queued", "running"} and (
        job is None
        or job.status not in {JobStatus.QUEUED.value, JobStatus.RUNNING.value}
    ):
        status = "stale" if _has_metrics(row) else "error"
    saved_result = (
        session.get(TrainingResultRow, row.evaluation_training_result_id)
        if row.evaluation_training_result_id is not None
        else None
    )
    target_result = current_primary_training_result(session, row.class_key)
    if job is not None and (job.config or {}).get("metric_target") == TEST_SAMPLE_EVALUATION_TARGET:
        try:
            queued_result_id = uuid.UUID(str((job.config or {}).get("training_result_id")))
        except (TypeError, ValueError):
            queued_result_id = None
        if queued_result_id is not None:
            target_result = session.get(TrainingResultRow, queued_result_id)
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
        pseudo_markup_result_id=(
            row.evaluation_pseudo_result_id
            if row.evaluation_training_result_id is None
            else None
        ),
        training_result_id=row.evaluation_training_result_id,
        target_training_result_id=target_result.id if target_result is not None else None,
        model_name=row.evaluation_model_name,
        target_model_name=target_result.model_name if target_result is not None else None,
        training_dataset_key=(
            saved_result.dataset_key or saved_result.class_key
            if saved_result is not None
            else None
        ),
        training_dataset_name=(
            saved_result.class_display_name if saved_result is not None else None
        ),
        target_training_dataset_key=(
            target_result.dataset_key or target_result.class_key
            if target_result is not None
            else None
        ),
        target_training_dataset_name=(
            target_result.class_display_name if target_result is not None else None
        ),
        threshold=row.evaluation_threshold,
        job_id=row.evaluation_job_id,
        progress=_test_f1_progress(job),
        markup_created_at=(
            row.evaluation_markup_created_at
            if row.evaluation_training_result_id is None
            else None
        ),
        evaluated_at=row.evaluated_at,
        error=row.evaluation_error,
        metrics=dict(row.evaluation_metrics or {}),
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
    "build_test_sample_download",
    "build_test_samples_download",
    "cleanup_test_sample_storage",
    "create_test_sample",
    "create_test_sample_batch",
    "current_primary_training_result",
    "delete_test_sample",
    "evaluate_test_sample_by_id",
    "evaluate_test_sample_preview",
    "evaluate_test_samples_for_pseudo_markup",
    "latest_test_sample_batch",
    "mark_test_samples_stale_for_pseudo_markup",
    "optimize_test_sample",
    "optimize_test_sample_preview",
    "primary_test_sample",
    "process_test_sample_batch_once",
    "queue_class_test_f1",
    "queue_test_sample_evaluation",
    "queue_training_result_test_f1",
    "reconcile_test_sample_evaluations",
    "reconcile_training_result_test_f1",
    "recover_test_sample_batches",
    "run_test_sample_batch_worker",
    "test_sample_batch_detail",
    "test_sample_catalog",
    "test_sample_detail",
    "test_sample_preview_path",
    "training_result_test_f1_info",
    "test_sample_model_compatibility_error",
    "update_test_sample",
    "update_test_sample_primary",
    "update_test_sample_tile",
]
