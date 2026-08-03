"""Servis AOI-raspoznavaniya vnutri training_ui_api."""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyproj import CRS as PyprojCRS
from pyproj import Geod, Transformer
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from sqlalchemy import select
from sqlalchemy.orm import Session

from mlsystem2.mlflow_adapter.api import get_usable_training_checkpoint
from mlsystem2.mlflow_adapter.contracts import MLflowAdapterError, MLflowBestCheckpoint

from ._config import TrainingUIAPIConfig
from ._dataset_catalog import find_managed_class, find_managed_dataset, list_managed_classes
from ._datasets import CUSTOM_KEY, imagery_images_dir
from ._models import JobRow, StoredFileRow, TrainingResultRow
from ._queueing import next_queue_position
from ._service import delete_job, inference_template_row_for_dataset
from ._templates import sanitize_inference_template_config
from .contracts import (
    JobSource,
    JobStatus,
    JobType,
    PseudolabelAPIError,
    PseudolabelClassInfo,
    PseudolabelClassListResponse,
    PseudolabelErrorInfo,
    PseudolabelJobCreate,
    PseudolabelJobInfo,
    ResultStatus,
    StoredFileKind,
)


PSEUDOLABEL_AOI_OPERATION = "pseudolabel_aoi"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SelectedModel:
    """Zafiksirovannaya model i ee effektivnyi inference-profile."""

    class_id: str
    class_name: str
    dataset_key: str
    dataset_name: str
    dataset_version: str | None
    imagery_type: str
    input_channels: int
    result: TrainingResultRow
    checkpoint: MLflowBestCheckpoint
    inference_template_id: uuid.UUID | None
    inference_template_config: dict[str, Any]


def pseudolabel_classes(
    session: Session,
    config: TrainingUIAPIConfig,
) -> PseudolabelClassListResponse:
    """Vernut klassy s poslednei prigodnoi modelyu."""

    items: list[PseudolabelClassInfo] = []
    for class_info in list_managed_classes(session, config, include_custom=False):
        selected = _select_model(session, config, class_info.key, required=False)
        if selected is None:
            continue
        items.append(_class_info(selected))
    items.sort(key=lambda item: (item.display_name.casefold(), item.class_id))
    return PseudolabelClassListResponse(classes=items)


def create_pseudolabel_job(
    session: Session,
    request: PseudolabelJobCreate,
    config: TrainingUIAPIConfig,
) -> PseudolabelJobInfo:
    """Proverit AOI, zafiksirovat model i postavit job v ochered."""

    aoi_wgs84, area_m2, vertex_count = _validated_aoi(request, config)
    selected = _select_model(session, config, request.class_id, required=True)
    assert selected is not None
    source_job = (
        session.get(JobRow, selected.result.job_id)
        if selected.result.job_id is not None
        else None
    )
    source_config = dict(source_job.config or {}) if source_job is not None else {}
    tile_size = _positive_int(source_config.get("tile_preparation.tile_size"), 768)
    stride = _positive_int(source_config.get("tile_preparation.stride"), tile_size)
    batch_size = _positive_int(source_config.get("train.batch_size"), 1)
    aoi_payload = mapping(aoi_wgs84)
    row = JobRow(
        type=JobType.INFERENCE.value,
        source=JobSource.MANUAL.value,
        status=JobStatus.QUEUED.value,
        queue_position=next_queue_position(session, JobType.INFERENCE, JobSource.MANUAL),
        dataset_key=selected.dataset_key,
        dataset_version=selected.dataset_version,
        dataset_name=selected.dataset_name,
        training_dataset_name=selected.dataset_name,
        inference_dataset_name="AOI",
        model_name=selected.result.model_name,
        architecture=selected.result.architecture,
        tile_size=tile_size,
        config={
            "operation": PSEUDOLABEL_AOI_OPERATION,
            "pseudolabel": {
                "class_id": selected.class_id,
                "class_name": selected.class_name,
                "aoi": aoi_payload,
                "aoi_crs": "EPSG:4326",
                "aoi_area_m2": area_m2,
                "aoi_vertex_count": vertex_count,
                "model_id": str(selected.result.id),
                "model_version": selected.result.mlflow_run_id,
                "model_name": selected.result.model_name,
                "architecture": selected.result.architecture,
                "training_result_id": str(selected.result.id),
                "mlflow_run_id": selected.result.mlflow_run_id,
                "checkpoint_artifact_path": selected.checkpoint.artifact_path,
                "checkpoint_uri": selected.checkpoint.artifact_uri,
                "checkpoint_threshold": selected.checkpoint.threshold,
                "checkpoint_f1_score": selected.checkpoint.f1_score,
                "checkpoint_epoch": selected.checkpoint.epoch,
                "imagery_type": selected.imagery_type,
                "input_channels": selected.input_channels,
                "images_root": str(imagery_images_dir(config.images_root, selected.imagery_type)),
                "inference_template_id": (
                    str(selected.inference_template_id)
                    if selected.inference_template_id is not None
                    else None
                ),
                "inference_template_config": selected.inference_template_config,
                "tile_size": tile_size,
                "stride": stride,
                "batch_size": batch_size,
                "timeout_seconds": config.pseudolabel_job_timeout_seconds,
                "warnings": [],
                "source_image_ids": [],
                "coverage_percent": None,
                "error": None,
                "result_file_id": None,
            },
        },
    )
    session.add(row)
    session.flush()
    LOGGER.info(
        "Задание AOI %s поставлено в очередь: класс %s, модель %s, run %s",
        row.id,
        selected.class_id,
        selected.result.id,
        selected.result.mlflow_run_id,
    )
    return _job_info(row)


def pseudolabel_job_info(session: Session, job_id: uuid.UUID) -> PseudolabelJobInfo:
    """Vernut tekushchii snapshot AOI job."""

    row = _pseudolabel_job(session, job_id)
    return _job_info(row)


def pseudolabel_result(
    session: Session,
    job_id: uuid.UUID,
    config: TrainingUIAPIConfig,
) -> dict[str, Any]:
    """Prochitat tolko zavershennyi sohranennyi FeatureCollection."""

    row = _pseudolabel_job(session, job_id)
    if row.status != JobStatus.COMPLETED.value:
        raise PseudolabelAPIError(
            "RESULT_NOT_READY",
            "Результат задания ещё не готов.",
            status_code=409,
            details={"status": _public_status(row.status)},
        )
    state = _state(row)
    raw_file_id = state.get("result_file_id")
    try:
        file_id = uuid.UUID(str(raw_file_id))
    except (TypeError, ValueError) as exc:
        raise PseudolabelAPIError(
            "RESULT_NOT_FOUND",
            "Файл результата задания не найден.",
            status_code=404,
        ) from exc
    file_row = session.get(StoredFileRow, file_id)
    if file_row is None or file_row.kind != StoredFileKind.PSEUDOLABEL_GEOJSON.value:
        raise PseudolabelAPIError(
            "RESULT_NOT_FOUND",
            "Файл результата задания не найден.",
            status_code=404,
        )
    path = Path(file_row.path)
    _ensure_stored_result_path(path, config)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PseudolabelAPIError(
            "RESULT_CORRUPTED",
            "Файл результата задания повреждён.",
            status_code=500,
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("type") != "FeatureCollection"
        or not isinstance(payload.get("features"), list)
    ):
        raise PseudolabelAPIError(
            "RESULT_CORRUPTED",
            "Результат не является GeoJSON FeatureCollection.",
            status_code=500,
        )
    return payload


def cancel_pseudolabel_job(session: Session, job_id: uuid.UUID) -> PseudolabelJobInfo:
    """Bezopasno ostanovit job i sohranit ego cancelled-sostoyanie."""

    row = _pseudolabel_job(session, job_id)
    if row.status not in {JobStatus.QUEUED.value, JobStatus.RUNNING.value}:
        raise PseudolabelAPIError(
            "JOB_NOT_CANCELLABLE",
            "Отменить можно только ожидающее или выполняющееся задание.",
            status_code=409,
        )
    delete_job(session, job_id, preserve_cancelled=True)
    LOGGER.info("Задание AOI %s отменено", job_id)
    return _job_info(row)


def _select_model(
    session: Session,
    config: TrainingUIAPIConfig,
    class_id: str,
    *,
    required: bool,
) -> _SelectedModel | None:
    """Vybrat poslednii prigodnyi best checkpoint osnovnogo dataseta."""

    class_info = find_managed_class(session, config, class_id)
    if class_info is None or class_info.is_custom or class_id == CUSTOM_KEY:
        if required:
            raise PseudolabelAPIError(
                "CLASS_NOT_FOUND",
                "Класс распознавания не найден.",
                status_code=404,
            )
        return None
    if not class_info.primary_dataset_key:
        if required:
            raise PseudolabelAPIError(
                "PRIMARY_DATASET_NOT_FOUND",
                "Для класса не выбран основной датасет.",
                status_code=409,
            )
        return None
    dataset = find_managed_dataset(session, config, class_info.primary_dataset_key)
    if (
        dataset is None
        or not dataset.source_available
        or dataset.images_dir is None
        or dataset.imagery_type is None
        or dataset.input_channels is None
    ):
        if required:
            raise PseudolabelAPIError(
                "PRIMARY_DATASET_UNAVAILABLE",
                "Основной датасет класса недоступен.",
                status_code=409,
            )
        return None
    rows = session.scalars(
        select(TrainingResultRow)
        .where(
            TrainingResultRow.class_key == dataset.key,
            TrainingResultRow.status == ResultStatus.OK.value,
            TrainingResultRow.trained_at.is_not(None),
            TrainingResultRow.mlflow_run_id.is_not(None),
        )
        .order_by(
            TrainingResultRow.trained_at.desc(),
            TrainingResultRow.created_at.desc(),
            TrainingResultRow.id.desc(),
        )
    ).all()
    for row in rows:
        source_job = session.get(JobRow, row.job_id) if row.job_id is not None else None
        if source_job is not None and source_job.status != JobStatus.COMPLETED.value:
            continue
        if _training_input_channels(source_job, dataset.input_channels) != dataset.input_channels:
            continue
        try:
            checkpoint = get_usable_training_checkpoint(config.mlflow_tracking_uri, row.mlflow_run_id or "")
        except MLflowAdapterError:
            continue
        if checkpoint is None:
            continue
        template = inference_template_row_for_dataset(session, row.architecture, dataset.key)
        template_config = (
            sanitize_inference_template_config(template.default_config)
            if template is not None
            else {}
        )
        return _SelectedModel(
            class_id=class_info.key,
            class_name=class_info.name,
            dataset_key=dataset.key,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            imagery_type=dataset.imagery_type.value,
            input_channels=dataset.input_channels,
            result=row,
            checkpoint=checkpoint,
            inference_template_id=template.id if template is not None else None,
            inference_template_config=template_config,
        )
    if required:
        raise PseudolabelAPIError(
            "USABLE_MODEL_NOT_FOUND",
            "Для класса нет успешно обученной и пригодной к инференсу модели.",
            status_code=404,
        )
    return None


def _validated_aoi(
    request: PseudolabelJobCreate,
    config: TrainingUIAPIConfig,
) -> tuple[Polygon | MultiPolygon, float, int]:
    """Proverit geometriyu, CRS, ploshchad i chislo vershin."""

    try:
        source_crs = PyprojCRS.from_user_input(request.aoi_crs)
    except Exception as exc:  # noqa: BLE001
        raise PseudolabelAPIError(
            "INVALID_CRS",
            "CRS зоны интереса не распознана.",
            status_code=422,
        ) from exc
    try:
        geometry = shape(request.aoi.model_dump())
    except Exception as exc:  # noqa: BLE001
        raise PseudolabelAPIError(
            "INVALID_GEOJSON",
            "Геометрия зоны интереса не является корректным GeoJSON.",
            status_code=422,
        ) from exc
    if not isinstance(geometry, (Polygon, MultiPolygon)) or geometry.is_empty:
        raise PseudolabelAPIError(
            "INVALID_GEOMETRY",
            "Зона интереса должна быть непустым Polygon или MultiPolygon.",
            status_code=422,
        )
    if not geometry.is_valid:
        raise PseudolabelAPIError(
            "INVALID_GEOMETRY",
            "Геометрия зоны интереса невалидна.",
            status_code=422,
        )
    vertex_count = _vertex_count(geometry)
    if vertex_count > config.pseudolabel_max_vertices:
        raise PseudolabelAPIError(
            "AOI_TOO_MANY_VERTICES",
            "Зона интереса содержит слишком много вершин.",
            status_code=422,
            details={"maximum": config.pseudolabel_max_vertices, "actual": vertex_count},
        )
    try:
        transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
        wgs84 = transform_geometry(transformer.transform, geometry)
    except Exception as exc:  # noqa: BLE001
        raise PseudolabelAPIError(
            "CRS_TRANSFORM_FAILED",
            "Не удалось преобразовать зону интереса в EPSG:4326.",
            status_code=422,
        ) from exc
    if not isinstance(wgs84, (Polygon, MultiPolygon)) or wgs84.is_empty or not wgs84.is_valid:
        raise PseudolabelAPIError(
            "INVALID_GEOMETRY",
            "После преобразования CRS зона интереса стала невалидной.",
            status_code=422,
        )
    min_x, min_y, max_x, max_y = wgs84.bounds
    if (
        not all(math.isfinite(value) for value in (min_x, min_y, max_x, max_y))
        or min_x < -180
        or max_x > 180
        or min_y < -90
        or max_y > 90
    ):
        raise PseudolabelAPIError(
            "INVALID_GEOMETRY",
            "Координаты зоны интереса выходят за допустимые границы EPSG:4326.",
            status_code=422,
        )
    area_m2 = _geodesic_area_m2(wgs84)
    if area_m2 <= 0:
        raise PseudolabelAPIError(
            "EMPTY_AOI",
            "Площадь зоны интереса должна быть больше нуля.",
            status_code=422,
        )
    if (
        config.pseudolabel_max_aoi_area_m2 is not None
        and area_m2 > config.pseudolabel_max_aoi_area_m2
    ):
        raise PseudolabelAPIError(
            "AOI_TOO_LARGE",
            "Площадь зоны интереса превышает допустимый предел.",
            status_code=422,
            details={"maximum_m2": config.pseudolabel_max_aoi_area_m2, "actual_m2": area_m2},
        )
    return wgs84, area_m2, vertex_count


def _class_info(selected: _SelectedModel) -> PseudolabelClassInfo:
    """Sobrat publichnyi DTO vybrannoi modeli."""

    assert selected.result.trained_at is not None
    assert selected.result.mlflow_run_id is not None
    return PseudolabelClassInfo(
        class_id=selected.class_id,
        display_name=selected.class_name,
        model_id=selected.result.id,
        model_version=selected.result.mlflow_run_id,
        model_name=selected.result.model_name,
        trained_at=selected.result.trained_at,
    )


def _job_info(row: JobRow) -> PseudolabelJobInfo:
    """Sobrat publichnyi snapshot bez servernyh putei."""

    state = _state(row)
    payload = _progress_payload(row)
    current = _integer(payload.get("current")) if payload else None
    total = _integer(payload.get("total")) if payload else None
    progress = None
    if current is not None and total is not None and total > 0:
        progress = min(100.0, max(0.0, current * 100.0 / total))
    error_payload = state.get("error")
    error = (
        PseudolabelErrorInfo.model_validate(error_payload)
        if isinstance(error_payload, dict)
        else None
    )
    source_ids = _string_list((payload or {}).get("source_image_ids")) or _string_list(
        state.get("source_image_ids")
    )
    coverage = (payload or {}).get("coverage_percent", state.get("coverage_percent"))
    try:
        coverage_percent = float(coverage) if coverage is not None else None
    except (TypeError, ValueError):
        coverage_percent = None
    stage = str((payload or {}).get("stage") or _default_stage(row.status))
    return PseudolabelJobInfo(
        job_id=row.id,
        status=_public_status(row.status),
        class_id=str(state.get("class_id") or ""),
        model_id=uuid.UUID(str(state.get("model_id"))),
        model_version=str(state.get("model_version") or ""),
        model_name=str(state.get("model_name") or row.model_name),
        created_at=row.created_at,
        finished_at=row.finished_at,
        progress=progress,
        current_stage=stage,
        error=error,
        warnings=_string_list(state.get("warnings")),
        source_image_ids=source_ids,
        coverage_percent=coverage_percent,
    )


def _pseudolabel_job(session: Session, job_id: uuid.UUID) -> JobRow:
    """Naiti tolko job nuzhnogo domena."""

    row = session.get(JobRow, job_id)
    if row is None or (row.config or {}).get("operation") != PSEUDOLABEL_AOI_OPERATION:
        raise PseudolabelAPIError(
            "JOB_NOT_FOUND",
            "Задание распознавания не найдено.",
            status_code=404,
        )
    return row


def _state(row: JobRow) -> dict[str, Any]:
    """Izvlech zafiksirovannoe sostoyanie iz JSON job."""

    value = (row.config or {}).get("pseudolabel")
    return value if isinstance(value, dict) else {}


def _progress_payload(row: JobRow) -> dict[str, Any] | None:
    """Bezopasno prochitat atomarno zapisannyi progress runner."""

    if row.tmp_path is None:
        return None
    path = Path(row.tmp_path) / "scratch" / "progress.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _public_status(value: str) -> str:
    """Preobrazovat v publichnye statusy AOI API."""

    return {
        JobStatus.COMPLETED.value: "succeeded",
        JobStatus.FAILED.value: "failed",
        JobStatus.CANCELLED.value: "cancelled",
    }.get(value, value)


def _default_stage(status: str) -> str:
    """Vernut etap, kogda progress-fail eshche nedostupen."""

    return {
        JobStatus.QUEUED.value: "queued",
        JobStatus.RUNNING.value: "running",
        JobStatus.COMPLETED.value: "succeeded",
        JobStatus.FAILED.value: "failed",
        JobStatus.CANCELLED.value: "cancelled",
    }.get(status, "unknown")


def _training_input_channels(row: JobRow | None, fallback: int) -> int:
    """Prochitat chislo kanalov iz zafiksirovannogo training job."""

    if row is None:
        return fallback
    return _positive_int((row.config or {}).get("train.input_channels"), fallback)


def _positive_int(value: object, fallback: int) -> int:
    """Normalizovat polozhitelnoe celoe znachenie."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _integer(value: object) -> int | None:
    """Bezopasno preobrazovat optional celoe."""

    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _string_list(value: object) -> list[str]:
    """Normalizovat JSON-massiv strok."""

    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _vertex_count(geometry: BaseGeometry) -> int:
    """Poschitat vershiny vseh kolec poligonalnoi geometrii."""

    if isinstance(geometry, Polygon):
        return len(geometry.exterior.coords) + sum(len(ring.coords) for ring in geometry.interiors)
    if isinstance(geometry, MultiPolygon):
        return sum(_vertex_count(item) for item in geometry.geoms)
    return 0


def _geodesic_area_m2(geometry: BaseGeometry) -> float:
    """Poschitat geodezicheskuyu ploshchad v kvadratnyh metrah."""

    area, _ = Geod(ellps="WGS84").geometry_area_perimeter(geometry)
    return abs(float(area))


def _ensure_stored_result_path(path: Path, config: TrainingUIAPIConfig) -> None:
    """Zapretit chtenie rezultata vne servernogo hranilishcha."""

    try:
        path.resolve(strict=True).relative_to(Path(config.stored_files_root).resolve())
    except (OSError, ValueError) as exc:
        raise PseudolabelAPIError(
            "RESULT_NOT_FOUND",
            "Файл результата задания недоступен.",
            status_code=404,
        ) from exc


__all__ = [
    "PSEUDOLABEL_AOI_OPERATION",
    "cancel_pseudolabel_job",
    "create_pseudolabel_job",
    "pseudolabel_classes",
    "pseudolabel_job_info",
    "pseudolabel_result",
]
