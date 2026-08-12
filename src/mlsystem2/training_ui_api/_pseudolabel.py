"""Servis AOI-raspoznavaniya vnutri training_ui_api."""

from __future__ import annotations

import json
import logging
import math
import statistics
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rasterio
from rasterio.warp import transform_bounds
from pyproj import CRS as PyprojCRS
from pyproj import Geod, Transformer
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from sqlalchemy import select
from sqlalchemy.orm import Session

from mlsystem2.dataset_preparing.api import resolve_scene_images
from mlsystem2.dataset_preparing.contracts import SceneImageResolutionRequest
from mlsystem2.mlflow_adapter.api import (
    get_finished_run_artifact,
    get_usable_training_checkpoint,
)
from mlsystem2.mlflow_adapter.contracts import MLflowAdapterError

from ._config import TrainingUIAPIConfig
from ._dataset_catalog import (
    find_managed_class,
    find_managed_dataset,
    dataset_class_row,
    list_managed_classes,
)
from ._datasets import CUSTOM_KEY, imagery_images_dir, resolve_scenes_file_images
from ._external_models import (
    ExternalModelError,
    ExternalModelManifest,
    external_model_payload,
    external_result_manifest,
)
from ._imagery_sources import find_imagery_source, list_imagery_sources
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
    PseudolabelSourceInfo,
    ResultStatus,
    StoredFileKind,
)


PSEUDOLABEL_AOI_OPERATION = "pseudolabel_aoi"
LOGGER = logging.getLogger(__name__)
_RESOLUTION_CACHE: dict[tuple[str, int, str], float | None] = {}


@dataclass(frozen=True)
class _SelectedCheckpoint:
    artifact_path: str
    artifact_uri: str | None
    threshold: float | None
    f1_score: float | None
    epoch: int | None


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
    target_resolution_m: float | None
    result: TrainingResultRow
    checkpoint: _SelectedCheckpoint
    external_model: ExternalModelManifest | None
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
    sources = [
        PseudolabelSourceInfo(
            source_id=item.source_id,
            display_name=item.display_name,
            kind=item.kind,
            protocol=item.protocol,
            native_channels=item.native_channels,
            imagery_type=item.imagery_type,
            attribution=item.attribution,
            license_url=item.license_url,
            available=item.available,
        )
        for item in list_imagery_sources(config)
    ]
    return PseudolabelClassListResponse(classes=items, sources=sources)


def create_pseudolabel_job(
    session: Session,
    request: PseudolabelJobCreate,
    config: TrainingUIAPIConfig,
) -> PseudolabelJobInfo:
    """Proverit AOI, zafiksirovat model i postavit job v ochered."""

    aoi_wgs84, area_m2, vertex_count = _validated_aoi(request, config)
    selected = _select_model(session, config, request.class_id, required=True)
    assert selected is not None
    source_id = request.source_id or selected.imagery_type
    source = find_imagery_source(config, source_id)
    if source is None:
        raise PseudolabelAPIError(
            "SOURCE_NOT_FOUND",
            "Источник снимков не найден.",
            status_code=404,
        )
    if not source.available:
        raise PseudolabelAPIError(
            "SOURCE_UNAVAILABLE",
            "Источник снимков сейчас недоступен.",
            status_code=409,
        )
    source_imagery_type = source.imagery_type or "external_rgb"
    cross_source = source_imagery_type != selected.imagery_type
    is_external = source.kind != "local"
    if (cross_source or is_external) and selected.target_resolution_m is None:
        raise PseudolabelAPIError(
            "MODEL_RESOLUTION_UNAVAILABLE",
            "Не удалось определить размер пикселя обучающих снимков модели.",
            status_code=409,
        )
    channel_mapping = _channel_mapping(selected.input_channels, source_imagery_type)
    source_images_root = (
        str(imagery_images_dir(config.images_root, source_imagery_type))
        if source.kind == "local"
        else ""
    )
    source_job = (
        session.get(JobRow, selected.result.job_id)
        if selected.result.job_id is not None
        else None
    )
    source_config = dict(source_job.config or {}) if source_job is not None else {}
    tile_size = (
        selected.external_model.tile_size
        if selected.external_model is not None
        else _positive_int(source_config.get("tile_preparation.tile_size"), 768)
    )
    context = (
        selected.external_model.context
        if selected.external_model is not None
        else (_integer(source_config.get("tile_preparation.context")) or 0)
    )
    core_size = tile_size - 2 * context
    if context < 0 or core_size <= 0:
        raise PseudolabelAPIError(
            "MODEL_WINDOW_INVALID",
            "Размер inference-тайла должен быть больше удвоенного context.",
            status_code=409,
        )
    stride = (
        selected.external_model.stride
        if selected.external_model is not None
        else (
            core_size
            if context
            else _positive_int(source_config.get("tile_preparation.stride"), tile_size)
        )
    )
    batch_size = (
        1
        if selected.external_model is not None
        else _positive_int(source_config.get("train.batch_size"), 1)
    )
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
        inference_dataset_name=f"AOI: {source.display_name}",
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
                "task": selected.result.task,
                "object_types": list(selected.result.class_schema or []),
                "architecture": selected.result.architecture,
                "training_result_id": str(selected.result.id),
                "mlflow_run_id": selected.result.mlflow_run_id,
                "checkpoint_artifact_path": selected.checkpoint.artifact_path,
                "checkpoint_uri": selected.checkpoint.artifact_uri,
                "checkpoint_threshold": selected.checkpoint.threshold,
                "checkpoint_f1_score": selected.checkpoint.f1_score,
                "checkpoint_epoch": selected.checkpoint.epoch,
                "external_model": external_model_payload(selected.external_model),
                "imagery_type": selected.imagery_type,
                "model_imagery_type": selected.imagery_type,
                "input_channels": selected.input_channels,
                "target_resolution_m": selected.target_resolution_m,
                "resample_to_resolution_m": (
                    selected.target_resolution_m
                    if selected.external_model is not None or cross_source or is_external
                    else None
                ),
                "source_id": source.source_id,
                "source_name": source.display_name,
                "source_kind": source.kind,
                "source_protocol": source.protocol,
                "source_imagery_type": source_imagery_type,
                "source_native_channels": source.native_channels,
                "source_attribution": source.attribution,
                "source_attributions": [source.attribution] if source.attribution else [],
                "source_license_url": source.license_url,
                "source_settings": source.settings,
                "channel_mapping": channel_mapping,
                "images_root": source_images_root,
                "inference_template_id": (
                    str(selected.inference_template_id)
                    if selected.inference_template_id is not None
                    else None
                ),
                "inference_template_config": selected.inference_template_config,
                "tile_size": tile_size,
                "context": context,
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
    class_row = dataset_class_row(session, class_info.key)
    if class_row is not None and class_row.primary_training_result_id is not None:
        primary_result = session.get(TrainingResultRow, class_row.primary_training_result_id)
        rows = [primary_result] if primary_result is not None else []
    else:
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
        if (
            row.status != ResultStatus.OK.value
            or row.trained_at is None
            or not row.mlflow_run_id
        ):
            continue
        source_job = session.get(JobRow, row.job_id) if row.job_id is not None else None
        if source_job is not None and source_job.status != JobStatus.COMPLETED.value:
            continue
        model_dataset = (
            find_managed_dataset(session, config, row.dataset_key)
            if row.dataset_key
            else dataset
        )
        if (
            model_dataset is None
            or model_dataset.images_dir is None
            or model_dataset.imagery_type is None
            or model_dataset.input_channels is None
        ):
            continue
        if (
            _training_input_channels(source_job, model_dataset.input_channels)
            != model_dataset.input_channels
        ):
            continue
        try:
            external_manifest = external_result_manifest(session, row)
        except ExternalModelError:
            continue
        if external_manifest is not None:
            try:
                artifact = get_finished_run_artifact(
                    config.mlflow_tracking_uri,
                    row.mlflow_run_id or "",
                    external_manifest.artifact_path,
                )
            except MLflowAdapterError:
                continue
            if artifact is None:
                continue
            checkpoint = _SelectedCheckpoint(
                artifact_path=artifact.artifact_path,
                artifact_uri=artifact.artifact_uri,
                threshold=external_manifest.score_threshold,
                f1_score=None,
                epoch=None,
            )
            target_resolution_m = external_manifest.target_resolution_m
            input_channels = external_manifest.input_channels
        else:
            try:
                native_checkpoint = get_usable_training_checkpoint(
                    config.mlflow_tracking_uri,
                    row.mlflow_run_id or "",
                )
            except MLflowAdapterError:
                continue
            if native_checkpoint is None:
                continue
            checkpoint = _SelectedCheckpoint(
                artifact_path=native_checkpoint.artifact_path,
                artifact_uri=native_checkpoint.artifact_uri,
                threshold=native_checkpoint.threshold,
                f1_score=native_checkpoint.f1_score,
                epoch=native_checkpoint.epoch,
            )
            target_resolution_m = _dataset_target_resolution_m(model_dataset)
            input_channels = model_dataset.input_channels
        template = inference_template_row_for_dataset(
            session,
            row.architecture,
            model_dataset.key,
        )
        template_config = (
            sanitize_inference_template_config(template.default_config)
            if template is not None
            else {}
        )
        return _SelectedModel(
            class_id=class_info.key,
            class_name=class_info.name,
            dataset_key=model_dataset.key,
            dataset_name=model_dataset.name,
            dataset_version=model_dataset.version,
            imagery_type=model_dataset.imagery_type.value,
            input_channels=input_channels,
            target_resolution_m=target_resolution_m,
            result=row,
            checkpoint=checkpoint,
            external_model=external_manifest,
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
        model_imagery_type=selected.imagery_type,
        input_channels=selected.input_channels,
        target_resolution_m=selected.target_resolution_m,
        task=selected.result.task,
        object_types=list(selected.result.class_schema or []),
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
        source_id=str(state.get("source_id") or state.get("imagery_type") or ""),
        source_name=str(state.get("source_name") or state.get("source_id") or ""),
        model_imagery_type=str(
            state.get("model_imagery_type") or state.get("imagery_type") or "kanopus"
        ),
        source_imagery_type=str(
            state.get("source_imagery_type") or state.get("imagery_type") or "kanopus"
        ),
        channel_mapping=str(state.get("channel_mapping") or "rgb_nir"),
        target_resolution_m=_optional_positive_number(state.get("target_resolution_m")),
        source_attributions=_string_list(state.get("source_attributions"))
        or _string_list([state.get("source_attribution")]),
        source_license_url=str(state.get("source_license_url") or ""),
        performance=(state.get("performance") if isinstance(state.get("performance"), dict) else {}),
        task=str(state.get("task") or "binary"),
        object_types=(
            [item for item in state.get("object_types", []) if isinstance(item, dict)]
            if isinstance(state.get("object_types"), list)
            else []
        ),
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


def _channel_mapping(model_channels: int, source_imagery_type: str) -> str:
    if model_channels not in {3, 4}:
        raise PseudolabelAPIError(
            "UNSUPPORTED_MODEL_CHANNELS",
            "AOI-инференс поддерживает модели с тремя или четырьмя входными каналами.",
            status_code=409,
        )
    if model_channels == 3:
        return "rgb"
    return "rgb_nir" if source_imagery_type == "kanopus" else "rgb_zero_nir"


def _dataset_target_resolution_m(dataset: Any) -> float | None:
    scenes_file = Path(dataset.scenes_file) if dataset.scenes_file else None
    annotations_dir = Path(dataset.annotations_dir) if dataset.annotations_dir else None
    images_dir = Path(dataset.images_dir) if dataset.images_dir else None
    if images_dir is None or not images_dir.is_dir():
        return None
    try:
        marker_path = scenes_file or annotations_dir
        marker = marker_path.stat().st_mtime_ns if marker_path and marker_path.exists() else 0
    except OSError:
        marker = 0
    dataset_source = scenes_file or annotations_dir
    key = (str(dataset_source or ""), int(marker), str(images_dir.resolve()))
    if key in _RESOLUTION_CACHE:
        return _RESOLUTION_CACHE[key]
    if annotations_dir is not None and annotations_dir.is_dir():
        try:
            resolution = resolve_scene_images(
                SceneImageResolutionRequest(
                    images_dir=str(images_dir),
                    annotations_dir=str(annotations_dir),
                )
            )
            paths = [Path(item.image_path) for item in resolution.images]
        except (OSError, ValueError):
            paths = []
    elif scenes_file is not None and scenes_file.is_file():
        paths = resolve_scenes_file_images(scenes_file, images_dir)
    else:
        paths = sorted(
            (
                path
                for path in images_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
            ),
            key=lambda path: path.as_posix().casefold(),
        )
    resolutions: list[float] = []
    for path in paths:
        try:
            with rasterio.open(path) as source:
                if source.crs is None or source.width <= 0 or source.height <= 0:
                    continue
                if PyprojCRS.from_user_input(source.crs).to_epsg() == 3857:
                    x_resolution = abs(float(source.res[0]))
                    y_resolution = abs(float(source.res[1]))
                else:
                    left, bottom, right, top = transform_bounds(
                        source.crs,
                        "EPSG:3857",
                        *source.bounds,
                        densify_pts=21,
                    )
                    x_resolution = abs(right - left) / source.width
                    y_resolution = abs(top - bottom) / source.height
                value = math.sqrt(x_resolution * y_resolution)
                if math.isfinite(value) and value > 0:
                    resolutions.append(value)
        except (OSError, ValueError, rasterio.errors.RasterioError):
            continue
    result = float(statistics.median(resolutions)) if resolutions else None
    _RESOLUTION_CACHE[key] = result
    return result


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
    return [str(item) for item in value if item is not None and str(item).strip()]


def _optional_positive_number(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


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
