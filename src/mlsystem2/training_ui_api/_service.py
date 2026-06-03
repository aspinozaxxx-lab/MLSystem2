"""Сервисные операции training UI API."""

from __future__ import annotations

import os
import shutil
import signal
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from mlsystem2.mlflow_adapter.api import create_experiment, list_experiments
from mlsystem2.mlflow_adapter.contracts import MLflowExperimentRequest
from mlsystem2.models.api import list_supported_models

from ._config import TrainingUIAPIConfig
from ._datasets import CUSTOM_KEY, CUSTOM_NAME, find_class, list_classes, list_datasets
from ._models import (
    CustomDatasetRow,
    JobRow,
    PseudoMarkupResultRow,
    QueueControlRow,
    StoredFileRow,
    TrainingResultRow,
    TrainingTemplateRow,
)
from ._templates import initial_templates
from .contracts import (
    AppLink,
    AppLinksResponse,
    ClassInfo,
    ClassListResponse,
    ClassResultsResponse,
    ConfigSchema,
    CustomDatasetInfo,
    DatasetInfo,
    DatasetListResponse,
    JobDetail,
    JobStatus,
    JobSummary,
    JobType,
    MLflowExperimentCreate,
    MLflowExperimentInfo,
    ModelInfo,
    ModelListResponse,
    PseudoMarkupResultInfo,
    QueueEnabledUpdate,
    QueueSnapshot,
    ResultStatus,
    StoredFileInfo,
    StoredFileKind,
    TemplateSource,
    TrainingJobCreate,
    TrainingResultInfo,
    TrainingTemplate,
    TrainingTemplateListResponse,
    TrainingTemplateUpdate,
    TrainingUIAPIError,
)


MODEL_DISPLAY_NAMES = {
    "smp_deeplabv3plus_resnet50": "deeplabV3+",
    "smp_segformer_b2": "segformer b2",
    "smp_segformer_b3": "segformer b3",
    "smp_unet_resnet34": "unet + resnet34",
    "smp_unet_resnet50": "unet + resnet50",
    "smp_unet_resnet101": "unet + resnet101",
    "smp_unet_resnet152": "unet + resnet152",
}
UI_ARCHITECTURES = tuple(MODEL_DISPLAY_NAMES)
ACTIVE_JOB_STATUSES = {JobStatus.QUEUED.value, JobStatus.RUNNING.value}


def app_links(config: TrainingUIAPIConfig) -> AppLinksResponse:
    return AppLinksResponse(
        links=[
            AppLink(key="grafana", title="Grafana", url=config.grafana_url),
            AppLink(key="mlflow", title="MLflow", url=config.mlflow_ui_url),
            AppLink(key="minio", title="MinIO", url=config.minio_ui_url),
        ]
    )


def mlflow_experiments(config: TrainingUIAPIConfig) -> list[MLflowExperimentInfo]:
    experiments = list_experiments(config.mlflow_tracking_uri)
    return [
        MLflowExperimentInfo(experiment_id=item.experiment_id, name=item.name)
        for item in sorted(experiments, key=lambda experiment: experiment.name.lower())
    ]


def create_mlflow_experiment(
    request: MLflowExperimentCreate,
    config: TrainingUIAPIConfig,
) -> MLflowExperimentInfo:
    experiment = create_experiment(
        MLflowExperimentRequest(tracking_uri=config.mlflow_tracking_uri, name=request.name)
    )
    return MLflowExperimentInfo(experiment_id=experiment.experiment_id, name=experiment.name)


def datasets(config: TrainingUIAPIConfig) -> DatasetListResponse:
    return DatasetListResponse(datasets=list_datasets(config.mlmarkup_root))


def classes(config: TrainingUIAPIConfig) -> ClassListResponse:
    return ClassListResponse(classes=list_classes(config.mlmarkup_root))


def models() -> ModelListResponse:
    supported = {item.name: item for item in list_supported_models()}
    ui_models = []
    for architecture in UI_ARCHITECTURES:
        spec = supported.get(architecture)
        if spec is None:
            continue
        ui_models.append(
            ModelInfo(
                architecture=architecture,
                display_name=MODEL_DISPLAY_NAMES[architecture],
                input_channels=spec.input_channels,
                output_channels=spec.output_channels,
                pretrained=spec.pretrained,
            )
        )
    return ModelListResponse(models=ui_models)


def ensure_seed_templates(session: Session) -> None:
    existing = set(session.scalars(select(TrainingTemplateRow.architecture)).all())
    for payload in initial_templates():
        if payload["architecture"] in existing:
            continue
        session.add(TrainingTemplateRow(**payload))
    _ensure_queue_control(session, JobType.TRAINING)
    _ensure_queue_control(session, JobType.INFERENCE)


def training_templates(session: Session) -> TrainingTemplateListResponse:
    ensure_seed_templates(session)
    rows = session.scalars(select(TrainingTemplateRow).order_by(TrainingTemplateRow.display_name)).all()
    return TrainingTemplateListResponse(templates=[_template_info(row) for row in rows])


def training_template(session: Session, architecture: str) -> TrainingTemplate:
    ensure_seed_templates(session)
    row = session.scalar(select(TrainingTemplateRow).where(TrainingTemplateRow.architecture == architecture))
    if row is None:
        raise TrainingUIAPIError(f"Шаблон не найден: {architecture}")
    return _template_info(row)


def update_training_template(
    session: Session,
    architecture: str,
    request: TrainingTemplateUpdate,
) -> TrainingTemplate:
    ensure_seed_templates(session)
    row = session.scalar(select(TrainingTemplateRow).where(TrainingTemplateRow.architecture == architecture))
    if row is None:
        raise TrainingUIAPIError(f"Шаблон не найден: {architecture}")
    if request.reset_to_baseline:
        row.default_config = row.baseline_default_config
        row.source = row.baseline_source
        row.source_mlflow_run_id = row.baseline_source_mlflow_run_id
        row.version += 1
    else:
        if request.default_config is not None:
            row.default_config = request.default_config
            row.source = TemplateSource.MANUAL.value
            row.source_mlflow_run_id = None
            row.version += 1
        if request.is_active is not None:
            row.is_active = request.is_active
    row.updated_at = _now()
    session.flush()
    return _template_info(row)


def create_custom_dataset(
    session: Session,
    *,
    name: str,
    scenes_name: str,
    scenes_content_type: str | None,
    scenes_bytes: bytes,
    annotation_name: str,
    annotation_content_type: str | None,
    annotation_bytes: bytes,
    config: TrainingUIAPIConfig,
) -> CustomDatasetInfo:
    _validate_upload_name(scenes_name, ".txt")
    _validate_upload_name(annotation_name, ".geojson")
    scenes_row = _store_file(
        session,
        kind=StoredFileKind.SCENES_TXT,
        original_name=scenes_name,
        content_type=scenes_content_type,
        content=scenes_bytes,
        config=config,
    )
    annotation_row = _store_file(
        session,
        kind=StoredFileKind.ANNOTATION_GEOJSON,
        original_name=annotation_name,
        content_type=annotation_content_type,
        content=annotation_bytes,
        config=config,
    )
    dataset = CustomDatasetRow(
        name=name.strip() or CUSTOM_NAME,
        scenes_file_id=scenes_row.id,
        annotation_file_id=annotation_row.id,
    )
    session.add(dataset)
    session.flush()
    return _custom_dataset_info(dataset)


def create_training_job(
    session: Session,
    request: TrainingJobCreate,
    config: TrainingUIAPIConfig,
) -> JobDetail:
    ensure_seed_templates(session)
    dataset = _resolve_dataset_name(session, request.dataset_key, request.custom_dataset_id, config)
    model_name = MODEL_DISPLAY_NAMES.get(request.architecture, request.architecture)
    tile_size = _int_or_none(request.config.get("tile_preparation.tile_size"))
    row = JobRow(
        type=JobType.TRAINING.value,
        status=JobStatus.QUEUED.value,
        queue_position=_next_queue_position(session, JobType.TRAINING),
        dataset_name=dataset.name,
        training_dataset_name=dataset.name,
        model_name=model_name,
        architecture=request.architecture,
        tile_size=tile_size,
        mlflow_experiment_id=request.mlflow_experiment_id,
        mlflow_experiment_name=request.mlflow_experiment_name,
        mlflow_run_name=request.mlflow_run_name,
        config=request.config,
        custom_dataset_id=request.custom_dataset_id,
    )
    session.add(row)
    session.flush()
    session.add(
        TrainingResultRow(
            class_key=request.dataset_key,
            class_display_name=dataset.name,
            architecture=request.architecture,
            model_name=model_name,
            status=ResultStatus.RUNNING.value,
            job_id=row.id,
        )
    )
    session.flush()
    return _job_detail(row)


def queues(session: Session) -> QueueSnapshot:
    ensure_seed_templates(session)
    training_control = _ensure_queue_control(session, JobType.TRAINING)
    inference_control = _ensure_queue_control(session, JobType.INFERENCE)
    return QueueSnapshot(
        training_enabled=training_control.enabled,
        inference_enabled=inference_control.enabled,
        training_jobs=_queue_jobs(session, JobType.TRAINING),
        inference_jobs=_queue_jobs(session, JobType.INFERENCE),
    )


def set_queue_enabled(
    session: Session,
    queue_name: JobType,
    request: QueueEnabledUpdate,
    config: TrainingUIAPIConfig,
) -> QueueSnapshot:
    control = _ensure_queue_control(session, queue_name)
    control.enabled = request.enabled
    control.updated_at = _now()
    if not request.enabled:
        running = session.scalars(
            select(JobRow).where(JobRow.type == queue_name.value, JobRow.status == JobStatus.RUNNING.value)
        ).all()
        for row in running:
            _stop_process_and_cleanup(row)
            row.status = JobStatus.QUEUED.value
            row.started_at = None
            row.finished_at = None
            row.queue_position = min(row.queue_position, 1)
    session.flush()
    return queues(session)


def job_detail(session: Session, job_id: uuid.UUID) -> JobDetail:
    row = session.get(JobRow, job_id)
    if row is None:
        raise TrainingUIAPIError(f"Задание не найдено: {job_id}")
    return _job_detail(row)


def delete_job(session: Session, job_id: uuid.UUID) -> JobDetail:
    row = session.get(JobRow, job_id)
    if row is None:
        raise TrainingUIAPIError(f"Задание не найдено: {job_id}")
    if row.status == JobStatus.RUNNING.value:
        _stop_process_and_cleanup(row)
    row.status = JobStatus.CANCELLED.value
    row.finished_at = _now()
    session.flush()
    return _job_detail(row)


def move_job(session: Session, job_id: uuid.UUID, *, direction: int) -> JobDetail:
    row = session.get(JobRow, job_id)
    if row is None:
        raise TrainingUIAPIError(f"Задание не найдено: {job_id}")
    if row.status != JobStatus.QUEUED.value:
        raise TrainingUIAPIError("Можно двигать только queued задания")
    queued = session.scalars(
        select(JobRow)
        .where(JobRow.type == row.type, JobRow.status == JobStatus.QUEUED.value)
        .order_by(JobRow.queue_position, JobRow.created_at)
    ).all()
    index = next((i for i, item in enumerate(queued) if item.id == row.id), None)
    if index is None:
        return _job_detail(row)
    target_index = index + direction
    if target_index < 0 or target_index >= len(queued):
        return _job_detail(row)
    target = queued[target_index]
    row.queue_position, target.queue_position = target.queue_position, row.queue_position
    session.flush()
    return _job_detail(row)


def class_results(
    session: Session,
    class_key: str,
    config: TrainingUIAPIConfig,
) -> ClassResultsResponse:
    class_info = find_class(config.mlmarkup_root, class_key)
    if class_info is None:
        class_info = ClassInfo(key=class_key, name=class_key)
    rows = session.scalars(
        select(TrainingResultRow)
        .where(TrainingResultRow.class_key == class_key)
        .order_by(TrainingResultRow.created_at.desc())
    ).all()
    return ClassResultsResponse(
        class_key=class_info.key,
        class_name=class_info.name,
        dataset_updated_at=class_info.updated_at,
        results=[_training_result_info(session, row) for row in rows],
    )


def create_pseudo_markup_job(
    session: Session,
    *,
    class_key: str,
    dataset_key: str | None,
    training_result_id: uuid.UUID | None,
    scenes_name: str | None,
    scenes_content_type: str | None,
    scenes_bytes: bytes | None,
    config: TrainingUIAPIConfig,
) -> JobDetail:
    class_info = find_class(config.mlmarkup_root, class_key)
    class_name = class_info.name if class_info else class_key
    scenes_file_id: uuid.UUID | None = None
    dataset_name = CUSTOM_NAME
    if scenes_bytes is not None and scenes_name is not None:
        _validate_upload_name(scenes_name, ".txt")
        scenes_row = _store_file(
            session,
            kind=StoredFileKind.SCENES_TXT,
            original_name=scenes_name,
            content_type=scenes_content_type,
            content=scenes_bytes,
            config=config,
        )
        scenes_file_id = scenes_row.id
        dataset_name = CUSTOM_NAME
    elif dataset_key:
        dataset = _resolve_dataset_name(session, dataset_key, None, config)
        dataset_name = dataset.name
        if dataset.scenes_file:
            scenes_row = _store_existing_file(
                session,
                kind=StoredFileKind.SCENES_TXT,
                path=Path(dataset.scenes_file),
            )
            scenes_file_id = scenes_row.id
    row = JobRow(
        type=JobType.INFERENCE.value,
        status=JobStatus.QUEUED.value,
        queue_position=_next_queue_position(session, JobType.INFERENCE),
        dataset_name=dataset_name,
        training_dataset_name=class_name,
        inference_dataset_name=dataset_name,
        model_name="pseudo-markup",
        architecture="pseudo-markup",
        config={
            "class_key": class_key,
            "dataset_key": dataset_key or CUSTOM_KEY,
            "training_result_id": str(training_result_id) if training_result_id else None,
        },
    )
    session.add(row)
    session.flush()
    session.add(
        PseudoMarkupResultRow(
            training_result_id=training_result_id,
            class_key=class_key,
            source_dataset_name=dataset_name,
            scenes_file_id=scenes_file_id,
            status=ResultStatus.RUNNING.value,
            job_id=row.id,
        )
    )
    session.flush()
    return _job_detail(row)


def stored_file(session: Session, file_id: uuid.UUID) -> StoredFileRow:
    row = session.get(StoredFileRow, file_id)
    if row is None:
        raise TrainingUIAPIError(f"Файл не найден: {file_id}")
    return row


def _resolve_dataset_name(
    session: Session,
    dataset_key: str,
    custom_dataset_id: uuid.UUID | None,
    config: TrainingUIAPIConfig,
) -> DatasetInfo:
    if dataset_key == CUSTOM_KEY:
        if custom_dataset_id is None:
            return DatasetInfo(key=CUSTOM_KEY, name=CUSTOM_NAME, is_custom=True)
        custom = session.get(CustomDatasetRow, custom_dataset_id)
        if custom is None:
            raise TrainingUIAPIError(f"Custom dataset не найден: {custom_dataset_id}")
        return DatasetInfo(key=CUSTOM_KEY, name=custom.name, is_custom=True)
    for dataset in list_datasets(config.mlmarkup_root):
        if dataset.key == dataset_key:
            return dataset
    raise TrainingUIAPIError(f"Датасет не найден: {dataset_key}")


def _store_file(
    session: Session,
    *,
    kind: StoredFileKind,
    original_name: str,
    content_type: str | None,
    content: bytes,
    config: TrainingUIAPIConfig,
) -> StoredFileRow:
    file_id = uuid.uuid4()
    target_dir = config.stored_files_root / kind.value
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{file_id}{Path(original_name).suffix.lower()}"
    target_path.write_bytes(content)
    row = StoredFileRow(
        id=file_id,
        kind=kind.value,
        original_name=original_name,
        content_type=content_type,
        path=str(target_path),
        size_bytes=len(content),
    )
    session.add(row)
    session.flush()
    return row


def _store_existing_file(session: Session, *, kind: StoredFileKind, path: Path) -> StoredFileRow:
    existing = session.scalar(select(StoredFileRow).where(StoredFileRow.path == str(path)))
    if existing is not None:
        return existing
    row = StoredFileRow(
        kind=kind.value,
        original_name=path.name,
        content_type="text/plain",
        path=str(path),
        size_bytes=path.stat().st_size if path.exists() else 0,
    )
    session.add(row)
    session.flush()
    return row


def _validate_upload_name(name: str, suffix: str) -> None:
    if Path(name).suffix.lower() != suffix:
        raise TrainingUIAPIError(f"Ожидался файл {suffix}: {name}")


def _next_queue_position(session: Session, queue_name: JobType) -> int:
    value = session.scalar(
        select(func.max(JobRow.queue_position)).where(
            JobRow.type == queue_name.value,
            JobRow.status.in_(ACTIVE_JOB_STATUSES),
        )
    )
    return int(value or 0) + 1


def _queue_jobs(session: Session, queue_name: JobType) -> list[JobSummary]:
    rows = session.scalars(
        select(JobRow)
        .where(JobRow.type == queue_name.value, JobRow.status.in_(ACTIVE_JOB_STATUSES))
        .order_by(JobRow.status.desc(), JobRow.queue_position, JobRow.created_at)
    ).all()
    running = [row for row in rows if row.status == JobStatus.RUNNING.value]
    queued = [row for row in rows if row.status == JobStatus.QUEUED.value]
    return [_job_summary(row) for row in [*running, *queued]]


def _ensure_queue_control(session: Session, queue_name: JobType) -> QueueControlRow:
    row = session.get(QueueControlRow, queue_name.value)
    if row is None:
        row = QueueControlRow(queue_name=queue_name.value, enabled=True)
        session.add(row)
        session.flush()
    return row


def _stop_process_and_cleanup(row: JobRow) -> None:
    if row.process_pid is not None:
        try:
            os.kill(row.process_pid, signal.SIGTERM)
        except OSError:
            pass
        row.process_pid = None
    if row.tmp_path:
        shutil.rmtree(row.tmp_path, ignore_errors=True)
        row.tmp_path = None


def _template_info(row: TrainingTemplateRow) -> TrainingTemplate:
    return TrainingTemplate(
        id=row.id,
        architecture=row.architecture,
        display_name=row.display_name,
        config_schema=ConfigSchema.model_validate(row.config_schema),
        default_config=row.default_config,
        source=TemplateSource(row.source),
        source_mlflow_run_id=row.source_mlflow_run_id,
        is_active=row.is_active,
        version=row.version,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _custom_dataset_info(row: CustomDatasetRow) -> CustomDatasetInfo:
    return CustomDatasetInfo(
        id=row.id,
        name=row.name,
        scenes_file=_stored_file_info(row.scenes_file),
        annotation_file=_stored_file_info(row.annotation_file),
        created_at=row.created_at,
    )


def _stored_file_info(row: StoredFileRow | None) -> StoredFileInfo | None:
    if row is None:
        return None
    return StoredFileInfo(
        id=row.id,
        kind=StoredFileKind(row.kind),
        original_name=row.original_name,
        size_bytes=row.size_bytes,
        created_at=row.created_at,
        download_url=f"/api/v1/files/{row.id}/download",
    )


def _job_summary(row: JobRow) -> JobSummary:
    return JobSummary(
        id=row.id,
        type=JobType(row.type),
        status=JobStatus(row.status),
        queue_position=row.queue_position,
        dataset_name=row.dataset_name,
        training_dataset_name=row.training_dataset_name,
        inference_dataset_name=row.inference_dataset_name,
        model_name=row.model_name,
        architecture=row.architecture,
        tile_size=row.tile_size,
        created_at=row.created_at,
        started_at=row.started_at,
        actions=_job_actions(row),
    )


def _job_detail(row: JobRow) -> JobDetail:
    return JobDetail(
        id=row.id,
        type=JobType(row.type),
        status=JobStatus(row.status),
        queue_position=row.queue_position,
        dataset_name=row.dataset_name,
        training_dataset_name=row.training_dataset_name,
        inference_dataset_name=row.inference_dataset_name,
        model_name=row.model_name,
        architecture=row.architecture,
        tile_size=row.tile_size,
        mlflow_experiment_name=row.mlflow_experiment_name,
        mlflow_run_name=row.mlflow_run_name,
        config=row.config,
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )


def _training_result_info(session: Session, row: TrainingResultRow) -> TrainingResultInfo:
    pseudo_rows = session.scalars(
        select(PseudoMarkupResultRow)
        .where(PseudoMarkupResultRow.training_result_id == row.id)
        .order_by(PseudoMarkupResultRow.created_at.desc())
    ).all()
    return TrainingResultInfo(
        id=row.id,
        model_name=row.model_name,
        architecture=row.architecture,
        f1_score=row.f1_score,
        epoch=row.epoch,
        trained_at=row.trained_at,
        mlflow_run_url=row.mlflow_run_url,
        status=ResultStatus(row.status),
        pseudo_markup_results=[_pseudo_markup_info(item) for item in pseudo_rows],
    )


def _pseudo_markup_info(row: PseudoMarkupResultRow) -> PseudoMarkupResultInfo:
    return PseudoMarkupResultInfo(
        id=row.id,
        source_dataset_name=row.source_dataset_name,
        scenes_file=_stored_file_info(row.scenes_file),
        geojson_file=_stored_file_info(row.geojson_file),
        status=ResultStatus(row.status),
        created_at=row.created_at,
    )


def _job_actions(row: JobRow) -> list[str]:
    if row.status == JobStatus.RUNNING.value:
        return ["delete"]
    if row.status == JobStatus.QUEUED.value:
        return ["move_up", "move_down", "delete"]
    return []


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _now() -> datetime:
    return datetime.now(timezone.utc)

