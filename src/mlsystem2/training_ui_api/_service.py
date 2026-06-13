"""Сервисные операции training UI API."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from mlsystem2.mlflow_adapter.api import (
    create_experiment,
    get_best_training_checkpoint,
    get_training_epoch_progress,
    list_experiments,
    mark_run_killed,
)
from mlsystem2.mlflow_adapter.contracts import (
    MLflowAdapterError,
    MLflowBestCheckpoint,
    MLflowExperimentRequest,
)

from ._automation import (
    automation_snapshot,
    ensure_automation_control,
    set_automation_enabled,
    sync_automation_once,
    update_automation_rule,
)
from ._catalog import MODEL_DISPLAY_NAMES, ui_model_infos
from ._config import TrainingUIAPIConfig, get_config
from ._datasets import (
    CUSTOM_KEY,
    CUSTOM_NAME,
    count_scenes_file_images,
    find_dataset,
    find_image_folder,
    list_classes,
    list_datasets,
    list_image_folders,
)
from ._models import (
    CustomDatasetRow,
    InferenceTemplateRow,
    JobRow,
    PseudoMarkupResultRow,
    QueueControlRow,
    StoredFileRow,
    TrainingResultRow,
    TrainingTemplateRow,
)
from ._processes import terminate_job_process
from ._queueing import ensure_queue_positions, next_queue_position, queue_sort_key
from ._templates import (
    initial_inference_templates,
    initial_templates,
    sanitize_inference_template_config,
    sanitize_template_config,
)
from .contracts import (
    AppLink,
    AppLinksResponse,
    AutomationEnabledUpdate,
    AutomationRuleInfo,
    AutomationRuleUpdate,
    AutomationSnapshot,
    ClassListResponse,
    ClassResultsResponse,
    ConfigSchema,
    CustomDatasetInfo,
    DatasetInfo,
    DatasetListResponse,
    ImageFolderListResponse,
    InferenceTemplate,
    InferenceTemplateApplyField,
    InferenceTemplateCreate,
    InferenceTemplateListResponse,
    InferenceTemplateUpdate,
    JobDetail,
    JobSource,
    JobStatus,
    JobSummary,
    JobType,
    MLflowExperimentCreate,
    MLflowExperimentInfo,
    ModelListResponse,
    PseudoMarkupResultInfo,
    QueueEnabledUpdate,
    QueueSnapshot,
    ResultStatus,
    ResultChangeInfo,
    ResultChangesResponse,
    RuntimeProgress,
    StoredFileInfo,
    StoredFileKind,
    TemplateSource,
    TrainingTemplateApplyField,
    TrainingTemplateCreate,
    TrainingJobCreate,
    TrainingResultInfo,
    TrainingTemplate,
    TrainingTemplateListResponse,
    TrainingTemplateUpdate,
    TrainingUIAPIError,
)


ACTIVE_JOB_STATUSES = {JobStatus.QUEUED.value, JobStatus.RUNNING.value}


def app_links(config: TrainingUIAPIConfig) -> AppLinksResponse:
    return AppLinksResponse(
        links=[
            AppLink(key="grafana", title="Grafana", url=config.grafana_url),
            AppLink(key="mlflow", title="MLflow", url=config.mlflow_ui_url),
            AppLink(key="minio", title="MinIO", url=config.minio_ui_url),
            AppLink(key="open_webui", title="Open WebUI", url=config.open_webui_url),
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
    return DatasetListResponse(datasets=list_datasets(config.mlmarkup_root, config.images_root))


def classes(config: TrainingUIAPIConfig) -> ClassListResponse:
    return ClassListResponse(classes=list_classes(config.mlmarkup_root, config.images_root))


def image_folders(config: TrainingUIAPIConfig) -> ImageFolderListResponse:
    return ImageFolderListResponse(folders=list_image_folders(config.images_root))


def models() -> ModelListResponse:
    return ModelListResponse(models=ui_model_infos())


def automation(session: Session, config: TrainingUIAPIConfig) -> AutomationSnapshot:
    ensure_seed_templates(session)
    return automation_snapshot(session, config)


def set_automation(
    session: Session,
    request: AutomationEnabledUpdate,
    config: TrainingUIAPIConfig,
) -> AutomationSnapshot:
    set_automation_enabled(session, request, config)
    if request.enabled:
        sync_automation_once(session, config)
    return automation_snapshot(session, config)


def update_automation(
    session: Session,
    request: AutomationRuleUpdate,
    config: TrainingUIAPIConfig,
) -> AutomationRuleInfo:
    ensure_seed_templates(session)
    return update_automation_rule(session, request, config)


def ensure_seed_templates(session: Session) -> None:
    existing = {
        (row.architecture, row.dataset_key): row
        for row in session.scalars(select(TrainingTemplateRow)).all()
    }
    seed_payloads = initial_templates()
    for payload in seed_payloads:
        row = existing.get((payload["architecture"], None))
        if row is None:
            session.add(TrainingTemplateRow(**payload))
            continue
        row.dataset_key = None
        row.dataset_name = None
        row.parent_template_id = None
        row.display_name = payload["display_name"]
        row.config_schema = payload["config_schema"]
        row.default_config = sanitize_template_config(
            row.default_config,
            fallback=payload["default_config"],
        )
        row.baseline_default_config = sanitize_template_config(
            row.baseline_default_config,
            fallback=payload["baseline_default_config"],
        )
    baselines = {payload["architecture"]: payload for payload in seed_payloads}
    dataset_rows = session.scalars(
        select(TrainingTemplateRow).where(TrainingTemplateRow.dataset_key.is_not(None))
    ).all()
    for row in dataset_rows:
        baseline = baselines.get(row.architecture)
        if baseline is None:
            continue
        row.config_schema = baseline["config_schema"]
        row.default_config = sanitize_template_config(
            row.default_config,
            fallback=baseline["default_config"],
        )
        row.baseline_default_config = sanitize_template_config(
            row.baseline_default_config,
            fallback=baseline["baseline_default_config"],
        )
    _ensure_seed_inference_templates(session)
    _ensure_queue_control(session, JobType.TRAINING)
    _ensure_queue_control(session, JobType.INFERENCE)
    ensure_automation_control(session)


def _ensure_seed_inference_templates(session: Session) -> None:
    existing = {
        (row.architecture, row.dataset_key): row
        for row in session.scalars(select(InferenceTemplateRow)).all()
    }
    seed_payloads = initial_inference_templates()
    base_payloads = [payload for payload in seed_payloads if payload.get("dataset_key") is None]
    dataset_payloads = [payload for payload in seed_payloads if payload.get("dataset_key") is not None]
    for payload in base_payloads:
        row = existing.get((payload["architecture"], None))
        if row is None:
            session.add(InferenceTemplateRow(**payload))
            continue
        row.dataset_key = None
        row.dataset_name = None
        row.parent_template_id = None
        row.display_name = payload["display_name"]
        row.config_schema = payload["config_schema"]
        row.default_config = sanitize_inference_template_config(
            row.default_config,
            fallback=payload["default_config"],
        )
        row.baseline_default_config = sanitize_inference_template_config(
            row.baseline_default_config,
            fallback=payload["baseline_default_config"],
        )
    session.flush()

    base_rows = {
        row.architecture: row
        for row in session.scalars(
            select(InferenceTemplateRow).where(InferenceTemplateRow.dataset_key.is_(None))
        ).all()
    }
    for payload in dataset_payloads:
        parent = base_rows.get(payload["architecture"])
        if parent is None:
            continue
        row = existing.get((payload["architecture"], payload["dataset_key"]))
        if row is None:
            row = InferenceTemplateRow(**payload)
            row.parent_template_id = parent.id
            session.add(row)
            continue
        row.parent_template_id = parent.id
        row.display_name = payload["display_name"]
        row.dataset_name = payload["dataset_name"]
        row.config_schema = parent.config_schema
        row.default_config = sanitize_inference_template_config(
            row.default_config,
            fallback=payload["default_config"],
        )
        row.baseline_default_config = sanitize_inference_template_config(
            row.baseline_default_config,
            fallback=payload["baseline_default_config"],
        )


def training_templates(session: Session) -> TrainingTemplateListResponse:
    ensure_seed_templates(session)
    rows = session.scalars(
        select(TrainingTemplateRow).order_by(
            TrainingTemplateRow.display_name,
            TrainingTemplateRow.dataset_name,
        )
    ).all()
    return TrainingTemplateListResponse(templates=[_template_info(row) for row in rows])


def training_template(session: Session, architecture: str) -> TrainingTemplate:
    ensure_seed_templates(session)
    row = _base_template_row(session, architecture)
    if row is None:
        raise TrainingUIAPIError(f"Шаблон не найден: {architecture}")
    return _template_info(row)


def update_training_template(
    session: Session,
    architecture: str,
    request: TrainingTemplateUpdate,
) -> TrainingTemplate:
    ensure_seed_templates(session)
    row = _base_template_row(session, architecture)
    if row is None:
        raise TrainingUIAPIError(f"Шаблон не найден: {architecture}")
    return update_training_template_by_id(session, row.id, request)


def create_training_template(
    session: Session,
    request: TrainingTemplateCreate,
    config: TrainingUIAPIConfig,
) -> TrainingTemplate:
    ensure_seed_templates(session)
    parent = _base_template_row(session, request.architecture)
    if parent is None:
        raise TrainingUIAPIError(f"Шаблон сети не найден: {request.architecture}")
    dataset = find_dataset(config.mlmarkup_root, request.dataset_key)
    if dataset is None or dataset.is_custom:
        raise TrainingUIAPIError(f"Датасет не найден: {request.dataset_key}")
    existing = _dataset_template_row(session, request.architecture, dataset.key)
    if existing is not None:
        raise TrainingUIAPIError(f"Шаблон для датасета уже существует: {dataset.name}")
    now = _now()
    row = TrainingTemplateRow(
        architecture=parent.architecture,
        dataset_key=dataset.key,
        dataset_name=dataset.name,
        parent_template_id=parent.id,
        display_name=f"{parent.display_name} / {dataset.name}",
        config_schema=parent.config_schema,
        default_config=sanitize_template_config(parent.default_config),
        baseline_default_config=sanitize_template_config(parent.default_config),
        source=parent.source,
        baseline_source=parent.source,
        source_mlflow_run_id=parent.source_mlflow_run_id,
        baseline_source_mlflow_run_id=parent.source_mlflow_run_id,
        is_active=True,
        version=1,
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    session.flush()
    return _template_info(row)


def update_training_template_by_id(
    session: Session,
    template_id: uuid.UUID,
    request: TrainingTemplateUpdate,
) -> TrainingTemplate:
    ensure_seed_templates(session)
    row = session.get(TrainingTemplateRow, template_id)
    if row is None:
        raise TrainingUIAPIError(f"Шаблон не найден: {template_id}")
    if request.reset_to_baseline:
        row.default_config = row.baseline_default_config
        row.source = row.baseline_source
        row.source_mlflow_run_id = row.baseline_source_mlflow_run_id
        row.version += 1
    else:
        if request.default_config is not None:
            row.default_config = sanitize_template_config(
                request.default_config,
                fallback=row.default_config,
            )
            row.source = TemplateSource.MANUAL.value
            row.source_mlflow_run_id = None
            row.version += 1
        if request.is_active is not None:
            row.is_active = request.is_active
    row.updated_at = _now()
    session.flush()
    return _template_info(row)


def delete_training_template(
    session: Session,
    template_id: uuid.UUID,
) -> TrainingTemplate:
    ensure_seed_templates(session)
    row = session.get(TrainingTemplateRow, template_id)
    if row is None:
        raise TrainingUIAPIError(f"Шаблон не найден: {template_id}")
    if row.dataset_key is None:
        raise TrainingUIAPIError("Базовый шаблон сети удалить нельзя")
    info = _template_info(row)
    session.delete(row)
    session.flush()
    return info


def apply_training_template_field_to_all(
    session: Session,
    template_id: uuid.UUID,
    request: TrainingTemplateApplyField,
) -> TrainingTemplateListResponse:
    ensure_seed_templates(session)
    row = session.get(TrainingTemplateRow, template_id)
    if row is None:
        raise TrainingUIAPIError(f"Шаблон не найден: {template_id}")
    if request.key not in {str(field["key"]) for field in row.config_schema.get("fields", [])}:
        raise TrainingUIAPIError(f"Параметр шаблона не найден: {request.key}")
    for template in session.scalars(select(TrainingTemplateRow)).all():
        current = dict(template.default_config)
        current[request.key] = request.value
        template.default_config = sanitize_template_config(current, fallback=template.default_config)
        template.source = TemplateSource.MANUAL.value
        template.source_mlflow_run_id = None
        template.version += 1
        template.updated_at = _now()
    session.flush()
    return training_templates(session)


def inference_templates(session: Session) -> InferenceTemplateListResponse:
    ensure_seed_templates(session)
    rows = session.scalars(
        select(InferenceTemplateRow).order_by(
            InferenceTemplateRow.display_name,
            InferenceTemplateRow.dataset_name,
        )
    ).all()
    return InferenceTemplateListResponse(templates=[_inference_template_info(row) for row in rows])


def inference_template(session: Session, architecture: str) -> InferenceTemplate:
    ensure_seed_templates(session)
    row = _base_inference_template_row(session, architecture)
    if row is None:
        raise TrainingUIAPIError(f"Шаблон инференса не найден: {architecture}")
    return _inference_template_info(row)


def update_inference_template(
    session: Session,
    architecture: str,
    request: InferenceTemplateUpdate,
) -> InferenceTemplate:
    ensure_seed_templates(session)
    row = _base_inference_template_row(session, architecture)
    if row is None:
        raise TrainingUIAPIError(f"Шаблон инференса не найден: {architecture}")
    return update_inference_template_by_id(session, row.id, request)


def create_inference_template(
    session: Session,
    request: InferenceTemplateCreate,
    config: TrainingUIAPIConfig,
) -> InferenceTemplate:
    ensure_seed_templates(session)
    parent = _base_inference_template_row(session, request.architecture)
    if parent is None:
        raise TrainingUIAPIError(f"Шаблон инференса сети не найден: {request.architecture}")
    dataset = find_dataset(config.mlmarkup_root, request.dataset_key)
    if dataset is None or dataset.is_custom:
        raise TrainingUIAPIError(f"Датасет не найден: {request.dataset_key}")
    existing = _dataset_inference_template_row(session, request.architecture, dataset.key)
    if existing is not None:
        raise TrainingUIAPIError(f"Шаблон инференса для датасета уже существует: {dataset.name}")
    now = _now()
    row = InferenceTemplateRow(
        architecture=parent.architecture,
        dataset_key=dataset.key,
        dataset_name=dataset.name,
        parent_template_id=parent.id,
        display_name=f"{parent.display_name} / {dataset.name}",
        config_schema=parent.config_schema,
        default_config=sanitize_inference_template_config(parent.default_config),
        baseline_default_config=sanitize_inference_template_config(parent.default_config),
        source=parent.source,
        baseline_source=parent.source,
        source_mlflow_run_id=parent.source_mlflow_run_id,
        baseline_source_mlflow_run_id=parent.source_mlflow_run_id,
        is_active=True,
        version=1,
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    session.flush()
    return _inference_template_info(row)


def update_inference_template_by_id(
    session: Session,
    template_id: uuid.UUID,
    request: InferenceTemplateUpdate,
) -> InferenceTemplate:
    ensure_seed_templates(session)
    row = session.get(InferenceTemplateRow, template_id)
    if row is None:
        raise TrainingUIAPIError(f"Шаблон инференса не найден: {template_id}")
    if request.reset_to_baseline:
        row.default_config = row.baseline_default_config
        row.source = row.baseline_source
        row.source_mlflow_run_id = row.baseline_source_mlflow_run_id
        row.version += 1
    else:
        if request.default_config is not None:
            row.default_config = sanitize_inference_template_config(
                request.default_config,
                fallback=row.default_config,
            )
            row.source = TemplateSource.MANUAL.value
            row.source_mlflow_run_id = None
            row.version += 1
        if request.is_active is not None:
            row.is_active = request.is_active
    row.updated_at = _now()
    session.flush()
    return _inference_template_info(row)


def delete_inference_template(
    session: Session,
    template_id: uuid.UUID,
) -> InferenceTemplate:
    ensure_seed_templates(session)
    row = session.get(InferenceTemplateRow, template_id)
    if row is None:
        raise TrainingUIAPIError(f"Шаблон инференса не найден: {template_id}")
    if row.dataset_key is None:
        raise TrainingUIAPIError("Базовый шаблон инференса сети удалить нельзя")
    info = _inference_template_info(row)
    session.delete(row)
    session.flush()
    return info


def apply_inference_template_field_to_all(
    session: Session,
    template_id: uuid.UUID,
    request: InferenceTemplateApplyField,
) -> InferenceTemplateListResponse:
    ensure_seed_templates(session)
    row = session.get(InferenceTemplateRow, template_id)
    if row is None:
        raise TrainingUIAPIError(f"Шаблон инференса не найден: {template_id}")
    if request.key not in {str(field["key"]) for field in row.config_schema.get("fields", [])}:
        raise TrainingUIAPIError(f"Параметр шаблона инференса не найден: {request.key}")
    for template in session.scalars(select(InferenceTemplateRow)).all():
        current = dict(template.default_config)
        current[request.key] = request.value
        template.default_config = sanitize_inference_template_config(
            current,
            fallback=template.default_config,
        )
        template.source = TemplateSource.MANUAL.value
        template.source_mlflow_run_id = None
        template.version += 1
        template.updated_at = _now()
    session.flush()
    return inference_templates(session)


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
    template_row = training_template_row_for_dataset(session, request.architecture, request.dataset_key)
    job_config = sanitize_template_config(
        request.config,
        fallback=template_row.default_config if template_row is not None else None,
    )
    tile_size = _int_or_none(job_config.get("tile_preparation.tile_size"))
    row = JobRow(
        type=JobType.TRAINING.value,
        source=JobSource.MANUAL.value,
        status=JobStatus.QUEUED.value,
        queue_position=_next_queue_position(session, JobType.TRAINING, JobSource.MANUAL),
        dataset_key=request.dataset_key,
        dataset_version=dataset.version,
        dataset_name=dataset.name,
        training_dataset_name=dataset.name,
        model_name=model_name,
        architecture=request.architecture,
        tile_size=tile_size,
        mlflow_experiment_id=request.mlflow_experiment_id,
        mlflow_experiment_name=request.mlflow_experiment_name,
        mlflow_run_name=request.mlflow_run_name,
        config=job_config,
        custom_dataset_id=request.custom_dataset_id,
    )
    session.add(row)
    session.flush()
    session.add(
        TrainingResultRow(
            source=JobSource.MANUAL.value,
            dataset_key=request.dataset_key,
            dataset_version=dataset.version,
            class_key=request.dataset_key,
            class_display_name=dataset.name,
            architecture=request.architecture,
            model_name=model_name,
            status=ResultStatus.RUNNING.value,
            job_id=row.id,
        )
    )
    session.flush()
    return _job_detail(session, row)


def queues(session: Session) -> QueueSnapshot:
    ensure_seed_templates(session)
    _delete_cancelled_manual_jobs(session)
    ensure_queue_positions(session)
    training_control = _ensure_queue_control(session, JobType.TRAINING)
    inference_control = _ensure_queue_control(session, JobType.INFERENCE)
    return QueueSnapshot(
        training_enabled=training_control.enabled,
        inference_enabled=inference_control.enabled,
        jobs=_queue_jobs(session),
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
    return _job_detail(session, row)


def delete_job(session: Session, job_id: uuid.UUID) -> JobDetail:
    row = session.get(JobRow, job_id)
    if row is None:
        raise TrainingUIAPIError(f"Задание не найдено: {job_id}")
    if row.source == JobSource.AUTOMATION.value:
        raise TrainingUIAPIError("Автоматические задания отменяются только через форму автоматизации")
    detail = _job_detail(session, row).model_copy(update={"status": JobStatus.CANCELLED})
    mlflow_run_id = _job_mlflow_run_id(session, row)
    if row.status == JobStatus.RUNNING.value:
        _stop_process_and_cleanup(row)
        if mlflow_run_id:
            _mark_mlflow_run_killed(mlflow_run_id)
    _delete_job_rows(session, row)
    session.flush()
    return detail


def move_job(session: Session, job_id: uuid.UUID, *, direction: int) -> JobDetail:
    row = session.get(JobRow, job_id)
    if row is None:
        raise TrainingUIAPIError(f"Задание не найдено: {job_id}")
    if row.source == JobSource.AUTOMATION.value:
        raise TrainingUIAPIError("Автоматические задания нельзя двигать вручную")
    if row.status != JobStatus.QUEUED.value:
        raise TrainingUIAPIError("Можно двигать только queued задания")
    ensure_queue_positions(session)
    queued = _queue_rows(session, manual_only=True, queued_only=True)
    index = next((i for i, item in enumerate(queued) if item.id == row.id), None)
    if index is None:
        return _job_detail(session, row)
    target_index = index + direction
    if target_index < 0 or target_index >= len(queued):
        return _job_detail(session, row)
    target = queued[target_index]
    row.queue_position, target.queue_position = target.queue_position, row.queue_position
    session.flush()
    return _job_detail(session, row)


def class_results(
    session: Session,
    class_key: str,
    config: TrainingUIAPIConfig,
) -> ClassResultsResponse:
    dataset_info = find_dataset(config.mlmarkup_root, class_key)
    if dataset_info is None:
        dataset_info = DatasetInfo(key=class_key, name=class_key)
    _delete_cancelled_manual_jobs(session)
    _delete_cancelled_results_for_class(session, class_key)
    rows = session.scalars(
        select(TrainingResultRow)
        .where(
            TrainingResultRow.class_key == class_key,
            TrainingResultRow.status != ResultStatus.CANCELLED.value,
        )
        .order_by(TrainingResultRow.created_at.desc())
    ).all()
    return ClassResultsResponse(
        class_key=dataset_info.key,
        class_name=dataset_info.name,
        dataset_updated_at=dataset_info.updated_at,
        results=[_training_result_info(session, row) for row in rows],
    )


def result_changes(session: Session, limit: int = 20) -> ResultChangesResponse:
    training_rows = session.scalars(
        select(TrainingResultRow)
        .where(
            TrainingResultRow.status == ResultStatus.OK.value,
        )
        .order_by(TrainingResultRow.updated_at.desc())
        .limit(limit)
    ).all()
    pseudo_rows = session.scalars(
        select(PseudoMarkupResultRow)
        .where(
            PseudoMarkupResultRow.status == ResultStatus.OK.value,
        )
        .order_by(PseudoMarkupResultRow.updated_at.desc())
        .limit(limit)
    ).all()
    changes: list[ResultChangeInfo] = []
    for row in training_rows:
        changes.append(
            ResultChangeInfo(
                id=row.id,
                class_key=row.class_key,
                dataset_name=row.class_display_name,
                model_name=row.model_name,
                action="обучена сеть",
                source=JobSource(row.source),
                status=ResultStatus(row.status),
                changed_at=row.updated_at or row.trained_at or row.created_at or _now(),
            )
        )
    for row in pseudo_rows:
        model_name = row.training_result.model_name if row.training_result is not None else "псевдоразметка"
        changes.append(
            ResultChangeInfo(
                id=row.id,
                class_key=row.class_key,
                dataset_name=row.source_dataset_name,
                model_name=model_name,
                action="создана разметка",
                source=JobSource(row.source),
                status=ResultStatus(row.status),
                changed_at=row.updated_at or row.created_at or _now(),
            )
        )
    changes.sort(key=lambda item: item.changed_at, reverse=True)
    return ResultChangesResponse(changes=changes[:limit])


def create_pseudo_markup_job(
    session: Session,
    *,
    class_key: str,
    dataset_key: str | None,
    image_folder_key: str | None,
    training_result_id: uuid.UUID | None,
    scenes_name: str | None,
    scenes_content_type: str | None,
    scenes_bytes: bytes | None,
    config: TrainingUIAPIConfig,
) -> JobDetail:
    ensure_seed_templates(session)
    dataset_key = (dataset_key or "").strip() or None
    image_folder_key = (image_folder_key or "").strip().strip("/").replace("\\", "/") or None
    has_uploaded_scenes = scenes_bytes is not None and scenes_name is not None
    source_count = sum(1 for value in (has_uploaded_scenes, bool(dataset_key), bool(image_folder_key)) if value)
    if source_count != 1:
        if source_count == 0:
            raise TrainingUIAPIError("Выберите датасет, папку снимков или загрузите txt со снимками")
        raise TrainingUIAPIError("Выберите только один источник снимков")
    class_dataset = find_dataset(config.mlmarkup_root, class_key)
    class_name = class_dataset.name if class_dataset else class_key
    training_result = _resolve_training_result(session, training_result_id)
    inference_template = (
        inference_template_row_for_dataset(session, training_result.architecture, dataset_key)
        if training_result is not None
        else None
    )
    inference_template_config = (
        sanitize_inference_template_config(inference_template.default_config)
        if inference_template is not None
        else {}
    )
    scenes_file_id: uuid.UUID | None = None
    dataset_name = CUSTOM_NAME
    inference_dataset_version: str | None = None
    if has_uploaded_scenes:
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
        inference_dataset_version = dataset.version
        if dataset.scenes_file:
            scenes_row = _store_existing_file(
                session,
                kind=StoredFileKind.SCENES_TXT,
                path=Path(dataset.scenes_file),
            )
            scenes_file_id = scenes_row.id
    elif image_folder_key:
        folder = find_image_folder(config.images_root, image_folder_key)
        if folder is None:
            raise TrainingUIAPIError(f"Папка снимков не найдена: {image_folder_key}")
        scenes_row = _store_file(
            session,
            kind=StoredFileKind.SCENES_TXT,
            original_name=f"{Path(image_folder_key).name or 'images'}.txt",
            content_type="text/plain",
            content=f"{image_folder_key}\n".encode("utf-8"),
            config=config,
        )
        scenes_file_id = scenes_row.id
        dataset_name = folder.name
    else:
        raise TrainingUIAPIError("Выберите датасет, папку снимков или загрузите txt со снимками")
    row = JobRow(
        type=JobType.INFERENCE.value,
        source=JobSource.MANUAL.value,
        status=JobStatus.QUEUED.value,
        queue_position=_next_queue_position(session, JobType.INFERENCE, JobSource.MANUAL),
        dataset_key=dataset_key or CUSTOM_KEY,
        dataset_version=inference_dataset_version,
        dataset_name=dataset_name,
        training_dataset_name=class_name,
        inference_dataset_name=dataset_name,
        model_name="pseudo-markup",
        architecture="pseudo-markup",
        config={
            "class_key": class_key,
            "dataset_key": dataset_key or CUSTOM_KEY,
            "image_folder_key": image_folder_key,
            "training_result_id": str(training_result_id) if training_result_id else None,
            "inference_template_id": str(inference_template.id) if inference_template is not None else None,
            "inference_template_config": inference_template_config,
            **_checkpoint_config(training_result, config),
        },
    )
    session.add(row)
    session.flush()
    session.add(
        PseudoMarkupResultRow(
            source=JobSource.MANUAL.value,
            dataset_key=dataset_key or CUSTOM_KEY,
            dataset_version=inference_dataset_version,
            training_result_id=training_result_id,
            class_key=class_key,
            source_dataset_name=dataset_name,
            scenes_file_id=scenes_file_id,
            status=ResultStatus.RUNNING.value,
            job_id=row.id,
        )
    )
    session.flush()
    return _job_detail(session, row)


def delete_pseudo_markup_result(
    session: Session,
    result_id: uuid.UUID,
    config: TrainingUIAPIConfig,
) -> PseudoMarkupResultInfo:
    row = session.get(PseudoMarkupResultRow, result_id)
    if row is None:
        raise TrainingUIAPIError(f"Результат псевдоразметки не найден: {result_id}")
    detail = _pseudo_markup_info(session, row)
    job = session.get(JobRow, row.job_id) if row.job_id is not None else None
    if job is not None and job.status == JobStatus.RUNNING.value:
        _stop_process_and_cleanup(job)
    stored_files = [row.scenes_file, row.geojson_file]
    session.delete(row)
    session.flush()
    if job is not None and job.type == JobType.INFERENCE.value:
        remaining = session.scalar(
            select(PseudoMarkupResultRow.id).where(PseudoMarkupResultRow.job_id == job.id).limit(1)
        )
        if remaining is None:
            session.delete(job)
            session.flush()
    for file_row in stored_files:
        _delete_owned_stored_file_if_unreferenced(session, file_row, config)
    session.flush()
    return detail


def _resolve_training_result(
    session: Session,
    training_result_id: uuid.UUID | None,
) -> TrainingResultRow | None:
    if training_result_id is None:
        return None
    row = session.get(TrainingResultRow, training_result_id)
    if row is None:
        raise TrainingUIAPIError(f"Результат обучения не найден: {training_result_id}")
    return row


def _checkpoint_config(
    row: TrainingResultRow | None,
    config: TrainingUIAPIConfig,
) -> dict[str, object]:
    if row is None or row.mlflow_run_id is None:
        return {}
    payload: dict[str, object] = {
        "mlflow_run_id": row.mlflow_run_id,
        "checkpoint_artifact_path": "checkpoints/best.pt",
    }
    checkpoint = _best_training_checkpoint(row, config)
    if checkpoint is not None:
        payload.update(
            {
                "checkpoint_uri": checkpoint.artifact_uri,
                "checkpoint_metric_name": checkpoint.metric_name,
                "checkpoint_f1_score": checkpoint.f1_score,
                "checkpoint_epoch": checkpoint.epoch,
                "checkpoint_threshold": checkpoint.threshold,
            }
        )
    return payload


def _best_training_checkpoint(
    row: TrainingResultRow,
    config: TrainingUIAPIConfig,
) -> MLflowBestCheckpoint | None:
    if row.mlflow_run_id is None:
        return None
    try:
        return get_best_training_checkpoint(config.mlflow_tracking_uri, row.mlflow_run_id)
    except MLflowAdapterError:
        return None


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


def training_template_row_for_dataset(
    session: Session,
    architecture: str,
    dataset_key: str | None,
) -> TrainingTemplateRow | None:
    if dataset_key and dataset_key != CUSTOM_KEY:
        row = _dataset_template_row(session, architecture, dataset_key)
        if row is not None and row.is_active:
            return row
    return _base_template_row(session, architecture)


def inference_template_row_for_dataset(
    session: Session,
    architecture: str,
    dataset_key: str | None,
) -> InferenceTemplateRow | None:
    if dataset_key:
        row = _dataset_inference_template_row(session, architecture, dataset_key)
        if row is not None and row.is_active:
            return row
    return _base_inference_template_row(session, architecture)


def _base_template_row(session: Session, architecture: str) -> TrainingTemplateRow | None:
    return session.scalar(
        select(TrainingTemplateRow).where(
            TrainingTemplateRow.architecture == architecture,
            TrainingTemplateRow.dataset_key.is_(None),
        )
    )


def _base_inference_template_row(session: Session, architecture: str) -> InferenceTemplateRow | None:
    return session.scalar(
        select(InferenceTemplateRow).where(
            InferenceTemplateRow.architecture == architecture,
            InferenceTemplateRow.dataset_key.is_(None),
        )
    )


def _dataset_template_row(
    session: Session,
    architecture: str,
    dataset_key: str,
) -> TrainingTemplateRow | None:
    return session.scalar(
        select(TrainingTemplateRow).where(
            TrainingTemplateRow.architecture == architecture,
            TrainingTemplateRow.dataset_key == dataset_key,
        )
    )


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


def _delete_owned_stored_file_if_unreferenced(
    session: Session,
    row: StoredFileRow | None,
    config: TrainingUIAPIConfig,
) -> None:
    if row is None or _stored_file_is_referenced(session, row.id):
        return
    if not _is_owned_stored_file(row, config):
        return
    Path(row.path).unlink(missing_ok=True)
    session.delete(row)


def _stored_file_is_referenced(session: Session, file_id: uuid.UUID) -> bool:
    pseudo_reference = session.scalar(
        select(PseudoMarkupResultRow.id)
        .where(
            (PseudoMarkupResultRow.scenes_file_id == file_id)
            | (PseudoMarkupResultRow.geojson_file_id == file_id)
        )
        .limit(1)
    )
    if pseudo_reference is not None:
        return True
    custom_reference = session.scalar(
        select(CustomDatasetRow.id)
        .where(
            (CustomDatasetRow.scenes_file_id == file_id)
            | (CustomDatasetRow.annotation_file_id == file_id)
        )
        .limit(1)
    )
    return custom_reference is not None


def _is_owned_stored_file(row: StoredFileRow, config: TrainingUIAPIConfig) -> bool:
    try:
        Path(row.path).resolve().relative_to(config.stored_files_root.resolve())
    except ValueError:
        return False
    return True


def _validate_upload_name(name: str, suffix: str) -> None:
    if Path(name).suffix.lower() != suffix:
        raise TrainingUIAPIError(f"Ожидался файл {suffix}: {name}")


def _next_queue_position(
    session: Session,
    queue_name: JobType,
    source: JobSource = JobSource.MANUAL,
) -> int:
    return next_queue_position(session, queue_name, source)


def _queue_jobs(session: Session, queue_name: JobType | None = None) -> list[JobSummary]:
    rows = _queue_rows(session, job_type=queue_name)
    return [_job_summary(session, row) for row in rows]


def _queue_rows(
    session: Session,
    *,
    job_type: JobType | None = None,
    manual_only: bool = False,
    queued_only: bool = False,
) -> list[JobRow]:
    conditions = [JobRow.status == JobStatus.QUEUED.value] if queued_only else [JobRow.status.in_(ACTIVE_JOB_STATUSES)]
    if job_type is not None:
        conditions.append(JobRow.type == job_type.value)
    if manual_only:
        conditions.append(JobRow.source == JobSource.MANUAL.value)
    rows = session.scalars(
        select(JobRow)
        .where(*conditions)
        .order_by(JobRow.queue_position, JobRow.created_at)
    ).all()
    rows.sort(key=queue_sort_key)
    return rows


def _ensure_queue_control(session: Session, queue_name: JobType) -> QueueControlRow:
    row = session.get(QueueControlRow, queue_name.value)
    if row is None:
        row = QueueControlRow(queue_name=queue_name.value, enabled=True)
        session.add(row)
        session.flush()
    return row


def _stop_process_and_cleanup(row: JobRow) -> None:
    terminate_job_process(row)
    if row.tmp_path:
        shutil.rmtree(row.tmp_path, ignore_errors=True)
        row.tmp_path = None


def _delete_job_rows(session: Session, row: JobRow) -> None:
    training_results = session.scalars(select(TrainingResultRow).where(TrainingResultRow.job_id == row.id)).all()
    training_result_ids = [item.id for item in training_results]
    pseudo_results = {
        result.id: result
        for result in session.scalars(select(PseudoMarkupResultRow).where(PseudoMarkupResultRow.job_id == row.id)).all()
    }
    if training_result_ids:
        for result in session.scalars(
            select(PseudoMarkupResultRow).where(PseudoMarkupResultRow.training_result_id.in_(training_result_ids))
        ).all():
            pseudo_results[result.id] = result
    for result in pseudo_results.values():
        session.delete(result)
    if pseudo_results:
        session.flush()
    for result in training_results:
        session.delete(result)
    if training_results:
        session.flush()
    session.delete(row)


def _delete_cancelled_results_for_class(session: Session, class_key: str) -> None:
    training_results = session.scalars(
        select(TrainingResultRow).where(
            TrainingResultRow.class_key == class_key,
            TrainingResultRow.status == ResultStatus.CANCELLED.value,
        )
    ).all()
    training_result_ids = [item.id for item in training_results]
    for result in session.scalars(
        select(PseudoMarkupResultRow).where(
            PseudoMarkupResultRow.class_key == class_key,
            PseudoMarkupResultRow.status == ResultStatus.CANCELLED.value,
        )
    ).all():
        session.delete(result)
    if training_result_ids:
        for result in session.scalars(
            select(PseudoMarkupResultRow).where(PseudoMarkupResultRow.training_result_id.in_(training_result_ids))
        ).all():
            session.delete(result)
    for result in training_results:
        session.delete(result)
    session.flush()


def _delete_cancelled_manual_jobs(session: Session) -> None:
    rows = session.scalars(
        select(JobRow).where(
            JobRow.source == JobSource.MANUAL.value,
            JobRow.status == JobStatus.CANCELLED.value,
        )
    ).all()
    for row in rows:
        _delete_job_rows(session, row)
    if rows:
        session.flush()


def _job_mlflow_run_id(session: Session, row: JobRow) -> str | None:
    if row.type != JobType.TRAINING.value:
        return None
    result_run_id = session.scalar(
        select(TrainingResultRow.mlflow_run_id)
        .where(
            TrainingResultRow.job_id == row.id,
            TrainingResultRow.mlflow_run_id.is_not(None),
        )
        .order_by(TrainingResultRow.updated_at.desc())
    )
    return str(result_run_id) if result_run_id else _runtime_mlflow_run_id(row)


def _mark_mlflow_run_killed(run_id: str) -> None:
    try:
        config = get_config()
        mark_run_killed(config.mlflow_tracking_uri, run_id)
    except Exception:
        return


def _template_info(row: TrainingTemplateRow) -> TrainingTemplate:
    return TrainingTemplate(
        id=row.id,
        architecture=row.architecture,
        dataset_key=row.dataset_key,
        dataset_name=row.dataset_name,
        parent_template_id=row.parent_template_id,
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


def _dataset_inference_template_row(
    session: Session,
    architecture: str,
    dataset_key: str,
) -> InferenceTemplateRow | None:
    return session.scalar(
        select(InferenceTemplateRow).where(
            InferenceTemplateRow.architecture == architecture,
            InferenceTemplateRow.dataset_key == dataset_key,
        )
    )


def _inference_template_info(row: InferenceTemplateRow) -> InferenceTemplate:
    return InferenceTemplate(
        id=row.id,
        architecture=row.architecture,
        dataset_key=row.dataset_key,
        dataset_name=row.dataset_name,
        parent_template_id=row.parent_template_id,
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


def _job_summary(session: Session, row: JobRow) -> JobSummary:
    return JobSummary(
        id=row.id,
        type=JobType(row.type),
        source=JobSource(row.source),
        status=JobStatus(row.status),
        queue_position=row.queue_position,
        dataset_key=row.dataset_key,
        dataset_version=row.dataset_version,
        dataset_name=row.dataset_name,
        training_dataset_name=row.training_dataset_name,
        inference_dataset_name=row.inference_dataset_name,
        model_name=row.model_name,
        architecture=row.architecture,
        tile_size=row.tile_size,
        created_at=row.created_at,
        started_at=row.started_at,
        progress=_job_progress(session, row),
        actions=_job_actions(row),
    )


def _job_detail(session: Session, row: JobRow) -> JobDetail:
    return JobDetail(
        id=row.id,
        type=JobType(row.type),
        source=JobSource(row.source),
        status=JobStatus(row.status),
        queue_position=row.queue_position,
        dataset_key=row.dataset_key,
        dataset_version=row.dataset_version,
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
        progress=_job_progress(session, row),
    )


def _training_result_info(session: Session, row: TrainingResultRow) -> TrainingResultInfo:
    pseudo_rows = session.scalars(
        select(PseudoMarkupResultRow)
        .where(PseudoMarkupResultRow.training_result_id == row.id)
        .order_by(PseudoMarkupResultRow.created_at.desc())
    ).all()
    return TrainingResultInfo(
        id=row.id,
        source=JobSource(row.source),
        dataset_key=row.dataset_key,
        dataset_version=row.dataset_version,
        model_name=row.model_name,
        architecture=row.architecture,
        f1_score=row.f1_score,
        epoch=row.epoch,
        trained_at=row.trained_at,
        mlflow_run_url=row.mlflow_run_url,
        status=ResultStatus(row.status),
        progress=_training_result_progress(session, row),
        pseudo_markup_results=[_pseudo_markup_info(session, item) for item in pseudo_rows],
    )


def _pseudo_markup_info(session: Session, row: PseudoMarkupResultRow) -> PseudoMarkupResultInfo:
    return PseudoMarkupResultInfo(
        id=row.id,
        source=JobSource(row.source),
        dataset_key=row.dataset_key,
        dataset_version=row.dataset_version,
        source_dataset_name=row.source_dataset_name,
        scenes_file=_stored_file_info(row.scenes_file),
        geojson_file=_stored_file_info(row.geojson_file),
        image_count=_pseudo_markup_image_count(row),
        status=ResultStatus(row.status),
        created_at=row.created_at,
        progress=_pseudo_result_progress(session, row),
    )


def _pseudo_markup_image_count(row: PseudoMarkupResultRow) -> int | None:
    if row.scenes_file is None:
        return None
    return count_scenes_file_images(Path(row.scenes_file.path), get_config().images_root)


def _job_progress(session: Session, row: JobRow) -> RuntimeProgress | None:
    if row.status != JobStatus.RUNNING.value:
        return None
    if row.type == JobType.TRAINING.value:
        return _training_job_progress(session, row)
    if row.type == JobType.INFERENCE.value:
        return _pseudo_job_progress(row)
    return None


def _training_job_progress(session: Session, row: JobRow) -> RuntimeProgress:
    run_id = _job_mlflow_run_id(session, row)
    return RuntimeProgress(
        current=_completed_training_epochs(run_id),
        total=_int_or_none((row.config or {}).get("train.epochs")),
        elapsed_minutes=_elapsed_minutes(row.started_at),
    )


def _training_result_progress(session: Session, row: TrainingResultRow) -> RuntimeProgress | None:
    if row.status != ResultStatus.RUNNING.value:
        return None
    job = session.get(JobRow, row.job_id) if row.job_id is not None else None
    run_id = row.mlflow_run_id or (_job_mlflow_run_id(session, job) if job is not None else None)
    return RuntimeProgress(
        current=_completed_training_epochs(run_id),
        total=_int_or_none((job.config or {}).get("train.epochs")) if job is not None else None,
        elapsed_minutes=_elapsed_minutes(job.started_at) if job is not None else None,
    )


def _completed_training_epochs(run_id: str | None) -> int:
    if not run_id:
        return 0
    try:
        progress = get_training_epoch_progress(get_config().mlflow_tracking_uri, run_id)
    except Exception:
        return 0
    return max(0, int(progress.completed_epochs))


def _pseudo_result_progress(session: Session, row: PseudoMarkupResultRow) -> RuntimeProgress | None:
    if row.status != ResultStatus.RUNNING.value or row.job_id is None:
        return None
    job = session.get(JobRow, row.job_id)
    if job is None:
        return None
    return _pseudo_job_progress(job)


def _pseudo_job_progress(row: JobRow) -> RuntimeProgress | None:
    payload = _pseudo_progress_payload(row)
    if payload is None:
        return None
    current = _int_or_none(payload.get("current"))
    total = _int_or_none(payload.get("total"))
    if current is None and total is None:
        return None
    if current is not None and total is not None:
        current = min(current, total)
    return RuntimeProgress(
        current=current,
        total=total,
        elapsed_minutes=_elapsed_minutes(row.started_at),
    )


def _pseudo_progress_payload(row: JobRow) -> dict[str, Any] | None:
    if row.tmp_path is None:
        return None
    path = Path(row.tmp_path) / "scratch" / "progress.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _runtime_mlflow_run_id(row: JobRow) -> str | None:
    if row.tmp_path is None:
        return None
    run_id_path = Path(row.tmp_path) / "mlflow_run_id"
    if not run_id_path.is_file():
        return None
    try:
        run_id = run_id_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return run_id or None


def _elapsed_minutes(started_at: datetime | None) -> int | None:
    if started_at is None:
        return None
    current = _now()
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    return max(0, int((current - started_at).total_seconds() // 60))


def _job_actions(row: JobRow) -> list[str]:
    if row.source == JobSource.AUTOMATION.value:
        return []
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
