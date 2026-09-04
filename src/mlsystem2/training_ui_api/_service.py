"""Сервисные операции training UI API."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from mlsystem2.mlflow_adapter.api import (
    create_experiment,
    download_run_artifact,
    get_best_training_checkpoint,
    get_finished_run_artifact,
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
from ._dataset_catalog import (
    create_dataset_class as _create_dataset_class,
    create_managed_dataset as _create_managed_dataset,
    dataset_class_row,
    find_managed_dataset,
    list_managed_classes,
    list_managed_datasets,
    managed_dataset_catalog,
    primary_training_result as primary_training_result,
    set_primary_dataset as _set_primary_dataset,
    synchronize_dataset_catalog,
    update_dataset_class as _update_dataset_class,
    update_managed_dataset as _update_managed_dataset,
)
from ._datasets import (
    CUSTOM_KEY,
    CUSTOM_NAME,
    IMAGERY_CHANNELS,
    RASTER_SUFFIXES,
    count_scenes_file_images,
    find_image_folder,
    imagery_images_dir,
    list_image_folders,
    per_image_annotation_files,
    per_image_scene_entries,
)
from ._external_models import (
    ExternalModelError,
    external_model_payload,
    external_result_manifest,
)
from ._model_export import (
    ModelExportArchive,
    build_external_triton_model_export_zip,
    build_triton_model_export_zip,
)
from ._models import (
    CustomDatasetRow,
    DatasetClassRow,
    DatasetRow,
    InferenceTemplateRow,
    JobRow,
    PseudoMarkupResultRow,
    QueueControlRow,
    StoredFileRow,
    TestSampleRow,
    TrainingResultTestMetricRow,
    TrainingResultRow,
    TrainingTemplateRow,
)
from ._processes import terminate_job_process
from ._queueing import (
    POST_TRAINING_INFERENCE_CONFIG_KEY,
    POST_TRAINING_INFERENCE_JOB_IDS_CONFIG_KEY,
    SECONDARY_PRIORITY_CONFIG_KEY,
    STOP_AND_SAVE_BEST_CONFIG_KEY,
    ensure_queue_positions,
    is_secondary_job,
    next_queue_position,
    queue_sort_key,
)
from ._templates import (
    initial_inference_templates,
    initial_templates,
    sanitize_inference_template_config,
    sanitize_template_config,
)
from ._template_selection import (
    dataset_inference_template_row,
    dataset_training_template_row,
    effective_inference_template_row,
    effective_training_template_row,
)
from ._test_samples import (
    TEST_SAMPLE_F1_OPERATION,
    dataset_test_sample_pseudo_markup,
    dataset_test_sample_training_result,
    mark_test_samples_stale_for_pseudo_markup,
    pseudo_markup_covers_dataset,
    primary_test_sample,
    queue_dataset_test_f1_all,
    reconcile_test_sample_evaluations,
    reconcile_training_result_test_f1,
    test_sample_source_pseudo_markup,
    test_sample_source_training_result,
    training_result_test_f1_info,
)
from .contracts import (
    AppLink,
    AppLinksResponse,
    AutomationEnabledUpdate,
    AutomationRuleInfo,
    AutomationRuleUpdate,
    AutomationSnapshot,
    BootstrapInfo,
    ClassListResponse,
    DatasetResultsResponse,
    ConfigSchema,
    CustomDatasetInfo,
    DatasetCatalogInfo,
    DatasetClassCreate,
    DatasetClassUpdate,
    DatasetInfo,
    DatasetListResponse,
    DatasetPrimaryDatasetUpdate,
    ImageFolderListResponse,
    InferenceTemplate,
    InferenceTemplateApplyField,
    InferenceTemplateCreate,
    InferenceTemplateListResponse,
    InferenceTemplateUpdate,
    JobDetail,
    JobLogInfo,
    JobSource,
    JobStatus,
    JobSummary,
    JobType,
    MLflowExperimentCreate,
    MLflowExperimentInfo,
    ModelListResponse,
    ManagedDatasetCreate,
    ManagedDatasetCompositionCreate,
    ManagedDatasetUpdate,
    PseudoMarkupResultInfo,
    PrimaryTestSampleInfo,
    QueueEnabledUpdate,
    QueueCountInfo,
    QueueSnapshot,
    ResultClassInfo,
    ResultClassListResponse,
    ResultDatasetInfo,
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
    TrainingResultBatchExportRequest,
    TrainingResultInfo,
    TrainingTemplate,
    TrainingTemplateListResponse,
    TrainingTemplateUpdate,
    TrainingUIAPIError,
)


ACTIVE_JOB_STATUSES = {
    JobStatus.QUEUED.value,
    JobStatus.RUNNING.value,
    JobStatus.PAUSED.value,
}
JOB_LOG_MAX_BYTES = 128 * 1024
JOB_CONTROL_DIR = "control"
STOP_AND_SAVE_BEST_REQUEST_FILE = "stop-and-save-best.request"


def app_links(config: TrainingUIAPIConfig) -> AppLinksResponse:
    return AppLinksResponse(
        links=[
            AppLink(key="grafana", title="Grafana", url=config.grafana_url),
            AppLink(key="mlflow", title="MLflow", url=config.mlflow_ui_url),
            AppLink(key="images", title="Снимки", url=config.images_ui_url),
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


def datasets(session: Session, config: TrainingUIAPIConfig) -> DatasetListResponse:
    return DatasetListResponse(datasets=list_managed_datasets(session, config))


def classes(session: Session, config: TrainingUIAPIConfig) -> ClassListResponse:
    return ClassListResponse(classes=list_managed_classes(session, config))


def dataset_catalog(session: Session, config: TrainingUIAPIConfig) -> DatasetCatalogInfo:
    return managed_dataset_catalog(session, config)


def sync_dataset_catalog(session: Session, config: TrainingUIAPIConfig) -> DatasetCatalogInfo:
    synchronize_dataset_catalog(session, config)
    return managed_dataset_catalog(session, config)


def create_dataset_class(
    session: Session,
    request: DatasetClassCreate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    return _create_dataset_class(session, request, config)


def update_dataset_class(
    session: Session,
    class_key: str,
    request: DatasetClassUpdate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    return _update_dataset_class(session, class_key, request, config)


def set_primary_dataset(
    session: Session,
    class_key: str,
    request: DatasetPrimaryDatasetUpdate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    return _set_primary_dataset(session, class_key, request, config)


def create_managed_dataset(
    session: Session,
    request: ManagedDatasetCreate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    return _create_managed_dataset(session, request, config)


def create_managed_dataset_composition(
    session: Session,
    request: ManagedDatasetCompositionCreate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    from ._dataset_catalog import create_managed_dataset_composition as create_composition

    return create_composition(session, request, config)


def update_managed_dataset(
    session: Session,
    dataset_key: str,
    request: ManagedDatasetUpdate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    return _update_managed_dataset(session, dataset_key, request, config)


def result_classes(
    session: Session,
    config: TrainingUIAPIConfig,
) -> ResultClassListResponse:
    catalog = list_managed_classes(session, config)
    output: list[ResultClassInfo] = []
    for class_info in catalog:
        selected_results = _result_card_training_results(
            session,
            class_info.key,
            [dataset.key for dataset in class_info.datasets],
        )
        result_datasets: list[ResultDatasetInfo] = []
        for dataset in class_info.datasets:
            test_f1 = None
            test_f1_metrics: dict[str, Any] = {}
            test_f1_status = None
            test_f1_training_result_id = None
            selected_result = selected_results.get(dataset.key)
            if selected_result is not None:
                info = training_result_test_f1_info(session, selected_result, config)
                if info is not None and info.f1 is not None:
                    test_f1 = info.f1
                    test_f1_metrics = dict(info.metrics or {})
                    test_f1_status = "current" if info.status == "current" else "stale"
                    test_f1_training_result_id = selected_result.id
            result_datasets.append(
                ResultDatasetInfo(
                    key=dataset.key,
                    name=dataset.name,
                    dataset_name=dataset.dataset_name,
                    class_key=dataset.class_key,
                    class_name=dataset.class_name,
                    quality_metric=dataset.quality_metric,
                    is_primary=dataset.is_primary,
                    image_count=dataset.image_count,
                    test_f1=test_f1,
                    test_f1_metrics=test_f1_metrics,
                    test_f1_status=test_f1_status,
                    test_f1_training_result_id=test_f1_training_result_id,
                )
            )
        output.append(
            ResultClassInfo(
                key=class_info.key,
                name=class_info.name,
                updated_at=class_info.updated_at,
                datasets=result_datasets,
                is_custom=class_info.is_custom,
                quality_metric=class_info.quality_metric,
            )
        )
    return ResultClassListResponse(classes=output)


def _result_card_training_results(
    session: Session,
    class_key: str,
    dataset_keys: list[str],
) -> dict[str, TrainingResultRow]:
    """Выбрать основную либо последнюю успешную сеть отдельно для каждого датасета."""

    ordered_dataset_keys = list(dict.fromkeys(dataset_keys))
    if not ordered_dataset_keys:
        return {}
    active_dataset_keys = set(ordered_dataset_keys)
    rows = session.scalars(
        select(TrainingResultRow)
        .where(
            (
                TrainingResultRow.dataset_key.in_(ordered_dataset_keys)
                | TrainingResultRow.class_key.in_(ordered_dataset_keys)
            ),
            TrainingResultRow.status == ResultStatus.OK.value,
        )
        .order_by(
            TrainingResultRow.trained_at.desc().nullslast(),
            TrainingResultRow.created_at.desc(),
            TrainingResultRow.id.desc(),
        )
    ).all()
    selected: dict[str, TrainingResultRow] = {}
    for row in rows:
        dataset_key = _training_result_dataset_key(row, active_dataset_keys)
        if dataset_key is not None:
            selected.setdefault(dataset_key, row)

    class_row = dataset_class_row(session, class_key)
    if class_row is None or class_row.primary_training_result_id is None:
        return selected
    primary = session.get(TrainingResultRow, class_row.primary_training_result_id)
    if primary is None or primary.status != ResultStatus.OK.value:
        return selected
    primary_dataset_key = _training_result_dataset_key(primary, active_dataset_keys)
    if primary_dataset_key is not None:
        selected[primary_dataset_key] = primary
    return selected


def _training_result_dataset_key(
    row: TrainingResultRow,
    active_dataset_keys: set[str],
) -> str | None:
    if row.dataset_key in active_dataset_keys:
        return row.dataset_key
    if row.class_key in active_dataset_keys:
        return row.class_key
    return None


def image_folders(config: TrainingUIAPIConfig) -> ImageFolderListResponse:
    return ImageFolderListResponse(folders=list_image_folders(config.images_root))


def models() -> ModelListResponse:
    return ModelListResponse(models=ui_model_infos())


def bootstrap(session: Session, config: TrainingUIAPIConfig) -> BootstrapInfo:
    all_datasets = list_managed_datasets(session, config)
    managed_datasets = [item for item in all_datasets if not item.is_custom]
    return BootstrapInfo(
        links=app_links(config).links,
        datasets=all_datasets,
        image_folders=image_folders(config).folders,
        classes=list_managed_classes(
            session,
            config,
            managed_datasets=managed_datasets,
        ),
        models=models().models,
        training_templates=training_templates(session).templates,
        inference_templates=inference_templates(session).templates,
    )


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
    seed_payloads = [
        _resolve_inference_seed_dataset(session, payload)
        for payload in initial_inference_templates()
    ]
    base_payloads = [payload for payload in seed_payloads if payload.get("dataset_key") is None]
    dataset_payloads = [
        payload for payload in seed_payloads if payload.get("dataset_key") is not None
    ]
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
        _reconcile_seed_inference_config(row, payload)
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
        _reconcile_seed_inference_config(row, payload)


def _reconcile_seed_inference_config(
    row: InferenceTemplateRow,
    payload: dict[str, Any],
) -> None:
    current = dict(row.default_config or {})
    previous_baseline = dict(row.baseline_default_config or {})
    next_baseline = sanitize_inference_template_config(payload["baseline_default_config"])
    reconciled = sanitize_inference_template_config(
        current,
        fallback=next_baseline,
    )
    for key, next_value in next_baseline.items():
        if key not in current or (
            key in previous_baseline and current[key] == previous_baseline[key]
        ):
            reconciled[key] = next_value
    row.default_config = reconciled
    row.baseline_default_config = next_baseline


def _resolve_inference_seed_dataset(
    session: Session,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Привязать именованный seed-шаблон к действующей строке каталога."""

    payload = dict(source)
    dataset_key = payload.get("dataset_key")
    if not isinstance(dataset_key, str) or not dataset_key:
        return payload
    active_key = session.scalar(
        select(DatasetRow.key).where(
            DatasetRow.key == dataset_key,
            DatasetRow.deleted_at.is_(None),
        )
    )
    if active_key is not None:
        return payload
    dataset_name = payload.get("dataset_name")
    display_name = dataset_name if isinstance(dataset_name, str) else dataset_key
    class_name, separator, short_name = display_name.partition("\\")
    if not separator or not class_name or not short_name:
        return payload
    active_key = session.scalar(
        select(DatasetRow.key)
        .join(DatasetClassRow, DatasetClassRow.id == DatasetRow.class_id)
        .where(
            DatasetClassRow.name == class_name,
            DatasetRow.name == short_name,
            DatasetRow.deleted_at.is_(None),
        )
        .limit(1)
    )
    if active_key is not None:
        payload["dataset_key"] = active_key
    return payload


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
    dataset = find_managed_dataset(session, config, request.dataset_key)
    if dataset is None or dataset.is_custom:
        raise TrainingUIAPIError(f"Датасет не найден: {request.dataset_key}")
    existing = dataset_training_template_row(session, request.architecture, dataset.key)
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
        template.default_config = sanitize_template_config(
            current, fallback=template.default_config
        )
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
    config: TrainingUIAPIConfig,
) -> InferenceTemplate:
    ensure_seed_templates(session)
    row = _base_inference_template_row(session, architecture)
    if row is None:
        raise TrainingUIAPIError(f"Шаблон инференса не найден: {architecture}")
    return update_inference_template_by_id(session, row.id, request, config)


def create_inference_template(
    session: Session,
    request: InferenceTemplateCreate,
    config: TrainingUIAPIConfig,
) -> InferenceTemplate:
    ensure_seed_templates(session)
    parent = _base_inference_template_row(session, request.architecture)
    if parent is None:
        raise TrainingUIAPIError(f"Шаблон инференса сети не найден: {request.architecture}")
    dataset = find_managed_dataset(session, config, request.dataset_key)
    if dataset is None or dataset.is_custom:
        raise TrainingUIAPIError(f"Датасет не найден: {request.dataset_key}")
    existing = dataset_inference_template_row(session, request.architecture, dataset.key)
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
    reconcile_test_sample_evaluations(session, config)
    reconcile_training_result_test_f1(session, config)
    return _inference_template_info(row)


def update_inference_template_by_id(
    session: Session,
    template_id: uuid.UUID,
    request: InferenceTemplateUpdate,
    config: TrainingUIAPIConfig,
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
    reconcile_test_sample_evaluations(session, config)
    reconcile_training_result_test_f1(session, config)
    return _inference_template_info(row)


def delete_inference_template(
    session: Session,
    template_id: uuid.UUID,
    config: TrainingUIAPIConfig,
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
    reconcile_test_sample_evaluations(session, config)
    reconcile_training_result_test_f1(session, config)
    return info


def apply_inference_template_field_to_all(
    session: Session,
    template_id: uuid.UUID,
    request: InferenceTemplateApplyField,
    config: TrainingUIAPIConfig,
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
    reconcile_test_sample_evaluations(session, config)
    reconcile_training_result_test_f1(session, config)
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
    template_row = training_template_row_for_dataset(
        session, request.architecture, request.dataset_key
    )
    job_config = sanitize_template_config(
        request.config,
        fallback=template_row.default_config if template_row is not None else None,
        normalize_factors=False,
    )
    job_config["train.quality_metric"] = dataset.quality_metric
    job_config["train.input_channels"] = dataset.input_channels or 4
    job_config["dataset.task"] = dataset.task
    job_config["dataset.object_types"] = [
        item.model_dump(mode="json") for item in dataset.object_types
    ]
    job_config["dataset.imagery_type"] = (
        dataset.imagery_type.value if dataset.imagery_type is not None else "kanopus"
    )
    if dataset.images_dir is not None:
        job_config["dataset.images_dir"] = dataset.images_dir
    job_config[POST_TRAINING_INFERENCE_CONFIG_KEY] = request.run_inference_after_training
    job_config[SECONDARY_PRIORITY_CONFIG_KEY] = request.secondary_priority
    _validate_tile_factor_config(job_config)
    _validate_training_pipeline_variant(job_config, request.architecture)
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
        mlflow_run_name=_blank_to_none(request.mlflow_run_name),
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
            quality_metric=dataset.quality_metric,
            task=dataset.task,
            class_schema=[item.model_dump(mode="json") for item in dataset.object_types],
            status=ResultStatus.RUNNING.value,
            job_id=row.id,
        )
    )
    session.flush()
    return _job_detail(session, row)


def _validate_training_pipeline_variant(
    job_config: dict[str, Any], architecture: str
) -> None:
    variant = str(job_config.get("train.pipeline_variant") or "legacy")
    if variant not in {"legacy", "next_gen"}:
        raise TrainingUIAPIError(f"Неизвестный вариант конвейера обучения: {variant}")
    if architecture == "segformer_b0" and variant != "next_gen":
        raise TrainingUIAPIError("SegFormer B0 HF доступен только в конвейере next-gen.")
    if variant == "legacy":
        return
    if job_config.get("dataset.task") != "binary":
        raise TrainingUIAPIError("next-gen v1 поддерживает только binary-датасеты.")
    if job_config.get("dataset.imagery_type") != "kanopus":
        raise TrainingUIAPIError("next-gen v1 поддерживает только снимки Kanopus.")
    if architecture not in {"smp_segformer_b0", "segformer_b0"}:
        raise TrainingUIAPIError(
            "next-gen v1 поддерживает только smp_segformer_b0 и SegFormer B0 HF."
        )
    if int(job_config.get("train.input_channels") or 0) != 4:
        raise TrainingUIAPIError("next-gen v1 требует четыре входных канала.")
    if job_config.get("train.max_val_batches_per_epoch") is not None:
        raise TrainingUIAPIError(
            "next-gen всегда выполняет полную validation: max_val_batches_per_epoch должен быть пустым."
        )
    if bool(job_config.get("train.pretrained")) and architecture != "segformer_b0":
        raise TrainingUIAPIError(
            "Предобученные веса next-gen доступны только для SegFormer B0 HF."
        )


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


def queue_count(session: Session) -> QueueCountInfo:
    active_jobs = session.scalar(
        select(func.count(JobRow.id)).where(JobRow.status.in_(ACTIVE_JOB_STATUSES))
    )
    return QueueCountInfo(active_jobs=int(active_jobs or 0))


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
            select(JobRow).where(
                JobRow.type == queue_name.value,
                JobRow.status.in_([JobStatus.RUNNING.value, JobStatus.PAUSED.value]),
            )
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


def job_log(
    session: Session,
    job_id: uuid.UUID,
    config: TrainingUIAPIConfig,
) -> JobLogInfo:
    row = session.get(JobRow, job_id)
    if row is None:
        raise TrainingUIAPIError(f"Задание не найдено: {job_id}")
    run_dir = _job_run_dir(row, config)
    if run_dir is not None:
        for path in _job_log_candidates(row, run_dir):
            if not path.is_file():
                continue
            try:
                content, truncated, size_bytes = _read_log_tail(path, JOB_LOG_MAX_BYTES)
            except OSError:
                continue
            if not content.strip():
                continue
            if _requires_journal_fallback(content):
                break
            return JobLogInfo(
                job_id=row.id,
                source_name=path.relative_to(run_dir).as_posix(),
                content=content,
                truncated=truncated,
                size_bytes=size_bytes,
            )
    if row.error:
        content, truncated, size_bytes = _trim_text_bytes(row.error, JOB_LOG_MAX_BYTES)
        return JobLogInfo(
            job_id=row.id,
            source_name="сохранённая ошибка",
            content=content,
            truncated=truncated,
            size_bytes=size_bytes,
        )
    journal = _journal_job_log(row, config)
    if journal is not None:
        return journal
    raise TrainingUIAPIError("Лог задания не найден.")


def delete_job(
    session: Session,
    job_id: uuid.UUID,
    *,
    preserve_cancelled: bool = False,
) -> JobDetail:
    row = session.get(JobRow, job_id)
    if row is None:
        raise TrainingUIAPIError(f"Задание не найдено: {job_id}")
    if row.source == JobSource.AUTOMATION.value:
        raise TrainingUIAPIError(
            "Автоматические задания отменяются только через форму автоматизации"
        )
    detail = _job_detail(session, row).model_copy(update={"status": JobStatus.CANCELLED})
    mlflow_run_id = _job_mlflow_run_id(session, row)
    if row.status in {JobStatus.RUNNING.value, JobStatus.PAUSED.value}:
        _stop_process_and_cleanup(row)
        if mlflow_run_id:
            _mark_mlflow_run_killed(mlflow_run_id)
    if preserve_cancelled:
        row.status = JobStatus.CANCELLED.value
        row.finished_at = _now()
        row.process_pid = None
        session.flush()
        return detail
    _delete_job_rows(session, row)
    session.flush()
    return detail


def stop_training_job_and_save_best(
    session: Session,
    job_id: uuid.UUID,
) -> JobDetail:
    """Запросить штатную остановку ручного обучения с публикацией лучшего чекпойнта."""

    row = session.get(JobRow, job_id)
    if row is None:
        raise TrainingUIAPIError(f"Задание не найдено: {job_id}")
    if row.type != JobType.TRAINING.value:
        raise TrainingUIAPIError("Сохранить лучший чекпойнт можно только при остановке обучения.")
    if row.source == JobSource.AUTOMATION.value:
        raise TrainingUIAPIError(
            "Автоматические задания останавливаются только через форму автоматизации."
        )
    if row.status not in {JobStatus.RUNNING.value, JobStatus.PAUSED.value}:
        raise TrainingUIAPIError("Сохранить чекпойнт можно только у выполняющегося обучения.")
    if _stop_and_save_best_requested(row):
        return _job_detail(session, row)
    if not _best_training_checkpoint_available(row):
        raise TrainingUIAPIError(
            "Лучший чекпойнт ещё не создан. Дождитесь завершения хотя бы одной эпохи "
            "или остановите обучение без сохранения результата."
        )
    if not row.tmp_path:
        raise TrainingUIAPIError("Рабочая папка обучения не найдена.")

    control_dir = Path(row.tmp_path) / JOB_CONTROL_DIR
    control_dir.mkdir(parents=True, exist_ok=True)
    request_path = control_dir / STOP_AND_SAVE_BEST_REQUEST_FILE
    temporary = control_dir / f".{STOP_AND_SAVE_BEST_REQUEST_FILE}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(f"{uuid.uuid4()}\n", encoding="utf-8")
    temporary.replace(request_path)
    row.config = {
        **dict(row.config or {}),
        STOP_AND_SAVE_BEST_CONFIG_KEY: True,
    }
    session.flush()
    return _job_detail(session, row)


def move_job(session: Session, job_id: uuid.UUID, *, direction: int) -> JobDetail:
    row = session.get(JobRow, job_id)
    if row is None:
        raise TrainingUIAPIError(f"Задание не найдено: {job_id}")
    if row.source == JobSource.AUTOMATION.value:
        raise TrainingUIAPIError("Автоматические задания нельзя двигать вручную")
    if row.status != JobStatus.QUEUED.value:
        raise TrainingUIAPIError("Можно двигать только queued задания")
    ensure_queue_positions(session)
    queued = [
        item
        for item in _queue_rows(session, manual_only=True, queued_only=True)
        if is_secondary_job(item) == is_secondary_job(row)
    ]
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


def dataset_results(
    session: Session,
    dataset_key: str,
    config: TrainingUIAPIConfig,
) -> DatasetResultsResponse:
    dataset_info = find_managed_dataset(session, config, dataset_key)
    if dataset_info is None:
        dataset_info = DatasetInfo(key=dataset_key, name=dataset_key, dataset_name=dataset_key)
    _delete_cancelled_manual_jobs(session)
    _delete_cancelled_results_for_class(session, dataset_key)
    rows = session.scalars(
        select(TrainingResultRow)
        .where(
            TrainingResultRow.class_key == dataset_key,
            TrainingResultRow.status != ResultStatus.CANCELLED.value,
        )
        .order_by(TrainingResultRow.created_at.desc())
    ).all()
    pseudo_by_training_id = _class_pseudo_results(session, [row.id for row in rows])
    job_rows = _result_jobs(
        session,
        [
            job_id
            for row in rows
            for job_id in [
                row.job_id,
                *[item.job_id for item in pseudo_by_training_id.get(row.id, [])],
            ]
            if job_id is not None
        ],
    )
    primary = primary_test_sample(session, dataset_key)
    result_infos = [
        _training_result_info(
            session,
            row,
            config=config,
            pseudo_rows=pseudo_by_training_id.get(row.id, []),
            jobs_by_id=job_rows,
        )
        for row in rows
    ]
    primary_test_samples = next(
        (
            list(info.test_f1.samples)
            for info in result_infos
            if info.test_f1 is not None and info.test_f1.samples
        ),
        [],
    )
    if not primary_test_samples and primary is not None:
        primary_test_samples = [
            PrimaryTestSampleInfo(
                id=primary.id,
                name=primary.name,
                content_revision=primary.content_revision,
                enabled_image_count=sum(tile.enabled for tile in primary.tiles),
                enabled_object_count=sum(
                    tile.object_count for tile in primary.tiles if tile.enabled
                ),
                class_key=primary.class_key,
                class_name=primary.class_name,
            )
        ]
    successful_test_statuses = [
        info.test_f1.status
        for row, info in zip(rows, result_infos, strict=True)
        if row.status == ResultStatus.OK.value and info.test_f1 is not None
    ]
    if not primary_test_samples:
        test_f1_status = "unavailable"
    elif (
        successful_test_statuses
        and all(item == "current" for item in successful_test_statuses)
        and len(successful_test_statuses)
        == sum(row.status == ResultStatus.OK.value for row in rows)
    ):
        test_f1_status = "current"
    elif any(item in {"queued", "running"} for item in successful_test_statuses):
        test_f1_status = "running"
    else:
        test_f1_status = "stale"
    return DatasetResultsResponse(
        dataset_key=dataset_info.key,
        dataset_name=dataset_info.name,
        class_key=dataset_info.class_key,
        class_name=dataset_info.class_name,
        quality_metric=dataset_info.quality_metric,
        dataset_updated_at=dataset_info.updated_at,
        primary_test_sample=(
            PrimaryTestSampleInfo(
                id=primary.id,
                name=primary.name,
                content_revision=primary.content_revision,
                enabled_image_count=sum(tile.enabled for tile in primary.tiles),
                enabled_object_count=sum(
                    tile.object_count for tile in primary.tiles if tile.enabled
                ),
                class_key=primary.class_key,
                class_name=primary.class_name,
            )
            if primary is not None
            else None
        ),
        primary_test_samples=primary_test_samples,
        test_f1_status=test_f1_status,
        results=result_infos,
    )


def recalculate_dataset_test_f1(
    session: Session,
    dataset_key: str,
    config: TrainingUIAPIConfig,
) -> DatasetResultsResponse:
    queue_dataset_test_f1_all(session, dataset_key, config)
    session.flush()
    return dataset_results(session, dataset_key, config)


def set_primary_training_result(
    session: Session,
    result_id: uuid.UUID,
    config: TrainingUIAPIConfig,
) -> TrainingResultInfo:
    """Назначить успешный результат основной сетью его класса."""

    row = session.get(TrainingResultRow, result_id)
    if row is None:
        raise TrainingUIAPIError(f"Результат обучения не найден: {result_id}")
    if row.status != ResultStatus.OK.value:
        raise TrainingUIAPIError("Основной можно назначить только успешно обученную сеть.")
    class_row = dataset_class_row(session, row.dataset_key or row.class_key)
    if class_row is None:
        synchronize_dataset_catalog(session, config)
        class_row = dataset_class_row(session, row.dataset_key or row.class_key)
    if class_row is None:
        raise TrainingUIAPIError(f"Класс результата не найден: {row.class_key}")
    previous_effective = primary_training_result(session, class_row.key)
    class_row.primary_training_result_id = row.id
    class_row.updated_at = datetime.now(timezone.utc)
    session.flush()
    if previous_effective is None or previous_effective.id != row.id:
        reconcile_test_sample_evaluations(
            session,
            config,
            class_keys={class_row.key},
        )
    return _training_result_info(session, row, config=config)


def clear_primary_training_result(
    session: Session,
    result_id: uuid.UUID,
    config: TrainingUIAPIConfig,
) -> TrainingResultInfo:
    """Снять явную отметку основной сети и включить выбор последней успешной."""

    row = session.get(TrainingResultRow, result_id)
    if row is None:
        raise TrainingUIAPIError(f"Результат обучения не найден: {result_id}")
    class_row = dataset_class_row(session, row.dataset_key or row.class_key)
    if class_row is None:
        raise TrainingUIAPIError(f"Класс результата не найден: {row.class_key}")
    if class_row.primary_training_result_id != row.id:
        raise TrainingUIAPIError("Эта сеть не отмечена основной для класса.")

    previous_effective = primary_training_result(session, class_row.key)
    class_row.primary_training_result_id = None
    class_row.updated_at = datetime.now(timezone.utc)
    session.flush()
    next_effective = primary_training_result(session, class_row.key)
    if (previous_effective.id if previous_effective is not None else None) != (
        next_effective.id if next_effective is not None else None
    ):
        reconcile_test_sample_evaluations(
            session,
            config,
            class_keys={class_row.key},
        )
    return _training_result_info(session, row, config=config)


def result_changes(
    session: Session,
    config: TrainingUIAPIConfig,
    limit: int = 20,
) -> ResultChangesResponse:
    ensure_queue_positions(session)
    datasets_by_key = {item.key: item for item in list_managed_datasets(session, config)}
    active_changes = [
        _job_change_info(session, row, datasets_by_key) for row in _queue_rows(session)
    ]
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
        dataset = datasets_by_key.get(row.class_key)
        changes.append(
            ResultChangeInfo(
                id=row.id,
                item_type="training_result",
                job_id=row.job_id,
                type=JobType.TRAINING,
                dataset_key=row.class_key,
                class_key=(dataset.class_key or row.class_key)
                if dataset is not None
                else row.class_key,
                class_name=(
                    dataset.class_name
                    if dataset is not None
                    else row.class_display_name.split("\\", maxsplit=1)[0]
                ),
                dataset_name=row.class_display_name,
                model_name=row.model_name,
                action="обучена сеть",
                source=JobSource(row.source),
                status=ResultStatus(row.status),
                changed_at=row.updated_at or row.trained_at or row.created_at or _now(),
                mlflow_run_url=row.mlflow_run_url,
            )
        )
    for row in pseudo_rows:
        dataset = datasets_by_key.get(row.class_key)
        model_name = (
            row.training_result.model_name if row.training_result is not None else "псевдоразметка"
        )
        changes.append(
            ResultChangeInfo(
                id=row.id,
                item_type="pseudo_markup_result",
                job_id=row.job_id,
                type=JobType.INFERENCE,
                dataset_key=row.class_key,
                class_key=(dataset.class_key or row.class_key)
                if dataset is not None
                else row.class_key,
                class_name=(
                    dataset.class_name
                    if dataset is not None
                    else row.source_dataset_name.split("\\", maxsplit=1)[0]
                ),
                dataset_name=row.source_dataset_name,
                model_name=model_name,
                action="создана разметка",
                source=JobSource(row.source),
                status=ResultStatus(row.status),
                changed_at=row.updated_at or row.created_at or _now(),
            )
        )
    changes.sort(key=lambda item: item.changed_at, reverse=True)
    return ResultChangesResponse(changes=[*active_changes, *changes[:limit]])


def _job_change_info(
    session: Session,
    row: JobRow,
    datasets_by_key: dict[str, DatasetInfo],
) -> ResultChangeInfo:
    job_type = JobType(row.type)
    dataset_key = _job_dataset_key(row)
    dataset = datasets_by_key.get(dataset_key)
    return ResultChangeInfo(
        id=row.id,
        item_type="job",
        job_id=row.id,
        type=job_type,
        dataset_key=dataset_key,
        class_key=(dataset.class_key or dataset_key) if dataset is not None else dataset_key,
        class_name=(
            dataset.class_name
            if dataset is not None
            else (row.training_dataset_name or row.dataset_name).split("\\", maxsplit=1)[0]
        ),
        dataset_name=row.training_dataset_name or row.dataset_name,
        model_name=row.model_name,
        action=_job_change_action(row),
        source=JobSource(row.source),
        status=row.status,
        changed_at=row.started_at or row.created_at or _now(),
        mlflow_run_url=_job_mlflow_run_url(session, row),
    )


def _job_dataset_key(row: JobRow) -> str:
    if row.type == JobType.INFERENCE.value:
        value = (row.config or {}).get("class_key")
        if isinstance(value, str) and value:
            return value
    return row.dataset_key or row.training_dataset_name or row.dataset_name


def _job_purpose(row: JobRow) -> str:
    if row.type == JobType.TRAINING.value:
        return "training"
    if (row.config or {}).get("operation") == TEST_SAMPLE_F1_OPERATION:
        return "test_sample_f1"
    return "pseudo_markup"


def _job_change_action(row: JobRow) -> str:
    running = row.status == JobStatus.RUNNING.value
    if row.type == JobType.TRAINING.value:
        if row.status == JobStatus.PAUSED.value:
            return "обучение приостановлено для более приоритетного задания"
        return "идёт обучение" if running else "запланировано обучение"
    if _job_purpose(row) == "test_sample_f1":
        return "считается тестовый F1" if running else "запланирован тестовый F1"
    if row.status == JobStatus.PAUSED.value:
        return "псевдоразметка приостановлена для более приоритетного задания"
    return "идёт псевдоразметка" if running else "запланирована псевдоразметка"


def _job_mlflow_run_url(session: Session, row: JobRow) -> str | None:
    if row.type != JobType.TRAINING.value:
        return None
    return session.scalar(
        select(TrainingResultRow.mlflow_run_url).where(TrainingResultRow.job_id == row.id)
    )


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
    secondary_priority: bool = False,
) -> JobDetail:
    ensure_seed_templates(session)
    dataset_key = (dataset_key or "").strip() or None
    image_folder_key = (image_folder_key or "").strip().strip("/").replace("\\", "/") or None
    has_uploaded_scenes = scenes_bytes is not None and scenes_name is not None
    source_count = sum(
        1 for value in (has_uploaded_scenes, bool(dataset_key), bool(image_folder_key)) if value
    )
    if source_count != 1:
        if source_count == 0:
            raise TrainingUIAPIError(
                "Выберите датасет, папку снимков или загрузите txt со снимками"
            )
        raise TrainingUIAPIError("Выберите только один источник снимков")
    class_dataset = find_managed_dataset(session, config, class_key)
    class_name = class_dataset.name if class_dataset else class_key
    training_result = _resolve_training_result(session, training_result_id)
    if training_result is not None and training_result.class_key != class_key:
        raise TrainingUIAPIError("Выбранная модель обучена для другого датасета")
    training_job = (
        session.get(JobRow, training_result.job_id)
        if training_result is not None and training_result.job_id is not None
        else None
    )
    input_channels = (
        _job_input_channels(training_job)
        if training_result is not None
        else (class_dataset.input_channels if class_dataset is not None else 4)
    )
    imagery_by_channels = {channels: imagery for imagery, channels in IMAGERY_CHANNELS.items()}
    imagery_type = imagery_by_channels.get(input_channels)
    if imagery_type is None:
        raise TrainingUIAPIError(
            f"Для модели с {input_channels} входными каналами тип снимков не поддерживается"
        )
    inference_template = (
        inference_template_row_for_dataset(
            session,
            training_result.architecture,
            training_result.class_key,
        )
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
    inference_images_root = str(imagery_images_dir(config.images_root, imagery_type))
    inference_annotation_files: list[str] = []
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
        if dataset.input_channels != input_channels:
            raise TrainingUIAPIError(
                "Выбранный датасет несовместим с числом входных каналов модели"
            )
        dataset_name = dataset.name
        inference_dataset_version = dataset.version
        inference_images_root = dataset.images_dir or str(config.images_root)
        if dataset.annotations_dir:
            try:
                scene_entries = per_image_scene_entries(
                    Path(dataset.annotations_dir),
                    Path(inference_images_root),
                )
            except (OSError, ValueError) as exc:
                raise TrainingUIAPIError(str(exc)) from exc
            scenes_row = _store_file(
                session,
                kind=StoredFileKind.SCENES_TXT,
                original_name=f"{dataset.dataset_name or 'dataset'}.txt",
                content_type="text/plain; charset=utf-8",
                content=("\n".join(scene_entries) + "\n").encode("utf-8"),
                config=config,
            )
            scenes_file_id = scenes_row.id
            inference_annotation_files = per_image_annotation_files(Path(dataset.annotations_dir))
        else:
            inference_annotation_files = [
                path
                for path in (dataset.annotation_file, dataset.hard_negative_annotation_file)
                if path
            ]
        if dataset.scenes_file and scenes_file_id is None:
            scenes_row = _store_existing_file(
                session,
                kind=StoredFileKind.SCENES_TXT,
                path=Path(dataset.scenes_file),
            )
            scenes_file_id = scenes_row.id
    elif image_folder_key:
        folder = find_image_folder(config.images_root, image_folder_key, imagery_type)
        if folder is None:
            existing_folder = find_image_folder(config.images_root, image_folder_key)
            if existing_folder is not None:
                raise TrainingUIAPIError("Папка снимков несовместима с выбранной моделью")
            raise TrainingUIAPIError(f"Папка снимков не найдена: {image_folder_key}")
        imagery_root = imagery_images_dir(config.images_root, imagery_type)
        scene_entries = [
            image.relative_to(imagery_root).as_posix()
            for image in sorted(Path(folder.path).iterdir())
            if image.is_file() and image.suffix.casefold() in RASTER_SUFFIXES
        ]
        scenes_row = _store_file(
            session,
            kind=StoredFileKind.SCENES_TXT,
            original_name=f"{Path(image_folder_key).name or 'images'}.txt",
            content_type="text/plain",
            content=("\n".join(scene_entries) + "\n").encode("utf-8"),
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
            "images_root": inference_images_root,
            "annotation_files": inference_annotation_files,
            "imagery_type": imagery_type,
            "input_channels": input_channels,
            "training_result_id": str(training_result_id) if training_result_id else None,
            "inference_template_id": str(inference_template.id)
            if inference_template is not None
            else None,
            "inference_template_config": inference_template_config,
            SECONDARY_PRIORITY_CONFIG_KEY: secondary_priority,
            **_checkpoint_config(session, training_result, config),
        },
    )
    session.add(row)
    session.flush()
    image_count = (
        count_scenes_file_images(
            Path(scenes_row.path),
            Path(inference_images_root),
            annotation_files=inference_annotation_files,
        )
        if scenes_file_id is not None
        else None
    )
    if scenes_file_id is not None and image_count == 0:
        raise TrainingUIAPIError(f"В выбранном источнике не найдены снимки типа «{imagery_type}»")
    session.add(
        PseudoMarkupResultRow(
            source=JobSource.MANUAL.value,
            dataset_key=dataset_key or CUSTOM_KEY,
            dataset_version=inference_dataset_version,
            training_result_id=training_result_id,
            class_key=class_key,
            source_dataset_name=dataset_name,
            image_count=image_count,
            scenes_file_id=scenes_file_id,
            status=ResultStatus.RUNNING.value,
            job_id=row.id,
        )
    )
    session.flush()
    return _job_detail(session, row)


def ensure_test_sample_pseudo_markup_job(
    session: Session,
    sample_id: uuid.UUID,
    config: TrainingUIAPIConfig,
) -> JobDetail:
    """Идемпотентно запустить штатную псевдоразметку источника тестового набора."""

    sample = session.scalar(
        select(TestSampleRow).where(TestSampleRow.id == sample_id).with_for_update()
    )
    if sample is None:
        raise TrainingUIAPIError("Тестовая разметка не найдена")
    primary = test_sample_source_training_result(session, sample)
    if primary is None:
        raise TrainingUIAPIError("Сеть, выбранная при создании разметки, недоступна")
    ready = test_sample_source_pseudo_markup(session, sample, config)
    if ready is not None and ready.job_id is not None:
        ready_job = session.get(JobRow, ready.job_id)
        if ready_job is not None and ready_job.status not in {
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        }:
            return _job_detail(session, ready_job)
    existing_rows = session.scalars(
        select(PseudoMarkupResultRow)
        .where(
            PseudoMarkupResultRow.dataset_key == sample.dataset_key,
            PseudoMarkupResultRow.training_result_id == primary.id,
            PseudoMarkupResultRow.status == "running",
        )
        .order_by(
            PseudoMarkupResultRow.updated_at.desc(),
            PseudoMarkupResultRow.created_at.desc(),
            PseudoMarkupResultRow.id.desc(),
        )
    ).all()
    for existing in existing_rows:
        if existing.job_id is None:
            continue
        job = session.get(JobRow, existing.job_id)
        if (
            job is not None
            and pseudo_markup_covers_dataset(session, existing, sample.dataset_key, config)
            and job.status
            not in {
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value,
            }
        ):
            return _job_detail(session, job)
    return create_pseudo_markup_job(
        session,
        class_key=primary.class_key,
        dataset_key=sample.dataset_key,
        image_folder_key=None,
        training_result_id=primary.id,
        scenes_name=None,
        scenes_content_type=None,
        scenes_bytes=None,
        config=config,
    )


def ensure_test_sample_batch_dataset_pseudo_markup_job(
    session: Session,
    dataset_key: str,
    config: TrainingUIAPIConfig,
) -> JobDetail:
    """Идемпотентно запустить псевдоразметку сети конкретного датасета."""

    dataset = find_managed_dataset(session, config, dataset_key)
    if dataset is None or dataset.is_custom:
        raise TrainingUIAPIError(f"Датасет не найден: {dataset_key}")
    training_result = dataset_test_sample_training_result(session, dataset.key)
    if training_result is None:
        raise TrainingUIAPIError("Для датасета нет успешной обученной сети")
    ready = dataset_test_sample_pseudo_markup(
        session,
        dataset.key,
        training_result.id,
        config=config,
    )
    if ready is not None:
        if ready.job_id is not None:
            job = session.get(JobRow, ready.job_id)
            if job is not None:
                return _job_detail(session, job)
        raise TrainingUIAPIError("Псевдоразметка этой пары датасет-сеть уже готова")

    existing_rows = session.scalars(
        select(PseudoMarkupResultRow)
        .where(
            PseudoMarkupResultRow.dataset_key == dataset.key,
            PseudoMarkupResultRow.training_result_id == training_result.id,
            PseudoMarkupResultRow.status == ResultStatus.RUNNING.value,
        )
        .order_by(
            PseudoMarkupResultRow.updated_at.desc(),
            PseudoMarkupResultRow.created_at.desc(),
            PseudoMarkupResultRow.id.desc(),
        )
    ).all()
    for existing in existing_rows:
        job = session.get(JobRow, existing.job_id) if existing.job_id is not None else None
        if job is not None and job.status not in {
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        }:
            return _job_detail(session, job)
    return create_pseudo_markup_job(
        session,
        class_key=training_result.class_key,
        dataset_key=dataset.key,
        image_folder_key=None,
        training_result_id=training_result.id,
        scenes_name=None,
        scenes_content_type=None,
        scenes_bytes=None,
        config=config,
    )


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
    if job is not None and job.status in {
        JobStatus.RUNNING.value,
        JobStatus.PAUSED.value,
    }:
        _stop_process_and_cleanup(job)
    mark_test_samples_stale_for_pseudo_markup(session, row.id)
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


def export_training_result_triton_zip(
    session: Session,
    *,
    result_id: uuid.UUID,
    model_name: str,
    sample_size: int | None,
    config: TrainingUIAPIConfig,
    context: int | None = None,
) -> ModelExportArchive:
    ensure_seed_templates(session)
    row = _training_result_row_for_export(session, result_id)
    return _build_training_result_export_archive(
        session,
        row,
        model_name=model_name,
        sample_size=sample_size,
        context=context,
        config=config,
    )


def export_training_results_triton_zip(
    session: Session,
    *,
    request: TrainingResultBatchExportRequest,
    config: TrainingUIAPIConfig,
) -> ModelExportArchive:
    if not request.items:
        raise TrainingUIAPIError("Выберите хотя бы одну модель для экспорта.")
    ensure_seed_templates(session)
    _validate_batch_export_uniqueness(request)

    temp_root = Path(tempfile.mkdtemp(prefix="mlsystem2-results-export-"))
    try:
        export_root = temp_root / "export"
        service_zip_dir = export_root / "models-serving-service"
        pipeline_dir = export_root / "pipelines"
        metadata_dir = export_root / "metadata"
        service_zip_dir.mkdir(parents=True)
        pipeline_dir.mkdir(parents=True)
        metadata_dir.mkdir(parents=True)

        exported_models: list[dict[str, object]] = []
        for item in request.items:
            row = _training_result_row_for_export(session, item.result_id)
            model_name = item.model_name.strip()
            archive = _build_training_result_export_archive(
                session,
                row,
                model_name=model_name,
                sample_size=item.sample_size,
                context=item.context,
                config=config,
            )
            try:
                _copy_model_export_files(
                    archive,
                    model_name=model_name,
                    service_zip_dir=service_zip_dir,
                    pipeline_dir=pipeline_dir,
                    metadata_dir=metadata_dir,
                )
            finally:
                archive.cleanup()
            exported_models.append(
                {
                    "result_id": str(row.id),
                    "model_name": model_name,
                    "class_key": row.class_key,
                    "dataset_key": row.dataset_key,
                    "trained_at": row.trained_at.isoformat()
                    if row.trained_at is not None
                    else None,
                    "created_at": row.created_at.isoformat()
                    if row.created_at is not None
                    else None,
                    "model_archive": f"models-serving-service/{model_name}.zip",
                    "pipeline": f"pipelines/{model_name}_triton.yaml",
                    "metadata": f"metadata/{model_name}_export_metadata.json",
                }
            )

        _write_export_json(
            export_root / "export_metadata.json",
            {
                "format": "mlsystem2_batch_triton_export",
                "models": exported_models,
            },
        )
        zip_path = temp_root / "models_export.zip"
        _zip_directory(export_root, zip_path)
        return ModelExportArchive(
            zip_path=zip_path,
            filename="models_export.zip",
            cleanup_root=temp_root,
        )
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def _validate_batch_export_uniqueness(request: TrainingResultBatchExportRequest) -> None:
    result_ids: set[uuid.UUID] = set()
    model_names: set[str] = set()
    for item in request.items:
        if item.result_id in result_ids:
            raise TrainingUIAPIError("Один результат обучения нельзя добавить в экспорт дважды.")
        result_ids.add(item.result_id)
        model_name = item.model_name.strip()
        if model_name in model_names:
            raise TrainingUIAPIError("Имена моделей в общем архиве должны быть уникальными.")
        model_names.add(model_name)


def _training_result_row_for_export(session: Session, result_id: uuid.UUID) -> TrainingResultRow:
    row = session.get(TrainingResultRow, result_id)
    if row is None:
        raise TrainingUIAPIError(f"Результат обучения не найден: {result_id}")
    if row.status != ResultStatus.OK.value:
        raise TrainingUIAPIError("Экспорт доступен только для успешного результата обучения.")
    if not row.mlflow_run_id:
        raise TrainingUIAPIError("У результата обучения нет MLflow run id для скачивания модели.")
    return row


def _build_training_result_export_archive(
    session: Session,
    row: TrainingResultRow,
    *,
    model_name: str,
    sample_size: int | None,
    config: TrainingUIAPIConfig,
    context: int | None = None,
) -> ModelExportArchive:
    try:
        external_manifest = external_result_manifest(session, row)
    except ExternalModelError as exc:
        raise TrainingUIAPIError(str(exc)) from exc
    artifact_path = (
        external_manifest.artifact_path if external_manifest is not None else "checkpoints/best.pt"
    )
    with tempfile.TemporaryDirectory(prefix="mlsystem2-result-export-") as temp_dir:
        try:
            downloaded = download_run_artifact(
                tracking_uri=config.mlflow_tracking_uri,
                run_id=row.mlflow_run_id,
                artifact_path=artifact_path,
                dst_dir=temp_dir,
            )
        except MLflowAdapterError as exc:
            raise TrainingUIAPIError(f"Не удалось скачать {artifact_path} из MLflow.") from exc

        checkpoint_path = Path(downloaded.local_path)
        if not checkpoint_path.is_file():
            raise TrainingUIAPIError(f"MLflow не вернул файл {artifact_path}.")
        if external_manifest is not None:
            return build_external_triton_model_export_zip(
                model_name=model_name,
                source_archive=checkpoint_path,
                manifest=external_manifest,
            )
        try:
            checkpoint_bytes = checkpoint_path.read_bytes()
        except OSError as exc:
            raise TrainingUIAPIError("Не удалось прочитать скачанный best.pt.") from exc

        inference_template = effective_inference_template_row(
            session,
            row.architecture,
            row.class_key,
        )
        postprocess_config = sanitize_inference_template_config(
            inference_template.default_config if inference_template is not None else None
        )

        return build_triton_model_export_zip(
            model_name=model_name,
            checkpoint_filename=checkpoint_path.name or "best.pt",
            checkpoint_bytes=checkpoint_bytes,
            sample_size=sample_size,
            context=context,
            postprocess_config=postprocess_config,
            class_schema_override=list(row.class_schema or []),
        )


def _copy_model_export_files(
    archive: ModelExportArchive,
    *,
    model_name: str,
    service_zip_dir: Path,
    pipeline_dir: Path,
    metadata_dir: Path,
) -> None:
    expected_files = {
        f"models-serving-service/{model_name}.zip": service_zip_dir / f"{model_name}.zip",
        f"pipelines/{model_name}_triton.yaml": pipeline_dir / f"{model_name}_triton.yaml",
        "export_metadata.json": metadata_dir / f"{model_name}_export_metadata.json",
    }
    with zipfile.ZipFile(archive.zip_path) as zip_file:
        names = set(zip_file.namelist())
        missing = [name for name in expected_files if name not in names]
        if missing:
            raise TrainingUIAPIError("Собранный архив модели не содержит ожидаемые файлы экспорта.")
        for source_name, target_path in expected_files.items():
            target_path.write_bytes(zip_file.read(source_name))


def _write_export_json(path: Path, content: dict[str, object]) -> None:
    path.write_text(
        json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _zip_directory(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir).as_posix())


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
    session: Session,
    row: TrainingResultRow | None,
    config: TrainingUIAPIConfig,
) -> dict[str, object]:
    if row is None or row.mlflow_run_id is None:
        return {}
    try:
        external_manifest = external_result_manifest(session, row)
    except ExternalModelError as exc:
        raise TrainingUIAPIError(str(exc)) from exc
    if external_manifest is not None:
        payload: dict[str, object] = {
            "mlflow_run_id": row.mlflow_run_id,
            "checkpoint_artifact_path": external_manifest.artifact_path,
            "checkpoint_threshold": external_manifest.score_threshold,
            "external_model": external_model_payload(external_manifest),
        }
        try:
            artifact = get_finished_run_artifact(
                config.mlflow_tracking_uri,
                row.mlflow_run_id,
                external_manifest.artifact_path,
            )
        except MLflowAdapterError:
            artifact = None
        if artifact is not None and artifact.artifact_uri is not None:
            payload["checkpoint_uri"] = artifact.artifact_uri
        return payload
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
    dataset = find_managed_dataset(session, config, dataset_key)
    if dataset is not None:
        ready = (dataset.scenes_file is not None and dataset.annotation_file is not None) or (
            dataset.annotations_dir is not None and (dataset.image_count or 0) > 0
        )
        if not dataset.source_available or dataset.images_dir is None:
            raise TrainingUIAPIError("Датасет недоступен: " + "; ".join(dataset.diagnostics))
        if not ready and not dataset.managed:
            raise TrainingUIAPIError("Датасет недоступен: " + "; ".join(dataset.diagnostics))
        return dataset
    raise TrainingUIAPIError(f"Датасет не найден: {dataset_key}")


def training_template_row_for_dataset(
    session: Session,
    architecture: str,
    dataset_key: str | None,
) -> TrainingTemplateRow | None:
    return effective_training_template_row(
        session,
        architecture,
        None if dataset_key == CUSTOM_KEY else dataset_key,
    )


def inference_template_row_for_dataset(
    session: Session,
    architecture: str,
    dataset_key: str | None,
) -> InferenceTemplateRow | None:
    return effective_inference_template_row(session, architecture, dataset_key)


def _base_template_row(session: Session, architecture: str) -> TrainingTemplateRow | None:
    return session.scalar(
        select(TrainingTemplateRow).where(
            TrainingTemplateRow.architecture == architecture,
            TrainingTemplateRow.dataset_key.is_(None),
        )
    )


def _base_inference_template_row(
    session: Session, architecture: str
) -> InferenceTemplateRow | None:
    return session.scalar(
        select(InferenceTemplateRow).where(
            InferenceTemplateRow.architecture == architecture,
            InferenceTemplateRow.dataset_key.is_(None),
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
    conditions = (
        [JobRow.status == JobStatus.QUEUED.value]
        if queued_only
        else [JobRow.status.in_(ACTIVE_JOB_STATUSES)]
    )
    if job_type is not None:
        conditions.append(JobRow.type == job_type.value)
    if manual_only:
        conditions.append(JobRow.source == JobSource.MANUAL.value)
    rows = session.scalars(
        select(JobRow).where(*conditions).order_by(JobRow.queue_position, JobRow.created_at)
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
    if _job_purpose(row) == "test_sample_f1":
        metric = session.scalar(
            select(TrainingResultTestMetricRow).where(TrainingResultTestMetricRow.job_id == row.id)
        )
        if metric is not None:
            metric.status = "stale" if metric.f1 is not None else "unavailable"
            metric.error = "Расчёт тестового F1 отменён."
            metric.job_id = None
            metric.updated_at = _now()
        session.delete(row)
        return
    training_results = session.scalars(
        select(TrainingResultRow).where(TrainingResultRow.job_id == row.id)
    ).all()
    training_result_ids = [item.id for item in training_results]
    pseudo_results = {
        result.id: result
        for result in session.scalars(
            select(PseudoMarkupResultRow).where(PseudoMarkupResultRow.job_id == row.id)
        ).all()
    }
    if training_result_ids:
        for result in session.scalars(
            select(PseudoMarkupResultRow).where(
                PseudoMarkupResultRow.training_result_id.in_(training_result_ids)
            )
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
            select(PseudoMarkupResultRow).where(
                PseudoMarkupResultRow.training_result_id.in_(training_result_ids)
            )
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
        original_name=stored_file_download_name(row),
        size_bytes=row.size_bytes,
        object_count=row.object_count,
        created_at=row.created_at,
        download_url=f"/api/v1/files/{row.id}/download",
    )


def stored_file_download_name(row: StoredFileRow) -> str:
    if row.kind != StoredFileKind.PSEUDO_MARKUP_GEOJSON.value:
        return row.original_name
    normalized = re.sub(r"[\\/]+", "_", row.original_name.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or row.original_name


def _job_summary(session: Session, row: JobRow) -> JobSummary:
    return JobSummary(
        id=row.id,
        type=JobType(row.type),
        purpose=_job_purpose(row),
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
        pipeline_variant=_job_pipeline_variant(row),
        validation_fold=_job_validation_fold(row),
        tile_size=row.tile_size,
        created_at=row.created_at,
        started_at=row.started_at,
        progress=_job_progress(session, row),
        actions=_job_actions(row),
        secondary_priority=is_secondary_job(row),
        best_checkpoint_available=_best_training_checkpoint_available(row),
        stop_and_save_best_requested=_stop_and_save_best_requested(row),
    )


def _job_input_channels(job: JobRow | None) -> int:
    if job is None:
        return 4
    value = (job.config or {}).get("train.input_channels")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 4
    return parsed if parsed > 0 else 4


def _job_pipeline_variant(job: JobRow | None) -> str:
    value = (job.config or {}).get("train.pipeline_variant") if job is not None else None
    return "next_gen" if value == "next_gen" else "legacy"


def _job_validation_fold(job: JobRow | None) -> int:
    value = (job.config or {}).get("next_gen.validation_fold") if job is not None else None
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _job_detail(session: Session, row: JobRow) -> JobDetail:
    return JobDetail(
        id=row.id,
        type=JobType(row.type),
        purpose=_job_purpose(row),
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
        pipeline_variant=_job_pipeline_variant(row),
        validation_fold=_job_validation_fold(row),
        tile_size=row.tile_size,
        mlflow_experiment_name=row.mlflow_experiment_name,
        mlflow_run_name=row.mlflow_run_name,
        config={
            key: value
            for key, value in (row.config or {}).items()
            if key
            not in {
                POST_TRAINING_INFERENCE_CONFIG_KEY,
                POST_TRAINING_INFERENCE_JOB_IDS_CONFIG_KEY,
                SECONDARY_PRIORITY_CONFIG_KEY,
                STOP_AND_SAVE_BEST_CONFIG_KEY,
            }
        },
        run_inference_after_training=bool(
            (row.config or {}).get(POST_TRAINING_INFERENCE_CONFIG_KEY, False)
        ),
        secondary_priority=is_secondary_job(row),
        best_checkpoint_available=_best_training_checkpoint_available(row),
        stop_and_save_best_requested=_stop_and_save_best_requested(row),
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        progress=_job_progress(session, row),
    )


def _job_run_dir(row: JobRow, config: TrainingUIAPIConfig) -> Path | None:
    candidates = []
    if row.tmp_path:
        candidates.append(Path(row.tmp_path))
    candidates.append(config.scratch_root / "jobs" / str(row.id))

    allowed_root = (config.scratch_root / "jobs").resolve()
    for path in candidates:
        try:
            resolved = path.resolve()
            resolved.relative_to(allowed_root)
        except (OSError, ValueError):
            continue
        if resolved.exists():
            return resolved
    return None


def _job_log_candidates(row: JobRow, run_dir: Path) -> list[Path]:
    if row.type == JobType.INFERENCE.value:
        if _job_purpose(row) == "test_sample_f1":
            return [
                run_dir / "worker_error.txt",
                run_dir / "logs" / "test_sample_f1.log",
                run_dir / "scratch" / "report.json",
            ]
        return [
            run_dir / "worker_error.txt",
            run_dir / "logs" / "pseudo_markup.log",
            run_dir / "scratch" / "report.json",
        ]
    return [
        run_dir / "worker_error.txt",
        run_dir / "train.log",
        run_dir / "logs" / "train.log",
    ]


def _requires_journal_fallback(content: str) -> bool:
    text = content.strip().lower()
    if not text:
        return True
    return "journalctl" in text


def _journal_job_log(row: JobRow, config: TrainingUIAPIConfig) -> JobLogInfo | None:
    if not config.journal_unit:
        return None
    since = _journal_timestamp(row.created_at - timedelta(minutes=2))
    until = _journal_timestamp((row.finished_at or _now()) + timedelta(minutes=2))
    try:
        completed = subprocess.run(
            [
                "journalctl",
                "-u",
                config.journal_unit,
                "--since",
                since,
                "--until",
                until,
                "--no-pager",
                "--output",
                "short-iso",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    content = _extract_journal_job_block(completed.stdout, row.id)
    if content is None:
        return None
    content, truncated, size_bytes = _trim_text_bytes(content, JOB_LOG_MAX_BYTES)
    return JobLogInfo(
        job_id=row.id,
        source_name=f"journalctl:{config.journal_unit}",
        content=content,
        truncated=truncated,
        size_bytes=size_bytes,
    )


def _journal_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _extract_journal_job_block(output: str, job_id: uuid.UUID) -> str | None:
    lines = output.splitlines()
    start = next((index for index, line in enumerate(lines) if str(job_id) in line), None)
    if start is None:
        return None
    collected: list[str] = []
    for line in lines[start:]:
        if collected and _journal_line_starts_unrelated_request(line):
            break
        collected.append(line)
    content = "\n".join(collected).strip()
    return content or None


def _journal_line_starts_unrelated_request(line: str) -> bool:
    message = line.rsplit("]: ", 1)[-1]
    return message.startswith("INFO:") or message.startswith("Started ")


def _trim_text_bytes(content: str, max_bytes: int) -> tuple[str, bool, int]:
    data = content.encode("utf-8")
    size_bytes = len(data)
    if size_bytes <= max_bytes:
        return content, False, size_bytes
    trimmed = data[-max_bytes:].decode("utf-8", errors="replace")
    return trimmed, True, size_bytes


def _read_log_tail(path: Path, max_bytes: int) -> tuple[str, bool, int]:
    size_bytes = path.stat().st_size
    truncated = size_bytes > max_bytes
    with path.open("rb") as stream:
        if truncated:
            stream.seek(-max_bytes, 2)
        data = stream.read()
    return data.decode("utf-8", errors="replace"), truncated, size_bytes


def _class_pseudo_results(
    session: Session,
    training_result_ids: list[uuid.UUID],
) -> dict[uuid.UUID, list[PseudoMarkupResultRow]]:
    if not training_result_ids:
        return {}
    rows = session.scalars(
        select(PseudoMarkupResultRow)
        .where(PseudoMarkupResultRow.training_result_id.in_(training_result_ids))
        .options(
            selectinload(PseudoMarkupResultRow.scenes_file),
            selectinload(PseudoMarkupResultRow.geojson_file),
        )
        .order_by(PseudoMarkupResultRow.created_at.desc())
    ).all()
    results: dict[uuid.UUID, list[PseudoMarkupResultRow]] = {}
    for row in rows:
        if row.training_result_id is not None:
            results.setdefault(row.training_result_id, []).append(row)
    return results


def _result_jobs(session: Session, job_ids: list[uuid.UUID]) -> dict[uuid.UUID, JobRow]:
    unique_ids = sorted(set(job_ids), key=str)
    if not unique_ids:
        return {}
    return {
        row.id: row
        for row in session.scalars(select(JobRow).where(JobRow.id.in_(unique_ids))).all()
    }


def _job_from_map(
    session: Session,
    job_id: uuid.UUID | None,
    jobs_by_id: dict[uuid.UUID, JobRow] | None,
) -> JobRow | None:
    if job_id is None:
        return None
    if jobs_by_id is not None:
        return jobs_by_id.get(job_id)
    return session.get(JobRow, job_id)


def _training_result_info(
    session: Session,
    row: TrainingResultRow,
    *,
    config: TrainingUIAPIConfig,
    pseudo_rows: list[PseudoMarkupResultRow] | None = None,
    jobs_by_id: dict[uuid.UUID, JobRow] | None = None,
) -> TrainingResultInfo:
    if pseudo_rows is None:
        pseudo_rows = session.scalars(
            select(PseudoMarkupResultRow)
            .where(PseudoMarkupResultRow.training_result_id == row.id)
            .options(
                selectinload(PseudoMarkupResultRow.scenes_file),
                selectinload(PseudoMarkupResultRow.geojson_file),
            )
            .order_by(PseudoMarkupResultRow.created_at.desc())
        ).all()
    job = _job_from_map(session, row.job_id, jobs_by_id)
    is_primary = _is_primary_training_result(session, row)
    return TrainingResultInfo(
        id=row.id,
        job_id=row.job_id,
        source=JobSource(row.source),
        dataset_key=row.dataset_key,
        dataset_version=row.dataset_version,
        model_name=row.model_name,
        architecture=row.architecture,
        pipeline_variant=_job_pipeline_variant(job),
        validation_fold=_job_validation_fold(job),
        is_primary=is_primary,
        input_channels=_job_input_channels(job),
        quality_metric=row.quality_metric,
        task=row.task,
        class_schema=list(row.class_schema or []),
        training_metrics=dict(row.training_metrics or {}),
        f1_score=row.f1_score,
        epoch=row.epoch,
        trained_at=row.trained_at,
        created_at=row.created_at,
        started_at=job.started_at if job is not None else None,
        mlflow_run_url=row.mlflow_run_url,
        sample_size_hint=job.tile_size if job is not None else None,
        status=_public_result_status(session, row.status, row.job_id, jobs_by_id),
        error=job.error if job is not None else None,
        progress=_training_result_progress(session, row, jobs_by_id),
        test_f1=training_result_test_f1_info(session, row, config),
        pseudo_markup_results=[
            _pseudo_markup_info(session, item, jobs_by_id) for item in pseudo_rows
        ],
    )


def _is_primary_training_result(session: Session, row: TrainingResultRow) -> bool:
    class_row = dataset_class_row(session, row.dataset_key or row.class_key)
    return class_row is not None and class_row.primary_training_result_id == row.id


def _pseudo_markup_info(
    session: Session,
    row: PseudoMarkupResultRow,
    jobs_by_id: dict[uuid.UUID, JobRow] | None = None,
) -> PseudoMarkupResultInfo:
    return PseudoMarkupResultInfo(
        id=row.id,
        job_id=row.job_id,
        source=JobSource(row.source),
        dataset_key=row.dataset_key,
        dataset_version=row.dataset_version,
        source_dataset_name=row.source_dataset_name,
        scenes_file=_stored_file_info(row.scenes_file),
        geojson_file=_stored_file_info(row.geojson_file),
        image_count=row.image_count,
        status=_public_result_status(session, row.status, row.job_id, jobs_by_id),
        created_at=row.created_at,
        runtime_minutes=_job_runtime_minutes(session, row.job_id, jobs_by_id),
        progress=_pseudo_result_progress(session, row, jobs_by_id),
        task=(row.training_result.task if row.training_result is not None else "binary"),
        class_schema=(
            list(row.training_result.class_schema or []) if row.training_result is not None else []
        ),
        by_type_download_url=(
            f"/api/v1/files/{row.geojson_file_id}/download-by-type"
            if row.geojson_file_id is not None
            and row.training_result is not None
            and row.training_result.task == "multiclass"
            else None
        ),
    )


def _public_result_status(
    session: Session,
    result_status: str,
    job_id: uuid.UUID | None,
    jobs_by_id: dict[uuid.UUID, JobRow] | None = None,
) -> str:
    if result_status != ResultStatus.RUNNING.value or job_id is None:
        return result_status
    job = _job_from_map(session, job_id, jobs_by_id)
    if job is not None and job.status in {
        JobStatus.QUEUED.value,
        JobStatus.RUNNING.value,
        JobStatus.PAUSED.value,
    }:
        return job.status
    return result_status


def _job_progress(session: Session, row: JobRow) -> RuntimeProgress | None:
    if row.status not in {JobStatus.RUNNING.value, JobStatus.PAUSED.value}:
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


def _training_result_progress(
    session: Session,
    row: TrainingResultRow,
    jobs_by_id: dict[uuid.UUID, JobRow] | None = None,
) -> RuntimeProgress | None:
    if row.status != ResultStatus.RUNNING.value:
        return None
    job = _job_from_map(session, row.job_id, jobs_by_id)
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


def _pseudo_result_progress(
    session: Session,
    row: PseudoMarkupResultRow,
    jobs_by_id: dict[uuid.UUID, JobRow] | None = None,
) -> RuntimeProgress | None:
    if row.status != ResultStatus.RUNNING.value or row.job_id is None:
        return None
    job = _job_from_map(session, row.job_id, jobs_by_id)
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


def _job_runtime_minutes(
    session: Session,
    job_id: uuid.UUID | None,
    jobs_by_id: dict[uuid.UUID, JobRow] | None = None,
) -> int | None:
    job = _job_from_map(session, job_id, jobs_by_id)
    if job is None or job.started_at is None or job.finished_at is None:
        return None
    started_at = job.started_at
    finished_at = job.finished_at
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    if finished_at.tzinfo is None:
        finished_at = finished_at.replace(tzinfo=timezone.utc)
    return max(0, int((finished_at - started_at).total_seconds() // 60))


def _job_actions(row: JobRow) -> list[str]:
    if row.source == JobSource.AUTOMATION.value:
        return []
    if row.status in {JobStatus.RUNNING.value, JobStatus.PAUSED.value}:
        actions = ["delete"]
        if (
            row.type == JobType.TRAINING.value
            and _best_training_checkpoint_available(row)
            and not _stop_and_save_best_requested(row)
        ):
            actions.insert(0, "stop_and_save_best")
        return actions
    if row.status == JobStatus.QUEUED.value:
        return ["move_up", "move_down", "delete"]
    return []


def _best_training_checkpoint_available(row: JobRow) -> bool:
    if row.type != JobType.TRAINING.value or not row.tmp_path:
        return False
    path = Path(row.tmp_path) / "scratch" / "checkpoints" / "best.pt"
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _stop_and_save_best_requested(row: JobRow) -> bool:
    return bool((row.config or {}).get(STOP_AND_SAVE_BEST_CONFIG_KEY, False))


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_tile_factor_config(config: dict[str, Any]) -> None:
    positive_factor = _float_or_error(config, "tile_preparation.positive_factor")
    hard_negative_factor = _float_or_error(config, "tile_preparation.hard_negative_factor")
    background_factor = _float_or_error(config, "tile_preparation.background_factor")
    for key, value in (
        ("tile_preparation.positive_factor", positive_factor),
        ("tile_preparation.hard_negative_factor", hard_negative_factor),
        ("tile_preparation.background_factor", background_factor),
    ):
        if value < 0.0 or value > 1.0:
            raise TrainingUIAPIError(f"{key} должен быть в диапазоне 0..1")
    if abs(positive_factor + hard_negative_factor + background_factor - 1.0) > 1e-6:
        raise TrainingUIAPIError(
            "Сумма positive_factor, hard_negative_factor и background_factor должна быть равна 1"
        )
    if positive_factor == 0.0 and hard_negative_factor == 0.0 and background_factor == 0.0:
        raise TrainingUIAPIError("Хотя бы один tile factor должен быть больше 0")


def _float_or_error(config: dict[str, Any], key: str) -> float:
    try:
        return float(config[key])
    except KeyError as exc:
        raise TrainingUIAPIError(f"Параметр {key} не задан") from exc
    except (TypeError, ValueError) as exc:
        raise TrainingUIAPIError(f"Параметр {key} должен быть числом") from exc


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None
