"""Автоматизация обучения и псевдоразметки."""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from mlsystem2.mlflow_adapter.api import create_experiment, get_best_training_checkpoint, mark_run_killed
from mlsystem2.mlflow_adapter.contracts import (
    MLflowAdapterError,
    MLflowBestCheckpoint,
    MLflowExperimentRequest,
)

from ._catalog import MODEL_DISPLAY_NAMES, UI_ARCHITECTURES, ui_model_infos
from ._config import TrainingUIAPIConfig
from ._datasets import CUSTOM_KEY, list_datasets
from ._models import (
    AutomationControlRow,
    AutomationRuleRow,
    JobRow,
    PseudoMarkupResultRow,
    StoredFileRow,
    TrainingResultRow,
    TrainingTemplateRow,
)
from ._processes import terminate_job_process
from ._templates import sanitize_template_config
from .contracts import (
    AutomationEnabledUpdate,
    AutomationRuleInfo,
    AutomationRuleUpdate,
    AutomationSnapshot,
    DatasetInfo,
    JobSource,
    JobStatus,
    JobType,
    ResultStatus,
    StoredFileKind,
    TrainingUIAPIError,
)


AUTOMATION_KEY = "automation"
AUTOMATION_EXPERIMENT_NAME = "MLSystem2 Automation"
ACTIVE_JOB_STATUSES = {JobStatus.QUEUED.value, JobStatus.RUNNING.value}
MLFLOW_RUN_ID_FILE = "mlflow_run_id"
LOGGER = logging.getLogger(__name__)


def ensure_automation_control(session: Session) -> AutomationControlRow:
    row = session.get(AutomationControlRow, AUTOMATION_KEY)
    if row is None:
        row = AutomationControlRow(key=AUTOMATION_KEY, enabled=False)
        session.add(row)
        session.flush()
    return row


def automation_snapshot(session: Session, config: TrainingUIAPIConfig) -> AutomationSnapshot:
    control = ensure_automation_control(session)
    datasets = _automation_datasets(config)
    models = ui_model_infos()
    rules = {
        (row.dataset_key, row.architecture): row
        for row in session.scalars(select(AutomationRuleRow)).all()
    }
    cells = []
    for dataset in datasets:
        for model in models:
            rule = rules.get((dataset.key, model.architecture))
            training_result = _current_training_result(
                session,
                dataset_key=dataset.key,
                architecture=model.architecture,
                dataset_version=dataset.version,
            )
            pseudo_result = (
                _current_pseudo_result(session, training_result, dataset.version)
                if training_result is not None
                else None
            )
            cells.append(
                AutomationRuleInfo(
                    id=rule.id if rule is not None else None,
                    dataset_key=dataset.key,
                    architecture=model.architecture,
                    training_enabled=rule.training_enabled if rule is not None else False,
                    pseudo_markup_enabled=rule.pseudo_markup_enabled if rule is not None else False,
                    dataset_version=dataset.version,
                    training_status=ResultStatus(training_result.status) if training_result else None,
                    pseudo_markup_status=ResultStatus(pseudo_result.status) if pseudo_result else None,
                    current_training_result_id=training_result.id if training_result else None,
                )
            )
    return AutomationSnapshot(enabled=control.enabled, datasets=datasets, models=models, rules=cells)


def set_automation_enabled(
    session: Session,
    request: AutomationEnabledUpdate,
    config: TrainingUIAPIConfig,
) -> None:
    control = ensure_automation_control(session)
    if not request.enabled:
        _cancel_all_automation_jobs(session, config)
    control.enabled = request.enabled
    control.updated_at = _now()
    session.flush()
    return None


def update_automation_rule(
    session: Session,
    request: AutomationRuleUpdate,
    config: TrainingUIAPIConfig,
) -> AutomationRuleInfo:
    dataset = _resolve_automation_dataset(config, request.dataset_key)
    if request.architecture not in UI_ARCHITECTURES:
        raise TrainingUIAPIError(f"Модель не найдена: {request.architecture}")

    row = session.scalar(
        select(AutomationRuleRow).where(
            AutomationRuleRow.dataset_key == request.dataset_key,
            AutomationRuleRow.architecture == request.architecture,
        )
    )
    old_training_enabled = row.training_enabled if row is not None else False
    old_pseudo_enabled = row.pseudo_markup_enabled if row is not None else False
    if row is None:
        row = AutomationRuleRow(
            dataset_key=request.dataset_key,
            architecture=request.architecture,
        )
        session.add(row)
        session.flush()

    row.training_enabled = request.training_enabled
    row.pseudo_markup_enabled = request.pseudo_markup_enabled
    row.updated_at = _now()

    if not request.training_enabled:
        _cancel_active_automation_jobs(session, row, job_type=JobType.TRAINING, config=config)
    if not request.pseudo_markup_enabled:
        _cancel_active_automation_jobs(session, row, job_type=JobType.INFERENCE, config=config)
    if not old_training_enabled and request.training_enabled:
        _reset_failed_training_attempts(session, row, dataset.version)
    if not old_pseudo_enabled and request.pseudo_markup_enabled:
        _reset_failed_pseudo_attempts(session, row, dataset.version)

    session.flush()
    training_result = _current_training_result(
        session,
        dataset_key=dataset.key,
        architecture=row.architecture,
        dataset_version=dataset.version,
    )
    pseudo_result = (
        _current_pseudo_result(session, training_result, dataset.version)
        if training_result is not None
        else None
    )
    return AutomationRuleInfo(
        id=row.id,
        dataset_key=row.dataset_key,
        architecture=row.architecture,
        training_enabled=row.training_enabled,
        pseudo_markup_enabled=row.pseudo_markup_enabled,
        dataset_version=dataset.version,
        training_status=ResultStatus(training_result.status) if training_result else None,
        pseudo_markup_status=ResultStatus(pseudo_result.status) if pseudo_result else None,
        current_training_result_id=training_result.id if training_result else None,
    )


def sync_automation_once(session: Session, config: TrainingUIAPIConfig) -> None:
    control = ensure_automation_control(session)
    if not control.enabled:
        return
    datasets = {item.key: item for item in _automation_datasets(config)}
    rules = session.scalars(
        select(AutomationRuleRow).where(
            (AutomationRuleRow.training_enabled.is_(True))
            | (AutomationRuleRow.pseudo_markup_enabled.is_(True))
        )
    ).all()
    for rule in rules:
        dataset = datasets.get(rule.dataset_key)
        if dataset is None or not dataset.version:
            continue
        _cancel_stale_automation_jobs(session, rule, dataset.version, config)
        if rule.training_enabled:
            _ensure_training_for_rule(session, rule, dataset, config)
        if rule.pseudo_markup_enabled:
            _ensure_pseudo_markup_for_rule(session, rule, dataset, config)


def _automation_datasets(config: TrainingUIAPIConfig) -> list[DatasetInfo]:
    return [
        item
        for item in list_datasets(config.mlmarkup_root)
        if item.key != CUSTOM_KEY and item.scenes_file and item.annotation_file
    ]


def _resolve_automation_dataset(config: TrainingUIAPIConfig, dataset_key: str) -> DatasetInfo:
    dataset = next((item for item in _automation_datasets(config) if item.key == dataset_key), None)
    if dataset is None:
        raise TrainingUIAPIError(f"Датасет не найден или неполный: {dataset_key}")
    return dataset


def _ensure_training_for_rule(
    session: Session,
    rule: AutomationRuleRow,
    dataset: DatasetInfo,
    config: TrainingUIAPIConfig,
) -> None:
    result = _current_training_result(
        session,
        dataset_key=dataset.key,
        architecture=rule.architecture,
        dataset_version=dataset.version,
    )
    if result is not None and result.status in {
        ResultStatus.RUNNING.value,
        ResultStatus.OK.value,
        ResultStatus.ERROR.value,
    }:
        return
    if _has_active_automation_job(session, JobType.TRAINING, rule, dataset.version):
        return
    template = session.scalar(
        select(TrainingTemplateRow).where(TrainingTemplateRow.architecture == rule.architecture)
    )
    if template is None or not template.is_active:
        return
    experiment = create_experiment(
        MLflowExperimentRequest(
            tracking_uri=config.mlflow_tracking_uri,
            name=AUTOMATION_EXPERIMENT_NAME,
        )
    )
    job_config = sanitize_template_config(template.default_config)
    model_name = MODEL_DISPLAY_NAMES.get(rule.architecture, rule.architecture)
    row = JobRow(
        type=JobType.TRAINING.value,
        source=JobSource.AUTOMATION.value,
        status=JobStatus.QUEUED.value,
        queue_position=_next_queue_position(session, JobType.TRAINING, JobSource.AUTOMATION),
        automation_rule_id=rule.id,
        dataset_key=dataset.key,
        dataset_version=dataset.version,
        dataset_name=dataset.name,
        training_dataset_name=dataset.name,
        model_name=model_name,
        architecture=rule.architecture,
        tile_size=_int_or_none(job_config.get("tile_preparation.tile_size")),
        mlflow_experiment_id=experiment.experiment_id,
        mlflow_experiment_name=experiment.name,
        mlflow_run_name=_automation_run_name(dataset, rule.architecture),
        config=job_config,
    )
    session.add(row)
    session.flush()
    session.add(
        TrainingResultRow(
            source=JobSource.AUTOMATION.value,
            automation_rule_id=rule.id,
            dataset_key=dataset.key,
            dataset_version=dataset.version,
            class_key=dataset.key,
            class_display_name=dataset.name,
            architecture=rule.architecture,
            model_name=model_name,
            status=ResultStatus.RUNNING.value,
            job_id=row.id,
        )
    )
    session.flush()


def _ensure_pseudo_markup_for_rule(
    session: Session,
    rule: AutomationRuleRow,
    dataset: DatasetInfo,
    config: TrainingUIAPIConfig,
) -> None:
    training_result = _current_training_result(
        session,
        dataset_key=dataset.key,
        architecture=rule.architecture,
        dataset_version=dataset.version,
    )
    if training_result is None or training_result.status != ResultStatus.OK.value:
        return
    if training_result.mlflow_run_id is None:
        return
    pseudo_result = _current_pseudo_result(session, training_result, dataset.version)
    if pseudo_result is not None and pseudo_result.status in {
        ResultStatus.RUNNING.value,
        ResultStatus.OK.value,
        ResultStatus.ERROR.value,
    }:
        return
    if _has_active_automation_job(session, JobType.INFERENCE, rule, dataset.version):
        return
    if not dataset.scenes_file:
        return
    scenes_row = _store_existing_file(session, kind=StoredFileKind.SCENES_TXT, path=Path(dataset.scenes_file))
    row = JobRow(
        type=JobType.INFERENCE.value,
        source=JobSource.AUTOMATION.value,
        status=JobStatus.QUEUED.value,
        queue_position=_next_queue_position(session, JobType.INFERENCE, JobSource.AUTOMATION),
        automation_rule_id=rule.id,
        dataset_key=dataset.key,
        dataset_version=dataset.version,
        dataset_name=dataset.name,
        training_dataset_name=dataset.name,
        inference_dataset_name=dataset.name,
        model_name=training_result.model_name,
        architecture=training_result.architecture,
        config={
            "class_key": dataset.key,
            "dataset_key": dataset.key,
            "training_result_id": str(training_result.id),
            **_checkpoint_config(training_result, config),
        },
    )
    session.add(row)
    session.flush()
    session.add(
        PseudoMarkupResultRow(
            source=JobSource.AUTOMATION.value,
            automation_rule_id=rule.id,
            dataset_key=dataset.key,
            dataset_version=dataset.version,
            training_result_id=training_result.id,
            class_key=dataset.key,
            source_dataset_name=dataset.name,
            scenes_file_id=scenes_row.id,
            status=ResultStatus.RUNNING.value,
            job_id=row.id,
        )
    )
    session.flush()


def _current_training_result(
    session: Session,
    *,
    dataset_key: str,
    architecture: str,
    dataset_version: str | None,
) -> TrainingResultRow | None:
    if dataset_version is None:
        return None
    return session.scalar(
        select(TrainingResultRow)
        .where(
            TrainingResultRow.source == JobSource.AUTOMATION.value,
            TrainingResultRow.dataset_key == dataset_key,
            TrainingResultRow.architecture == architecture,
            TrainingResultRow.dataset_version == dataset_version,
        )
        .order_by(TrainingResultRow.created_at.desc())
    )


def _current_pseudo_result(
    session: Session,
    training_result: TrainingResultRow,
    dataset_version: str | None,
) -> PseudoMarkupResultRow | None:
    if dataset_version is None:
        return None
    return session.scalar(
        select(PseudoMarkupResultRow)
        .where(
            PseudoMarkupResultRow.source == JobSource.AUTOMATION.value,
            PseudoMarkupResultRow.training_result_id == training_result.id,
            PseudoMarkupResultRow.dataset_version == dataset_version,
        )
        .order_by(PseudoMarkupResultRow.created_at.desc())
    )


def _has_active_automation_job(
    session: Session,
    job_type: JobType,
    rule: AutomationRuleRow,
    dataset_version: str | None,
) -> bool:
    return (
        session.scalar(
            select(JobRow.id).where(
                JobRow.type == job_type.value,
                JobRow.source == JobSource.AUTOMATION.value,
                JobRow.automation_rule_id == rule.id,
                JobRow.dataset_version == dataset_version,
                JobRow.status.in_(ACTIVE_JOB_STATUSES),
            )
        )
        is not None
    )


def _cancel_stale_automation_jobs(
    session: Session,
    rule: AutomationRuleRow,
    current_version: str,
    config: TrainingUIAPIConfig,
) -> None:
    rows = session.scalars(
        select(JobRow).where(
            JobRow.source == JobSource.AUTOMATION.value,
            JobRow.automation_rule_id == rule.id,
            JobRow.status.in_(ACTIVE_JOB_STATUSES),
            JobRow.dataset_version != current_version,
        )
    ).all()
    for row in rows:
        _cancel_job(session, row, config)


def _cancel_active_automation_jobs(
    session: Session,
    rule: AutomationRuleRow,
    *,
    job_type: JobType,
    config: TrainingUIAPIConfig,
) -> None:
    rows = session.scalars(
        select(JobRow).where(
            JobRow.type == job_type.value,
            JobRow.source == JobSource.AUTOMATION.value,
            JobRow.automation_rule_id == rule.id,
            JobRow.status.in_(ACTIVE_JOB_STATUSES),
        )
    ).all()
    for row in rows:
        _cancel_job(session, row, config)


def _cancel_all_automation_jobs(session: Session, config: TrainingUIAPIConfig) -> None:
    rows = session.scalars(
        select(JobRow).where(
            JobRow.source == JobSource.AUTOMATION.value,
            JobRow.status.in_(ACTIVE_JOB_STATUSES),
        )
    ).all()
    for row in rows:
        _cancel_job(session, row, config)


def _cancel_job(session: Session, row: JobRow, config: TrainingUIAPIConfig) -> None:
    mlflow_run_id = _training_job_mlflow_run_id(session, row)
    if row.status == JobStatus.RUNNING.value:
        terminate_job_process(row)
        if mlflow_run_id:
            _mark_mlflow_run_killed(config, mlflow_run_id)
        if row.tmp_path:
            shutil.rmtree(row.tmp_path, ignore_errors=True)
            row.tmp_path = None
    row.status = JobStatus.CANCELLED.value
    row.finished_at = _now()
    for result in session.scalars(select(TrainingResultRow).where(TrainingResultRow.job_id == row.id)).all():
        result.status = ResultStatus.CANCELLED.value
        result.updated_at = _now()
    for result in session.scalars(select(PseudoMarkupResultRow).where(PseudoMarkupResultRow.job_id == row.id)).all():
        result.status = ResultStatus.CANCELLED.value
        result.updated_at = _now()


def _training_job_mlflow_run_id(session: Session, row: JobRow) -> str | None:
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
    if result_run_id:
        return str(result_run_id)
    if not row.tmp_path:
        return None
    run_id_file = Path(row.tmp_path) / MLFLOW_RUN_ID_FILE
    if run_id_file.is_file():
        run_id = run_id_file.read_text(encoding="utf-8").strip()
        if run_id:
            return run_id
    log_path = Path(row.tmp_path) / "train.log"
    if not log_path.is_file():
        return None
    match = re.search(r"MLflow run id:\s*([0-9a-fA-F]+)", log_path.read_text(encoding="utf-8", errors="ignore"))
    return match.group(1) if match else None


def _mark_mlflow_run_killed(config: TrainingUIAPIConfig, run_id: str) -> None:
    try:
        mark_run_killed(config.mlflow_tracking_uri, run_id)
    except MLflowAdapterError:
        LOGGER.warning("Не удалось пометить MLflow run %s как killed", run_id, exc_info=True)


def _reset_failed_training_attempts(
    session: Session,
    rule: AutomationRuleRow,
    dataset_version: str | None,
) -> None:
    if dataset_version is None:
        return
    rows = session.scalars(
        select(TrainingResultRow).where(
            TrainingResultRow.source == JobSource.AUTOMATION.value,
            TrainingResultRow.automation_rule_id == rule.id,
            TrainingResultRow.dataset_version == dataset_version,
            TrainingResultRow.status == ResultStatus.ERROR.value,
        )
    ).all()
    for row in rows:
        row.status = ResultStatus.CANCELLED.value
        row.updated_at = _now()


def _reset_failed_pseudo_attempts(
    session: Session,
    rule: AutomationRuleRow,
    dataset_version: str | None,
) -> None:
    if dataset_version is None:
        return
    rows = session.scalars(
        select(PseudoMarkupResultRow).where(
            PseudoMarkupResultRow.source == JobSource.AUTOMATION.value,
            PseudoMarkupResultRow.automation_rule_id == rule.id,
            PseudoMarkupResultRow.dataset_version == dataset_version,
            PseudoMarkupResultRow.status == ResultStatus.ERROR.value,
        )
    ).all()
    for row in rows:
        row.status = ResultStatus.CANCELLED.value
        row.updated_at = _now()


def _checkpoint_config(
    row: TrainingResultRow,
    config: TrainingUIAPIConfig,
) -> dict[str, object]:
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


def _next_queue_position(session: Session, queue_name: JobType, source: JobSource) -> int:
    rows = session.scalars(
        select(JobRow.queue_position).where(
            JobRow.type == queue_name.value,
            JobRow.source == source.value,
            JobRow.status.in_(ACTIVE_JOB_STATUSES),
        )
    ).all()
    return (max(rows) if rows else 0) + 1


def _automation_run_name(dataset: DatasetInfo, architecture: str) -> str:
    version = (dataset.version or "unknown").replace("git:", "").replace("fs:", "")[:8]
    return f"auto_{_slug(dataset.name)}_{_slug(architecture)}_{version}"


def _slug(value: str) -> str:
    text = re.sub(r"[\\/\s]+", "_", value.strip().lower())
    return re.sub(r"[^\w-]+", "", text, flags=re.UNICODE)


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _now() -> datetime:
    return datetime.now(timezone.utc)
