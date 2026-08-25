"""Фоновый исполнитель очереди training UI API."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import re
import shutil
import shlex
import subprocess
import sys
import traceback
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from mlsystem2.dataset_preparing.api import (
    is_per_image_footprint_name,
    load_dataset_manifest,
)
from mlsystem2.mlflow_adapter.api import (
    get_best_training_checkpoint,
    get_finished_run_artifact,
    list_experiments,
)
from mlsystem2.mlflow_adapter.contracts import MLflowAdapterError, MLflowBestCheckpoint
from mlsystem2.settings.api import load_settings

from ._automation import AUTOMATION_KEY, sync_automation_once
from ._config import TrainingUIAPIConfig
from ._dataset_catalog import (
    dataset_class_row,
    find_managed_dataset,
    list_managed_datasets,
)
from ._datasets import (
    CUSTOM_KEY,
    count_scenes_file_images,
    imagery_images_dir,
    per_image_annotation_files,
)
from ._external_models import (
    ExternalModelError,
    external_model_payload,
    external_result_manifest,
)
from ._inference_backend import (
    GEOALERT_INFERENCE_BACKEND,
    inference_backend_for_imagery,
)
from ._managed_datasets import (
    has_pending_managed_materialization,
    process_next_managed_materialization,
)
from ._models import (
    AutomationControlRow,
    CustomDatasetRow,
    JobRow,
    PseudoMarkupResultRow,
    QueueControlRow,
    StoredFileRow,
    TestSampleRow,
    TrainingResultRow,
    TrainingResultTestMetricRow,
)
from ._processes import terminate_job_process
from ._pseudolabel import PSEUDOLABEL_AOI_OPERATION
from ._queueing import (
    DATASET_EDITOR_PSEUDO_OPERATION,
    POST_TRAINING_INFERENCE_CONFIG_KEY,
    POST_TRAINING_INFERENCE_JOB_IDS_CONFIG_KEY,
    SECONDARY_PRIORITY_CONFIG_KEY,
    dispatch_sort_key,
    ensure_queue_positions,
    is_secondary_job,
)
from ._templates import normalize_tile_factors
from ._test_samples import (
    TEST_SAMPLE_F1_OPERATION,
    TEST_SAMPLE_EVALUATION_TARGET,
    _training_result_test_plan,
    _training_result_test_scope,
    current_primary_training_result,
    evaluate_test_samples_for_pseudo_markup,
    primary_test_sample,
    queue_training_result_test_f1,
    reconcile_test_sample_evaluations,
    reconcile_training_result_test_f1,
    test_sample_model_compatibility_error,
)
from .contracts import JobSource, JobStatus, JobType, ResultStatus, StoredFileKind


LOGGER = logging.getLogger(__name__)
MLFLOW_RUN_ID_FILE = "mlflow_run_id"
JOB_ERROR_MAX_BYTES = 8 * 1024
JOB_CONTROL_DIR = "control"
JOB_PAUSE_REQUEST = "pause.request"
JOB_PAUSED_MARKER = "paused"
URGENT_JOB_PRIORITY = "urgent"


class _ManagedDatasetNotReady(RuntimeError):
    """Материализация уже запрошена, поэтому queued job нужно оставить в очереди."""


class _StartedProcess(Protocol):
    pid: int


ProcessLauncher = Callable[..., _StartedProcess]


def _is_test_sample_f1_job(row: JobRow) -> bool:
    return (row.config or {}).get("operation") == TEST_SAMPLE_F1_OPERATION


def _is_saved_test_sample_evaluation_job(row: JobRow) -> bool:
    return (
        _is_test_sample_f1_job(row)
        and (row.config or {}).get("metric_target") == TEST_SAMPLE_EVALUATION_TARGET
    )


def _is_pseudolabel_aoi_job(row: JobRow) -> bool:
    """Proverit domennoe naznachenie inference job."""

    return (row.config or {}).get("operation") == PSEUDOLABEL_AOI_OPERATION


def _is_dataset_editor_pseudo_job(row: JobRow) -> bool:
    return (row.config or {}).get("operation") == DATASET_EDITOR_PSEUDO_OPERATION


async def run_queue_worker(
    session_factory: sessionmaker[Session],
    config: TrainingUIAPIConfig,
) -> None:
    interval = max(1, config.worker_interval_seconds)
    automation_interval = max(interval, config.automation_sync_interval_seconds)
    next_automation_at = 0.0
    materialization_task: asyncio.Task[bool] | None = None
    LOGGER.info(
        "Training UI worker started: queue interval %s sec, automation interval %s sec",
        interval,
        automation_interval,
    )
    try:
        while True:
            if materialization_task is not None and materialization_task.done():
                try:
                    materialization_task.result()
                except Exception:
                    LOGGER.exception("Managed dataset materialization failed")
                materialization_task = None
            try:
                with session_factory() as session:
                    now = asyncio.get_running_loop().time()
                    if now >= next_automation_at:
                        sync_automation_once(session, config)
                        next_automation_at = now + automation_interval
                    dispatch_queue_once(session, config)
                    session.commit()
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("Training UI worker tick failed")
            if materialization_task is None and has_pending_managed_materialization(config):
                materialization_task = asyncio.create_task(
                    asyncio.to_thread(
                        _process_managed_materialization_request,
                        session_factory,
                        config,
                    )
                )
            await asyncio.sleep(interval)
    finally:
        if materialization_task is not None:
            try:
                await asyncio.shield(materialization_task)
            except asyncio.CancelledError:
                pass
            except Exception:
                LOGGER.exception("Managed dataset materialization stopped with worker")


def _process_managed_materialization_request(
    session_factory: sessionmaker[Session],
    config: TrainingUIAPIConfig,
) -> bool:
    with session_factory() as session:
        processed = process_next_managed_materialization(session, config)
        session.commit()
        return processed


def dispatch_queue_once(
    session: Session,
    config: TrainingUIAPIConfig,
    *,
    popen_factory: ProcessLauncher = subprocess.Popen,
) -> None:
    _reconcile_running_training_jobs(session, config)
    _reconcile_running_inference_jobs(session, config)
    ensure_queue_positions(session)
    if _coordinate_job_preemption(session, config, popen_factory=popen_factory):
        return
    if _has_running_job(session):
        return
    job = _next_dispatch_job(session)
    if job is None:
        return
    if job.type == JobType.INFERENCE.value:
        _start_inference_job(session, job, config, popen_factory=popen_factory)
        return
    _start_training_job(session, job, config, popen_factory=popen_factory)


def dispatch_training_queue_once(
    session: Session,
    config: TrainingUIAPIConfig,
    *,
    popen_factory: ProcessLauncher = subprocess.Popen,
) -> None:
    _reconcile_running_training_jobs(session, config)
    _reconcile_running_inference_jobs(session, config)
    ensure_queue_positions(session)
    if _coordinate_job_preemption(session, config, popen_factory=popen_factory):
        return
    if _has_running_job(session):
        return
    next_job = _next_dispatch_job(session)
    if next_job is not None and next_job.type != JobType.TRAINING.value:
        return
    control = session.get(QueueControlRow, JobType.TRAINING.value)
    if control is not None and not control.enabled:
        return
    job = _next_dispatch_job(session, JobType.TRAINING)
    if job is None:
        return
    _start_training_job(session, job, config, popen_factory=popen_factory)


def dispatch_inference_queue_once(
    session: Session,
    config: TrainingUIAPIConfig,
    *,
    popen_factory: ProcessLauncher = subprocess.Popen,
) -> None:
    _reconcile_running_training_jobs(session, config)
    _reconcile_running_inference_jobs(session, config)
    ensure_queue_positions(session)
    if _coordinate_job_preemption(session, config, popen_factory=popen_factory):
        return
    if _has_running_job(session):
        return
    next_job = _next_dispatch_job(session)
    if next_job is not None and next_job.type != JobType.INFERENCE.value:
        return
    control = session.get(QueueControlRow, JobType.INFERENCE.value)
    if control is not None and not control.enabled:
        return
    job = _next_dispatch_job(session, JobType.INFERENCE)
    if job is None:
        return
    _start_inference_job(session, job, config, popen_factory=popen_factory)


def _next_dispatch_job(session: Session, job_type: JobType | None = None) -> JobRow | None:
    conditions = [JobRow.status == JobStatus.QUEUED.value]
    if job_type is not None:
        conditions.append(JobRow.type == job_type.value)
    rows = session.scalars(select(JobRow).where(*conditions)).all()
    automation_enabled = _automation_enabled(session)
    candidates = [row for row in rows if _dispatch_allowed(session, row, automation_enabled)]
    candidates.sort(key=dispatch_sort_key)
    return candidates[0] if candidates else None


def _coordinate_job_preemption(
    session: Session,
    config: TrainingUIAPIConfig,
    *,
    popen_factory: ProcessLauncher,
) -> bool:
    """Освободить ресурс для более приоритетной работы и затем продолжить прежний job."""

    running = session.scalar(
        select(JobRow)
        .where(JobRow.status == JobStatus.RUNNING.value)
        .order_by(JobRow.started_at, JobRow.created_at)
    )
    paused = session.scalar(
        select(JobRow)
        .where(JobRow.status == JobStatus.PAUSED.value)
        .order_by(JobRow.started_at, JobRow.created_at)
    )
    if running is not None:
        preemptor = _preempting_job(session, running)
        if preemptor is None:
            if running.tmp_path is not None and _job_pause_request(running).is_file():
                _request_job_resume(running)
            return True
        token = _request_job_pause(running)
        if not _job_pause_confirmed(running, token):
            return True
        running.status = JobStatus.PAUSED.value
        session.flush()
        _start_job(session, preemptor, config, popen_factory=popen_factory)
        return True

    if paused is None:
        return False
    if is_secondary_job(paused):
        preemptor = _next_urgent_inference_job(session) or _next_non_secondary_job(session)
        if preemptor is not None:
            token = _request_job_pause(paused)
            if not _job_pause_confirmed(paused, token):
                return True
            _start_job(session, preemptor, config, popen_factory=popen_factory)
            return True
    elif paused.type == JobType.TRAINING.value:
        urgent = _next_urgent_inference_job(session)
        if urgent is not None:
            token = _request_job_pause(paused)
            if not _job_pause_confirmed(paused, token):
                return True
            _start_inference_job(session, urgent, config, popen_factory=popen_factory)
            return True
    _request_job_resume(paused)
    if not _job_paused_marker(paused).is_file():
        paused.status = JobStatus.RUNNING.value
        session.flush()
    return True


def _preempting_job(session: Session, running: JobRow) -> JobRow | None:
    if running.type == JobType.TRAINING.value or is_secondary_job(running):
        urgent = _next_urgent_inference_job(session)
        if urgent is not None:
            return urgent
    if is_secondary_job(running):
        return _next_non_secondary_job(session)
    return None


def _next_non_secondary_job(session: Session) -> JobRow | None:
    rows = session.scalars(select(JobRow).where(JobRow.status == JobStatus.QUEUED.value)).all()
    automation_enabled = _automation_enabled(session)
    candidates = [
        row
        for row in rows
        if not is_secondary_job(row) and _dispatch_allowed(session, row, automation_enabled)
    ]
    candidates.sort(key=dispatch_sort_key)
    return candidates[0] if candidates else None


def _start_job(
    session: Session,
    row: JobRow,
    config: TrainingUIAPIConfig,
    *,
    popen_factory: ProcessLauncher,
) -> None:
    if row.type == JobType.INFERENCE.value:
        _start_inference_job(session, row, config, popen_factory=popen_factory)
    else:
        _start_training_job(session, row, config, popen_factory=popen_factory)


def _next_urgent_inference_job(session: Session) -> JobRow | None:
    rows = session.scalars(
        select(JobRow).where(
            JobRow.type == JobType.INFERENCE.value,
            JobRow.status == JobStatus.QUEUED.value,
        )
    ).all()
    candidates = [
        row
        for row in rows
        if _is_urgent_job(row) and _dispatch_allowed(session, row, _automation_enabled(session))
    ]
    candidates.sort(key=dispatch_sort_key)
    return candidates[0] if candidates else None


def _is_urgent_job(row: JobRow) -> bool:
    return (row.config or {}).get("priority") == URGENT_JOB_PRIORITY


def _job_control_dir(row: JobRow) -> Path:
    if row.tmp_path is None:
        raise RuntimeError("У выполняющегося задания отсутствует рабочая директория.")
    return Path(row.tmp_path) / JOB_CONTROL_DIR


def _job_pause_request(row: JobRow) -> Path:
    return _job_control_dir(row) / JOB_PAUSE_REQUEST


def _job_paused_marker(row: JobRow) -> Path:
    return _job_control_dir(row) / JOB_PAUSED_MARKER


def _request_job_pause(row: JobRow) -> str:
    request_path = _job_pause_request(row)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        token = request_path.read_text(encoding="utf-8").strip()
    except OSError:
        token = ""
    if token:
        return token
    token = uuid.uuid4().hex
    temporary = request_path.with_suffix(".tmp")
    temporary.write_text(f"{token}\n", encoding="utf-8")
    os.replace(temporary, request_path)
    return token


def _job_pause_confirmed(row: JobRow, token: str) -> bool:
    try:
        return _job_paused_marker(row).read_text(encoding="utf-8").strip() == token
    except OSError:
        return False


def _request_job_resume(row: JobRow) -> None:
    _job_pause_request(row).unlink(missing_ok=True)


def _dispatch_allowed(session: Session, row: JobRow, automation_enabled: bool) -> bool:
    if (
        row.source == JobSource.AUTOMATION.value
        and not automation_enabled
        and not _is_test_sample_f1_job(row)
    ):
        return False
    control = session.get(QueueControlRow, row.type)
    if control is not None and not control.enabled:
        return False
    return True


def _automation_enabled(session: Session) -> bool:
    row = session.get(AutomationControlRow, AUTOMATION_KEY)
    return bool(row and row.enabled)


def _reconcile_running_training_jobs(session: Session, config: TrainingUIAPIConfig) -> None:
    rows = session.scalars(
        select(JobRow).where(
            JobRow.type == JobType.TRAINING.value,
            JobRow.status.in_([JobStatus.RUNNING.value, JobStatus.PAUSED.value]),
        )
    ).all()
    for row in rows:
        run_dir = Path(row.tmp_path) if row.tmp_path else None
        _sync_training_run_id(session, row, config)
        exit_code = _read_exit_code(run_dir) if run_dir is not None else None
        if exit_code is not None:
            _finish_training_job(session, row, config, succeeded=exit_code == 0)
            continue
        if row.process_pid is not None and _pid_is_alive(row.process_pid):
            continue
        _finish_training_job(session, row, config, succeeded=False)


def _reconcile_running_inference_jobs(session: Session, config: TrainingUIAPIConfig) -> None:
    rows = session.scalars(
        select(JobRow).where(
            JobRow.type == JobType.INFERENCE.value,
            JobRow.status.in_([JobStatus.RUNNING.value, JobStatus.PAUSED.value]),
        )
    ).all()
    for row in rows:
        run_dir = Path(row.tmp_path) if row.tmp_path else None
        exit_code = _read_exit_code(run_dir) if run_dir is not None else None
        if exit_code is not None:
            _finish_inference_job(session, row, config, succeeded=exit_code == 0)
            continue
        if _pseudolabel_timed_out(row):
            terminate_job_process(row)
            _set_pseudolabel_error(
                row,
                "TIMEOUT",
                "Превышено допустимое время распознавания зоны интереса.",
            )
            _finish_inference_job(session, row, config, succeeded=False)
            continue
        if row.process_pid is not None and _pid_is_alive(row.process_pid):
            continue
        _finish_inference_job(session, row, config, succeeded=False)


def _has_running_training_job(session: Session) -> bool:
    return (
        session.scalar(
            select(JobRow.id).where(
                JobRow.type == JobType.TRAINING.value,
                JobRow.status.in_([JobStatus.RUNNING.value, JobStatus.PAUSED.value]),
            )
        )
        is not None
    )


def _has_running_job(session: Session) -> bool:
    return (
        session.scalar(
            select(JobRow.id).where(
                JobRow.status.in_([JobStatus.RUNNING.value, JobStatus.PAUSED.value])
            )
        )
        is not None
    )


def _has_running_inference_job(session: Session) -> bool:
    return (
        session.scalar(
            select(JobRow.id).where(
                JobRow.type == JobType.INFERENCE.value,
                JobRow.status == JobStatus.RUNNING.value,
            )
        )
        is not None
    )


def _start_training_job(
    session: Session,
    row: JobRow,
    config: TrainingUIAPIConfig,
    *,
    popen_factory: ProcessLauncher,
) -> None:
    run_dir = config.scratch_root / "jobs" / str(row.id)
    row.error = None
    row.tmp_path = str(run_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "scratch").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / JOB_CONTROL_DIR).mkdir(parents=True, exist_ok=True)

    try:
        config_path = run_dir / "run.yml"
        payload = _build_training_config(session, row, config, run_dir)
        _write_yaml(config_path, payload)
        load_settings(config.training_settings_path, config_path)
        script_path = _write_run_script(row, config, run_dir, config_path)
        process = popen_factory(
            ["bash", str(script_path)],
            cwd=str(config.project_root),
            start_new_session=True,
        )
    except _ManagedDatasetNotReady:
        shutil.rmtree(run_dir, ignore_errors=True)
        row.tmp_path = None
        row.error = None
        session.flush()
        LOGGER.info("Training job %s waits for managed dataset materialization", row.id)
        return
    except Exception:
        LOGGER.exception("Failed to start training job %s", row.id)
        _write_worker_error(
            run_dir,
            f"Не удалось запустить обучение.\n\n{traceback.format_exc()}",
        )
        _finish_training_job(session, row, config, succeeded=False)
        return

    row.status = JobStatus.RUNNING.value
    row.started_at = _now()
    row.finished_at = None
    row.process_pid = process.pid
    training_results = _training_results(session, row)
    for result in training_results:
        result.status = ResultStatus.RUNNING.value
        result.updated_at = _now()
    session.flush()
    LOGGER.info("Started training job %s with pid %s", row.id, process.pid)


def _start_inference_job(
    session: Session,
    row: JobRow,
    config: TrainingUIAPIConfig,
    *,
    popen_factory: ProcessLauncher,
) -> None:
    is_test_f1 = _is_test_sample_f1_job(row)
    is_pseudolabel_aoi = _is_pseudolabel_aoi_job(row)
    run_dir = config.scratch_root / "jobs" / str(row.id)
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / JOB_CONTROL_DIR).mkdir(parents=True, exist_ok=True)
    row.tmp_path = str(run_dir)
    try:
        config_path = run_dir / (
            "test_f1_config.yaml"
            if is_test_f1
            else "pseudolabel_config.yaml"
            if is_pseudolabel_aoi
            else "pseudo_config.yaml"
        )
        if is_test_f1:
            payload = _build_test_sample_f1_config(session, row, config, run_dir)
        elif is_pseudolabel_aoi:
            payload = _build_pseudolabel_aoi_config(row, config, run_dir)
        else:
            payload = _build_pseudo_markup_config(session, row, config, run_dir)
        payload["control_dir"] = str(run_dir / JOB_CONTROL_DIR)
        _write_yaml(config_path, payload)
        script_path = _write_pseudo_run_script(
            config,
            run_dir,
            config_path,
            test_f1=is_test_f1,
            inference_backend=str(payload.get("inference_backend") or ""),
        )
        process = popen_factory(
            ["bash", str(script_path)],
            cwd=str(config.project_root),
            start_new_session=True,
        )
    except Exception:
        LOGGER.exception("Failed to start inference job %s", row.id)
        _write_worker_error(
            run_dir,
            (
                "Не удалось запустить расчёт F1 на тестовой разметке.\n\n"
                if is_test_f1
                else "Не удалось запустить распознавание зоны интереса.\n\n"
                if is_pseudolabel_aoi
                else "Не удалось запустить псевдоразметку.\n\n"
            )
            + traceback.format_exc(),
        )
        if is_pseudolabel_aoi:
            _set_pseudolabel_error(
                row,
                "START_FAILED",
                "Не удалось запустить распознавание зоны интереса.",
            )
        _finish_inference_job(session, row, config, succeeded=False)
        return

    row.status = JobStatus.RUNNING.value
    row.started_at = _now()
    row.finished_at = None
    row.process_pid = process.pid
    if is_test_f1:
        if _is_saved_test_sample_evaluation_job(row):
            sample = _saved_test_sample_evaluation(session, row)
            if sample is not None and sample.evaluation_job_id == row.id:
                sample.metric_status = "running"
                sample.updated_at = _now()
        else:
            metric = _test_sample_f1_metric(session, row)
            if metric is not None:
                metric.status = "running"
                metric.updated_at = _now()
    elif not is_pseudolabel_aoi:
        for result in _pseudo_markup_results(session, row):
            result.status = ResultStatus.RUNNING.value
            result.updated_at = _now()
    session.flush()
    LOGGER.info("Started inference job %s with pid %s", row.id, process.pid)


def _build_training_config(
    session: Session,
    row: JobRow,
    config: TrainingUIAPIConfig,
    run_dir: Path,
) -> dict[str, Any]:
    flat = dict(row.config or {})
    positive_factor = _float_value(flat, "tile_preparation.positive_factor", 0.5)
    hard_negative_factor = _float_value(flat, "tile_preparation.hard_negative_factor", 0.0)
    background_factor = _float_value(
        flat,
        "tile_preparation.background_factor",
        1.0 - positive_factor - hard_negative_factor,
    )
    tile_factors = {
        "tile_preparation.positive_factor": positive_factor,
        "tile_preparation.hard_negative_factor": hard_negative_factor,
        "tile_preparation.background_factor": background_factor,
    }
    normalize_tile_factors(tile_factors)
    dataset_config = _dataset_config(session, row, config, run_dir)
    manifest = (
        load_dataset_manifest(str(dataset_config["annotations_dir"]))
        if dataset_config.get("annotations_dir")
        else None
    )
    task = "multiclass" if manifest is not None else "binary"
    raw_loss = str(_flat_value(flat, "train.loss", "bce_dice"))
    if task == "multiclass":
        loss = (
            raw_loss
            if raw_loss in {"cross_entropy", "cross_entropy_dice"}
            else "cross_entropy_dice"
        )
    else:
        loss = raw_loss if raw_loss in {"bce_dice", "focal_dice", "focal_tversky"} else "bce_dice"
    return {
        "runtime": {
            "project_root": str(config.project_root),
            "scratch_root": str(run_dir / "scratch"),
            "logs_root": str(run_dir / "logs"),
            "cleanup_scratch_after_mlflow_log": True,
        },
        "dataset": {
            **dataset_config,
            "val_fraction": _float_value(flat, "dataset.val_fraction", 0.2),
        },
        "tile_preparation": {
            "tile_size": _int_value(flat, "tile_preparation.tile_size", row.tile_size or 512),
            "stride": _int_value(flat, "tile_preparation.stride", row.tile_size or 512),
            "context": _int_value(flat, "tile_preparation.context", 0),
            "augmentation_level": _int_value(flat, "tile_preparation.augmentation_level", 0),
            "positive_factor": tile_factors["tile_preparation.positive_factor"],
            "hard_negative_factor": tile_factors["tile_preparation.hard_negative_factor"],
            "background_factor": tile_factors["tile_preparation.background_factor"],
            "class_balance": task == "multiclass",
        },
        "train": {
            "task": task,
            "quality_metric": (
                "pixel"
                if task == "multiclass"
                else str(_flat_value(flat, "train.quality_metric", "pixel"))
            ),
            "model_name": row.architecture,
            "input_channels": _int_value(flat, "train.input_channels", 4),
            "output_channels": len(manifest.classes) + 1 if manifest is not None else 1,
            "initial_checkpoint_uri": _blank_to_none(
                _flat_value(flat, "train.initial_checkpoint_uri", None)
            ),
            "epochs": _int_value(flat, "train.epochs", 1),
            "batch_size": _int_value(flat, "train.batch_size", 1),
            "learning_rate": _float_value(flat, "train.learning_rate", 0.0001),
            "weight_decay": _float_value(flat, "train.weight_decay", 0.0),
            "loss": loss,
            "focal_alpha": _float_value(flat, "train.focal_alpha", 0.6),
            "pos_weight": _float_value(flat, "train.pos_weight", 1.0),
            "hard_negative_weight": _float_value(flat, "train.hard_negative_weight", 1.0),
            "tversky_alpha": _float_value(flat, "train.tversky_alpha", 0.4),
            "tversky_beta": _float_value(flat, "train.tversky_beta", 0.6),
            "threshold": _float_value(flat, "train.threshold", 0.5),
            "early_stopping_patience": _int_value(flat, "train.early_stopping_patience", 10),
            "max_train_batches_per_epoch": _optional_int(flat, "train.max_train_batches_per_epoch"),
            "max_val_batches_per_epoch": _optional_int(flat, "train.max_val_batches_per_epoch"),
            "max_training_time_sec": _optional_int(flat, "train.max_training_time_sec"),
        },
        "mlflow": {
            "experiment_name": row.mlflow_experiment_name or "MLSystem2",
        },
    }


def _build_pseudo_markup_config(
    session: Session,
    row: JobRow,
    config: TrainingUIAPIConfig,
    run_dir: Path,
) -> dict[str, Any]:
    if _is_dataset_editor_pseudo_job(row):
        state = (row.config or {}).get("editor_pseudo")
        if not isinstance(state, dict):
            raise RuntimeError("В задании редактора отсутствуют параметры псевдоразметки.")
        try:
            training_result_id = uuid.UUID(str(state.get("training_result_id")))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "В задании редактора повреждён идентификатор основной сети."
            ) from exc
        training_result = session.get(TrainingResultRow, training_result_id)
        if training_result is None or training_result.status != ResultStatus.OK.value:
            raise RuntimeError("Основная сеть для псевдоразметки снимка больше недоступна.")
        scenes_path = run_dir / "editor_scene.txt"
        scenes_path.write_text(f"{state.get('image_relative')}\n", encoding="utf-8")
        snapshot = state
        scenes_file = str(scenes_path)
        images_root = str(state.get("images_root") or config.images_root)
        annotation_files: list[str] = []
        class_key = training_result.class_key
        class_name = training_result.class_display_name
        source_model = training_result.model_name
    else:
        result = _first_pseudo_markup_result(session, row)
        if result is None or result.scenes_file is None:
            raise RuntimeError("Для псевдоразметки не найден txt со снимками.")
        training_result = (
            session.get(TrainingResultRow, result.training_result_id)
            if result.training_result_id is not None
            else None
        )
        snapshot = dict(row.config or {})
        scenes_file = result.scenes_file.path
        images_root = str(snapshot.get("images_root") or config.images_root)
        annotation_files = [
            str(value) for value in (snapshot.get("annotation_files") or []) if value
        ]
        if not annotation_files and result.dataset_key and result.dataset_key != CUSTOM_KEY:
            dataset = find_managed_dataset(session, config, result.dataset_key)
            if dataset is not None:
                annotation_files = (
                    per_image_annotation_files(Path(dataset.annotations_dir))
                    if dataset.annotations_dir
                    else [
                        path
                        for path in (dataset.annotation_file, dataset.hard_negative_annotation_file)
                        if path
                    ]
                )
        class_key = result.class_key
        class_name = (
            training_result.class_display_name if training_result is not None else result.class_key
        )
        source_model = training_result.model_name if training_result is not None else row.model_name

    source_training_job = (
        session.get(JobRow, training_result.job_id)
        if training_result is not None and training_result.job_id is not None
        else None
    )
    flat = dict(source_training_job.config or {}) if source_training_job is not None else {}
    try:
        external_manifest = (
            external_result_manifest(session, training_result)
            if training_result is not None
            else None
        )
    except ExternalModelError as exc:
        raise RuntimeError(str(exc)) from exc
    tile_size = (
        external_manifest.tile_size
        if external_manifest is not None
        else _int_value(flat, "tile_preparation.tile_size", 768)
    )
    context = (
        external_manifest.context
        if external_manifest is not None
        else _int_value(flat, "tile_preparation.context", 0)
    )
    core_size = tile_size - 2 * context
    if context < 0 or core_size <= 0:
        raise RuntimeError("Размер inference-тайла должен быть больше удвоенного context.")
    threshold = _optional_float(snapshot, "checkpoint_threshold")
    if threshold is None and training_result is not None and external_manifest is None:
        raise RuntimeError(
            "Для псевдоразметки по результату обучения не найден MLflow val/best_threshold "
            "на эпохе best checkpoint."
        )
    if threshold is None:
        threshold = _float_value(flat, "train.threshold", 0.5)
    class_row = dataset_class_row(session, class_key)
    imagery_type = str(
        snapshot.get("imagery_type")
        or (class_row.imagery_type if class_row is not None else "kanopus")
    )
    inference_backend = inference_backend_for_imagery(imagery_type)
    return {
        "run_root": str(run_dir / "scratch"),
        "inference_backend": inference_backend,
        "output_geojson": str(run_dir / "scratch" / "pseudo_markup.geojson"),
        "report_path": str(run_dir / "scratch" / "report.json"),
        "scenes_file": scenes_file,
        "images_root": images_root,
        "annotation_files": annotation_files,
        "class_key": class_key,
        "class_name": class_name,
        "source_model": source_model,
        "task": training_result.task if training_result is not None else "binary",
        "object_types": (
            list(training_result.class_schema or []) if training_result is not None else []
        ),
        "mlflow_tracking_uri": config.mlflow_tracking_uri,
        "mlflow_run_id": snapshot.get("mlflow_run_id"),
        "checkpoint_uri": snapshot.get("checkpoint_uri"),
        "checkpoint_artifact_path": (
            snapshot.get("checkpoint_artifact_path") or "checkpoints/best.pt"
        ),
        "checkpoint_f1_score": snapshot.get("checkpoint_f1_score"),
        "checkpoint_epoch": snapshot.get("checkpoint_epoch"),
        "external_model": external_model_payload(external_manifest),
        "imagery_type": imagery_type,
        "model_imagery_type": imagery_type,
        "input_channels": _int_value(snapshot, "input_channels", 4),
        "postprocess_config": snapshot.get("inference_template_config") or {},
        "threshold": threshold,
        "tile_size": tile_size,
        "context": context,
        "stride": (
            external_manifest.stride
            if external_manifest is not None
            else (core_size if context else _int_value(flat, "tile_preparation.stride", tile_size))
        ),
        "batch_size": (
            1 if external_manifest is not None else _int_value(flat, "train.batch_size", 1)
        ),
        "device": "cuda",
        **_geoalert_runtime_config(config, inference_backend),
    }


def _build_pseudolabel_aoi_config(
    row: JobRow,
    config: TrainingUIAPIConfig,
    run_dir: Path,
) -> dict[str, Any]:
    """Sobrat runner-config tolko iz servernogo snapshot job."""

    state = (row.config or {}).get("pseudolabel")
    if not isinstance(state, dict):
        raise RuntimeError("В задании распознавания отсутствуют зафиксированные параметры.")
    threshold = state.get("checkpoint_threshold")
    external_model = state.get("external_model")
    if threshold is None and not isinstance(external_model, dict):
        raise RuntimeError("У зафиксированной модели отсутствует порог распознавания.")
    images_root = str(state.get("images_root") or "")
    index_key = hashlib.sha256(images_root.encode("utf-8")).hexdigest()[:20] if images_root else ""
    imagery_type = str(state.get("model_imagery_type") or state.get("imagery_type") or "kanopus")
    inference_backend = inference_backend_for_imagery(imagery_type)
    return {
        "operation": PSEUDOLABEL_AOI_OPERATION,
        "job_id": str(row.id),
        "run_root": str(run_dir / "scratch"),
        "inference_backend": inference_backend,
        "output_geojson": str(run_dir / "scratch" / "pseudo_markup.geojson"),
        "report_path": str(run_dir / "scratch" / "report.json"),
        "images_root": images_root,
        "raster_index_path": (
            str(
                config.stored_files_root / "cache" / "pseudolabel-image-index" / f"{index_key}.json"
            )
            if index_key
            else None
        ),
        "aoi": state.get("aoi"),
        "aoi_crs": "EPSG:4326",
        "aoi_area_m2": state.get("aoi_area_m2"),
        "class_key": state.get("class_id"),
        "class_name": state.get("class_name"),
        "model_id": state.get("model_id"),
        "model_version": state.get("model_version"),
        "source_model": state.get("model_name"),
        "mlflow_tracking_uri": config.mlflow_tracking_uri,
        "mlflow_run_id": state.get("mlflow_run_id"),
        "checkpoint_uri": state.get("checkpoint_uri"),
        "checkpoint_artifact_path": state.get("checkpoint_artifact_path") or "checkpoints/best.pt",
        "checkpoint_f1_score": state.get("checkpoint_f1_score"),
        "checkpoint_epoch": state.get("checkpoint_epoch"),
        "external_model": external_model,
        "imagery_type": state.get("imagery_type"),
        "model_imagery_type": state.get("model_imagery_type") or state.get("imagery_type"),
        "target_resolution_m": state.get("target_resolution_m"),
        "resample_to_resolution_m": state.get("resample_to_resolution_m"),
        "source_id": state.get("source_id"),
        "source_name": state.get("source_name"),
        "source_kind": state.get("source_kind"),
        "source_protocol": state.get("source_protocol"),
        "source_imagery_type": state.get("source_imagery_type"),
        "source_native_channels": state.get("source_native_channels"),
        "source_attribution": state.get("source_attribution"),
        "source_license_url": state.get("source_license_url"),
        "source_settings": state.get("source_settings") or {},
        "channel_mapping": state.get("channel_mapping"),
        "input_channels": _int_value(state, "input_channels", 4),
        "postprocess_config": state.get("inference_template_config") or {},
        "threshold": float(threshold) if threshold is not None else None,
        "tile_size": _int_value(state, "tile_size", 768),
        "context": _int_value(state, "context", 0),
        "stride": _int_value(state, "stride", 768),
        "batch_size": _int_value(state, "batch_size", 1),
        "image_scan_workers": config.pseudolabel_image_scan_workers,
        "tile_read_workers": config.pseudolabel_tile_read_workers,
        "prefetch_batches": config.pseudolabel_prefetch_batches,
        "external_http_workers": config.pseudolabel_external_http_workers,
        "device": "cuda",
        **_geoalert_runtime_config(config, inference_backend),
    }


def _build_test_sample_f1_config(
    session: Session,
    row: JobRow,
    config: TrainingUIAPIConfig,
    run_dir: Path,
) -> dict[str, Any]:
    try:
        training_result_id = uuid.UUID(str(row.config.get("training_result_id")))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("В задании F1 повреждён идентификатор сети.") from exc
    training_result = session.get(TrainingResultRow, training_result_id)
    if training_result is None or training_result.status != ResultStatus.OK.value:
        raise RuntimeError("Успешный результат обучения для расчёта F1 не найден.")
    saved_evaluation = _is_saved_test_sample_evaluation_job(row)
    plan = None
    managed_targets: tuple[Any, ...] = ()
    managed_scope: list[dict[str, Any]] = []
    managed_full_scope: list[dict[str, Any]] = []
    if saved_evaluation:
        try:
            sample_id = uuid.UUID(str(row.config.get("test_sample_id")))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("В задании F1 повреждён идентификатор разметки.") from exc
        sample = session.get(TestSampleRow, sample_id)
        if sample is None:
            raise RuntimeError("Тестовая разметка была удалена.")
        primary = current_primary_training_result(session, sample.class_key)
        if primary is None or primary.id != training_result.id:
            raise RuntimeError("Основная сеть класса была заменена.")
        compatibility_error = test_sample_model_compatibility_error(
            session,
            sample,
            training_result,
        )
        if compatibility_error is not None:
            raise RuntimeError(compatibility_error)
    else:
        plan = _training_result_test_plan(session, training_result)
        if plan.error is not None or not plan.targets:
            raise RuntimeError(plan.error or "Основные тестовые разметки были удалены.")
        if plan.managed:
            managed_full_scope = _training_result_test_scope(plan)
            expected_full_scope = list(
                row.config.get("managed_full_test_samples") or row.config.get("test_samples") or []
            )
            if expected_full_scope != managed_full_scope:
                raise RuntimeError(
                    "Состав основных тестовых разметок управляемого датасета изменён."
                )
            managed_scope = list(row.config.get("test_samples") or [])
            managed_targets = _managed_targets_for_scope(plan, managed_scope)
            if not managed_targets:
                raise RuntimeError("В задании не выбран класс управляемого датасета.")
            sample = managed_targets[0].sample
        else:
            sample = plan.targets[0].sample
            compatibility_error = test_sample_model_compatibility_error(
                session,
                sample,
                training_result,
            )
            if compatibility_error is not None:
                raise RuntimeError(compatibility_error)
    if saved_evaluation or not (plan and plan.managed):
        expected_revision = int(row.config.get("test_sample_revision") or 0)
        if sample.content_revision != expected_revision:
            raise RuntimeError("Состав основной тестовой разметки изменён.")
        expected_indices = {
            int(value) for value in (row.config.get("test_sample_tile_indices") or [])
        }
        enabled_tiles = [tile for tile in sample.tiles if tile.enabled]
        if not enabled_tiles or {tile.tile_index for tile in enabled_tiles} != expected_indices:
            raise RuntimeError("Состав включённых тайлов не совпадает с ревизией задания.")
    if not training_result.mlflow_run_id:
        raise RuntimeError("У результата обучения отсутствует MLflow run id.")
    source_training_job = (
        session.get(JobRow, training_result.job_id) if training_result.job_id is not None else None
    )
    flat = dict(source_training_job.config or {}) if source_training_job is not None else {}
    class_row = dataset_class_row(session, training_result.class_key)
    imagery_type = str(
        _flat_value(flat, "dataset.imagery_type", None)
        or (class_row.imagery_type if class_row is not None else "kanopus")
    )
    inference_backend = inference_backend_for_imagery(imagery_type)
    try:
        external_manifest = external_result_manifest(session, training_result)
    except ExternalModelError as exc:
        raise RuntimeError(str(exc)) from exc
    if external_manifest is not None:
        try:
            artifact = get_finished_run_artifact(
                config.mlflow_tracking_uri,
                training_result.mlflow_run_id,
                external_manifest.artifact_path,
            )
        except MLflowAdapterError as exc:
            raise RuntimeError("Не удалось проверить ZIP внешней модели в MLflow.") from exc
        if artifact is None:
            raise RuntimeError("ZIP внешней модели отсутствует в завершённом MLflow run.")
        checkpoint_uri = artifact.artifact_uri
        checkpoint_artifact_path = artifact.artifact_path
        checkpoint_f1_score = None
        checkpoint_epoch = None
        checkpoint_threshold = external_manifest.score_threshold
        inference_tile_size = external_manifest.tile_size
        inference_context = external_manifest.context
        inference_stride = external_manifest.stride
        input_channels = external_manifest.input_channels
        batch_size = 1
    else:
        checkpoint = _best_training_checkpoint(config, training_result.mlflow_run_id)
        if checkpoint is None:
            raise RuntimeError("Не удалось получить best checkpoint из MLflow.")
        if checkpoint.threshold is None:
            raise RuntimeError("У best checkpoint отсутствует порог val/best_threshold.")
        checkpoint_uri = checkpoint.artifact_uri
        checkpoint_artifact_path = checkpoint.artifact_path
        checkpoint_f1_score = checkpoint.f1_score
        checkpoint_epoch = checkpoint.epoch
        checkpoint_threshold = checkpoint.threshold
        inference_tile_size = _int_value(flat, "tile_preparation.tile_size", 768)
        inference_context = _int_value(flat, "tile_preparation.context", 0)
        inference_core_size = inference_tile_size - 2 * inference_context
        if inference_context < 0 or inference_core_size <= 0:
            raise RuntimeError("Размер inference-тайла должен быть больше удвоенного context.")
        inference_stride = (
            inference_core_size
            if inference_context
            else _int_value(flat, "tile_preparation.stride", inference_tile_size)
        )
        input_channels = _int_value(flat, "train.input_channels", 4)
        batch_size = _int_value(flat, "train.batch_size", 1)
    tiles: list[dict[str, Any]] = []
    targets = list(managed_targets) if plan is not None and plan.managed else [None]
    for target in targets:
        target_sample = target.sample if target is not None else sample
        sample_root = Path(config.stored_files_root) / "test-samples" / str(target_sample.id)
        target_tiles = sorted(
            (tile for tile in target_sample.tiles if tile.enabled),
            key=lambda item: item.tile_index,
        )
        for tile in target_tiles:
            base_name = f"tile_{tile.tile_index:03d}"
            tif_path = sample_root / f"{base_name}.tif"
            mask_path = sample_root / f"{base_name}_mask.png"
            geojson_path = sample_root / f"{base_name}.geojson"
            if not tif_path.is_file() or not mask_path.is_file() or not geojson_path.is_file():
                raise RuntimeError(
                    "Не найдены TIFF, GeoJSON или маска тестового тайла "
                    f"{target_sample.name}/{base_name}."
                )
            tiles.append(
                {
                    "index": len(tiles) + 1 if target is not None else tile.tile_index,
                    "source_tile_index": tile.tile_index,
                    "test_sample_id": str(target_sample.id),
                    "image_path": str(tif_path),
                    "mask_path": str(mask_path),
                    "geojson_path": str(geojson_path),
                    **(
                        {
                            "target_class_id": target.class_id,
                            "target_class_slug": target.class_slug,
                        }
                        if target is not None
                        else {}
                    ),
                }
            )
    return {
        "operation": TEST_SAMPLE_F1_OPERATION,
        "inference_backend": inference_backend,
        "run_root": str(run_dir / "scratch"),
        "report_path": str(run_dir / "scratch" / "report.json"),
        "metric_target": row.config.get("metric_target"),
        "class_key": sample.class_key if saved_evaluation else training_result.class_key,
        "class_name": sample.class_name if saved_evaluation else training_result.class_display_name,
        "task": training_result.task,
        "class_schema": list(training_result.class_schema or []),
        "object_types": list(training_result.class_schema or []),
        "source_model": training_result.model_name,
        "training_result_id": str(training_result.id),
        "test_sample_id": (None if plan is not None and plan.managed else str(sample.id)),
        "test_sample_revision": (
            None if plan is not None and plan.managed else sample.content_revision
        ),
        "managed_test_samples": bool(plan is not None and plan.managed),
        "test_samples": managed_scope if plan is not None and plan.managed else [],
        "managed_full_test_samples": (
            managed_full_scope if plan is not None and plan.managed else []
        ),
        "f1_aggregation": ("macro" if plan is not None and plan.managed else "foreground"),
        "tiles": tiles,
        "mlflow_tracking_uri": config.mlflow_tracking_uri,
        "mlflow_run_id": training_result.mlflow_run_id,
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_artifact_path": checkpoint_artifact_path,
        "checkpoint_f1_score": checkpoint_f1_score,
        "checkpoint_epoch": checkpoint_epoch,
        "external_model": external_model_payload(external_manifest),
        "imagery_type": imagery_type,
        "model_imagery_type": imagery_type,
        "input_channels": input_channels,
        "postprocess_config": row.config.get("inference_template_config") or {},
        "postprocess_profile": row.config.get("postprocess_profile"),
        "test_f1_evaluator_version": row.config.get("test_f1_evaluator_version"),
        "threshold": checkpoint_threshold,
        "tile_size": inference_tile_size,
        "context": inference_context,
        "stride": inference_stride,
        "batch_size": batch_size,
        "device": "cuda",
        **_geoalert_runtime_config(config, inference_backend),
    }


def _managed_targets_for_scope(
    plan: Any,
    requested_scope: list[dict[str, Any]],
) -> tuple[Any, ...]:
    full_scope = _training_result_test_scope(plan)
    requested_indices: list[int] = []
    for requested in requested_scope:
        matches = [
            index
            for index, current in enumerate(full_scope)
            if current == requested and index not in requested_indices
        ]
        if not matches:
            raise RuntimeError(
                "Выбранный класс или ревизия тестовой разметки управляемого датасета изменены."
            )
        requested_indices.append(matches[0])
    if requested_indices != sorted(requested_indices):
        raise RuntimeError("Порядок тестовых разметок управляемого датасета повреждён.")
    return tuple(plan.targets[index] for index in requested_indices)


def _geoalert_runtime_config(
    config: TrainingUIAPIConfig,
    inference_backend: str,
) -> dict[str, str]:
    if inference_backend != GEOALERT_INFERENCE_BACKEND:
        return {}
    return {
        "geoalert_python_path": str(config.geoalert_python_path),
        "geoalert_inference_root": str(config.geoalert_inference_root),
        "geoalert_model_repository": str(config.geoalert_model_repository),
        "geoalert_pipeline_root": str(config.geoalert_pipeline_root),
        "geoalert_triton_http_url": config.geoalert_triton_http_url,
        "geoalert_triton_python_site_packages": config.geoalert_triton_python_site_packages,
    }


def _dataset_config(
    session: Session,
    row: JobRow,
    config: TrainingUIAPIConfig,
    run_dir: Path,
) -> dict[str, Any]:
    result = _first_training_result(session, row)
    dataset_key = result.class_key if result is not None else row.dataset_name
    if row.custom_dataset_id is not None or dataset_key == CUSTOM_KEY:
        custom = row.custom_dataset or session.get(CustomDatasetRow, row.custom_dataset_id)
        if custom is None:
            raise RuntimeError(f"Custom dataset не найден: {row.custom_dataset_id}")
        return {
            "images_dir": str(
                row.config.get("dataset.images_dir")
                or imagery_images_dir(config.images_root, "kanopus")
            ),
            "scenes_file": custom.scenes_file.path,
            "annotation_file": custom.annotation_file.path,
        }

    dataset = find_managed_dataset(session, config, dataset_key)
    if dataset is None:
        dataset = next(
            (
                item
                for item in list_managed_datasets(session, config, include_custom=False)
                if item.name == row.dataset_name
            ),
            None,
        )
    if dataset is None or not dataset.source_available or dataset.images_dir is None:
        raise RuntimeError(f"Датасет не найден или неполный: {row.dataset_name}")
    if dataset.managed and dataset.annotations_dir is None:
        raise _ManagedDatasetNotReady(f"Управляемый датасет подготавливается: {row.dataset_name}")
    images_dir = str(
        row.config.get("dataset.images_dir") or dataset.images_dir or config.images_root
    )
    snapshot_dir = run_dir / "dataset_snapshot"
    if dataset.annotations_dir:
        source_dir = Path(dataset.annotations_dir).resolve()
        geojson_files = sorted(
            (
                path
                for path in source_dir.iterdir()
                if path.is_file() and path.suffix.casefold() == ".geojson"
            ),
            key=lambda path: path.name.casefold(),
        )
        annotation_files = [
            path for path in geojson_files if not is_per_image_footprint_name(path.name)
        ]
        if not annotation_files:
            raise RuntimeError(f"Per-image датасет пуст: {row.dataset_name}")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for source_file in geojson_files:
            shutil.copy2(source_file, snapshot_dir / source_file.name)
        manifest_file = source_dir / ".mlsystem2-dataset.json"
        if manifest_file.is_file():
            shutil.copy2(manifest_file, snapshot_dir / manifest_file.name)
        return {
            "images_dir": images_dir,
            "annotations_dir": str(snapshot_dir),
        }
    if not dataset.scenes_file or not dataset.annotation_file:
        raise RuntimeError(f"Legacy-датасет не найден или неполный: {row.dataset_name}")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    scenes_file = _snapshot_dataset_file(Path(dataset.scenes_file), snapshot_dir)
    annotation_file = _snapshot_dataset_file(Path(dataset.annotation_file), snapshot_dir)
    hard_negative_file = (
        _snapshot_dataset_file(Path(dataset.hard_negative_annotation_file), snapshot_dir)
        if dataset.hard_negative_annotation_file
        else None
    )
    return {
        "images_dir": images_dir,
        "scenes_file": str(scenes_file),
        "annotation_file": str(annotation_file),
        **(
            {"hard_negative_annotation_file": str(hard_negative_file)}
            if hard_negative_file is not None
            else {}
        ),
    }


def _snapshot_dataset_file(source: Path, snapshot_dir: Path) -> Path:
    resolved = source.resolve(strict=True)
    destination = snapshot_dir / resolved.name
    shutil.copy2(resolved, destination)
    return destination


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)


def _write_run_script(
    row: JobRow,
    config: TrainingUIAPIConfig,
    run_dir: Path,
    config_path: Path,
) -> Path:
    script_path = run_dir / "run_training.sh"
    log_path = run_dir / "train.log"
    exit_code_path = run_dir / "exit_code"
    mlflow_run_id_path = run_dir / MLFLOW_RUN_ID_FILE
    command = [
        sys.executable,
        "-m",
        "mlsystem2.cli.train",
        "--settings",
        str(config.training_settings_path),
        "--run",
        str(config_path),
    ]
    if row.mlflow_run_name:
        command.extend(["--run-name", row.mlflow_run_name])
    quoted_command = " ".join(shlex.quote(item) for item in command)
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -o pipefail",
                f"cd {shlex.quote(str(config.project_root))}",
                f"export MLSYSTEM2_MLFLOW_RUN_ID_FILE={shlex.quote(str(mlflow_run_id_path))}",
                "export MLSYSTEM2_TRAINING_CONTROL_DIR="
                f"{shlex.quote(str(run_dir / JOB_CONTROL_DIR))}",
                f"export MLSYSTEM2_TORCH_NUM_THREADS={config.training_torch_num_threads}",
                "export MLSYSTEM2_TORCH_NUM_INTEROP_THREADS="
                f"{config.training_torch_num_interop_threads}",
                f"export OMP_NUM_THREADS={config.training_torch_num_threads}",
                f"export MKL_NUM_THREADS={config.training_torch_num_threads}",
                "export OPENBLAS_NUM_THREADS=1",
                "export NUMEXPR_NUM_THREADS=1",
                "command -v renice >/dev/null 2>&1 && "
                f"renice -n {config.training_process_nice} -p $$ >/dev/null 2>&1 || true",
                "command -v ionice >/dev/null 2>&1 && "
                f"ionice -c 2 -n {config.training_process_io_priority} -p $$ "
                ">/dev/null 2>&1 || true",
                f"{quoted_command} > {shlex.quote(str(log_path))} 2>&1",
                "code=$?",
                f"printf '%s\\n' \"$code\" > {shlex.quote(str(exit_code_path))}",
                'exit "$code"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o750)
    return script_path


def _write_pseudo_run_script(
    config: TrainingUIAPIConfig,
    run_dir: Path,
    config_path: Path,
    *,
    test_f1: bool = False,
    inference_backend: str = "",
) -> Path:
    script_path = run_dir / ("run_test_f1.sh" if test_f1 else "run_pseudo_markup.sh")
    log_path = run_dir / "logs" / ("test_sample_f1.log" if test_f1 else "pseudo_markup.log")
    exit_code_path = run_dir / "exit_code"
    command = [
        sys.executable,
        "-m",
        (
            "mlsystem2.training_ui_api._geoalert_runner"
            if inference_backend == GEOALERT_INFERENCE_BACKEND
            else "mlsystem2.training_ui_api._pseudo_runner"
        ),
        "--config",
        str(config_path),
    ]
    quoted_command = " ".join(shlex.quote(item) for item in command)
    script_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -o pipefail",
                f"cd {shlex.quote(str(config.project_root))}",
                f"{quoted_command} > {shlex.quote(str(log_path))} 2>&1",
                "code=$?",
                f"printf '%s\\n' \"$code\" > {shlex.quote(str(exit_code_path))}",
                'exit "$code"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o750)
    return script_path


def _finish_training_job(
    session: Session,
    row: JobRow,
    config: TrainingUIAPIConfig,
    *,
    succeeded: bool,
) -> None:
    row.status = JobStatus.COMPLETED.value if succeeded else JobStatus.FAILED.value
    row.error = None if succeeded else _training_job_error(row)
    row.finished_at = _now()
    row.process_pid = None
    mlflow_run_id = _extract_mlflow_run_id(row)
    mlflow_experiment_id = (
        _resolve_mlflow_experiment_id(row, config) if mlflow_run_id else row.mlflow_experiment_id
    )
    best_checkpoint = (
        _best_training_checkpoint(config, mlflow_run_id) if succeeded and mlflow_run_id else None
    )
    checkpoint_metadata = _local_training_checkpoint_metadata(row) if succeeded else {}
    training_results = _training_results(session, row)
    for result in training_results:
        result.status = ResultStatus.OK.value if succeeded else ResultStatus.ERROR.value
        result.trained_at = row.finished_at if succeeded else result.trained_at
        result.mlflow_run_id = mlflow_run_id or result.mlflow_run_id
        if best_checkpoint is not None:
            result.f1_score = best_checkpoint.f1_score
            result.epoch = best_checkpoint.epoch
        task = checkpoint_metadata.get("task")
        class_schema = checkpoint_metadata.get("class_schema")
        if task in {"binary", "multiclass"}:
            result.task = str(task)
        if isinstance(class_schema, list):
            result.class_schema = [item for item in class_schema if isinstance(item, dict)]
        if checkpoint_metadata:
            result.training_metrics = {
                key: value
                for key, value in checkpoint_metadata.items()
                if key.startswith("val_")
                or key in {"quality_metric", "confidence_threshold", "epoch"}
            }
        result.mlflow_run_url = (
            _mlflow_run_url(config, mlflow_experiment_id, mlflow_run_id)
            if mlflow_run_id
            else result.mlflow_run_url
        )
        result.updated_at = _now()
    session.flush()
    if succeeded:
        affected_class_keys: set[str] = set()
        for result in training_results:
            class_row = dataset_class_row(session, result.dataset_key or result.class_key)
            if class_row is not None:
                affected_class_keys.add(class_row.key)
            try:
                queue_training_result_test_f1(
                    session,
                    result,
                    config,
                    source=JobSource(result.source),
                )
            except Exception:  # noqa: BLE001
                LOGGER.exception(
                    "Не удалось поставить автоматический расчёт тестового F1 для сети %s",
                    result.id,
                )
        if affected_class_keys:
            try:
                reconcile_test_sample_evaluations(
                    session,
                    config,
                    class_keys=affected_class_keys,
                )
            except Exception:  # noqa: BLE001
                LOGGER.exception(
                    "Не удалось сверить прямые оценки тестовых разметок после обучения сети"
                )
        _queue_post_training_inference(session, row, training_results, config)
        session.flush()
    LOGGER.info("Finished training job %s with status %s", row.id, row.status)


def _queue_post_training_inference(
    session: Session,
    row: JobRow,
    training_results: list[TrainingResultRow],
    config: TrainingUIAPIConfig,
) -> None:
    job_config = dict(row.config or {})
    if not bool(job_config.get(POST_TRAINING_INFERENCE_CONFIG_KEY, False)):
        return
    if job_config.get(POST_TRAINING_INFERENCE_JOB_IDS_CONFIG_KEY):
        return

    from ._service import create_pseudo_markup_job

    custom = (
        row.custom_dataset or session.get(CustomDatasetRow, row.custom_dataset_id)
        if row.custom_dataset_id is not None
        else None
    )
    queued_job_ids: list[str] = []
    for result in training_results:
        try:
            scenes_name: str | None = None
            scenes_content_type: str | None = None
            scenes_bytes: bytes | None = None
            inference_dataset_key: str | None = result.dataset_key or row.dataset_key
            if custom is not None:
                scenes_name = custom.scenes_file.original_name
                scenes_content_type = custom.scenes_file.content_type
                scenes_bytes = Path(custom.scenes_file.path).read_bytes()
                inference_dataset_key = None
            with session.begin_nested():
                detail = create_pseudo_markup_job(
                    session,
                    class_key=result.class_key,
                    dataset_key=inference_dataset_key,
                    image_folder_key=None,
                    training_result_id=result.id,
                    scenes_name=scenes_name,
                    scenes_content_type=scenes_content_type,
                    scenes_bytes=scenes_bytes,
                    config=config,
                    secondary_priority=bool(job_config.get(SECONDARY_PRIORITY_CONFIG_KEY, False)),
                )
            queued_job_ids.append(str(detail.id))
        except Exception:  # noqa: BLE001
            LOGGER.exception(
                "Не удалось поставить инференс после обучения сети %s по датасету %s",
                result.id,
                result.dataset_key,
            )
    if queued_job_ids:
        row.config = {
            **job_config,
            POST_TRAINING_INFERENCE_JOB_IDS_CONFIG_KEY: queued_job_ids,
        }


def _local_training_checkpoint_metadata(row: JobRow) -> dict[str, Any]:
    if not row.tmp_path:
        return {}
    path = Path(row.tmp_path) / "scratch" / "checkpoints" / "best.pt"
    if not path.is_file():
        return {}
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        LOGGER.exception("Не удалось прочитать metadata локального checkpoint %s", path)
        return {}
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    return dict(metadata) if isinstance(metadata, dict) else {}


def _sync_training_run_id(session: Session, row: JobRow, config: TrainingUIAPIConfig) -> None:
    mlflow_run_id = _extract_mlflow_run_id(row)
    if not mlflow_run_id:
        return
    mlflow_experiment_id = _resolve_mlflow_experiment_id(row, config)
    changed = False
    for result in _training_results(session, row):
        if result.mlflow_run_id != mlflow_run_id:
            result.mlflow_run_id = mlflow_run_id
            changed = True
        run_url = _mlflow_run_url(config, mlflow_experiment_id, mlflow_run_id)
        if result.mlflow_run_url != run_url:
            result.mlflow_run_url = run_url
            changed = True
        if changed:
            result.updated_at = _now()
    if changed:
        session.flush()


def _finish_inference_job(
    session: Session,
    row: JobRow,
    config: TrainingUIAPIConfig,
    *,
    succeeded: bool,
) -> None:
    if _is_test_sample_f1_job(row):
        _finish_test_sample_f1_job(session, row, config, succeeded=succeeded)
        return
    if _is_pseudolabel_aoi_job(row):
        _finish_pseudolabel_aoi_job(session, row, config, succeeded=succeeded)
        return
    if _is_dataset_editor_pseudo_job(row):
        _finish_dataset_editor_pseudo_job(session, row, config, succeeded=succeeded)
        return
    row.finished_at = _now()
    row.process_pid = None
    output_path = _pseudo_output_path(row)
    report = _pseudo_report(row)
    report_allows_success = _pseudo_report_allows_success(report)
    succeeded = succeeded and report_allows_success
    row.status = JobStatus.COMPLETED.value if succeeded else JobStatus.FAILED.value
    row.error = None if succeeded else _pseudo_markup_error(report, row)
    has_geojson = output_path is not None and output_path.is_file()
    pseudo_results = _pseudo_markup_results(session, row)
    if succeeded and has_geojson:
        file_row = _store_generated_geojson(
            session,
            output_path,
            config,
            original_name=_pseudo_geojson_download_name(row, pseudo_results, row.finished_at),
            object_count=_pseudo_geojson_object_count(output_path, report),
        )
    else:
        file_row = None
    for result in pseudo_results:
        result.status = ResultStatus.OK.value if file_row is not None else ResultStatus.ERROR.value
        result.geojson_file_id = file_row.id if file_row is not None else result.geojson_file_id
        if file_row is not None:
            result.geojson_file = file_row
        if result.image_count is None:
            images_root = Path(str(row.config.get("images_root") or config.images_root))
            result.image_count = _stored_scenes_image_count(
                result.scenes_file,
                images_root,
            )
        result.updated_at = _now()
    if succeeded and file_row is None:
        row.status = JobStatus.FAILED.value
        row.error = _pseudo_markup_error(report, row)
    session.flush()
    evaluated_dataset_keys: set[str] = set()
    for result in pseudo_results:
        if result.status != ResultStatus.OK.value:
            continue
        try:
            evaluate_test_samples_for_pseudo_markup(session, result, config)
            if result.class_key:
                evaluated_dataset_keys.add(result.class_key)
        except Exception:  # noqa: BLE001
            LOGGER.exception(
                "Не удалось автоматически пересчитать тестовые разметки для результата %s",
                result.id,
            )
    if evaluated_dataset_keys:
        try:
            reconcile_training_result_test_f1(
                session,
                config,
                dataset_keys=evaluated_dataset_keys,
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Не удалось восстановить тестовый F1 после новой псевдоразметки")
    _cleanup_inference_scratch(row)
    LOGGER.info(
        "Finished pseudo-markup job %s with status %s report=%s",
        row.id,
        row.status,
        report,
    )


def _finish_dataset_editor_pseudo_job(
    session: Session,
    row: JobRow,
    config: TrainingUIAPIConfig,
    *,
    succeeded: bool,
) -> None:
    row.finished_at = _now()
    row.process_pid = None
    output_path = _pseudo_output_path(row)
    report = _pseudo_report(row)
    succeeded = succeeded and _pseudo_report_allows_success(report)
    file_row = None
    if succeeded and output_path is not None and output_path.is_file():
        file_row = _store_generated_geojson(
            session,
            output_path,
            config,
            original_name=f"dataset_editor_pseudo_{row.id}.geojson",
            object_count=_pseudo_geojson_object_count(output_path, report),
            kind=StoredFileKind.PSEUDOLABEL_GEOJSON,
        )
    succeeded = file_row is not None
    row.status = JobStatus.COMPLETED.value if succeeded else JobStatus.FAILED.value
    state = dict((row.config or {}).get("editor_pseudo") or {})
    state["result_file_id"] = str(file_row.id) if file_row is not None else None
    state["error"] = None if succeeded else _dataset_editor_pseudo_error(report, row)
    row.config = {**(row.config or {}), "editor_pseudo": state}
    row.error = None if succeeded else str(state["error"])
    session.flush()
    _cleanup_inference_scratch(row)
    LOGGER.info(
        "Завершена псевдоразметка снимка редактора %s со статусом %s",
        row.id,
        row.status,
    )


def _dataset_editor_pseudo_error(report: dict[str, Any] | None, row: JobRow) -> str:
    if report is not None:
        if report.get("error"):
            return f"Ошибка инференса: {report['error']}"
        if report.get("failures"):
            return f"Ошибка инференса: {report['failures']}"
    if row.tmp_path:
        path = Path(row.tmp_path) / "worker_error.txt"
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            value = ""
        if value:
            return value[:4000]
    return "Не удалось получить псевдоразметку снимка."


def _finish_pseudolabel_aoi_job(
    session: Session,
    row: JobRow,
    config: TrainingUIAPIConfig,
    *,
    succeeded: bool,
) -> None:
    """Sohranit rezultat i publichnoe sostoyanie AOI job."""

    row.finished_at = _now()
    row.process_pid = None
    output_path = _pseudo_output_path(row)
    report = _pseudo_report(row)
    succeeded = succeeded and _pseudolabel_aoi_report_allows_success(report)
    has_geojson = output_path is not None and output_path.is_file()
    file_row = None
    if succeeded and has_geojson:
        file_row = _store_generated_geojson(
            session,
            output_path,
            config,
            original_name=f"pseudolabel_{row.id}.geojson",
            object_count=_pseudo_geojson_object_count(output_path, report),
            kind=StoredFileKind.PSEUDOLABEL_GEOJSON,
        )
    succeeded = file_row is not None
    row.status = JobStatus.COMPLETED.value if succeeded else JobStatus.FAILED.value
    state = dict((row.config or {}).get("pseudolabel") or {})
    if report is not None:
        state["source_image_ids"] = _string_values(report.get("source_image_ids"))
        state["coverage_percent"] = _optional_number(report.get("coverage_percent"))
        state["warnings"] = _string_values(report.get("warnings"))
        state["source_attributions"] = _string_values(
            report.get("source_attributions")
        ) or state.get("source_attributions", [])
        if isinstance(report.get("performance"), dict):
            state["performance"] = report["performance"]
    state["result_file_id"] = str(file_row.id) if file_row is not None else None
    if succeeded:
        state["error"] = None
    elif not isinstance(state.get("error"), dict):
        report_error = report.get("error") if report is not None else None
        if isinstance(report_error, dict):
            state["error"] = {
                "code": str(report_error.get("code") or "INFERENCE_FAILED"),
                "message": str(
                    report_error.get("message")
                    or "Не удалось выполнить распознавание зоны интереса."
                ),
                "details": (
                    report_error.get("details")
                    if isinstance(report_error.get("details"), dict)
                    else {}
                ),
            }
        else:
            state["error"] = {
                "code": "INFERENCE_FAILED",
                "message": "Не удалось выполнить распознавание зоны интереса.",
                "details": {},
            }
    row.config = {**(row.config or {}), "pseudolabel": state}
    session.flush()
    _cleanup_inference_scratch(row)
    LOGGER.info(
        "Завершено задание распознавания AOI %s со статусом %s, отчёт=%s",
        row.id,
        row.status,
        report,
    )


def _finish_test_sample_f1_job(
    session: Session,
    row: JobRow,
    config: TrainingUIAPIConfig,
    *,
    succeeded: bool,
) -> None:
    if _is_saved_test_sample_evaluation_job(row):
        _finish_saved_test_sample_evaluation_job(
            session,
            row,
            config,
            succeeded=succeeded,
        )
        return
    row.finished_at = _now()
    row.process_pid = None
    report = _pseudo_report(row)
    report_ok = _test_sample_f1_report_allows_success(report)
    succeeded = succeeded and report_ok
    row.status = JobStatus.COMPLETED.value if succeeded else JobStatus.FAILED.value
    metric = _test_sample_f1_metric(session, row)
    if metric is not None:
        training_result = _test_sample_f1_training_result(session, row)
        managed_evaluation = bool((row.config or {}).get("managed_test_samples"))
        partial_managed_evaluation = False
        managed_full_scope: list[dict[str, Any]] = []
        managed_selected_slugs: list[str] = []
        if managed_evaluation and training_result is not None:
            plan = _training_result_test_plan(session, training_result)
            managed_full_scope = list(
                (row.config or {}).get("managed_full_test_samples")
                or (row.config or {}).get("test_samples")
                or []
            )
            managed_scope = list((row.config or {}).get("test_samples") or [])
            try:
                selected_targets = _managed_targets_for_scope(plan, managed_scope)
            except RuntimeError:
                selected_targets = ()
            managed_selected_slugs = [
                str(target.class_slug)
                for target in selected_targets
                if target.class_slug is not None
            ]
            partial_managed_evaluation = bool(
                managed_scope != managed_full_scope and selected_targets
            )
            still_current = bool(
                plan.managed
                and plan.error is None
                and managed_full_scope == _training_result_test_scope(plan)
                and selected_targets
                and metric.sample_id is None
                and metric.sample_revision is None
                and metric.job_id == row.id
            )
        else:
            sample = (
                session.get(TestSampleRow, metric.sample_id)
                if metric.sample_id is not None
                else None
            )
            primary_sample = (
                primary_test_sample(session, training_result.class_key)
                if training_result is not None
                else None
            )
            compatibility_error = (
                test_sample_model_compatibility_error(session, sample, training_result)
                if sample is not None and training_result is not None
                else "Сеть или тестовая разметка больше не существуют."
            )
            expected_revision = int(row.config.get("test_sample_revision") or 0)
            still_current = bool(
                sample
                and training_result
                and primary_sample is not None
                and primary_sample.id == sample.id
                and compatibility_error is None
                and sample.content_revision == expected_revision
                and metric.sample_revision == expected_revision
                and metric.job_id == row.id
            )
        merged_metrics: dict[str, Any] | None = None
        if succeeded and still_current and report is not None and partial_managed_evaluation:
            try:
                merged_metrics = _merge_partial_managed_metrics(
                    metric.metrics,
                    report.get("metrics"),
                    full_scope=managed_full_scope,
                    selected_slugs=managed_selected_slugs,
                )
            except RuntimeError as exc:
                succeeded = False
                row.status = JobStatus.FAILED.value
                row.error = str(exc)
        if succeeded and still_current and report is not None:
            if merged_metrics is not None:
                pixel_values = _managed_metric_values(merged_metrics, "pixel")
                object_values = _managed_metric_values(merged_metrics, "objects")
            else:
                pixel_values = _report_metric_values(
                    report,
                    section="pixel" if managed_evaluation else None,
                )
                object_values = _report_metric_values(
                    report,
                    prefix="object_",
                    section="objects" if managed_evaluation else None,
                )
            (
                metric.precision,
                metric.recall,
                metric.f1,
                metric.true_positive,
                metric.false_positive,
                metric.false_negative,
            ) = pixel_values
            (
                metric.object_precision,
                metric.object_recall,
                metric.object_f1,
                metric.object_true_positive,
                metric.object_false_positive,
                metric.object_false_negative,
            ) = object_values
            report_threshold = report.get("threshold")
            metric.threshold = float(report_threshold) if report_threshold is not None else None
            metric.metrics = merged_metrics or dict(report.get("metrics") or {})
            metric.status = "current"
            metric.evaluated_at = row.finished_at
            metric.error = None
        elif not still_current:
            metric.status = "stale" if metric.f1 is not None else "unavailable"
            metric.error = "Основная тестовая разметка изменилась во время расчёта."
            metric.job_id = None
        else:
            metric.status = "stale" if metric.f1 is not None else "error"
            metric.error = _test_sample_f1_error(report, row)
        metric.updated_at = _now()
    session.flush()
    _cleanup_inference_scratch(row)
    LOGGER.info(
        "Finished test-sample F1 job %s with status %s report=%s",
        row.id,
        row.status,
        report,
    )


def _finish_saved_test_sample_evaluation_job(
    session: Session,
    row: JobRow,
    config: TrainingUIAPIConfig,
    *,
    succeeded: bool,
) -> None:
    row.finished_at = _now()
    row.process_pid = None
    report = _pseudo_report(row)
    succeeded = succeeded and _test_sample_f1_report_allows_success(report)
    row.status = JobStatus.COMPLETED.value if succeeded else JobStatus.FAILED.value
    sample = _saved_test_sample_evaluation(session, row)
    training_result = _test_sample_f1_training_result(session, row)
    expected_revision = int((row.config or {}).get("test_sample_revision") or 0)
    expected_indices = {
        int(value) for value in (row.config or {}).get("test_sample_tile_indices") or []
    }
    current_primary = (
        current_primary_training_result(session, sample.class_key) if sample is not None else None
    )
    compatibility_error = (
        test_sample_model_compatibility_error(session, sample, training_result)
        if sample is not None and training_result is not None
        else "Основная сеть или тестовая разметка больше не существуют."
    )
    still_current = bool(
        sample is not None
        and training_result is not None
        and sample.evaluation_job_id == row.id
        and sample.content_revision == expected_revision
        and {tile.tile_index for tile in sample.tiles if tile.enabled} == expected_indices
        and current_primary is not None
        and current_primary.id == training_result.id
        and compatibility_error is None
    )
    reconcile_sample_id: uuid.UUID | None = None
    if sample is not None and sample.evaluation_job_id == row.id:
        if succeeded and still_current and report is not None and training_result is not None:
            pixel = _report_metric_values(report)
            objects = _report_metric_values(report, prefix="object_")
            (
                sample.pixel_precision,
                sample.pixel_recall,
                sample.pixel_f1,
                sample.pixel_true_positive,
                sample.pixel_false_positive,
                sample.pixel_false_negative,
            ) = pixel
            (
                sample.object_precision,
                sample.object_recall,
                sample.object_f1,
                sample.object_true_positive,
                sample.object_false_positive,
                sample.object_false_negative,
            ) = objects
            sample.evaluation_metrics = dict(report.get("metrics") or {})
            sample.evaluation_training_result_id = training_result.id
            sample.evaluation_model_name = training_result.model_name
            sample.evaluated_revision = sample.content_revision
            sample.evaluation_inference_template_id = _optional_uuid(
                (row.config or {}).get("inference_template_id")
            )
            sample.evaluation_inference_template_version = _optional_scalar_int(
                (row.config or {}).get("inference_template_version")
            )
            sample.evaluation_inference_config_hash = (
                str((row.config or {}).get("inference_config_hash") or "") or None
            )
            sample.evaluation_evaluator_version = _optional_scalar_int(
                (row.config or {}).get("test_f1_evaluator_version")
            )
            sample.evaluation_threshold = _optional_number(report.get("threshold"))
            sample.metric_status = "current"
            sample.evaluated_at = row.finished_at
            sample.evaluation_error = None
            row.error = None
        elif not still_current:
            sample.evaluation_job_id = None
            sample.metric_status = (
                "stale"
                if sample.pixel_f1 is not None and sample.object_f1 is not None
                else "unavailable"
            )
            sample.evaluation_error = (
                "Основная сеть или состав тестовой разметки изменились во время расчёта."
            )
            reconcile_sample_id = sample.id
        else:
            sample.metric_status = "error"
            sample.evaluation_error = _test_sample_f1_error(report, row)
            row.error = sample.evaluation_error
        sample.updated_at = _now()
    session.flush()
    if reconcile_sample_id is not None:
        reconcile_test_sample_evaluations(
            session,
            config,
            sample_ids={reconcile_sample_id},
        )
    _cleanup_inference_scratch(row)
    LOGGER.info(
        "Finished saved test-sample evaluation job %s with status %s report=%s",
        row.id,
        row.status,
        report,
    )


def _report_metric_values(
    report: dict[str, Any],
    *,
    prefix: str = "",
    section: str | None = None,
) -> tuple[float, float, float, int, int, int]:
    true_positive = int(report.get(f"{prefix}true_positive") or 0)
    false_positive = int(report.get(f"{prefix}false_positive") or 0)
    false_negative = int(report.get(f"{prefix}false_negative") or 0)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    denominator = precision + recall
    f1 = 2.0 * precision * recall / denominator if denominator else 0.0
    if section is not None:
        metrics = report.get("metrics")
        section_metrics = metrics.get(section) if isinstance(metrics, dict) else None
        macro = section_metrics.get("macro") if isinstance(section_metrics, dict) else None
        if isinstance(macro, dict):
            try:
                precision = float(macro["precision"])
                recall = float(macro["recall"])
                f1 = float(macro["f1"])
            except (KeyError, TypeError, ValueError):
                pass
    return (
        precision,
        recall,
        f1,
        true_positive,
        false_positive,
        false_negative,
    )


def _merge_partial_managed_metrics(
    stored_metrics: Any,
    report_metrics: Any,
    *,
    full_scope: list[dict[str, Any]],
    selected_slugs: list[str],
) -> dict[str, Any]:
    stored = copy.deepcopy(stored_metrics) if isinstance(stored_metrics, dict) else {}
    incoming = copy.deepcopy(report_metrics) if isinstance(report_metrics, dict) else {}
    full_slugs = [
        str(item.get("class_slug")) for item in full_scope if item.get("class_slug") is not None
    ]
    selected = set(selected_slugs)
    if not full_slugs or not selected or not selected.issubset(full_slugs):
        raise RuntimeError("Не удалось определить классы частичного пересчёта F1.")

    merged = stored
    for section_name in ("pixel", "objects"):
        stored_section = stored.get(section_name)
        incoming_section = incoming.get(section_name)
        stored_per_class = (
            stored_section.get("per_class") if isinstance(stored_section, dict) else None
        )
        incoming_per_class = (
            incoming_section.get("per_class") if isinstance(incoming_section, dict) else None
        )
        if not isinstance(stored_per_class, dict) or not isinstance(incoming_per_class, dict):
            raise RuntimeError("Сохранённые метрики не позволяют выполнить частичный пересчёт.")
        merged_per_class: dict[str, dict[str, Any]] = {}
        for slug in full_slugs:
            source = incoming_per_class if slug in selected else stored_per_class
            value = source.get(slug)
            if not isinstance(value, dict):
                raise RuntimeError(f"В результате отсутствует метрика класса {slug}.")
            merged_per_class[slug] = _normalized_metric_payload(value)
        summary = _managed_section_summary(merged_per_class)
        merged[section_name] = {
            **(stored_section if isinstance(stored_section, dict) else {}),
            "per_class": merged_per_class,
            **summary,
        }

    warnings = [
        str(value)
        for source in (stored.get("warnings"), incoming.get("warnings"))
        for value in (source if isinstance(source, list) else [])
    ]
    merged["warnings"] = list(dict.fromkeys(warnings))
    merged["aggregation"] = "macro"
    merged["aggregation_label"] = "Среднее F1 по основным выборкам классов"
    merged["test_samples"] = full_scope
    return merged


def _normalized_metric_payload(value: dict[str, Any]) -> dict[str, Any]:
    try:
        true_positive = int(value["true_positive"])
        false_positive = int(value["false_positive"])
        false_negative = int(value["false_negative"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("В поклассовой метрике отсутствуют TP/FP/FN.") from exc
    calculated = _metric_payload_from_counts(
        true_positive,
        false_positive,
        false_negative,
    )
    return {**value, **calculated}


def _metric_payload_from_counts(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> dict[str, float | int]:
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    iou_denominator = true_positive + false_positive + false_negative
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / f1_denominator if f1_denominator else 0.0,
        "iou": true_positive / iou_denominator if iou_denominator else 0.0,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def _managed_section_summary(
    per_class: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    values = list(per_class.values())
    macro = {
        key: sum(float(value[key]) for value in values) / len(values)
        for key in ("precision", "recall", "f1", "iou")
    }
    micro = _metric_payload_from_counts(
        sum(int(value["true_positive"]) for value in values),
        sum(int(value["false_positive"]) for value in values),
        sum(int(value["false_negative"]) for value in values),
    )
    return {"macro": macro, "micro": micro, "foreground": dict(micro)}


def _managed_metric_values(
    metrics: dict[str, Any],
    section_name: str,
) -> tuple[float, float, float, int, int, int]:
    section = metrics.get(section_name)
    macro = section.get("macro") if isinstance(section, dict) else None
    micro = section.get("micro") if isinstance(section, dict) else None
    if not isinstance(macro, dict) or not isinstance(micro, dict):
        raise RuntimeError("Не удалось собрать итоговую метрику управляемого датасета.")
    return (
        float(macro["precision"]),
        float(macro["recall"]),
        float(macro["f1"]),
        int(micro["true_positive"]),
        int(micro["false_positive"]),
        int(micro["false_negative"]),
    )


def _saved_test_sample_evaluation(
    session: Session,
    row: JobRow,
) -> TestSampleRow | None:
    sample_id = _optional_uuid((row.config or {}).get("test_sample_id"))
    return session.get(TestSampleRow, sample_id) if sample_id is not None else None


def _test_sample_f1_training_result(
    session: Session,
    row: JobRow,
) -> TrainingResultRow | None:
    result_id = _optional_uuid((row.config or {}).get("training_result_id"))
    return session.get(TrainingResultRow, result_id) if result_id is not None else None


def _optional_uuid(value: Any) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_scalar_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _test_sample_f1_metric(
    session: Session,
    row: JobRow,
) -> TrainingResultTestMetricRow | None:
    return session.scalar(
        select(TrainingResultTestMetricRow).where(TrainingResultTestMetricRow.job_id == row.id)
    )


def _test_sample_f1_report_allows_success(report: dict[str, Any] | None) -> bool:
    if report is None or report.get("status") != "ok":
        return False
    try:
        return int(report.get("processed") or 0) > 0
    except (TypeError, ValueError):
        return False


def _test_sample_f1_error(report: dict[str, Any] | None, row: JobRow) -> str:
    if report is not None:
        error = report.get("error")
        if error:
            return f"Не удалось рассчитать тестовый F1: {error}"
        failures = report.get("failures")
        if failures:
            return f"Не удалось рассчитать тестовый F1: {failures}"
    if row.tmp_path:
        path = Path(row.tmp_path) / "worker_error.txt"
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            text = ""
        if text:
            return text[:4000]
    return "Не удалось рассчитать F1 на основной тестовой разметке."


def _pseudo_output_path(row: JobRow) -> Path | None:
    if row.tmp_path is None:
        return None
    return Path(row.tmp_path) / "scratch" / "pseudo_markup.geojson"


def _pseudo_report(row: JobRow) -> dict[str, Any] | None:
    if row.tmp_path is None:
        return None
    path = Path(row.tmp_path) / "scratch" / "report.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _pseudo_report_allows_success(report: dict[str, Any] | None) -> bool:
    if report is None:
        return False
    if report.get("status") != "ok":
        return False
    try:
        processed = int(report.get("processed") or 0)
        failed = int(report.get("failed") or 0)
        missing = int(report.get("missing_images") or 0)
    except (TypeError, ValueError):
        return False
    if processed <= 0 or failed != 0 or missing != 0:
        return False
    expected = report.get("unique_image_count")
    if expected is None:
        expected = report.get("input_scene_count")
    if expected is None:
        return True
    try:
        return processed == int(expected)
    except (TypeError, ValueError):
        return False


def _pseudo_markup_error(report: dict[str, Any] | None, row: JobRow) -> str:
    if report is not None:

        def report_count(key: str, default: int = 0) -> int:
            try:
                return int(report.get(key) or default)
            except (TypeError, ValueError):
                return default

        error = report.get("error")
        if error:
            return f"Ошибка инференса: {error}"[:4000]
        failures = report.get("failures")
        if failures:
            processed = report_count("processed")
            expected = report_count(
                "unique_image_count",
                report_count("input_scene_count", processed),
            )
            first = failures[0] if isinstance(failures, list) else failures
            if isinstance(first, dict):
                first = first.get("error") or first
            return (
                f"Инференс обработал только {processed} из {expected} снимков. "
                f"Первая ошибка: {first}"
            )[:4000]
        if report.get("status") == "partial":
            processed = report_count("processed")
            expected = report_count(
                "unique_image_count",
                report_count("input_scene_count", processed),
            )
            return f"Инференс обработал только {processed} из {expected} снимков."
    if row.tmp_path:
        path = Path(row.tmp_path) / "worker_error.txt"
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError:
            text = ""
        if text:
            return text[:4000]
    return "Не удалось получить полную псевдоразметку."


def _pseudolabel_aoi_report_allows_success(report: dict[str, Any] | None) -> bool:
    """Не публиковать геометрически неполный результат AOI."""

    if not _pseudo_report_allows_success(report) or report is None:
        return False
    try:
        processed = int(report.get("processed") or 0)
        selected = int(report.get("unique_image_count") or 0)
        failed = int(report.get("failed") or 0)
    except (TypeError, ValueError):
        return False
    return selected > 0 and processed == selected and failed == 0


def _store_generated_geojson(
    session: Session,
    source_path: Path,
    config: TrainingUIAPIConfig,
    *,
    original_name: str,
    object_count: int | None,
    kind: StoredFileKind = StoredFileKind.PSEUDO_MARKUP_GEOJSON,
) -> StoredFileRow:
    file_id = uuid.uuid4()
    target_dir = config.stored_files_root / kind.value
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{file_id}.geojson"
    tmp_path = target_dir / f".{file_id}.geojson.tmp"
    try:
        shutil.copy2(source_path, tmp_path)
        tmp_path.replace(target_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        target_path.unlink(missing_ok=True)
        raise
    row = StoredFileRow(
        id=file_id,
        kind=kind.value,
        original_name=original_name,
        content_type="application/geo+json",
        path=str(target_path),
        size_bytes=target_path.stat().st_size,
        object_count=object_count,
    )
    session.add(row)
    try:
        session.flush()
    except Exception:
        target_path.unlink(missing_ok=True)
        raise
    return row


def _pseudo_geojson_object_count(source_path: Path, report: dict[str, Any] | None) -> int | None:
    count = _report_feature_count(report)
    return count if count is not None else _geojson_feature_count(source_path)


def _report_feature_count(report: dict[str, Any] | None) -> int | None:
    if report is None:
        return None
    try:
        count = int(report.get("feature_count"))
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _geojson_feature_count(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    features = payload.get("features")
    return len(features) if isinstance(features, list) else None


def _stored_scenes_image_count(
    row: StoredFileRow | None,
    images_root: Path,
) -> int | None:
    if row is None:
        return None
    return count_scenes_file_images(Path(row.path), images_root)


def _cleanup_inference_scratch(row: JobRow) -> None:
    if row.tmp_path is None:
        return
    shutil.rmtree(Path(row.tmp_path) / "scratch", ignore_errors=True)


def _pseudo_geojson_download_name(
    row: JobRow,
    results: list[PseudoMarkupResultRow],
    created_at: datetime | None,
) -> str:
    result = results[0] if results else None
    dataset_name = (
        result.training_result.class_display_name
        if result is not None and result.training_result is not None
        else row.training_dataset_name
        or (result.class_key if result is not None else None)
        or row.dataset_name
    )
    model_name = (
        result.training_result.model_name
        if result is not None and result.training_result is not None
        else row.model_name
    )
    timestamp = (created_at or _now()).strftime("%H_%M_%d_%m")
    return f"{_filename_part(dataset_name)}_{_filename_part(model_name)}_{timestamp}.geojson"


def _filename_part(value: str) -> str:
    normalized = re.sub(r"[\\/]+", "_", str(value).strip())
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unknown"


def _best_training_checkpoint(
    config: TrainingUIAPIConfig,
    mlflow_run_id: str,
) -> MLflowBestCheckpoint | None:
    try:
        return get_best_training_checkpoint(config.mlflow_tracking_uri, mlflow_run_id)
    except MLflowAdapterError:
        LOGGER.warning(
            "Failed to read best MLflow checkpoint summary for run %s",
            mlflow_run_id,
            exc_info=True,
        )
        return None


def _training_results(session: Session, row: JobRow) -> list[TrainingResultRow]:
    return session.scalars(
        select(TrainingResultRow).where(TrainingResultRow.job_id == row.id)
    ).all()


def _pseudo_markup_results(session: Session, row: JobRow) -> list[PseudoMarkupResultRow]:
    return session.scalars(
        select(PseudoMarkupResultRow).where(PseudoMarkupResultRow.job_id == row.id)
    ).all()


def _first_training_result(session: Session, row: JobRow) -> TrainingResultRow | None:
    return session.scalar(
        select(TrainingResultRow)
        .where(TrainingResultRow.job_id == row.id)
        .order_by(TrainingResultRow.created_at)
    )


def _first_pseudo_markup_result(session: Session, row: JobRow) -> PseudoMarkupResultRow | None:
    return session.scalar(
        select(PseudoMarkupResultRow)
        .where(PseudoMarkupResultRow.job_id == row.id)
        .order_by(PseudoMarkupResultRow.created_at)
    )


def _extract_mlflow_run_id(row: JobRow) -> str | None:
    if not row.tmp_path:
        return None
    run_dir = Path(row.tmp_path)
    run_id_path = run_dir / MLFLOW_RUN_ID_FILE
    try:
        run_id = run_id_path.read_text(encoding="utf-8").strip()
    except OSError:
        run_id = ""
    if run_id:
        return run_id
    log_path = run_dir / "train.log"
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r"^mlflow_run=(?P<run_id>[a-zA-Z0-9_-]+)\s*$", text, re.MULTILINE)
    return match.group("run_id") if match else None


def _mlflow_run_url(
    config: TrainingUIAPIConfig,
    experiment_id: str | None,
    run_id: str,
) -> str:
    base = config.mlflow_ui_url.rstrip("/")
    if not experiment_id:
        return base
    return f"{base}/#/experiments/{experiment_id}/runs/{run_id}"


def _resolve_mlflow_experiment_id(
    row: JobRow,
    config: TrainingUIAPIConfig,
) -> str | None:
    if row.mlflow_experiment_id:
        return row.mlflow_experiment_id
    experiment_name = (row.mlflow_experiment_name or "").strip()
    if not experiment_name:
        return None
    try:
        experiment = next(
            (
                candidate
                for candidate in list_experiments(config.mlflow_tracking_uri)
                if candidate.name == experiment_name
            ),
            None,
        )
    except MLflowAdapterError:
        LOGGER.warning(
            "Не удалось определить id MLflow experiment %s для задания %s",
            experiment_name,
            row.id,
            exc_info=True,
        )
        return None
    if experiment is None:
        return None
    row.mlflow_experiment_id = experiment.experiment_id
    return experiment.experiment_id


def _read_exit_code(run_dir: Path | None) -> int | None:
    if run_dir is None:
        return None
    path = run_dir / "exit_code"
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 1


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _pseudolabel_timed_out(row: JobRow) -> bool:
    """Proverit zafiksirovannyi timeout s uchetom timezone."""

    if not _is_pseudolabel_aoi_job(row) or row.started_at is None:
        return False
    state = (row.config or {}).get("pseudolabel")
    if not isinstance(state, dict):
        return False
    timeout_seconds = _int_value(state, "timeout_seconds", 0)
    if timeout_seconds <= 0:
        return False
    started_at = row.started_at
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    return (_now() - started_at).total_seconds() >= timeout_seconds


def _set_pseudolabel_error(row: JobRow, code: str, message: str) -> None:
    """Atomarno zapisat strukturirovannuyu oshibku v JSON job."""

    state = dict((row.config or {}).get("pseudolabel") or {})
    state["error"] = {"code": code, "message": message, "details": {}}
    row.config = {**(row.config or {}), "pseudolabel": state}


def _string_values(value: object) -> list[str]:
    """Normalizovat optional JSON-massiv strok."""

    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _optional_number(value: object) -> float | None:
    """Bezopasno preobrazovat optional chislo."""

    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _write_worker_error(run_dir: Path, message: str) -> None:
    try:
        (run_dir / "worker_error.txt").write_text(message, encoding="utf-8")
    except OSError:
        pass


def _training_job_error(row: JobRow) -> str:
    run_dir = Path(row.tmp_path) if row.tmp_path else None
    if run_dir is not None:
        for path in (
            run_dir / "worker_error.txt",
            run_dir / "train.log",
            run_dir / "logs" / "train.log",
        ):
            try:
                if not path.is_file():
                    continue
                size_bytes = path.stat().st_size
                with path.open("rb") as stream:
                    if size_bytes > JOB_ERROR_MAX_BYTES:
                        stream.seek(-JOB_ERROR_MAX_BYTES, 2)
                    content = stream.read().decode("utf-8", errors="replace").strip()
            except OSError:
                continue
            if content:
                return content
    return "Обучение завершилось с ошибкой, но журнал процесса недоступен."


def _flat_value(flat: dict[str, Any], key: str, default: Any) -> Any:
    value = flat.get(key, default)
    return default if value == "" else value


def _int_value(flat: dict[str, Any], key: str, default: int) -> int:
    value = _flat_value(flat, key, default)
    return int(value)


def _optional_int(flat: dict[str, Any], key: str) -> int | None:
    value = _flat_value(flat, key, None)
    return None if value is None else int(value)


def _float_value(flat: dict[str, Any], key: str, default: float) -> float:
    value = _flat_value(flat, key, default)
    return float(value)


def _optional_float(flat: dict[str, Any], key: str) -> float | None:
    value = _flat_value(flat, key, None)
    return None if value is None else float(value)


def _blank_to_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _now() -> datetime:
    return datetime.now(timezone.utc)
