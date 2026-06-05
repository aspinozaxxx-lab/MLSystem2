"""Фоновый исполнитель очереди training UI API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import shlex
import subprocess
import sys
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from mlsystem2.mlflow_adapter.api import get_best_training_checkpoint
from mlsystem2.mlflow_adapter.contracts import MLflowAdapterError, MLflowBestCheckpoint
from mlsystem2.settings.api import load_settings

from ._automation import AUTOMATION_KEY, sync_automation_once
from ._config import TrainingUIAPIConfig
from ._datasets import CUSTOM_KEY, list_datasets
from ._models import (
    AutomationControlRow,
    CustomDatasetRow,
    JobRow,
    PseudoMarkupResultRow,
    QueueControlRow,
    StoredFileRow,
    TrainingResultRow,
)
from .contracts import JobSource, JobStatus, JobType, ResultStatus, StoredFileKind


LOGGER = logging.getLogger(__name__)
MLFLOW_RUN_ID_FILE = "mlflow_run_id"


class _StartedProcess(Protocol):
    pid: int


ProcessLauncher = Callable[..., _StartedProcess]


async def run_queue_worker(
    session_factory: sessionmaker[Session],
    config: TrainingUIAPIConfig,
) -> None:
    interval = max(1, config.worker_interval_seconds)
    LOGGER.info("Training UI worker started with interval %s sec", interval)
    while True:
        try:
            with session_factory() as session:
                sync_automation_once(session, config)
                dispatch_training_queue_once(session, config)
                dispatch_inference_queue_once(session, config)
                session.commit()
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Training UI worker tick failed")
        await asyncio.sleep(interval)


def dispatch_training_queue_once(
    session: Session,
    config: TrainingUIAPIConfig,
    *,
    popen_factory: ProcessLauncher = subprocess.Popen,
) -> None:
    _reconcile_running_training_jobs(session, config)
    if _has_running_training_job(session):
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
    _reconcile_running_inference_jobs(session, config)
    if _has_running_inference_job(session):
        return
    control = session.get(QueueControlRow, JobType.INFERENCE.value)
    if control is not None and not control.enabled:
        return
    job = _next_dispatch_job(session, JobType.INFERENCE)
    if job is None:
        return
    _start_inference_job(session, job, config, popen_factory=popen_factory)


def _next_dispatch_job(session: Session, job_type: JobType) -> JobRow | None:
    rows = session.scalars(
        select(JobRow).where(
            JobRow.type == job_type.value,
            JobRow.status == JobStatus.QUEUED.value,
        )
    ).all()
    automation_enabled = _automation_enabled(session)
    candidates = [
        row
        for row in rows
        if row.source != JobSource.AUTOMATION.value or automation_enabled
    ]
    candidates.sort(
        key=lambda row: (
            0 if row.source == JobSource.MANUAL.value else 1,
            row.queue_position,
            row.created_at,
        )
    )
    return candidates[0] if candidates else None


def _automation_enabled(session: Session) -> bool:
    row = session.get(AutomationControlRow, AUTOMATION_KEY)
    return bool(row and row.enabled)


def _reconcile_running_training_jobs(session: Session, config: TrainingUIAPIConfig) -> None:
    rows = session.scalars(
        select(JobRow).where(
            JobRow.type == JobType.TRAINING.value,
            JobRow.status == JobStatus.RUNNING.value,
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
            JobRow.status == JobStatus.RUNNING.value,
        )
    ).all()
    for row in rows:
        run_dir = Path(row.tmp_path) if row.tmp_path else None
        exit_code = _read_exit_code(run_dir) if run_dir is not None else None
        if exit_code is not None:
            _finish_inference_job(session, row, config, succeeded=exit_code == 0)
            continue
        if row.process_pid is not None and _pid_is_alive(row.process_pid):
            continue
        _finish_inference_job(session, row, config, succeeded=False)


def _has_running_training_job(session: Session) -> bool:
    return (
        session.scalar(
            select(JobRow.id).where(
                JobRow.type == JobType.TRAINING.value,
                JobRow.status == JobStatus.RUNNING.value,
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
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "scratch").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

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
    except Exception:
        LOGGER.exception("Failed to start training job %s", row.id)
        _write_worker_error(run_dir, "Не удалось запустить обучение. Подробности в journalctl.")
        _finish_training_job(session, row, config, succeeded=False)
        return

    row.status = JobStatus.RUNNING.value
    row.started_at = _now()
    row.finished_at = None
    row.process_pid = process.pid
    row.tmp_path = str(run_dir)
    for result in _training_results(session, row):
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
    run_dir = config.scratch_root / "jobs" / str(row.id)
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    try:
        config_path = run_dir / "pseudo_config.yaml"
        payload = _build_pseudo_markup_config(session, row, config, run_dir)
        _write_yaml(config_path, payload)
        script_path = _write_pseudo_run_script(config, run_dir, config_path)
        process = popen_factory(
            ["bash", str(script_path)],
            cwd=str(config.project_root),
            start_new_session=True,
        )
    except Exception:
        LOGGER.exception("Failed to start pseudo-markup job %s", row.id)
        _write_worker_error(run_dir, "Не удалось запустить псевдоразметку. Подробности в journalctl.")
        _finish_inference_job(session, row, config, succeeded=False)
        return

    row.status = JobStatus.RUNNING.value
    row.started_at = _now()
    row.finished_at = None
    row.process_pid = process.pid
    row.tmp_path = str(run_dir)
    for result in _pseudo_markup_results(session, row):
        result.status = ResultStatus.RUNNING.value
        result.updated_at = _now()
    session.flush()
    LOGGER.info("Started pseudo-markup job %s with pid %s", row.id, process.pid)


def _build_training_config(
    session: Session,
    row: JobRow,
    config: TrainingUIAPIConfig,
    run_dir: Path,
) -> dict[str, Any]:
    flat = dict(row.config or {})
    train_batch_size = _int_value(flat, "train.batch_size", 1)
    train_threshold = _float_value(flat, "train.threshold", 0.5)
    return {
        "runtime": {
            "project_root": str(config.project_root),
            "scratch_root": str(run_dir / "scratch"),
            "logs_root": str(run_dir / "logs"),
            "cleanup_scratch_after_mlflow_log": True,
        },
        "dataset": {
            **_dataset_config(session, row, config),
            "val_fraction": _float_value(flat, "dataset.val_fraction", 0.2),
        },
        "tile_preparation": {
            "tile_size": _int_value(flat, "tile_preparation.tile_size", row.tile_size or 512),
            "stride": _int_value(flat, "tile_preparation.stride", row.tile_size or 512),
            "augmentation_level": _int_value(flat, "tile_preparation.augmentation_level", 0),
            "positive_factor": _float_value(flat, "tile_preparation.positive_factor", 0.5),
        },
        "train": {
            "model_name": row.architecture,
            "initial_checkpoint_uri": _blank_to_none(
                _flat_value(flat, "train.initial_checkpoint_uri", None)
            ),
            "epochs": _int_value(flat, "train.epochs", 1),
            "batch_size": train_batch_size,
            "learning_rate": _float_value(flat, "train.learning_rate", 0.0001),
            "weight_decay": _float_value(flat, "train.weight_decay", 0.0),
            "loss": str(_flat_value(flat, "train.loss", "bce_dice")),
            "focal_alpha": _float_value(flat, "train.focal_alpha", 0.6),
            "pos_weight": _float_value(flat, "train.pos_weight", 1.0),
            "tversky_alpha": _float_value(flat, "train.tversky_alpha", 0.4),
            "tversky_beta": _float_value(flat, "train.tversky_beta", 0.6),
            "threshold": train_threshold,
            "early_stopping_patience": _int_value(flat, "train.early_stopping_patience", 10),
            "max_train_batches_per_epoch": _optional_int(flat, "train.max_train_batches_per_epoch"),
            "max_val_batches_per_epoch": _optional_int(flat, "train.max_val_batches_per_epoch"),
            "max_training_time_sec": _optional_int(flat, "train.max_training_time_sec"),
        },
        "inference": {
            "checkpoint_uri": str(run_dir / "scratch" / "checkpoints" / "best.pt"),
            "threshold": train_threshold,
            "batch_size": train_batch_size,
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
    result = _first_pseudo_markup_result(session, row)
    if result is None or result.scenes_file is None:
        raise RuntimeError("Для псевдоразметки не найден txt со снимками.")
    training_result = (
        session.get(TrainingResultRow, result.training_result_id)
        if result.training_result_id is not None
        else None
    )
    source_training_job = (
        session.get(JobRow, training_result.job_id)
        if training_result is not None and training_result.job_id is not None
        else None
    )
    flat = dict(source_training_job.config or {}) if source_training_job is not None else {}
    tile_size = _int_value(flat, "tile_preparation.tile_size", 768)
    threshold = _optional_float(row.config, "checkpoint_threshold")
    if threshold is None:
        threshold = _float_value(flat, "train.threshold", 0.5)
    return {
        "run_root": str(run_dir / "scratch"),
        "output_geojson": str(run_dir / "scratch" / "pseudo_markup.geojson"),
        "report_path": str(run_dir / "scratch" / "report.json"),
        "scenes_file": result.scenes_file.path,
        "images_root": str(config.images_root),
        "class_key": result.class_key,
        "class_name": training_result.class_display_name if training_result is not None else result.class_key,
        "source_model": training_result.model_name if training_result is not None else row.model_name,
        "mlflow_tracking_uri": config.mlflow_tracking_uri,
        "mlflow_run_id": row.config.get("mlflow_run_id"),
        "checkpoint_uri": row.config.get("checkpoint_uri"),
        "checkpoint_artifact_path": row.config.get("checkpoint_artifact_path") or "checkpoints/best.pt",
        "checkpoint_f1_score": row.config.get("checkpoint_f1_score"),
        "checkpoint_epoch": row.config.get("checkpoint_epoch"),
        "threshold": threshold,
        "tile_size": tile_size,
        "stride": _int_value(flat, "tile_preparation.stride", tile_size),
        "batch_size": _int_value(flat, "train.batch_size", 1),
        "device": "cuda",
    }


def _dataset_config(
    session: Session,
    row: JobRow,
    config: TrainingUIAPIConfig,
) -> dict[str, Any]:
    result = _first_training_result(session, row)
    dataset_key = result.class_key if result is not None else row.dataset_name
    if row.custom_dataset_id is not None or dataset_key == CUSTOM_KEY:
        custom = row.custom_dataset or session.get(CustomDatasetRow, row.custom_dataset_id)
        if custom is None:
            raise RuntimeError(f"Custom dataset не найден: {row.custom_dataset_id}")
        return {
            "scenes_file": custom.scenes_file.path,
            "annotation_file": custom.annotation_file.path,
        }

    dataset = next((item for item in list_datasets(config.mlmarkup_root) if item.key == dataset_key), None)
    if dataset is None:
        dataset = next((item for item in list_datasets(config.mlmarkup_root) if item.name == row.dataset_name), None)
    if dataset is None or not dataset.scenes_file or not dataset.annotation_file:
        raise RuntimeError(f"Датасет не найден или неполный: {row.dataset_name}")
    return {
        "scenes_file": dataset.scenes_file,
        "annotation_file": dataset.annotation_file,
    }


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
                f"{quoted_command} > {shlex.quote(str(log_path))} 2>&1",
                "code=$?",
                f"printf '%s\\n' \"$code\" > {shlex.quote(str(exit_code_path))}",
                "exit \"$code\"",
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
) -> Path:
    script_path = run_dir / "run_pseudo_markup.sh"
    log_path = run_dir / "logs" / "pseudo_markup.log"
    exit_code_path = run_dir / "exit_code"
    command = [
        sys.executable,
        "-m",
        "mlsystem2.training_ui_api._pseudo_runner",
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
                "exit \"$code\"",
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
    row.finished_at = _now()
    row.process_pid = None
    mlflow_run_id = _extract_mlflow_run_id(row)
    best_checkpoint = (
        _best_training_checkpoint(config, mlflow_run_id)
        if succeeded and mlflow_run_id
        else None
    )
    for result in _training_results(session, row):
        result.status = ResultStatus.OK.value if succeeded else ResultStatus.ERROR.value
        result.trained_at = row.finished_at if succeeded else result.trained_at
        result.mlflow_run_id = mlflow_run_id or result.mlflow_run_id
        if best_checkpoint is not None:
            result.f1_score = best_checkpoint.f1_score
            result.epoch = best_checkpoint.epoch
        result.mlflow_run_url = (
            _mlflow_run_url(config, row.mlflow_experiment_id, mlflow_run_id)
            if mlflow_run_id
            else result.mlflow_run_url
        )
        result.updated_at = _now()
    session.flush()
    LOGGER.info("Finished training job %s with status %s", row.id, row.status)


def _sync_training_run_id(session: Session, row: JobRow, config: TrainingUIAPIConfig) -> None:
    mlflow_run_id = _extract_mlflow_run_id(row)
    if not mlflow_run_id:
        return
    changed = False
    for result in _training_results(session, row):
        if result.mlflow_run_id != mlflow_run_id:
            result.mlflow_run_id = mlflow_run_id
            changed = True
        run_url = _mlflow_run_url(config, row.mlflow_experiment_id, mlflow_run_id)
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
    row.status = JobStatus.COMPLETED.value if succeeded else JobStatus.FAILED.value
    row.finished_at = _now()
    row.process_pid = None
    output_path = _pseudo_output_path(row)
    report = _pseudo_report(row)
    has_geojson = output_path is not None and output_path.is_file()
    pseudo_results = _pseudo_markup_results(session, row)
    if succeeded and has_geojson:
        file_row = _store_generated_geojson(
            session,
            output_path,
            config,
            original_name=_pseudo_geojson_download_name(row, pseudo_results, row.finished_at),
        )
    else:
        file_row = None
    for result in pseudo_results:
        result.status = ResultStatus.OK.value if file_row is not None else ResultStatus.ERROR.value
        result.geojson_file_id = file_row.id if file_row is not None else result.geojson_file_id
        result.updated_at = _now()
    if succeeded and file_row is None:
        row.status = JobStatus.FAILED.value
    session.flush()
    LOGGER.info(
        "Finished pseudo-markup job %s with status %s report=%s",
        row.id,
        row.status,
        report,
    )


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


def _store_generated_geojson(
    session: Session,
    source_path: Path,
    config: TrainingUIAPIConfig,
    *,
    original_name: str,
) -> StoredFileRow:
    file_id = uuid.uuid4()
    target_dir = config.stored_files_root / StoredFileKind.PSEUDO_MARKUP_GEOJSON.value
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{file_id}.geojson"
    shutil.copy2(source_path, target_path)
    row = StoredFileRow(
        id=file_id,
        kind=StoredFileKind.PSEUDO_MARKUP_GEOJSON.value,
        original_name=original_name,
        content_type="application/geo+json",
        path=str(target_path),
        size_bytes=target_path.stat().st_size,
    )
    session.add(row)
    session.flush()
    return row


def _pseudo_geojson_download_name(
    row: JobRow,
    results: list[PseudoMarkupResultRow],
    created_at: datetime | None,
) -> str:
    result = results[0] if results else None
    dataset_name = result.class_key if result is not None else row.dataset_name
    model_name = result.training_result.model_name if result is not None and result.training_result is not None else row.model_name
    timestamp = (created_at or _now()).strftime("%H_%M_%d_%m")
    return f"{_filename_part(dataset_name)}_{_filename_part(model_name)}_{timestamp}.geojson"


def _filename_part(value: str) -> str:
    return " ".join(str(value).strip().split()) or "unknown"


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
    return session.scalars(select(TrainingResultRow).where(TrainingResultRow.job_id == row.id)).all()


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


def _write_worker_error(run_dir: Path, message: str) -> None:
    try:
        (run_dir / "worker_error.txt").write_text(message, encoding="utf-8")
    except OSError:
        pass


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
