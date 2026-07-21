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
import traceback
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from mlsystem2.mlflow_adapter.api import get_best_training_checkpoint, list_experiments
from mlsystem2.mlflow_adapter.contracts import MLflowAdapterError, MLflowBestCheckpoint
from mlsystem2.settings.api import load_settings

from ._automation import AUTOMATION_KEY, sync_automation_once
from ._config import TrainingUIAPIConfig
from ._dataset_catalog import find_managed_dataset, list_managed_datasets
from ._datasets import CUSTOM_KEY, count_scenes_file_images, imagery_images_dir
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
from ._queueing import dispatch_sort_key, ensure_queue_positions
from ._templates import normalize_tile_factors
from ._test_samples import (
    TEST_SAMPLE_F1_OPERATION,
    evaluate_test_samples_for_pseudo_markup,
    queue_training_result_test_f1,
    reconcile_training_result_test_f1,
)
from .contracts import JobSource, JobStatus, JobType, ResultStatus, StoredFileKind


LOGGER = logging.getLogger(__name__)
MLFLOW_RUN_ID_FILE = "mlflow_run_id"


class _StartedProcess(Protocol):
    pid: int


ProcessLauncher = Callable[..., _StartedProcess]


def _is_test_sample_f1_job(row: JobRow) -> bool:
    return (row.config or {}).get("operation") == TEST_SAMPLE_F1_OPERATION


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
                dispatch_queue_once(session, config)
                session.commit()
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Training UI worker tick failed")
        await asyncio.sleep(interval)


def dispatch_queue_once(
    session: Session,
    config: TrainingUIAPIConfig,
    *,
    popen_factory: ProcessLauncher = subprocess.Popen,
) -> None:
    _reconcile_running_training_jobs(session, config)
    _reconcile_running_inference_jobs(session, config)
    ensure_queue_positions(session)
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
    rows = session.scalars(
        select(JobRow).where(*conditions)
    ).all()
    automation_enabled = _automation_enabled(session)
    candidates = [
        row
        for row in rows
        if _dispatch_allowed(session, row, automation_enabled)
    ]
    candidates.sort(key=dispatch_sort_key)
    return candidates[0] if candidates else None


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


def _has_running_job(session: Session) -> bool:
    return (
        session.scalar(
            select(JobRow.id).where(JobRow.status == JobStatus.RUNNING.value)
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
        _write_worker_error(
            run_dir,
            "Не удалось запустить обучение.\n\n"
            f"{traceback.format_exc()}",
        )
        _finish_training_job(session, row, config, succeeded=False)
        return

    row.status = JobStatus.RUNNING.value
    row.started_at = _now()
    row.finished_at = None
    row.process_pid = process.pid
    row.tmp_path = str(run_dir)
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
    run_dir = config.scratch_root / "jobs" / str(row.id)
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    try:
        config_path = run_dir / ("test_f1_config.yaml" if is_test_f1 else "pseudo_config.yaml")
        payload = (
            _build_test_sample_f1_config(session, row, config, run_dir)
            if is_test_f1
            else _build_pseudo_markup_config(session, row, config, run_dir)
        )
        _write_yaml(config_path, payload)
        script_path = _write_pseudo_run_script(
            config,
            run_dir,
            config_path,
            test_f1=is_test_f1,
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
                else "Не удалось запустить псевдоразметку.\n\n"
            )
            + traceback.format_exc(),
        )
        _finish_inference_job(session, row, config, succeeded=False)
        return

    row.status = JobStatus.RUNNING.value
    row.started_at = _now()
    row.finished_at = None
    row.process_pid = process.pid
    row.tmp_path = str(run_dir)
    if is_test_f1:
        metric = _test_sample_f1_metric(session, row)
        if metric is not None:
            metric.status = "running"
            metric.updated_at = _now()
    else:
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
            "positive_factor": tile_factors["tile_preparation.positive_factor"],
            "hard_negative_factor": tile_factors["tile_preparation.hard_negative_factor"],
            "background_factor": tile_factors["tile_preparation.background_factor"],
        },
        "train": {
            "quality_metric": str(_flat_value(flat, "train.quality_metric", "pixel")),
            "model_name": row.architecture,
            "input_channels": _int_value(flat, "train.input_channels", 4),
            "initial_checkpoint_uri": _blank_to_none(
                _flat_value(flat, "train.initial_checkpoint_uri", None)
            ),
            "epochs": _int_value(flat, "train.epochs", 1),
            "batch_size": _int_value(flat, "train.batch_size", 1),
            "learning_rate": _float_value(flat, "train.learning_rate", 0.0001),
            "weight_decay": _float_value(flat, "train.weight_decay", 0.0),
            "loss": str(_flat_value(flat, "train.loss", "bce_dice")),
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
    if threshold is None and training_result is not None:
        raise RuntimeError(
            "Для псевдоразметки по результату обучения не найден MLflow val/best_threshold "
            "на эпохе best checkpoint."
        )
    if threshold is None:
        threshold = _float_value(flat, "train.threshold", 0.5)
    return {
        "run_root": str(run_dir / "scratch"),
        "inference_backend": "pytorch_one_off",
        "output_geojson": str(run_dir / "scratch" / "pseudo_markup.geojson"),
        "report_path": str(run_dir / "scratch" / "report.json"),
        "scenes_file": result.scenes_file.path,
        "images_root": str(row.config.get("images_root") or config.images_root),
        "class_key": result.class_key,
        "class_name": training_result.class_display_name if training_result is not None else result.class_key,
        "source_model": training_result.model_name if training_result is not None else row.model_name,
        "mlflow_tracking_uri": config.mlflow_tracking_uri,
        "mlflow_run_id": row.config.get("mlflow_run_id"),
        "checkpoint_uri": row.config.get("checkpoint_uri"),
        "checkpoint_artifact_path": row.config.get("checkpoint_artifact_path") or "checkpoints/best.pt",
        "checkpoint_f1_score": row.config.get("checkpoint_f1_score"),
        "checkpoint_epoch": row.config.get("checkpoint_epoch"),
        "imagery_type": row.config.get("imagery_type"),
        "input_channels": _int_value(row.config, "input_channels", 4),
        "postprocess_config": row.config.get("inference_template_config") or {},
        "threshold": threshold,
        "tile_size": tile_size,
        "stride": _int_value(flat, "tile_preparation.stride", tile_size),
        "batch_size": _int_value(flat, "train.batch_size", 1),
        "device": "cuda",
    }


def _build_test_sample_f1_config(
    session: Session,
    row: JobRow,
    config: TrainingUIAPIConfig,
    run_dir: Path,
) -> dict[str, Any]:
    try:
        training_result_id = uuid.UUID(str(row.config.get("training_result_id")))
        sample_id = uuid.UUID(str(row.config.get("test_sample_id")))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("В задании F1 повреждены идентификаторы сети или разметки.") from exc
    training_result = session.get(TrainingResultRow, training_result_id)
    sample = session.get(TestSampleRow, sample_id)
    if training_result is None or training_result.status != ResultStatus.OK.value:
        raise RuntimeError("Успешный результат обучения для расчёта F1 не найден.")
    if sample is None or not sample.is_primary or sample.dataset_key != training_result.class_key:
        raise RuntimeError("Основная тестовая разметка была заменена или удалена.")
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
    checkpoint = _best_training_checkpoint(config, training_result.mlflow_run_id)
    if checkpoint is None:
        raise RuntimeError("Не удалось получить best checkpoint из MLflow.")
    if checkpoint.threshold is None:
        raise RuntimeError("У best checkpoint отсутствует порог val/best_threshold.")

    source_training_job = (
        session.get(JobRow, training_result.job_id)
        if training_result.job_id is not None
        else None
    )
    flat = dict(source_training_job.config or {}) if source_training_job is not None else {}
    inference_tile_size = _int_value(flat, "tile_preparation.tile_size", 768)
    sample_root = Path(config.stored_files_root) / "test-samples" / str(sample.id)
    tiles: list[dict[str, Any]] = []
    for tile in sorted(enabled_tiles, key=lambda item: item.tile_index):
        base_name = f"tile_{tile.tile_index:03d}"
        tif_path = sample_root / f"{base_name}.tif"
        mask_path = sample_root / f"{base_name}_mask.png"
        geojson_path = sample_root / f"{base_name}.geojson"
        if not tif_path.is_file() or not mask_path.is_file() or not geojson_path.is_file():
            raise RuntimeError(
                f"Не найдены TIFF, GeoJSON или маска тестового тайла {base_name}."
            )
        tiles.append(
            {
                "index": tile.tile_index,
                "image_path": str(tif_path),
                "mask_path": str(mask_path),
                "geojson_path": str(geojson_path),
            }
        )
    return {
        "operation": TEST_SAMPLE_F1_OPERATION,
        "run_root": str(run_dir / "scratch"),
        "report_path": str(run_dir / "scratch" / "report.json"),
        "class_key": training_result.class_key,
        "class_name": training_result.class_display_name,
        "source_model": training_result.model_name,
        "training_result_id": str(training_result.id),
        "test_sample_id": str(sample.id),
        "test_sample_revision": sample.content_revision,
        "tiles": tiles,
        "mlflow_tracking_uri": config.mlflow_tracking_uri,
        "mlflow_run_id": training_result.mlflow_run_id,
        "checkpoint_uri": checkpoint.artifact_uri,
        "checkpoint_artifact_path": checkpoint.artifact_path,
        "checkpoint_f1_score": checkpoint.f1_score,
        "checkpoint_epoch": checkpoint.epoch,
        "input_channels": _int_value(flat, "train.input_channels", 4),
        "postprocess_config": row.config.get("inference_template_config") or {},
        "postprocess_profile": row.config.get("postprocess_profile"),
        "test_f1_evaluator_version": row.config.get("test_f1_evaluator_version"),
        "threshold": checkpoint.threshold,
        "tile_size": inference_tile_size,
        "stride": _int_value(flat, "tile_preparation.stride", inference_tile_size),
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
    if (
        dataset is None
        or not dataset.source_available
        or dataset.images_dir is None
        or not dataset.scenes_file
        or not dataset.annotation_file
    ):
        raise RuntimeError(f"Датасет не найден или неполный: {row.dataset_name}")
    return {
        "images_dir": str(
            row.config.get("dataset.images_dir")
            or dataset.images_dir
            or config.images_root
        ),
        "scenes_file": dataset.scenes_file,
        "annotation_file": dataset.annotation_file,
        **(
            {"hard_negative_annotation_file": dataset.hard_negative_annotation_file}
            if dataset.hard_negative_annotation_file
            else {}
        ),
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
    *,
    test_f1: bool = False,
) -> Path:
    script_path = run_dir / ("run_test_f1.sh" if test_f1 else "run_pseudo_markup.sh")
    log_path = run_dir / "logs" / ("test_sample_f1.log" if test_f1 else "pseudo_markup.log")
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
    mlflow_experiment_id = (
        _resolve_mlflow_experiment_id(row, config)
        if mlflow_run_id
        else row.mlflow_experiment_id
    )
    best_checkpoint = (
        _best_training_checkpoint(config, mlflow_run_id)
        if succeeded and mlflow_run_id
        else None
    )
    training_results = _training_results(session, row)
    for result in training_results:
        result.status = ResultStatus.OK.value if succeeded else ResultStatus.ERROR.value
        result.trained_at = row.finished_at if succeeded else result.trained_at
        result.mlflow_run_id = mlflow_run_id or result.mlflow_run_id
        if best_checkpoint is not None:
            result.f1_score = best_checkpoint.f1_score
            result.epoch = best_checkpoint.epoch
        result.mlflow_run_url = (
            _mlflow_run_url(config, mlflow_experiment_id, mlflow_run_id)
            if mlflow_run_id
            else result.mlflow_run_url
        )
        result.updated_at = _now()
    session.flush()
    if succeeded:
        for result in training_results:
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
        session.flush()
    LOGGER.info("Finished training job %s with status %s", row.id, row.status)


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
        _finish_test_sample_f1_job(session, row, succeeded=succeeded)
        return
    row.status = JobStatus.COMPLETED.value if succeeded else JobStatus.FAILED.value
    row.finished_at = _now()
    row.process_pid = None
    output_path = _pseudo_output_path(row)
    report = _pseudo_report(row)
    report_allows_success = _pseudo_report_allows_success(report)
    succeeded = succeeded and report_allows_success
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
            LOGGER.exception(
                "Не удалось восстановить тестовый F1 после новой псевдоразметки"
            )
    _cleanup_inference_scratch(row)
    LOGGER.info(
        "Finished pseudo-markup job %s with status %s report=%s",
        row.id,
        row.status,
        report,
    )


def _finish_test_sample_f1_job(
    session: Session,
    row: JobRow,
    *,
    succeeded: bool,
) -> None:
    row.finished_at = _now()
    row.process_pid = None
    report = _pseudo_report(row)
    report_ok = _test_sample_f1_report_allows_success(report)
    succeeded = succeeded and report_ok
    row.status = JobStatus.COMPLETED.value if succeeded else JobStatus.FAILED.value
    metric = _test_sample_f1_metric(session, row)
    if metric is not None:
        sample = session.get(TestSampleRow, metric.sample_id) if metric.sample_id is not None else None
        expected_revision = int(row.config.get("test_sample_revision") or 0)
        still_current = bool(
            sample
            and sample.is_primary
            and sample.dataset_key == row.dataset_key
            and sample.content_revision == expected_revision
            and metric.sample_revision == expected_revision
            and metric.job_id == row.id
        )
        if succeeded and still_current and report is not None:
            true_positive = int(report.get("true_positive") or 0)
            false_positive = int(report.get("false_positive") or 0)
            false_negative = int(report.get("false_negative") or 0)
            precision_denominator = true_positive + false_positive
            recall_denominator = true_positive + false_negative
            precision = true_positive / precision_denominator if precision_denominator else 0.0
            recall = true_positive / recall_denominator if recall_denominator else 0.0
            denominator = precision + recall
            metric.precision = precision
            metric.recall = recall
            metric.f1 = 2.0 * precision * recall / denominator if denominator else 0.0
            metric.true_positive = true_positive
            metric.false_positive = false_positive
            metric.false_negative = false_negative
            object_true_positive = int(report.get("object_true_positive") or 0)
            object_false_positive = int(report.get("object_false_positive") or 0)
            object_false_negative = int(report.get("object_false_negative") or 0)
            object_precision_denominator = object_true_positive + object_false_positive
            object_recall_denominator = object_true_positive + object_false_negative
            object_precision = (
                object_true_positive / object_precision_denominator
                if object_precision_denominator
                else 0.0
            )
            object_recall = (
                object_true_positive / object_recall_denominator
                if object_recall_denominator
                else 0.0
            )
            object_denominator = object_precision + object_recall
            metric.object_precision = object_precision
            metric.object_recall = object_recall
            metric.object_f1 = (
                2.0 * object_precision * object_recall / object_denominator
                if object_denominator
                else 0.0
            )
            metric.object_true_positive = object_true_positive
            metric.object_false_positive = object_false_positive
            metric.object_false_negative = object_false_negative
            metric.threshold = float(report.get("threshold"))
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


def _test_sample_f1_metric(
    session: Session,
    row: JobRow,
) -> TrainingResultTestMetricRow | None:
    return session.scalar(
        select(TrainingResultTestMetricRow).where(
            TrainingResultTestMetricRow.job_id == row.id
        )
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
    if report.get("status") not in {"ok", "partial"}:
        return False
    try:
        processed = int(report.get("processed") or 0)
    except (TypeError, ValueError):
        return False
    return processed > 0


def _store_generated_geojson(
    session: Session,
    source_path: Path,
    config: TrainingUIAPIConfig,
    *,
    original_name: str,
    object_count: int | None,
) -> StoredFileRow:
    file_id = uuid.uuid4()
    target_dir = config.stored_files_root / StoredFileKind.PSEUDO_MARKUP_GEOJSON.value
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
        kind=StoredFileKind.PSEUDO_MARKUP_GEOJSON.value,
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
    model_name = result.training_result.model_name if result is not None and result.training_result is not None else row.model_name
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
