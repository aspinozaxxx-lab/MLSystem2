"""Публичный фасад адаптера MLflow."""

from __future__ import annotations

from pathlib import Path

from mlsystem2.dataset_preparing.contracts import DatasetPreparationReport
from mlsystem2.train.contracts import EpochMetrics, TrainResult
from mlsystem2.train_pipeline.contracts import PipelineReport, TimingReport

from ._client import end_run as _end_run
from ._client import create_experiment as _create_experiment
from ._client import download_run_artifact as _download_run_artifact
from ._client import get_best_training_checkpoint as _get_best_training_checkpoint
from ._client import get_training_epoch_progress as _get_training_epoch_progress
from ._client import list_experiments as _list_experiments
from ._client import log_dataset_artifacts as _log_dataset_artifacts
from ._client import log_dataset_preparation as _log_dataset_preparation
from ._client import log_pipeline_report as _log_pipeline_report
from ._client import log_run_config as _log_run_config
from ._client import log_tile_preparation as _log_tile_preparation
from ._client import log_timing_report as _log_timing_report
from ._client import log_training_epoch as _log_training_epoch
from ._client import log_training_artifacts as _log_training_artifacts
from ._client import log_training_metrics as _log_training_metrics
from ._client import mark_run_killed as _mark_run_killed
from ._client import start_run as _start_run
from .contracts import (
    MLflowExperiment,
    MLflowExperimentRequest,
    MLflowBestCheckpoint,
    MLflowDownloadedArtifact,
    MLflowRunRef,
    MLflowRunStatus,
    MLflowStartRunRequest,
    MLflowTrainingProgress,
)


def list_experiments(tracking_uri: str) -> list[MLflowExperiment]:
    return _list_experiments(tracking_uri)


def create_experiment(request: MLflowExperimentRequest) -> MLflowExperiment:
    return _create_experiment(request)


def get_best_training_checkpoint(
    tracking_uri: str,
    run_id: str,
) -> MLflowBestCheckpoint | None:
    return _get_best_training_checkpoint(tracking_uri, run_id)


def get_training_epoch_progress(
    tracking_uri: str,
    run_id: str,
) -> MLflowTrainingProgress:
    return _get_training_epoch_progress(tracking_uri, run_id)


def download_run_artifact(
    *,
    tracking_uri: str,
    run_id: str,
    artifact_path: str,
    dst_dir: str | Path,
) -> MLflowDownloadedArtifact:
    return _download_run_artifact(
        tracking_uri=tracking_uri,
        run_id=run_id,
        artifact_path=artifact_path,
        dst_dir=dst_dir,
    )


def start_run(request: MLflowStartRunRequest) -> MLflowRunRef:
    return _start_run(request)


def log_dataset_preparation(run: MLflowRunRef, report: DatasetPreparationReport) -> None:
    _log_dataset_preparation(run, report)


def log_dataset_artifacts(run: MLflowRunRef, files: dict[str, str | Path]) -> None:
    _log_dataset_artifacts(run, files)


def log_tile_preparation(run: MLflowRunRef, report: dict[str, object]) -> None:
    _log_tile_preparation(run, report)


def log_run_config(run: MLflowRunRef, config_path: str | Path) -> None:
    _log_run_config(run, config_path)


def log_training_epoch(run: MLflowRunRef, metrics: EpochMetrics) -> None:
    _log_training_epoch(run, metrics)


def log_training_metrics(run: MLflowRunRef, result: TrainResult) -> None:
    _log_training_metrics(run, result)


def log_training_artifacts(run: MLflowRunRef, result: TrainResult) -> None:
    _log_training_artifacts(run, result)


def log_timing_report(run: MLflowRunRef, report: TimingReport) -> None:
    _log_timing_report(run, report)


def log_pipeline_report(run: MLflowRunRef, report: PipelineReport) -> None:
    _log_pipeline_report(run, report)


def end_run(run: MLflowRunRef, status: MLflowRunStatus) -> None:
    _end_run(run, status)


def mark_run_killed(tracking_uri: str, run_id: str) -> None:
    _mark_run_killed(tracking_uri, run_id)


__all__ = [
    "list_experiments",
    "create_experiment",
    "get_best_training_checkpoint",
    "get_training_epoch_progress",
    "download_run_artifact",
    "start_run",
    "log_dataset_preparation",
    "log_dataset_artifacts",
    "log_tile_preparation",
    "log_run_config",
    "log_training_epoch",
    "log_training_metrics",
    "log_training_artifacts",
    "log_timing_report",
    "log_pipeline_report",
    "end_run",
    "mark_run_killed",
]
