"""Публичный фасад адаптера MLflow."""

from __future__ import annotations

from pathlib import Path

from mlsystem2.dataset_preparing.contracts import DatasetPreparationReport
from mlsystem2.train.contracts import EpochMetrics, TrainResult
from mlsystem2.train_pipeline.contracts import PipelineReport, TimingReport

from ._client import end_run as _end_run
from ._client import log_dataset_preparation as _log_dataset_preparation
from ._client import log_pipeline_report as _log_pipeline_report
from ._client import log_run_config as _log_run_config
from ._client import log_tile_preparation as _log_tile_preparation
from ._client import log_timing_report as _log_timing_report
from ._client import log_training_epoch as _log_training_epoch
from ._client import log_training_artifacts as _log_training_artifacts
from ._client import log_training_metrics as _log_training_metrics
from ._client import start_run as _start_run
from .contracts import MLflowRunRef, MLflowRunStatus, MLflowStartRunRequest


def start_run(request: MLflowStartRunRequest) -> MLflowRunRef:
    return _start_run(request)


def log_dataset_preparation(run: MLflowRunRef, report: DatasetPreparationReport) -> None:
    _log_dataset_preparation(run, report)


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


__all__ = [
    "start_run",
    "log_dataset_preparation",
    "log_tile_preparation",
    "log_run_config",
    "log_training_epoch",
    "log_training_metrics",
    "log_training_artifacts",
    "log_timing_report",
    "log_pipeline_report",
    "end_run",
]
