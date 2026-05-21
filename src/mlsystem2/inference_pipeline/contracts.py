"""Публичные контракты конвейера инференса."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from mlsystem2.mlflow_adapter.contracts import MLflowRunRef
from mlsystem2.train_pipeline.contracts import PipelineReport, PipelineStatus, TimingReport


class InferencePipelineError(RuntimeError):
    """Невосстановимая ошибка конвейера инференса."""


class InferencePipelineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str | None = None


class InferencePipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: PipelineStatus
    mlflow_run: MLflowRunRef | None
    timings: TimingReport
    report: PipelineReport


__all__ = [
    "InferencePipelineError",
    "InferencePipelineRequest",
    "InferencePipelineResult",
]
