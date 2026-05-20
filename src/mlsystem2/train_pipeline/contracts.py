"""Публичные контракты конвейера обучения."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mlsystem2.mlflow_adapter.contracts import MLflowRunRef


class TrainPipelineError(RuntimeError):
    """Невосстановимая ошибка конвейера обучения."""


class PipelineStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ModuleTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module: str
    elapsed_sec: float = Field(ge=0.0)
    details: dict[str, object] = Field(default_factory=dict)


class TimingReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_pipeline_time_sec: float = Field(ge=0.0)
    modules: list[ModuleTiming]


class PipelineReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: PipelineStatus
    message: str
    dataset_status: str | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    artifacts: dict[str, object] = Field(default_factory=dict)


class TrainPipelineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_path: str | Path
    run_name: str | None = None


class TrainPipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: PipelineStatus
    mlflow_run: MLflowRunRef | None
    timings: TimingReport
    report: PipelineReport


__all__ = [
    "ModuleTiming",
    "PipelineReport",
    "PipelineStatus",
    "TimingReport",
    "TrainPipelineError",
    "TrainPipelineRequest",
    "TrainPipelineResult",
]
