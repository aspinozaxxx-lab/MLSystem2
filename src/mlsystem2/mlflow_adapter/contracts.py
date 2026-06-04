"""Публичные контракты адаптера MLflow."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class MLflowAdapterError(RuntimeError):
    """Ошибка адаптера MLflow."""


class MLflowRunStatus(StrEnum):
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"


class MLflowStartRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    tracking_uri: str
    experiment_name: str
    dataset: str | None = None
    run_name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class MLflowRunRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    experiment_name: str
    tracking_uri: str
    active: bool


class MLflowExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    name: str
    lifecycle_stage: str | None = None


class MLflowExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tracking_uri: str
    name: str


class MLflowArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str
    artifact_path: str


class MLflowBestCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tracking_uri: str
    run_id: str
    metric_name: str
    f1_score: float
    epoch: int
    threshold: float | None = None
    artifact_path: str
    artifact_uri: str | None = None


class MLflowDownloadedArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    artifact_path: str
    local_path: str


__all__ = [
    "MLflowAdapterError",
    "MLflowArtifactRef",
    "MLflowBestCheckpoint",
    "MLflowDownloadedArtifact",
    "MLflowExperiment",
    "MLflowExperimentRequest",
    "MLflowRunRef",
    "MLflowRunStatus",
    "MLflowStartRunRequest",
]
