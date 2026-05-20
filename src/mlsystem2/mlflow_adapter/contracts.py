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
    run_name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class MLflowRunRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    experiment_name: str
    tracking_uri: str
    active: bool


class MLflowArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str
    artifact_path: str


__all__ = [
    "MLflowAdapterError",
    "MLflowArtifactRef",
    "MLflowRunRef",
    "MLflowRunStatus",
    "MLflowStartRunRequest",
]
