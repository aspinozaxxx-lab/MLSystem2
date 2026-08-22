"""Job and queue contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .common import JobSource, JobStatus, JobType, RuntimeProgress


class TrainingJobCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mlflow_experiment_id: str | None = None
    mlflow_experiment_name: str
    mlflow_run_name: str | None = None
    dataset_key: str
    custom_dataset_id: UUID | None = None
    architecture: str
    config: dict[str, Any]
    run_inference_after_training: bool = False


class QueueEnabledUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


class QueueControlInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queue_name: JobType
    enabled: bool
    updated_at: datetime


class QueueCountInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_jobs: int = Field(ge=0)


class JobSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    type: JobType
    purpose: Literal["training", "pseudo_markup", "test_sample_f1"]
    source: JobSource = JobSource.MANUAL
    status: JobStatus
    queue_position: int
    dataset_key: str | None = None
    dataset_version: str | None = None
    dataset_name: str
    training_dataset_name: str | None = None
    inference_dataset_name: str | None = None
    model_name: str
    architecture: str
    tile_size: int | None = None
    created_at: datetime
    started_at: datetime | None = None
    progress: RuntimeProgress | None = None
    actions: list[str] = Field(default_factory=list)


class QueueSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    training_enabled: bool
    inference_enabled: bool
    jobs: list[JobSummary] = Field(default_factory=list)
    training_jobs: list[JobSummary]
    inference_jobs: list[JobSummary]


class JobDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    type: JobType
    purpose: Literal["training", "pseudo_markup", "test_sample_f1"]
    source: JobSource = JobSource.MANUAL
    status: JobStatus
    queue_position: int
    dataset_key: str | None = None
    dataset_version: str | None = None
    dataset_name: str
    training_dataset_name: str | None = None
    inference_dataset_name: str | None = None
    model_name: str
    architecture: str
    tile_size: int | None = None
    mlflow_experiment_name: str | None = None
    mlflow_run_name: str | None = None
    config: dict[str, Any]
    run_inference_after_training: bool = False
    readonly: bool = True
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: RuntimeProgress | None = None


class JobLogInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    source_name: str
    content: str
    truncated: bool = False
    size_bytes: int


__all__ = [
    "JobDetail",
    "JobLogInfo",
    "JobSummary",
    "QueueControlInfo",
    "QueueCountInfo",
    "QueueEnabledUpdate",
    "QueueSnapshot",
    "TrainingJobCreate",
]
