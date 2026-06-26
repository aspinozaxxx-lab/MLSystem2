"""Training result, pseudo-markup, and automation contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .catalog import DatasetInfo, ModelInfo
from .common import JobSource, JobType, ResultStatus, RuntimeProgress, StoredFileInfo


class CustomDatasetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    name: str
    scenes_file: StoredFileInfo
    annotation_file: StoredFileInfo
    created_at: datetime


class PseudoMarkupResultInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    job_id: UUID | None = None
    source: JobSource = JobSource.MANUAL
    dataset_key: str | None = None
    dataset_version: str | None = None
    source_dataset_name: str
    scenes_file: StoredFileInfo | None = None
    geojson_file: StoredFileInfo | None = None
    image_count: int | None = None
    status: Literal["queued", "running", "ok", "error", "cancelled"]
    created_at: datetime
    runtime_minutes: int | None = None
    progress: RuntimeProgress | None = None


class TrainingResultInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    job_id: UUID | None = None
    source: JobSource = JobSource.MANUAL
    dataset_key: str | None = None
    dataset_version: str | None = None
    model_name: str
    architecture: str
    f1_score: float | None = None
    epoch: int | None = None
    trained_at: datetime | None = None
    created_at: datetime
    started_at: datetime | None = None
    mlflow_run_url: str | None = None
    status: Literal["queued", "running", "ok", "error", "cancelled"]
    progress: RuntimeProgress | None = None
    pseudo_markup_results: list[PseudoMarkupResultInfo] = Field(default_factory=list)


class ClassResultsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_key: str
    class_name: str
    dataset_updated_at: datetime | None = None
    results: list[TrainingResultInfo]


class ResultChangeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    item_type: Literal["job", "training_result", "pseudo_markup_result"] = "training_result"
    job_id: UUID | None = None
    type: JobType | None = None
    class_key: str
    dataset_name: str
    model_name: str
    action: str
    source: JobSource = JobSource.MANUAL
    status: Literal["queued", "running", "ok", "error", "cancelled"]
    changed_at: datetime
    mlflow_run_url: str | None = None


class ResultChangesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    changes: list[ResultChangeInfo]


class AutomationEnabledUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


class AutomationRuleUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_key: str
    architecture: str
    training_enabled: bool
    pseudo_markup_enabled: bool


class AutomationRuleInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID | None = None
    dataset_key: str
    architecture: str
    training_enabled: bool = False
    pseudo_markup_enabled: bool = False
    dataset_version: str | None = None
    training_status: ResultStatus | None = None
    pseudo_markup_status: ResultStatus | None = None
    current_training_result_id: UUID | None = None


class AutomationSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    datasets: list[DatasetInfo]
    models: list[ModelInfo]
    rules: list[AutomationRuleInfo]


__all__ = [
    "AutomationEnabledUpdate",
    "AutomationRuleInfo",
    "AutomationRuleUpdate",
    "AutomationSnapshot",
    "ClassResultsResponse",
    "CustomDatasetInfo",
    "PseudoMarkupResultInfo",
    "ResultChangeInfo",
    "ResultChangesResponse",
    "TrainingResultInfo",
]
