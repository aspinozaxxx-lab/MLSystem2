"""Публичные контракты training UI API."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TrainingUIAPIError(RuntimeError):
    """Ошибка сервиса training UI API."""


class TemplateSource(StrEnum):
    HPO_BEST = "hpo_best"
    ANALOGY = "analogy"
    MANUAL = "manual"


class JobType(StrEnum):
    TRAINING = "training"
    INFERENCE = "inference"


class JobSource(StrEnum):
    MANUAL = "manual"
    AUTOMATION = "automation"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResultStatus(StrEnum):
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"


class StoredFileKind(StrEnum):
    SCENES_TXT = "scenes_txt"
    ANNOTATION_GEOJSON = "annotation_geojson"
    PSEUDO_MARKUP_GEOJSON = "pseudo_markup_geojson"


class AppLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    title: str
    url: str


class AppLinksResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    links: list[AppLink]


class MLflowExperimentInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    name: str


class MLflowExperimentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)


class DatasetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    class_key: str | None = None
    class_name: str | None = None
    variant_key: str | None = None
    variant_name: str | None = None
    path: str | None = None
    is_custom: bool = False
    scenes_file: str | None = None
    annotation_file: str | None = None
    version: str | None = None
    updated_at: datetime | None = None


class DatasetListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets: list[DatasetInfo]


class ClassInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    updated_at: datetime | None = None
    variants: list[DatasetInfo] = Field(default_factory=list)
    is_custom: bool = False


class ClassListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classes: list[ClassInfo]


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture: str
    display_name: str
    input_channels: int
    output_channels: int
    pretrained: bool


class ModelListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelInfo]


class ConfigField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    value_type: str
    tooltip: str
    required: bool = True
    options: list[str] | None = None
    min_value: float | None = None
    max_value: float | None = None


class ConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: list[ConfigField]


class TrainingTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    architecture: str
    dataset_key: str | None = None
    dataset_name: str | None = None
    parent_template_id: UUID | None = None
    display_name: str
    config_schema: ConfigSchema
    default_config: dict[str, Any]
    source: TemplateSource
    source_mlflow_run_id: str | None = None
    is_active: bool
    version: int
    created_at: datetime
    updated_at: datetime


class TrainingTemplateListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    templates: list[TrainingTemplate]


class InferenceTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    architecture: str
    dataset_key: str | None = None
    dataset_name: str | None = None
    parent_template_id: UUID | None = None
    display_name: str
    config_schema: ConfigSchema
    default_config: dict[str, Any]
    source: TemplateSource
    source_mlflow_run_id: str | None = None
    is_active: bool
    version: int
    created_at: datetime
    updated_at: datetime


class InferenceTemplateListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    templates: list[InferenceTemplate]


class TrainingTemplateUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_config: dict[str, Any] | None = None
    is_active: bool | None = None
    reset_to_baseline: bool = False


class TrainingTemplateCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture: str
    dataset_key: str


class TrainingTemplateApplyField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    value: Any


InferenceTemplateUpdate = TrainingTemplateUpdate
InferenceTemplateCreate = TrainingTemplateCreate
InferenceTemplateApplyField = TrainingTemplateApplyField


class StoredFileInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    kind: StoredFileKind
    original_name: str
    size_bytes: int
    created_at: datetime
    download_url: str


class CustomDatasetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    name: str
    scenes_file: StoredFileInfo
    annotation_file: StoredFileInfo
    created_at: datetime


class TrainingJobCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mlflow_experiment_id: str | None = None
    mlflow_experiment_name: str
    mlflow_run_name: str
    dataset_key: str
    custom_dataset_id: UUID | None = None
    architecture: str
    config: dict[str, Any]


class AutomationEnabledUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


class AutomationRuleUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_key: str
    architecture: str
    training_enabled: bool
    pseudo_markup_enabled: bool


class QueueEnabledUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


class QueueControlInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queue_name: JobType
    enabled: bool
    updated_at: datetime


class RuntimeProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current: int | None = None
    total: int | None = None
    elapsed_minutes: int | None = None


class JobSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    type: JobType
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
    readonly: bool = True
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress: RuntimeProgress | None = None


class PseudoMarkupResultInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    source: JobSource = JobSource.MANUAL
    dataset_key: str | None = None
    dataset_version: str | None = None
    source_dataset_name: str
    scenes_file: StoredFileInfo | None = None
    geojson_file: StoredFileInfo | None = None
    status: ResultStatus
    created_at: datetime
    progress: RuntimeProgress | None = None


class TrainingResultInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    source: JobSource = JobSource.MANUAL
    dataset_key: str | None = None
    dataset_version: str | None = None
    model_name: str
    architecture: str
    f1_score: float | None = None
    epoch: int | None = None
    trained_at: datetime | None = None
    mlflow_run_url: str | None = None
    status: ResultStatus
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
    class_key: str
    dataset_name: str
    model_name: str
    action: str
    source: JobSource = JobSource.MANUAL
    status: ResultStatus
    changed_at: datetime


class ResultChangesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    changes: list[ResultChangeInfo]


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
    "AppLink",
    "AppLinksResponse",
    "AutomationEnabledUpdate",
    "AutomationRuleInfo",
    "AutomationRuleUpdate",
    "AutomationSnapshot",
    "ClassInfo",
    "ClassListResponse",
    "ClassResultsResponse",
    "ConfigField",
    "ConfigSchema",
    "CustomDatasetInfo",
    "DatasetInfo",
    "DatasetListResponse",
    "JobDetail",
    "JobSource",
    "JobStatus",
    "JobSummary",
    "JobType",
    "InferenceTemplate",
    "InferenceTemplateApplyField",
    "InferenceTemplateCreate",
    "InferenceTemplateListResponse",
    "InferenceTemplateUpdate",
    "MLflowExperimentCreate",
    "MLflowExperimentInfo",
    "ModelInfo",
    "ModelListResponse",
    "PseudoMarkupResultInfo",
    "QueueControlInfo",
    "QueueEnabledUpdate",
    "QueueSnapshot",
    "ResultStatus",
    "RuntimeProgress",
    "StoredFileInfo",
    "StoredFileKind",
    "TemplateSource",
    "TrainingJobCreate",
    "TrainingTemplateApplyField",
    "TrainingTemplateCreate",
    "TrainingResultInfo",
    "ResultChangeInfo",
    "ResultChangesResponse",
    "TrainingTemplate",
    "TrainingTemplateListResponse",
    "TrainingTemplateUpdate",
    "TrainingUIAPIError",
]
