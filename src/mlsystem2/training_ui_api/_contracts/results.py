"""Training result, pseudo-markup, and automation contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    checkpoint_epoch: int | None = None
    source: JobSource = JobSource.MANUAL
    dataset_key: str | None = None
    dataset_version: str | None = None
    source_dataset_name: str
    scenes_file: StoredFileInfo | None = None
    geojson_file: StoredFileInfo | None = None
    image_count: int | None = None
    status: Literal["queued", "running", "paused", "ok", "error", "cancelled"]
    created_at: datetime
    runtime_minutes: int | None = None
    progress: RuntimeProgress | None = None
    task: Literal["binary", "multiclass"] = "binary"
    class_schema: list[dict[str, Any]] = Field(default_factory=list)
    by_type_download_url: str | None = None


class PseudoMarkupSceneInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    raster_url: str
    footprint_url: str
    bounds: tuple[float, float, float, float]
    has_alpha: bool = False
    has_nir: bool = False
    nodata: float | None = None


class PseudoMarkupViewInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    training_result_id: UUID | None = None
    checkpoint_epoch: int | None = None
    model_name: str
    source_dataset_name: str
    training_dataset_name: str
    class_name: str | None = None
    created_at: datetime
    geojson_url: str
    object_count: int | None = None
    expected_image_count: int | None = None
    scenes: list[PseudoMarkupSceneInfo]
    warnings: list[str]


class PseudoMarkupComparisonRequest(BaseModel):
    """Активные сохранённые разметки и ревизии подложки."""

    model_config = ConfigDict(extra="forbid")

    result_ids: list[UUID] = Field(min_length=1, max_length=12)
    scene_revisions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_results(self):
        if len(set(self.result_ids)) != len(self.result_ids):
            raise ValueError("Псевдоразметки в сравнении не должны повторяться.")
        return self


class PseudoMarkupPixelComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intersection: int = Field(ge=0)
    only_first: int = Field(ge=0)
    only_second: int = Field(ge=0)


class PseudoMarkupComparisonCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_ids: list[UUID]
    scenes: dict[str, PseudoMarkupPixelComparison]
    total: PseudoMarkupPixelComparison
    warnings: list[str]


class PseudoMarkupComparisonLayers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    available_result_ids: list[UUID]
    geojson: dict[str, Any]
    counts: PseudoMarkupPixelComparison | None = None


class TrainingResultTestF1Info(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["current", "stale", "queued", "running", "error", "unavailable"]
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    f1: float | None = Field(default=None, ge=0.0, le=1.0)
    true_positive: int | None = Field(default=None, ge=0)
    false_positive: int | None = Field(default=None, ge=0)
    false_negative: int | None = Field(default=None, ge=0)
    quality_metric: Literal["pixel", "objects"] = "pixel"
    pixel_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    pixel_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    pixel_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    pixel_true_positive: int | None = Field(default=None, ge=0)
    pixel_false_positive: int | None = Field(default=None, ge=0)
    pixel_false_negative: int | None = Field(default=None, ge=0)
    object_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    object_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    object_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    object_true_positive: int | None = Field(default=None, ge=0)
    object_false_positive: int | None = Field(default=None, ge=0)
    object_false_negative: int | None = Field(default=None, ge=0)
    metrics: dict[str, Any] = Field(default_factory=dict)
    sample_id: UUID | None = None
    sample_name: str | None = None
    sample_revision: int | None = Field(default=None, ge=1)
    samples: list[PrimaryTestSampleInfo] = Field(default_factory=list)
    aggregation: Literal["foreground", "macro"] = "foreground"
    job_id: UUID | None = None
    evaluated_at: datetime | None = None
    error: str | None = None
    progress: RuntimeProgress | None = None


class PrimaryTestSampleInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    name: str
    content_revision: int = Field(ge=1)
    enabled_image_count: int = Field(ge=0)
    enabled_object_count: int = Field(ge=0)
    class_key: str | None = None
    class_name: str | None = None
    class_slug: str | None = None
    class_id: int | None = Field(default=None, ge=1)
    color: str | None = None


class TestF1ScoreInfo(BaseModel):
    """Оценка одного снимка в сохранённом расчёте."""

    model_config = ConfigDict(extra="forbid")
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)


class TestF1SceneInfo(PseudoMarkupSceneInfo):
    """Независимый тестовый TIFF и его сохранённые оценки."""

    layers_url: str
    raster_available: bool = True
    sample_name: str
    sample_revision: int = Field(ge=1)
    target_class_id: int | None = None
    class_schema: list[dict[str, Any]] = Field(default_factory=list)
    pixel: TestF1ScoreInfo
    objects: TestF1ScoreInfo
    metrics: dict[str, Any] = Field(default_factory=dict)


class TestF1ViewInfo(BaseModel):
    """Готовность карты и снимки оценки конкретной сети."""

    model_config = ConfigDict(extra="forbid")
    training_result_id: UUID
    model_name: str
    training_dataset_name: str
    status: Literal["ready", "missing", "queued", "running", "error", "unavailable"]
    metric: TrainingResultTestF1Info | None = None
    scenes: list[TestF1SceneInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TrainingResultInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    job_id: UUID | None = None
    source: JobSource = JobSource.MANUAL
    dataset_key: str | None = None
    dataset_version: str | None = None
    model_name: str
    architecture: str
    pipeline_variant: Literal["legacy", "next_gen", "next_gen2", "object_f1"] = "legacy"
    validation_fold: int = Field(default=0, ge=0)
    is_primary: bool = False
    can_continue_training: bool = False
    continued_from_result_id: UUID | None = None
    continued_from_epoch: int | None = None
    continued_from_checkpoint: Literal["best", "last"] | None = None
    input_channels: int = Field(default=4, gt=0)
    quality_metric: Literal["pixel", "objects"] = "pixel"
    task: Literal["binary", "multiclass"] = "binary"
    class_schema: list[dict[str, object]] = Field(default_factory=list)
    training_metrics: dict[str, object] = Field(default_factory=dict)
    f1_score: float | None = None
    epoch: int | None = None
    trained_at: datetime | None = None
    created_at: datetime
    started_at: datetime | None = None
    mlflow_run_url: str | None = None
    sample_size_hint: int | None = None
    status: Literal["queued", "running", "paused", "ok", "error", "cancelled"]
    error: str | None = None
    progress: RuntimeProgress | None = None
    test_f1: TrainingResultTestF1Info | None = None
    pseudo_markup_results: list[PseudoMarkupResultInfo] = Field(default_factory=list)


class TrainingResultExportItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_id: UUID
    model_name: str
    sample_size: int | None = None
    context: int | None = Field(default=None, ge=0)


class TrainingResultBatchExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[TrainingResultExportItem] = Field(default_factory=list)


class DatasetResultsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_key: str
    dataset_name: str
    class_key: str | None = None
    class_name: str | None = None
    quality_metric: Literal["pixel", "objects"] = "pixel"
    dataset_updated_at: datetime | None = None
    primary_test_sample: PrimaryTestSampleInfo | None = None
    primary_test_samples: list[PrimaryTestSampleInfo] = Field(default_factory=list)
    test_f1_status: Literal["current", "stale", "running", "unavailable"] = "unavailable"
    results: list[TrainingResultInfo]


class ResultDatasetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    dataset_name: str | None = None
    class_key: str | None = None
    class_name: str | None = None
    quality_metric: Literal["pixel", "objects"] = "pixel"
    is_primary: bool = False
    image_count: int | None = None
    test_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    test_f1_metrics: dict[str, Any] = Field(default_factory=dict)
    test_f1_status: Literal["current", "stale"] | None = None
    test_f1_training_result_id: UUID | None = None


class ResultClassInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    updated_at: datetime | None = None
    datasets: list[ResultDatasetInfo] = Field(default_factory=list)
    is_custom: bool = False
    quality_metric: Literal["pixel", "objects"] = "pixel"


class ResultClassListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classes: list[ResultClassInfo] = Field(default_factory=list)


class ResultChangeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    item_type: Literal["job", "training_result", "pseudo_markup_result"] = "training_result"
    job_id: UUID | None = None
    type: JobType | None = None
    dataset_key: str
    class_key: str
    class_name: str | None = None
    dataset_name: str
    model_name: str
    action: str
    source: JobSource = JobSource.MANUAL
    status: Literal["queued", "running", "paused", "ok", "error", "cancelled"]
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
    "DatasetResultsResponse",
    "CustomDatasetInfo",
    "PrimaryTestSampleInfo",
    "PseudoMarkupResultInfo",
    "PseudoMarkupSceneInfo",
    "PseudoMarkupViewInfo",
    "PseudoMarkupComparisonRequest",
    "PseudoMarkupPixelComparison",
    "PseudoMarkupComparisonCounts",
    "PseudoMarkupComparisonLayers",
    "ResultClassInfo",
    "ResultClassListResponse",
    "ResultChangeInfo",
    "ResultChangesResponse",
    "ResultDatasetInfo",
    "TrainingResultBatchExportRequest",
    "TrainingResultExportItem",
    "TrainingResultInfo",
    "TrainingResultTestF1Info",
    "TestF1ScoreInfo",
    "TestF1SceneInfo",
    "TestF1ViewInfo",
]
