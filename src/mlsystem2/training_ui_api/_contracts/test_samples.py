"""Контракты постоянных тестовых разметок."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import RuntimeProgress


class TestSampleCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="", max_length=180)
    dataset_key: str = Field(min_length=1)
    tile_width: int = Field(default=1536, gt=0)
    tile_height: int = Field(default=1536, gt=0)
    image_count: int = Field(default=10, gt=0)
    object_count: int = Field(default=150, gt=0)


class TestSampleUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=180)
    is_primary: bool | None = None
    enabled_tile_indices: list[int] | None = None

    @model_validator(mode="after")
    def validate_not_empty(self) -> Self:
        if (
            self.name is None
            and self.is_primary is None
            and self.enabled_tile_indices is None
        ):
            raise ValueError("Нужно передать хотя бы одно изменение тестовой разметки.")
        return self


class TestSampleTileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


class TestSamplePrimaryUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_primary: bool


class TestSampleOptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_tile_count: int = Field(gt=0)
    max_tile_count: int = Field(gt=0)
    min_object_count: int = Field(gt=0)
    metric: Literal["pixel", "objects"] = "pixel"


class TestSampleEvaluationPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled_tile_indices: list[int] = Field(default_factory=list)


class TestSampleDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled_tile_indices: list[int]
    include_previews: bool = True


class TestSampleBulkDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_ids: list[UUID] = Field(
        min_length=1,
        description=(
            "Уникальные идентификаторы сохранённых тестовых разметок; "
            "не более одной разметки для каждого dataset_key."
        ),
        json_schema_extra={"uniqueItems": True},
    )
    include_previews: bool = True

    @model_validator(mode="after")
    def validate_unique_sample_ids(self) -> Self:
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Идентификаторы тестовых разметок не должны повторяться.")
        return self


class TestSampleMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)


class TestSampleEvaluationInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["current", "stale", "queued", "running", "unavailable", "error"]
    pixel: TestSampleMetric | None = None
    objects: TestSampleMetric | None = None
    object_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    pseudo_markup_result_id: UUID | None = None
    training_result_id: UUID | None = None
    target_training_result_id: UUID | None = None
    model_name: str | None = None
    target_model_name: str | None = None
    threshold: float | None = None
    job_id: UUID | None = None
    progress: RuntimeProgress | None = None
    markup_created_at: datetime | None = None
    evaluated_at: datetime | None = None
    error: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class TestSampleSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    name: str
    dataset_key: str
    dataset_name: str
    dataset_version: str | None = None
    class_key: str
    class_name: str
    task: Literal["binary", "multiclass"] = "binary"
    class_schema: list[dict[str, Any]] = Field(default_factory=list)
    class_object_counts: dict[str, int] = Field(default_factory=dict)
    quality_metric: Literal["pixel", "objects"] = "pixel"
    image_count: int = Field(gt=0)
    enabled_image_count: int = Field(ge=0)
    actual_object_count: int = Field(gt=0)
    enabled_object_count: int = Field(ge=0)
    is_primary: bool = False
    evaluation: TestSampleEvaluationInfo
    created_at: datetime
    updated_at: datetime


class TestSampleDatasetGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    samples: list[TestSampleSummary] = Field(default_factory=list)


class TestSampleClassGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    datasets: list[TestSampleDatasetGroup] = Field(default_factory=list)


class TestSampleCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classes: list[TestSampleClassGroup] = Field(default_factory=list)


class TestSampleTileInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(gt=0)
    source_name: str
    territory: str
    object_count: int = Field(gt=0)
    class_object_counts: dict[str, int] = Field(default_factory=dict)
    evaluation_metrics: dict[str, Any] = Field(default_factory=dict)
    f1_score: float | None = Field(default=None, ge=0.0, le=1.0)
    enabled: bool
    preview_url: str


class TestSampleDetail(TestSampleSummary):
    model_config = ConfigDict(extra="forbid")

    tile_width: int = Field(gt=0)
    tile_height: int = Field(gt=0)
    requested_object_count: int = Field(gt=0)
    territory_count: int = Field(gt=0)
    warnings: list[str] = Field(default_factory=list)
    download_url: str
    tiles: list[TestSampleTileInfo] = Field(default_factory=list)


class TestSampleDraftPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled_tile_indices: list[int] = Field(default_factory=list)
    enabled_image_count: int = Field(ge=0)
    enabled_object_count: int = Field(ge=0)
    evaluation: TestSampleEvaluationInfo


class TestSampleBatchItemCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_key: str = Field(min_length=1)
    min_object_count: int = Field(default=150, gt=0)
    metric: Literal["pixel", "objects"] = "pixel"


class TestSampleBatchCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tile_size: Literal[512, 768, 1024, 1536, 2048, 2560, 3072, 3584] = 1536
    min_image_count: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Минимальное число включённых тайлов; без поля используется image_count."
        ),
    )
    image_count: int = Field(
        default=10,
        gt=0,
        description="Максимальное число включённых тайлов.",
    )
    items: list[TestSampleBatchItemCreate] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_image_count_range(self) -> Self:
        if self.min_image_count is None:
            self.min_image_count = self.image_count
        if self.min_image_count > self.image_count:
            raise ValueError(
                "Минимальное число снимков не может быть больше максимального."
            )
        return self


class TestSampleBatchItemInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    position: int = Field(ge=1)
    dataset_key: str
    dataset_name: str
    dataset_version: str | None = None
    class_key: str
    class_name: str
    min_object_count: int = Field(gt=0)
    metric: Literal["pixel", "objects"]
    status: Literal["queued", "running", "ok", "error"]
    pool_tile_count: int | None = Field(default=None, gt=0)
    pool_object_count: int | None = Field(default=None, gt=0)
    sample_id: UUID | None = None
    sample_name: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class TestSampleBatchInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    status: Literal["queued", "running", "ok", "partial", "error"]
    tile_size: int = Field(gt=0)
    min_image_count: int = Field(gt=0)
    image_count: int = Field(gt=0)
    completed_count: int = Field(ge=0)
    total_count: int = Field(gt=0)
    elapsed_seconds: int = Field(ge=0)
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    items: list[TestSampleBatchItemInfo] = Field(default_factory=list)


__all__ = [
    "TestSampleBatchCreate",
    "TestSampleBatchInfo",
    "TestSampleBatchItemCreate",
    "TestSampleBatchItemInfo",
    "TestSampleBulkDownloadRequest",
    "TestSampleCatalogResponse",
    "TestSampleClassGroup",
    "TestSampleDatasetGroup",
    "TestSampleCreate",
    "TestSampleDetail",
    "TestSampleDownloadRequest",
    "TestSampleDraftPreview",
    "TestSampleEvaluationPreviewRequest",
    "TestSampleEvaluationInfo",
    "TestSampleMetric",
    "TestSampleOptimizeRequest",
    "TestSamplePrimaryUpdate",
    "TestSampleSummary",
    "TestSampleTileInfo",
    "TestSampleTileUpdate",
    "TestSampleUpdate",
]
