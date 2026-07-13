"""Контракты постоянных тестовых выборок."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TestSampleCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="", max_length=180)
    dataset_key: str = Field(min_length=1)
    tile_width: int = Field(default=1024, gt=0)
    tile_height: int = Field(default=1024, gt=0)
    image_count: int = Field(default=10, gt=0)
    object_count: int = Field(default=150, gt=0)


class TestSampleUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=180)


class TestSampleTileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


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

    status: Literal["current", "stale", "unavailable", "error"]
    pixel: TestSampleMetric | None = None
    objects: TestSampleMetric | None = None
    object_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    pseudo_markup_result_id: UUID | None = None
    model_name: str | None = None
    markup_created_at: datetime | None = None
    evaluated_at: datetime | None = None
    error: str | None = None


class TestSampleSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    name: str
    dataset_key: str
    dataset_name: str
    dataset_version: str | None = None
    class_key: str
    class_name: str
    variant_key: str
    variant_name: str
    image_count: int = Field(gt=0)
    enabled_image_count: int = Field(ge=0)
    actual_object_count: int = Field(gt=0)
    enabled_object_count: int = Field(ge=0)
    evaluation: TestSampleEvaluationInfo
    created_at: datetime
    updated_at: datetime


class TestSampleVariantGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    samples: list[TestSampleSummary] = Field(default_factory=list)


class TestSampleClassGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    variants: list[TestSampleVariantGroup] = Field(default_factory=list)


class TestSampleCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classes: list[TestSampleClassGroup] = Field(default_factory=list)


class TestSampleTileInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(gt=0)
    source_name: str
    territory: str
    object_count: int = Field(gt=0)
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


__all__ = [
    "TestSampleCatalogResponse",
    "TestSampleClassGroup",
    "TestSampleCreate",
    "TestSampleDetail",
    "TestSampleEvaluationInfo",
    "TestSampleMetric",
    "TestSampleSummary",
    "TestSampleTileInfo",
    "TestSampleTileUpdate",
    "TestSampleUpdate",
    "TestSampleVariantGroup",
]
