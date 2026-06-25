"""Shared public contracts for the training UI API."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


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


class RuntimeProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current: int | None = None
    total: int | None = None
    elapsed_minutes: int | None = None


class StoredFileInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    kind: StoredFileKind
    original_name: str
    size_bytes: int
    object_count: int | None = None
    created_at: datetime
    download_url: str


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
    recommended_range: str | None = None


class ConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: list[ConfigField]


JsonDict = dict[str, Any]


__all__ = [
    "ConfigField",
    "ConfigSchema",
    "JobSource",
    "JobStatus",
    "JobType",
    "JsonDict",
    "ResultStatus",
    "RuntimeProgress",
    "StoredFileInfo",
    "StoredFileKind",
    "TemplateSource",
    "TrainingUIAPIError",
]
