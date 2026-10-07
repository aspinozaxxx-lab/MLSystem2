"""Template contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .common import ConfigSchema, TemplateSource


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
    display_name: str
    description: str | None = None
    class_keys: list[str]
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


class InferenceTemplateUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, min_length=1, max_length=240)
    description: str | None = Field(default=None, max_length=2000)
    default_config: dict[str, Any] | None = None
    is_active: bool | None = None
    reset_to_baseline: bool = False


class InferenceTemplateCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(min_length=1, max_length=240)
    description: str | None = Field(default=None, max_length=2000)


class InferenceTemplateClassUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template_id: UUID | None


InferenceTemplateApplyField = TrainingTemplateApplyField


__all__ = [
    "InferenceTemplate",
    "InferenceTemplateApplyField",
    "InferenceTemplateCreate",
    "InferenceTemplateClassUpdate",
    "InferenceTemplateListResponse",
    "InferenceTemplateUpdate",
    "TrainingTemplate",
    "TrainingTemplateApplyField",
    "TrainingTemplateCreate",
    "TrainingTemplateListResponse",
    "TrainingTemplateUpdate",
]
