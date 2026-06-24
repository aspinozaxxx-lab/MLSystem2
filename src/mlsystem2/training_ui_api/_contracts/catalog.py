"""Catalog and static metadata contracts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


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
    hard_negative_annotation_file: str | None = None
    image_count: int | None = None
    version: str | None = None
    updated_at: datetime | None = None
    diagnostics: list[str] = Field(default_factory=list)


class DatasetListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets: list[DatasetInfo]


class ImageFolderInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    path: str
    image_count: int


class ImageFolderListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folders: list[ImageFolderInfo]


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


__all__ = [
    "AppLink",
    "AppLinksResponse",
    "ClassInfo",
    "ClassListResponse",
    "DatasetInfo",
    "DatasetListResponse",
    "ImageFolderInfo",
    "ImageFolderListResponse",
    "MLflowExperimentCreate",
    "MLflowExperimentInfo",
    "ModelInfo",
    "ModelListResponse",
]
