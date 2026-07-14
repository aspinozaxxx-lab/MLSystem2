"""Catalog and static metadata contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

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
    subclass_key: str | None = None
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
    quality_metric: Literal["pixel", "objects"] = "pixel"
    image_type: str = "all"
    images_dir: str | None = None
    source_type: str | None = None
    source_path: str | None = None
    source_available: bool = True
    is_primary: bool = False
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


class DatasetSubclassInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    is_primary: bool = False
    dataset: DatasetInfo | None = None


class ClassInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    updated_at: datetime | None = None
    variants: list[DatasetInfo] = Field(default_factory=list)
    subclasses: list[DatasetSubclassInfo] = Field(default_factory=list)
    is_custom: bool = False
    quality_metric: Literal["pixel", "objects"] = "pixel"
    primary_subclass_key: str | None = None


class QualityMetric(StrEnum):
    PIXEL = "pixel"
    OBJECTS = "objects"


class DatasetSourceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    path: str
    assigned_dataset_key: str | None = None
    diagnostics: list[str] = Field(default_factory=list)


class ImageTypeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    path: str
    image_count: int = Field(ge=0)


class DatasetCatalogInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classes: list[ClassInfo] = Field(default_factory=list)
    sources: list[DatasetSourceInfo] = Field(default_factory=list)
    image_types: list[ImageTypeInfo] = Field(default_factory=list)


class DatasetClassCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=240)


class DatasetClassUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=240)
    quality_metric: QualityMetric | None = None


class DatasetPrimarySubclassUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subclass_key: str = Field(min_length=1, max_length=180)


class DatasetSubclassCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_key: str = Field(min_length=1, max_length=180)
    name: str = Field(min_length=1, max_length=240)


class DatasetSubclassUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=240)


class ManagedDatasetCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subclass_key: str = Field(min_length=1, max_length=180)
    source_path: str = Field(min_length=1, max_length=1024)
    image_type: str = Field(default="all", min_length=1, max_length=240)


class ManagedDatasetUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(min_length=1, max_length=1024)
    image_type: str = Field(default="all", min_length=1, max_length=240)


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
    "DatasetCatalogInfo",
    "DatasetClassCreate",
    "DatasetClassUpdate",
    "DatasetInfo",
    "DatasetListResponse",
    "DatasetPrimarySubclassUpdate",
    "DatasetSourceInfo",
    "DatasetSubclassCreate",
    "DatasetSubclassInfo",
    "DatasetSubclassUpdate",
    "ImageFolderInfo",
    "ImageFolderListResponse",
    "ImageTypeInfo",
    "ManagedDatasetCreate",
    "ManagedDatasetUpdate",
    "MLflowExperimentCreate",
    "MLflowExperimentInfo",
    "ModelInfo",
    "ModelListResponse",
    "QualityMetric",
]
