"""Catalog and static metadata contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImageryType(StrEnum):
    KANOPUS = "kanopus"
    ORTHO = "ortho"


class DatasetFormat(StrEnum):
    LEGACY = "legacy"
    PER_IMAGE = "per_image"
    PER_IMAGE_MULTICLASS = "per_image_multiclass"


class DatasetObjectType(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(gt=0)
    slug: str
    name: str
    color: str
    priority: int = 0


class ManagedDatasetSourceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_key: str
    dataset_name: str
    class_key: str
    class_name: str
    priority: int
    object_type_id: int = Field(gt=0)
    object_type_slug: str
    color: str


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
    dataset_name: str | None = None
    class_key: str | None = None
    class_name: str | None = None
    path: str | None = None
    is_custom: bool = False
    scenes_file: str | None = None
    annotation_file: str | None = None
    hard_negative_annotation_file: str | None = None
    format: DatasetFormat | None = None
    annotations_dir: str | None = None
    image_count: int | None = None
    version: str | None = None
    updated_at: datetime | None = None
    quality_metric: Literal["pixel", "objects"] = "pixel"
    imagery_type: ImageryType | None = None
    input_channels: int | None = Field(default=None, gt=0)
    images_dir: str | None = None
    source_type: str | None = None
    source_path: str | None = None
    source_available: bool = True
    is_primary: bool = False
    diagnostics: list[str] = Field(default_factory=list)
    task: Literal["binary", "multiclass"] = "binary"
    object_types: list[DatasetObjectType] = Field(default_factory=list)
    combined: bool = False
    managed: bool = False
    managed_sources: list[ManagedDatasetSourceInfo] = Field(default_factory=list)
    source_status: Literal["current", "stale", "unknown", "unavailable"] = "unknown"
    source_changes: list[str] = Field(default_factory=list)
    class_counts: dict[str, int] = Field(default_factory=dict)
    hard_negative_count: int = Field(default=0, ge=0)
    manifest_path: str | None = None


class DatasetListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets: list[DatasetInfo]


class ImageFolderInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    path: str
    image_count: int
    imagery_type: ImageryType


class ImageFolderListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folders: list[ImageFolderInfo]


class ClassInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    updated_at: datetime | None = None
    datasets: list[DatasetInfo] = Field(default_factory=list)
    is_custom: bool = False
    quality_metric: Literal["pixel", "objects"] = "pixel"
    imagery_type: ImageryType | None = None
    primary_dataset_key: str | None = None


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


class ImageryTypeInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: ImageryType
    name: str
    folder: str
    path: str
    input_channels: int = Field(gt=0)
    image_count: int = Field(ge=0)


class DatasetCatalogInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classes: list[ClassInfo] = Field(default_factory=list)
    sources: list[DatasetSourceInfo] = Field(default_factory=list)
    imagery_types: list[ImageryTypeInfo] = Field(default_factory=list)


class DatasetClassCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=240)
    imagery_type: ImageryType


class DatasetClassUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=240)
    quality_metric: QualityMetric | None = None
    imagery_type: ImageryType | None = None


class DatasetPrimaryDatasetUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_key: str = Field(min_length=1, max_length=180)


class ManagedDatasetCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_key: str = Field(min_length=1, max_length=180)
    name: str = Field(min_length=1, max_length=240)
    source_path: str = Field(min_length=1, max_length=1024)


class ManagedDatasetCompositionSourceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_key: str = Field(min_length=1, max_length=180)
    priority: int = 0
    color: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")


class ManagedDatasetUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=240)
    source_path: str | None = Field(default=None, min_length=1, max_length=1024)
    sources: list[ManagedDatasetCompositionSourceCreate] | None = Field(
        default=None,
        min_length=2,
    )

    @model_validator(mode="after")
    def validate_update(self) -> Self:
        if self.name is None and self.source_path is None and self.sources is None:
            raise ValueError("Нужно передать хотя бы одно изменение датасета.")
        if self.sources is not None:
            keys = [item.dataset_key for item in self.sources]
            if len(keys) != len(set(keys)):
                raise ValueError("sources не должен содержать повторяющиеся датасеты")
        return self


class ManagedDatasetCompositionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_key: str = Field(min_length=1, max_length=180)
    name: str = Field(min_length=1, max_length=240)
    sources: list[ManagedDatasetCompositionSourceCreate] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_unique_sources(self) -> Self:
        keys = [item.dataset_key for item in self.sources]
        if len(keys) != len(set(keys)):
            raise ValueError("sources не должен содержать повторяющиеся датасеты")
        return self


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
    "DatasetFormat",
    "DatasetInfo",
    "DatasetListResponse",
    "DatasetObjectType",
    "DatasetPrimaryDatasetUpdate",
    "DatasetSourceInfo",
    "ImageryType",
    "ImageryTypeInfo",
    "ImageFolderInfo",
    "ImageFolderListResponse",
    "ManagedDatasetCreate",
    "ManagedDatasetCompositionCreate",
    "ManagedDatasetCompositionSourceCreate",
    "ManagedDatasetSourceInfo",
    "ManagedDatasetUpdate",
    "MLflowExperimentCreate",
    "MLflowExperimentInfo",
    "ModelInfo",
    "ModelListResponse",
    "QualityMetric",
]
