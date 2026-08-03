"""Публичные контракты подготовки датасета."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetPreparationError(RuntimeError):
    """Невосстановимая ошибка подготовки датасета."""


class DatasetClassRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str
    name: str
    scenes_file: str
    annotation_file: str
    hard_negative_annotation_file: str | None = None
    priority: int = 0


class DatasetPreparationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    images_dir: str
    scenes_file: str | None = None
    annotation_file: str | None = None
    hard_negative_annotation_file: str | None = None
    classes: list[DatasetClassRequest] | None = None
    val_fraction: float = Field(gt=0.0, lt=1.0)
    expected_band_count: int | None = Field(default=None, gt=0)
    expected_dtype: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_dataset_mode(self) -> Self:
        classes = self.classes or []
        has_binary_paths = (
            self.scenes_file is not None
            or self.annotation_file is not None
            or self.hard_negative_annotation_file is not None
        )
        if has_binary_paths and classes:
            raise ValueError(
                "DatasetPreparationRequest должен задавать либо classes, "
                "либо scenes_file + annotation_file"
            )
        if classes:
            _validate_unique_values([item.slug for item in classes], "slug")
            _validate_unique_values([item.name for item in classes], "name")
            return self
        if not self.scenes_file or not self.annotation_file:
            raise ValueError(
                "binary DatasetPreparationRequest должен задавать scenes_file и annotation_file"
            )
        return self


class DatasetClassAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: int = Field(gt=0)
    slug: str
    name: str
    annotation_file: str
    hard_negative_annotation_file: str | None = None
    priority: int = 0


class PreparedDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_vrt_xml: str
    val_vrt_xml: str
    pool_vrt_xml: str | None = None
    annotation_file: str | None = None
    hard_negative_annotation_file: str | None = None
    class_annotations: list[DatasetClassAnnotation] = Field(default_factory=list)


class DatasetSceneReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    image_path: str | None
    positive_objects: int = Field(ge=0)
    hard_negative_objects: int = Field(ge=0)
    object_count: int = Field(ge=0)


class DatasetPreparationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error"]
    scenes_total: int = Field(ge=0)
    scenes_found: int = Field(ge=0)
    positive_objects: int = Field(ge=0)
    hard_negative_objects: int = Field(ge=0)
    objects_total: int = Field(ge=0)
    band_count: int | None = Field(default=None, gt=0)
    dtypes: list[str] = Field(default_factory=list)
    scenes: list[DatasetSceneReport]
    missing_files: list[str]
    errors: list[str]


class DatasetPreparationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: PreparedDataset | None
    report: DatasetPreparationReport


class SceneImageResolutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    images_dir: str
    scenes_file: str
    annotation_files: list[str] = Field(default_factory=list)


class ResolvedSceneImage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    image_path: str
    request_scenes: list[str] = Field(default_factory=list)


class SceneImageResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_scene_count: int = Field(ge=0)
    images: list[ResolvedSceneImage] = Field(default_factory=list)
    missing_scenes: list[str] = Field(default_factory=list)
    ambiguous_scenes: dict[str, list[str]] = Field(default_factory=dict)


def _validate_unique_values(values: list[str], field_name: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        joined = ", ".join(sorted(duplicates))
        raise ValueError(f"classes должен иметь уникальные {field_name}: {joined}")


__all__ = [
    "DatasetClassAnnotation",
    "DatasetClassRequest",
    "DatasetPreparationError",
    "DatasetPreparationRequest",
    "PreparedDataset",
    "DatasetSceneReport",
    "DatasetPreparationReport",
    "DatasetPreparationResult",
    "ResolvedSceneImage",
    "SceneImageResolution",
    "SceneImageResolutionRequest",
]
