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
    annotations_dir: str | None = None
    classes: list[DatasetClassRequest] | None = None
    val_fraction: float = Field(gt=0.0, lt=1.0)
    expected_band_count: int | None = Field(default=None, gt=0)
    expected_dtype: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_dataset_mode(self) -> Self:
        classes = self.classes or []
        has_legacy_binary_paths = (
            self.scenes_file is not None
            or self.annotation_file is not None
            or self.hard_negative_annotation_file is not None
        )
        has_per_image_binary = self.annotations_dir is not None
        mode_count = sum((has_legacy_binary_paths, has_per_image_binary, bool(classes)))
        if mode_count != 1:
            raise ValueError(
                "DatasetPreparationRequest должен задавать ровно один режим: classes, "
                "scenes_file + annotation_file или annotations_dir"
            )
        if classes:
            _validate_unique_values([item.slug for item in classes], "slug")
            _validate_unique_values([item.name for item in classes], "name")
            return self
        if has_per_image_binary:
            if not self.annotations_dir:
                raise ValueError("annotations_dir не должен быть пустым")
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


class DatasetClassDefinition(BaseModel):
    """Класс объектов в per-image multiclass-датасете."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(gt=0)
    slug: str = Field(min_length=1)
    name: str = Field(min_length=1)
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    priority: int = 0


class DatasetSourceRevision(BaseModel):
    """Зафиксированное состояние исходной папки комбинированного датасета."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    class_slug: str = Field(min_length=1)
    git_revision: str = Field(min_length=1)
    tree_revision: str = Field(min_length=1)
    file_hashes: dict[str, str] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    """Содержимое `.mlsystem2-dataset.json`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    task: Literal["multiclass"]
    combined: bool = False
    classes: list[DatasetClassDefinition] = Field(min_length=2)
    sources: list[DatasetSourceRevision] = Field(default_factory=list)
    build_id: str | None = None
    built_at: str | None = None
    code_revision: str | None = None
    scene_ids: list[str] = Field(default_factory=list)
    source_warnings: list[str] = Field(default_factory=list)
    baseline_hashes: dict[str, dict[str, str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_classes(self) -> Self:
        _validate_unique_values([str(item.id) for item in self.classes], "id")
        _validate_unique_values([item.slug for item in self.classes], "slug")
        _validate_unique_values([item.name for item in self.classes], "name")
        expected_ids = list(range(1, len(self.classes) + 1))
        actual_ids = sorted(item.id for item in self.classes)
        if actual_ids != expected_ids:
            raise ValueError(
                "classes должен использовать последовательные id от 1 до количества классов"
            )
        if self.combined:
            source_slugs = [item.class_slug for item in self.sources]
            _validate_unique_values(source_slugs, "source class_slug")
            unknown = sorted(set(source_slugs) - {item.slug for item in self.classes})
            if unknown:
                raise ValueError(
                    "sources содержит неизвестные class_slug: " + ", ".join(unknown)
                )
        return self


class PreparedScene(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    image_path: str
    annotation_file: str | None = None
    footprint_file: str | None = None


class PreparedDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal[
        "legacy_binary",
        "per_image_binary",
        "legacy_multiclass",
        "per_image_multiclass",
    ]
    scenes: list[PreparedScene] = Field(min_length=1)
    annotation_file: str | None = None
    hard_negative_annotation_file: str | None = None
    class_annotations: list[DatasetClassAnnotation] = Field(default_factory=list)
    classes: list[DatasetClassDefinition] = Field(default_factory=list)
    manifest_file: str | None = None


class DatasetSceneReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    image_path: str | None
    positive_objects: int = Field(ge=0)
    hard_negative_objects: int = Field(ge=0)
    object_count: int = Field(ge=0)
    class_counts: dict[str, int] = Field(default_factory=dict)


class DatasetPreparationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error"]
    scenes_total: int = Field(ge=0)
    scenes_found: int = Field(ge=0)
    positive_objects: int = Field(ge=0)
    hard_negative_objects: int = Field(ge=0)
    objects_total: int = Field(ge=0)
    class_counts: dict[str, int] = Field(default_factory=dict)
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
    scenes_file: str | None = None
    annotations_dir: str | None = None
    annotation_files: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_resolution_mode(self) -> Self:
        if bool(self.scenes_file) == bool(self.annotations_dir):
            raise ValueError(
                "SceneImageResolutionRequest должен задавать scenes_file или annotations_dir"
            )
        if self.annotations_dir and self.annotation_files:
            raise ValueError("annotation_files допустимы только вместе со scenes_file")
        return self


class ResolvedSceneImage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    image_path: str
    annotation_file: str | None = None
    footprint_file: str | None = None
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
    "DatasetClassDefinition",
    "DatasetClassRequest",
    "DatasetManifest",
    "DatasetPreparationError",
    "DatasetPreparationRequest",
    "DatasetSourceRevision",
    "PreparedDataset",
    "DatasetSceneReport",
    "DatasetPreparationReport",
    "DatasetPreparationResult",
    "PreparedScene",
    "ResolvedSceneImage",
    "SceneImageResolution",
    "SceneImageResolutionRequest",
]
