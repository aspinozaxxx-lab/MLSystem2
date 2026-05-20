"""Публичные контракты подготовки датасета."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetPreparationError(RuntimeError):
    """Невосстановимая ошибка подготовки датасета."""


class SceneRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    image_uri: str
    annotation_uri: str | None = None
    object_count: int = Field(ge=0)


class SceneFootprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    image_uri: str
    geometry_wkt: str


class ObjectCountByScene(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    object_count: int = Field(ge=0)


class DatasetSplit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_scenes: list[str]
    val_scenes: list[str]


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    images_uri: str
    scenes: list[SceneRecord]
    split: DatasetSplit
    footprints: list[SceneFootprint]
    object_counts: list[ObjectCountByScene]
    footprints_artifact_uri: str | None = None


class DatasetPreparationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["succeeded", "failed"]
    scenes_total: int = Field(ge=0)
    train_scenes: list[str]
    val_scenes: list[str]
    object_counts: list[ObjectCountByScene]
    missing_files: list[str]
    warnings: list[str]
    errors: list[str]
    footprints_artifact_uri: str | None = None


class DatasetPreparationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    images_uri: str
    scenes_file: str
    annotation_file: str
    default_class_dir: str
    val_fraction: float = Field(gt=0.0, lt=1.0)
    split_strategy: Literal["object_count_balanced"]
    output_uri: str | None = None


class DatasetPreparationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["succeeded", "failed"]
    manifest: DatasetManifest | None = None
    report: DatasetPreparationReport
    errors: list[str]


__all__ = [
    "DatasetManifest",
    "DatasetPreparationError",
    "DatasetPreparationReport",
    "DatasetPreparationRequest",
    "DatasetPreparationResult",
    "DatasetSplit",
    "ObjectCountByScene",
    "SceneFootprint",
    "SceneRecord",
]
