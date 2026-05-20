"""Публичные контракты подготовки датасета."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetPreparationError(RuntimeError):
    """Невосстановимая ошибка подготовки датасета."""


class DatasetPreparationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    images_dir: str
    scenes_file: str
    annotation_file: str
    val_fraction: float = Field(gt=0.0, lt=1.0)


class PreparedDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_vrt_xml: str
    val_vrt_xml: str
    annotation_file: str


class DatasetSceneReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    image_path: str | None
    object_count: int = Field(ge=0)
    split: Literal["train", "val", "missing"]


class DatasetPreparationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "error"]
    scenes_total: int = Field(ge=0)
    scenes_found: int = Field(ge=0)
    objects_total: int = Field(ge=0)
    train_scenes_count: int = Field(ge=0)
    train_objects_count: int = Field(ge=0)
    val_scenes_count: int = Field(ge=0)
    val_objects_count: int = Field(ge=0)
    scenes: list[DatasetSceneReport]
    missing_files: list[str]
    errors: list[str]


class DatasetPreparationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: PreparedDataset | None
    report: DatasetPreparationReport


__all__ = [
    "DatasetPreparationError",
    "DatasetPreparationRequest",
    "PreparedDataset",
    "DatasetSceneReport",
    "DatasetPreparationReport",
    "DatasetPreparationResult",
]
