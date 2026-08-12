"""Публичные контракты подготовки тайлов."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TilePreparationError(RuntimeError):
    """Ошибка подготовки тайлов."""


HARD_NEGATIVE_LABEL = -1


class TileClassAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: int = Field(gt=0)
    slug: str
    name: str
    annotation_file: str | Path
    hard_negative_annotation_file: str | Path | None = None
    priority: int = 0


class TileClassDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: int = Field(gt=0)
    slug: str = Field(min_length=1)
    name: str = Field(min_length=1)
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    priority: int = 0


class TileSceneSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    image_path: str | Path
    annotation_file: str | Path | None = None


class TileSplitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    val_fraction: float = Field(gt=0.0, lt=1.0)
    seed: int = 42


class TileDataloaderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenes: list[TileSceneSource] = Field(min_length=1)
    annotation_file: str | Path | None = None
    hard_negative_annotation_file: str | Path | None = None
    class_annotations: list[TileClassAnnotation] = Field(default_factory=list)
    classes: list[TileClassDefinition] = Field(default_factory=list)
    batch_size: int = Field(gt=0)
    mode: Literal["train", "val"]
    tile_split: TileSplitRequest | None = None
    max_batches_per_epoch: int | None = Field(default=None, gt=0)
    include_object_instances: bool = False

    @model_validator(mode="after")
    def validate_annotation_mode(self) -> Self:
        has_per_image = any(item.annotation_file is not None for item in self.scenes)
        if has_per_image and not all(item.annotation_file is not None for item in self.scenes):
            raise ValueError("Все per-image сцены должны содержать annotation_file")
        has_global_binary = self.annotation_file is not None
        has_legacy_multiclass = bool(self.class_annotations)
        has_per_image_multiclass = has_per_image and bool(self.classes)
        mode_count = sum((has_global_binary, has_legacy_multiclass, has_per_image))
        if mode_count != 1:
            raise ValueError(
                "TileDataloaderRequest должен задавать один режим: глобальный binary, "
                "legacy multiclass или per-image"
            )
        if has_per_image and (
            self.annotation_file is not None
            or self.hard_negative_annotation_file is not None
            or has_legacy_multiclass
        ):
            raise ValueError(
                "Per-image TileDataloaderRequest не смешивается с глобальной разметкой"
            )
        if self.classes and not has_per_image:
            raise ValueError("classes допустимы только для per-image разметки")
        if has_legacy_multiclass and self.hard_negative_annotation_file is not None:
            raise ValueError(
                "hard_negative_annotation_file задается в TileDataloaderRequest только для binary режима"
            )
        if self.include_object_instances and (
            self.mode != "val" or has_legacy_multiclass or has_per_image_multiclass
        ):
            raise ValueError(
                "include_object_instances поддерживается только для binary val loader"
            )
        return self


__all__ = [
    "HARD_NEGATIVE_LABEL",
    "TileClassAnnotation",
    "TileClassDefinition",
    "TileDataloaderRequest",
    "TilePreparationError",
    "TileSceneSource",
    "TileSplitRequest",
]
