"""Публичные контракты подготовки тайлов."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TilePreparationError(RuntimeError):
    """Ошибка подготовки тайлов."""


class TileClassAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: int = Field(gt=0)
    slug: str
    name: str
    annotation_file: str | Path
    priority: int = 0


class TileSplitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    val_fraction: float = Field(gt=0.0, lt=1.0)
    seed: int = 42


class TileDataloaderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vrt_xml: str
    annotation_file: str | Path | None = None
    class_annotations: list[TileClassAnnotation] = Field(default_factory=list)
    batch_size: int = Field(gt=0)
    mode: Literal["train", "val"]
    tile_split: TileSplitRequest | None = None
    max_batches_per_epoch: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_annotation_mode(self) -> Self:
        has_binary = self.annotation_file is not None
        has_multiclass = bool(self.class_annotations)
        if has_binary == has_multiclass:
            raise ValueError(
                "TileDataloaderRequest должен задавать либо annotation_file, "
                "либо class_annotations"
            )
        return self


__all__ = [
    "TileClassAnnotation",
    "TileDataloaderRequest",
    "TilePreparationError",
    "TileSplitRequest",
]
