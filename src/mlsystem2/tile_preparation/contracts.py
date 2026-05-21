"""Публичные контракты подготовки тайлов."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TilePreparationError(RuntimeError):
    """Ошибка подготовки тайлов."""


class TileDataloaderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vrt_xml: str
    annotation_file: str | Path
    batch_size: int = Field(gt=0)
    mode: Literal["train", "val"]


__all__ = [
    "TileDataloaderRequest",
    "TilePreparationError",
]
