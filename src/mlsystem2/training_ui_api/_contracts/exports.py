"""Контракты временного экспорта тестовой разметки."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MarkupExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_key: str = Field(min_length=1)
    tile_width: int = Field(default=1536, gt=0)
    tile_height: int = Field(default=1536, gt=0)
    image_count: int = Field(default=10, gt=0)
    object_count: int = Field(default=150, gt=0)
    exclude_boundary_objects: bool = False


class MarkupExportTileInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(gt=0)
    source_name: str
    territory: str
    object_count: int = Field(gt=0)
    preview_url: str


class MarkupExportInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    dataset_key: str
    dataset_name: str
    dataset_version: str | None = None
    tile_width: int = Field(gt=0)
    tile_height: int = Field(gt=0)
    image_count: int = Field(gt=0)
    requested_object_count: int = Field(gt=0)
    actual_object_count: int = Field(gt=0)
    exclude_boundary_objects: bool = False
    territory_count: int = Field(gt=0)
    warnings: list[str] = Field(default_factory=list)
    expires_at: datetime
    download_url: str
    tiles: list[MarkupExportTileInfo]


__all__ = [
    "MarkupExportInfo",
    "MarkupExportRequest",
    "MarkupExportTileInfo",
]
