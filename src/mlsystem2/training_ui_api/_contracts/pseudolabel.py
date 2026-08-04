"""Publichnye kontrakty AOI pseudolabel API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PseudolabelAPIError(RuntimeError):
    """Strukturirovannaya domennaya oshibka AOI API."""

    # Sohranyaet stabilnyi kod, HTTP-status i bezopasnye detali.
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class PseudolabelGeometry(BaseModel):
    """Poligonalnaya geometriya AOI."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["Polygon", "MultiPolygon"]
    coordinates: list[Any]


class PseudolabelJobCreate(BaseModel):
    """Minimalnyi zapros na servernoe raspoznavanie."""

    model_config = ConfigDict(extra="forbid")

    class_id: str = Field(min_length=1, max_length=180)
    source_id: str | None = Field(default=None, min_length=1, max_length=180)
    aoi: PseudolabelGeometry
    aoi_crs: str = Field(min_length=1, max_length=128)


class PseudolabelClassInfo(BaseModel):
    """Klass i poslednyaya prigodnaya model."""

    model_config = ConfigDict(extra="forbid")

    class_id: str
    display_name: str
    model_id: UUID
    model_version: str
    model_name: str
    trained_at: datetime
    model_imagery_type: Literal["kanopus", "ortho"]
    input_channels: Literal[3, 4]
    target_resolution_m: float | None = Field(default=None, gt=0)


class PseudolabelSourceInfo(BaseModel):
    """Доступный серверу источник снимков."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    display_name: str
    kind: Literal["local", "catalog", "xyz", "tms", "wmts"]
    protocol: str
    native_channels: int = Field(gt=0)
    imagery_type: str | None = None
    attribution: str
    license_url: str
    available: bool


class PseudolabelClassListResponse(BaseModel):
    """Spisok dostupnyh klassov."""

    model_config = ConfigDict(extra="forbid")

    classes: list[PseudolabelClassInfo] = Field(default_factory=list)
    sources: list[PseudolabelSourceInfo] = Field(default_factory=list)


class PseudolabelErrorInfo(BaseModel):
    """Strukturirovannaya oshibka job."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class PseudolabelJobInfo(BaseModel):
    """Publichnoe sostoyanie AOI job."""

    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    class_id: str
    model_id: UUID
    model_version: str
    model_name: str
    created_at: datetime
    finished_at: datetime | None = None
    progress: float | None = Field(default=None, ge=0.0, le=100.0)
    current_stage: str
    error: PseudolabelErrorInfo | None = None
    warnings: list[str] = Field(default_factory=list)
    source_image_ids: list[str] = Field(default_factory=list)
    coverage_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    source_id: str
    source_name: str
    model_imagery_type: Literal["kanopus", "ortho"]
    source_imagery_type: str
    channel_mapping: str
    target_resolution_m: float | None = Field(default=None, gt=0)
    source_attributions: list[str] = Field(default_factory=list)
    source_license_url: str = ""
    performance: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "PseudolabelAPIError",
    "PseudolabelClassInfo",
    "PseudolabelClassListResponse",
    "PseudolabelErrorInfo",
    "PseudolabelGeometry",
    "PseudolabelJobCreate",
    "PseudolabelJobInfo",
    "PseudolabelSourceInfo",
]
