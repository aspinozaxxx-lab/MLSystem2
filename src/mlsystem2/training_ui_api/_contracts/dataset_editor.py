"""Контракты редактора per-image датасетов."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .catalog import ManagedDatasetSourceInfo


class DatasetEditorObjectType(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(gt=0)
    slug: str
    name: str
    color: str
    priority: int = 0


class DatasetEditorDatasetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    class_key: str
    class_name: str
    dataset_name: str
    imagery_type: Literal["kanopus", "ortho"]
    scene_count: int = Field(ge=0)
    task: Literal["binary", "multiclass"] = "binary"
    object_types: list[DatasetEditorObjectType] = Field(default_factory=list)
    combined: bool = False
    managed: bool = False
    managed_sources: list[ManagedDatasetSourceInfo] = Field(default_factory=list)
    source_status: Literal["current", "stale", "unknown", "unavailable"] = "unknown"
    source_changes: list[str] = Field(default_factory=list)
    class_counts: dict[str, int] = Field(default_factory=dict)
    hard_negative_count: int = Field(default=0, ge=0)
    primary_training_result_id: UUID | None = None


class DatasetEditorDatasetListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets: list[DatasetEditorDatasetInfo]


class DatasetEditorDraftSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    annotation_name: str
    base_revision: str
    deleted: bool = False
    stale: bool = False
    total_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    hard_negative_count: int = Field(ge=0)
    class_counts: dict[str, int] = Field(default_factory=dict)
    updated_at: datetime


class DatasetEditorDraftInfo(DatasetEditorDraftSummary):
    model_config = ConfigDict(extra="forbid")

    geojson: dict[str, Any]


class DatasetEditorSceneInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    annotation_name: str
    image_name: str
    raster_url: str
    total_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    hard_negative_count: int = Field(ge=0)
    revision: str
    class_counts: dict[str, int] = Field(default_factory=dict)
    draft: DatasetEditorDraftSummary | None = None


class DatasetEditorSceneListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetEditorDatasetInfo
    scenes: list[DatasetEditorSceneInfo]


class DatasetEditorSceneDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene: DatasetEditorSceneInfo
    geojson: dict[str, Any]
    valid_data_footprint: dict[str, Any]
    draft: DatasetEditorDraftInfo | None = None


class DatasetEditorPseudoMarkupInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["unavailable", "ready", "queued", "running", "failed"]
    source: Literal["dataset", "scene"] | None = None
    training_result_id: UUID | None = None
    model_name: str | None = None
    job_id: UUID | None = None
    progress_current: int | None = Field(default=None, ge=0)
    progress_total: int | None = Field(default=None, ge=0)
    object_count: int = Field(default=0, ge=0)
    message: str | None = None
    geojson: dict[str, Any] | None = None
    can_retry: bool = False


class DatasetEditorSaveDraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_revision: str = Field(min_length=1, max_length=128)
    geojson: dict[str, Any]
    deleted: bool = False


class DatasetEditorDiscardDraftsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deleted_count: int = Field(ge=0)


class DatasetEditorRasterFolderInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    path: str


class DatasetEditorRasterInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    path: str
    annotation_name: str
    size_bytes: int = Field(ge=0)


class DatasetEditorRasterBrowserResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folder: str
    parent: str | None = None
    folders: list[DatasetEditorRasterFolderInfo]
    rasters: list[DatasetEditorRasterInfo]


class DatasetEditorAddScenesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_paths: list[str] = Field(default_factory=list)
    folder_path: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if bool(self.image_paths) == bool(self.folder_path):
            raise ValueError("Передайте image_paths или folder_path")
        if len(self.image_paths) != len(set(self.image_paths)):
            raise ValueError("image_paths не должен содержать повторов")
        return self


class DatasetEditorSaveSceneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: str = Field(min_length=1, max_length=128)
    geojson: dict[str, Any]


class DatasetEditorPublishSceneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    annotation_name: str = Field(min_length=1, max_length=512)
    revision: str = Field(min_length=1, max_length=128)
    geojson: dict[str, Any]


class DatasetEditorPublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenes: list[DatasetEditorPublishSceneRequest] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_scenes(self) -> Self:
        names = [scene.annotation_name.casefold() for scene in self.scenes]
        if len(names) != len(set(names)):
            raise ValueError("scenes не должен содержать повторяющиеся annotation_name")
        return self


class DatasetEditorDeleteSceneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: str = Field(min_length=1, max_length=128)


class DatasetEditorMutationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commit: str
    publication_status: Literal["publishing", "published"]
    scenes: list[DatasetEditorSceneInfo] = Field(default_factory=list)


class DatasetEditorPublicationInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commit: str
    live_commit: str | None = None
    status: Literal["publishing", "published"]


class DatasetEditorRebuildPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetEditorRebuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preview_token: str = Field(min_length=1, max_length=128)
    mode: Literal["merge", "replace"]


class DatasetEditorRebuildChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["added", "edited", "deleted", "source_added", "source_edited", "source_deleted"]
    annotation_name: str
    origin_key: str | None = None
    detail: str | None = None


class DatasetEditorRebuildPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preview_token: str
    dataset_key: str
    source_status: Literal["current", "stale", "unknown", "unavailable"]
    source_changes: list[str] = Field(default_factory=list)
    local_changes: list[DatasetEditorRebuildChange] = Field(default_factory=list)
    conflicts: list[DatasetEditorRebuildChange] = Field(default_factory=list)
    replacement_scene_count: int = Field(ge=0)
    replacement_class_counts: dict[str, int] = Field(default_factory=dict)
    replacement_hard_negative_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class DatasetEditorRebuildResult(DatasetEditorMutationResult):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["merge", "replace"]
    conflicts: list[DatasetEditorRebuildChange] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


__all__ = [
    "DatasetEditorAddScenesRequest",
    "DatasetEditorDatasetInfo",
    "DatasetEditorDatasetListResponse",
    "DatasetEditorDeleteSceneRequest",
    "DatasetEditorDiscardDraftsResult",
    "DatasetEditorDraftInfo",
    "DatasetEditorDraftSummary",
    "DatasetEditorObjectType",
    "DatasetEditorMutationResult",
    "DatasetEditorPublishRequest",
    "DatasetEditorPublishSceneRequest",
    "DatasetEditorPublicationInfo",
    "DatasetEditorPseudoMarkupInfo",
    "DatasetEditorRasterBrowserResponse",
    "DatasetEditorRasterFolderInfo",
    "DatasetEditorRasterInfo",
    "DatasetEditorRebuildChange",
    "DatasetEditorRebuildPreview",
    "DatasetEditorRebuildPreviewRequest",
    "DatasetEditorRebuildRequest",
    "DatasetEditorRebuildResult",
    "DatasetEditorSaveSceneRequest",
    "DatasetEditorSaveDraftRequest",
    "DatasetEditorSceneDetail",
    "DatasetEditorSceneInfo",
    "DatasetEditorSceneListResponse",
]
