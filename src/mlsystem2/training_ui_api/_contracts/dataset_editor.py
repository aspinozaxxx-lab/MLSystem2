"""Контракты редактора per-image датасетов."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetEditorDatasetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    name: str
    class_key: str
    class_name: str
    dataset_name: str
    imagery_type: Literal["kanopus", "ortho"]
    scene_count: int = Field(ge=0)


class DatasetEditorDatasetListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets: list[DatasetEditorDatasetInfo]


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


class DatasetEditorSceneListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetEditorDatasetInfo
    scenes: list[DatasetEditorSceneInfo]


class DatasetEditorSceneDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene: DatasetEditorSceneInfo
    geojson: dict[str, Any]


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


__all__ = [
    "DatasetEditorAddScenesRequest",
    "DatasetEditorDatasetInfo",
    "DatasetEditorDatasetListResponse",
    "DatasetEditorDeleteSceneRequest",
    "DatasetEditorMutationResult",
    "DatasetEditorPublishRequest",
    "DatasetEditorPublishSceneRequest",
    "DatasetEditorPublicationInfo",
    "DatasetEditorRasterBrowserResponse",
    "DatasetEditorRasterFolderInfo",
    "DatasetEditorRasterInfo",
    "DatasetEditorSaveSceneRequest",
    "DatasetEditorSceneDetail",
    "DatasetEditorSceneInfo",
    "DatasetEditorSceneListResponse",
]
