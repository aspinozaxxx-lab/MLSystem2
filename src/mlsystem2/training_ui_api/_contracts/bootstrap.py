"""Bootstrap contract used by the React frontend."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .catalog import AppLink, ClassInfo, DatasetInfo, ImageFolderInfo, ModelInfo
from .templates import InferenceTemplate, TrainingTemplate


class BootstrapInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    links: list[AppLink]
    datasets: list[DatasetInfo]
    image_folders: list[ImageFolderInfo]
    classes: list[ClassInfo]
    models: list[ModelInfo]
    training_templates: list[TrainingTemplate]
    inference_templates: list[InferenceTemplate]


__all__ = ["BootstrapInfo"]
