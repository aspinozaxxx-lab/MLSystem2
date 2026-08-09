"""Публичный фасад подготовки датасета."""

from __future__ import annotations

from ._prepare import prepare_dataset as _prepare_dataset
from ._per_image import per_image_annotation_name as _per_image_annotation_name
from ._scene_resolution import resolve_scene_images as _resolve_scene_images
from .contracts import (
    DatasetPreparationRequest,
    DatasetPreparationResult,
    SceneImageResolution,
    SceneImageResolutionRequest,
)


def prepare_dataset(request: DatasetPreparationRequest) -> DatasetPreparationResult:
    return _prepare_dataset(request)


def resolve_scene_images(request: SceneImageResolutionRequest) -> SceneImageResolution:
    return _resolve_scene_images(request)


def per_image_annotation_name(image_path: str) -> str:
    return _per_image_annotation_name(image_path)


__all__ = ["prepare_dataset", "resolve_scene_images", "per_image_annotation_name"]
