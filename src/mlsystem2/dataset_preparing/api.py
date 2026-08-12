"""Публичный фасад подготовки датасета."""

from __future__ import annotations

from ._prepare import prepare_dataset as _prepare_dataset
from ._manifest import load_dataset_manifest as _load_dataset_manifest
from ._per_image import per_image_annotation_name as _per_image_annotation_name
from ._scene_resolution import resolve_scene_images as _resolve_scene_images
from .contracts import (
    DatasetPreparationRequest,
    DatasetPreparationResult,
    DatasetManifest,
    SceneImageResolution,
    SceneImageResolutionRequest,
)


def prepare_dataset(request: DatasetPreparationRequest) -> DatasetPreparationResult:
    return _prepare_dataset(request)


def resolve_scene_images(request: SceneImageResolutionRequest) -> SceneImageResolution:
    return _resolve_scene_images(request)


def per_image_annotation_name(image_path: str) -> str:
    return _per_image_annotation_name(image_path)


def load_dataset_manifest(annotations_dir_or_file: str) -> DatasetManifest | None:
    return _load_dataset_manifest(annotations_dir_or_file)


__all__ = [
    "prepare_dataset",
    "resolve_scene_images",
    "per_image_annotation_name",
    "load_dataset_manifest",
]
