"""Публичный фасад подготовки датасета."""

from __future__ import annotations

from ._prepare import prepare_dataset as _prepare_dataset
from ._manifest import load_dataset_manifest as _load_dataset_manifest
from ._per_image import (
    PerImageAnnotationResolution,
    footprint_name_for_annotation as _footprint_name_for_annotation,
    is_per_image_footprint_name as _is_per_image_footprint_name,
    per_image_annotation_files as _per_image_annotation_files,
    per_image_annotation_name as _per_image_annotation_name,
    per_image_footprint_name as _per_image_footprint_name,
    resolve_per_image_annotations as _resolve_per_image_annotations,
)
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


def per_image_footprint_name(image_path: str) -> str:
    return _per_image_footprint_name(image_path)


def footprint_name_for_annotation(annotation_file: str) -> str:
    return _footprint_name_for_annotation(annotation_file)


def is_per_image_footprint_name(value: str) -> bool:
    return _is_per_image_footprint_name(value)


def per_image_annotation_files(annotations_dir: str) -> list[str]:
    return [path.as_posix() for path in _per_image_annotation_files(annotations_dir)]


def resolve_per_image_annotations(
    images_dir: str,
    annotations_dir: str,
) -> PerImageAnnotationResolution:
    return _resolve_per_image_annotations(images_dir, annotations_dir)


def load_dataset_manifest(annotations_dir_or_file: str) -> DatasetManifest | None:
    return _load_dataset_manifest(annotations_dir_or_file)


__all__ = [
    "prepare_dataset",
    "resolve_scene_images",
    "footprint_name_for_annotation",
    "is_per_image_footprint_name",
    "per_image_annotation_name",
    "per_image_annotation_files",
    "per_image_footprint_name",
    "resolve_per_image_annotations",
    "load_dataset_manifest",
]
