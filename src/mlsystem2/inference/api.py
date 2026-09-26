"""Публичный фасад инференса."""

from __future__ import annotations

from .contracts import (
    InferenceRequest, InferenceResult, ObjectSceneAccumulator, ObjectSceneRequest,
    ObjectSeparationConfig,
)


def run_inference(request: InferenceRequest) -> InferenceResult:
    from ._runner import run_inference as _run_inference

    return _run_inference(request)


def create_object_scene(request: ObjectSceneRequest) -> ObjectSceneAccumulator:
    """Создать дисковый накопитель карт области и границ."""
    from ._objects import SceneAccumulator

    return SceneAccumulator(request)


def separate_objects(foreground, boundary, valid_pixels, config: ObjectSeparationConfig | None = None):
    """Выделить объекты одной связной области или небольшого массива."""
    from ._objects import separate_objects as separate

    return separate(foreground, boundary, valid_pixels, config or ObjectSeparationConfig())


def object_window_origins(length: int, tile_size: int) -> list[int]:
    """Покрыть ось окнами с половинным шагом, включая последний пиксель."""
    if length <= 0 or tile_size < 2:
        raise ValueError("Размер снимка положителен, размер окна не меньше двух")
    last = max(0, length - tile_size)
    return sorted({*range(0, last + 1, tile_size // 2), last})


__all__ = ["run_inference", "create_object_scene", "separate_objects", "object_window_origins"]
