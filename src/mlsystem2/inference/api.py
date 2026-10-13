"""Публичный фасад инференса."""

from __future__ import annotations

from .contracts import (
    InferenceError, InferenceRequest, InferenceResult, ObjectSceneAccumulator, ObjectSceneRequest,
    ObjectSeparationConfig,
)
from mlsystem2.models.contracts import ModelSpec


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


def object_separation_config(model_spec: ModelSpec) -> ObjectSeparationConfig:
    """Прочитать зафиксированный в checkpoint профиль разделения экземпляров."""
    parameters = model_spec.parameters
    variant = parameters.get("pipeline_variant", "legacy")
    if variant not in {"object_f1", "object_f1_v2"}:
        raise InferenceError("Модель не относится к объектному конвейеру")
    raw = parameters.get("object_separation")
    if variant == "object_f1_v2" and not isinstance(raw, dict):
        raise InferenceError("Checkpoint object f1 v2 не содержит параметров разделения")
    try:
        result = ObjectSeparationConfig.model_validate(raw or {})
    except ValueError as exc:
        raise InferenceError("Checkpoint содержит некорректные параметры разделения") from exc
    if result.profile != variant:
        raise InferenceError("Профиль разделения не соответствует конвейеру checkpoint")
    return result


def object_window_origins(length: int, tile_size: int) -> list[int]:
    """Покрыть ось окнами с половинным шагом, включая последний пиксель."""
    if length <= 0 or tile_size < 2:
        raise ValueError("Размер снимка положителен, размер окна не меньше двух")
    last = max(0, length - tile_size)
    return sorted({*range(0, last + 1, tile_size // 2), last})


__all__ = ["run_inference", "create_object_scene", "separate_objects", "object_window_origins", "object_separation_config"]
