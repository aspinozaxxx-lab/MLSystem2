"""Заглушка фабрики моделей."""

from __future__ import annotations

from .contracts import ModelHandle, ModelSpec, ModelsError


_SUPPORTED_NAMES = {"unet"}


def list_supported_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="unet",
            input_channels=3,
            output_channels=1,
            pretrained=False,
            parameters={},
        )
    ]


def create_model(spec: ModelSpec) -> ModelHandle:
    if spec.name not in _SUPPORTED_NAMES:
        raise ModelsError(f"Неподдерживаемая архитектура модели: {spec.name}")
    placeholder_model = {
        "name": spec.name,
        "input_channels": spec.input_channels,
        "output_channels": spec.output_channels,
        "pretrained": spec.pretrained,
        "parameters": spec.parameters,
        "implementation": "placeholder",
    }
    return ModelHandle(spec=spec, model=placeholder_model)
