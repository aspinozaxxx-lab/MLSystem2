"""Фабрика поддерживаемых моделей."""

from __future__ import annotations

from .contracts import ModelHandle, ModelSpec, ModelsError


_SEGFORMER_B2 = "segformer_b2"
_SUPPORTED_NAMES = {_SEGFORMER_B2}
_PRETRAINED_B2 = "nvidia/segformer-b2-finetuned-ade-512-512"


def list_supported_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name=_SEGFORMER_B2,
            input_channels=4,
            output_channels=1,
            pretrained=False,
            parameters={},
        )
    ]


def create_model(spec: ModelSpec) -> ModelHandle:
    if spec.name not in _SUPPORTED_NAMES:
        raise ModelsError(f"Неподдерживаемая архитектура модели: {spec.name}")
    return ModelHandle(spec=spec, model=_create_segformer_b2(spec))


def _create_segformer_b2(spec: ModelSpec):
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as exc:
        raise ModelsError(
            "Для создания segformer_b2 требуется optional dependency transformers. "
            "Установите пакет через `pip install -e .[torch]`."
        ) from exc

    config = SegformerConfig(
        num_channels=spec.input_channels,
        num_labels=spec.output_channels,
        depths=[3, 4, 6, 3],
        hidden_sizes=[64, 128, 320, 512],
        decoder_hidden_size=768,
    )
    if spec.pretrained:
        try:
            return SegformerForSemanticSegmentation.from_pretrained(
                _PRETRAINED_B2,
                config=config,
                ignore_mismatched_sizes=True,
            )
        except Exception as exc:
            raise ModelsError("Не удалось загрузить pretrained segformer_b2 из Hugging Face") from exc
    return SegformerForSemanticSegmentation(config)
