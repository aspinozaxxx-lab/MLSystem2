"""Фабрика поддерживаемых моделей."""

from __future__ import annotations

from .contracts import ModelHandle, ModelSpec, ModelsError


_SEGFORMER_B0 = "segformer_b0"
_SEGFORMER_B2 = "segformer_b2"
_SUPPORTED_NAMES = {_SEGFORMER_B0, _SEGFORMER_B2}
_PRETRAINED_B0 = "nvidia/segformer-b0-finetuned-ade-512-512"
_PRETRAINED_B2 = "nvidia/segformer-b2-finetuned-ade-512-512"
_SEGFORMER_CONFIGS = {
    _SEGFORMER_B0: {
        "depths": [2, 2, 2, 2],
        "hidden_sizes": [32, 64, 160, 256],
        "decoder_hidden_size": 256,
        "pretrained": _PRETRAINED_B0,
    },
    _SEGFORMER_B2: {
        "depths": [3, 4, 6, 3],
        "hidden_sizes": [64, 128, 320, 512],
        "decoder_hidden_size": 768,
        "pretrained": _PRETRAINED_B2,
    },
}


def list_supported_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name=_SEGFORMER_B0,
            input_channels=4,
            output_channels=1,
            pretrained=False,
            parameters={},
        ),
        ModelSpec(
            name=_SEGFORMER_B2,
            input_channels=4,
            output_channels=1,
            pretrained=False,
            parameters={},
        ),
    ]


def create_model(spec: ModelSpec) -> ModelHandle:
    if spec.name not in _SUPPORTED_NAMES:
        raise ModelsError(f"Неподдерживаемая архитектура модели: {spec.name}")
    return ModelHandle(spec=spec, model=_create_segformer(spec))


def _create_segformer(spec: ModelSpec):
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as exc:
        raise ModelsError(
            "Для создания SegFormer требуется optional dependency transformers. "
            "Установите пакет через `pip install -e .[torch]`."
        ) from exc

    model_config = _SEGFORMER_CONFIGS[spec.name]
    config = SegformerConfig(
        num_channels=spec.input_channels,
        num_labels=spec.output_channels,
        depths=model_config["depths"],
        hidden_sizes=model_config["hidden_sizes"],
        decoder_hidden_size=model_config["decoder_hidden_size"],
    )
    if spec.pretrained:
        try:
            return SegformerForSemanticSegmentation.from_pretrained(
                model_config["pretrained"],
                config=config,
                ignore_mismatched_sizes=True,
            )
        except Exception as exc:
            raise ModelsError(f"Не удалось загрузить pretrained {spec.name} из Hugging Face") from exc
    return SegformerForSemanticSegmentation(config)
