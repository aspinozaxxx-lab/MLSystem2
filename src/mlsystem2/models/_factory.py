"""Фабрика поддерживаемых моделей."""

from __future__ import annotations

from .contracts import ModelHandle, ModelSpec, ModelsError

try:
    import torch
except ImportError:
    torch = None


_SEGFORMER_B0 = "segformer_b0"
_SEGFORMER_B2 = "segformer_b2"
_SMP_SEGFORMER_B0 = "smp_segformer_b0"
_SMP_SEGFORMER_B2 = "smp_segformer_b2"
_SUPPORTED_NAMES = {_SEGFORMER_B0, _SEGFORMER_B2, _SMP_SEGFORMER_B0, _SMP_SEGFORMER_B2}
_SMP_ENCODERS = {
    _SMP_SEGFORMER_B0: "mit_b0",
    _SMP_SEGFORMER_B2: "mit_b2",
}
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
        ModelSpec(
            name=_SMP_SEGFORMER_B0,
            input_channels=4,
            output_channels=1,
            pretrained=False,
            parameters={},
        ),
        ModelSpec(
            name=_SMP_SEGFORMER_B2,
            input_channels=4,
            output_channels=1,
            pretrained=False,
            parameters={},
        ),
    ]


def create_model(spec: ModelSpec) -> ModelHandle:
    if spec.name not in _SUPPORTED_NAMES:
        raise ModelsError(f"Неподдерживаемая архитектура модели: {spec.name}")
    if spec.name in _SMP_ENCODERS:
        return ModelHandle(spec=spec, model=_create_smp_segformer(spec))
    return ModelHandle(spec=spec, model=_create_segformer(spec))


def _create_smp_segformer(spec: ModelSpec):
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ModelsError(
            "Для создания SMP SegFormer требуется optional dependency segmentation_models_pytorch. "
            "Установите пакет через `pip install segmentation-models-pytorch`."
        ) from exc
    if torch is None:
        raise ModelsError(
            "Для создания SMP SegFormer требуется optional dependency torch. "
            "Установите пакет через `pip install -e .[torch]`."
        )
    if spec.pretrained:
        raise ModelsError("SMP SegFormer в MLSystem2 поддерживает только encoder_weights=None.")
    return smp.Segformer(
        encoder_name=_SMP_ENCODERS[spec.name],
        encoder_weights=None,
        in_channels=spec.input_channels,
        classes=spec.output_channels,
        activation=None,
    )


def _create_segformer(spec: ModelSpec):
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as exc:
        raise ModelsError(
            "Для создания SegFormer требуется optional dependency transformers. "
            "Установите пакет через `pip install -e .[torch]`."
        ) from exc
    if torch is None:
        raise ModelsError(
            "Для создания SegFormer требуется optional dependency torch. "
            "Установите пакет через `pip install -e .[torch]`."
        )

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
            model = SegformerForSemanticSegmentation.from_pretrained(
                model_config["pretrained"],
                config=config,
                ignore_mismatched_sizes=True,
            )
            return _SegFormerRawInputWrapper(model)
        except Exception as exc:
            raise ModelsError(f"Не удалось загрузить pretrained {spec.name} из Hugging Face") from exc
    return _SegFormerRawInputWrapper(SegformerForSemanticSegmentation(config))


if torch is not None:

    class _SegFormerRawInputWrapper(torch.nn.Module):
        def __init__(self, model, input_scale: float = 255.0) -> None:
            super().__init__()
            self.model = model
            self.input_scale = float(input_scale)

        def forward(self, x):
            return self.model(x.float() / self.input_scale)

else:

    class _SegFormerRawInputWrapper:
        def __init__(self, model, input_scale: float = 255.0) -> None:
            raise ModelsError(
                "Для создания SegFormer требуется optional dependency torch. "
                "Установите пакет через `pip install -e .[torch]`."
            )
