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
_SMP_SEGFORMER_B1 = "smp_segformer_b1"
_SMP_SEGFORMER_B2 = "smp_segformer_b2"
_SMP_SEGFORMER_B3 = "smp_segformer_b3"
_SMP_DEEPLABV3PLUS_RESNET50 = "smp_deeplabv3plus_resnet50"
_SMP_UNET_RESNET34 = "smp_unet_resnet34"
_SMP_UNET_RESNET50 = "smp_unet_resnet50"
_SMP_UNET_RESNET101 = "smp_unet_resnet101"
_SMP_UNET_RESNET152 = "smp_unet_resnet152"
_SUPPORTED_NAMES = {
    _SEGFORMER_B0,
    _SEGFORMER_B2,
    _SMP_SEGFORMER_B0,
    _SMP_SEGFORMER_B1,
    _SMP_SEGFORMER_B2,
    _SMP_SEGFORMER_B3,
    _SMP_DEEPLABV3PLUS_RESNET50,
    _SMP_UNET_RESNET34,
    _SMP_UNET_RESNET50,
    _SMP_UNET_RESNET101,
    _SMP_UNET_RESNET152,
}
_SMP_ENCODERS = {
    _SMP_SEGFORMER_B0: "mit_b0",
    _SMP_SEGFORMER_B1: "mit_b1",
    _SMP_SEGFORMER_B2: "mit_b2",
    _SMP_SEGFORMER_B3: "mit_b3",
}
_SMP_UNET_ENCODERS = {
    _SMP_UNET_RESNET34: "resnet34",
    _SMP_UNET_RESNET50: "resnet50",
    _SMP_UNET_RESNET101: "resnet101",
    _SMP_UNET_RESNET152: "resnet152",
}
_PRETRAINED_B0 = "nvidia/segformer-b0-finetuned-ade-512-512"
_PRETRAINED_B0_REVISION = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"
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
            name=_SMP_SEGFORMER_B1,
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
        ModelSpec(
            name=_SMP_SEGFORMER_B3,
            input_channels=4,
            output_channels=1,
            pretrained=False,
            parameters={},
        ),
        ModelSpec(
            name=_SMP_DEEPLABV3PLUS_RESNET50,
            input_channels=4,
            output_channels=1,
            pretrained=False,
            parameters={},
        ),
        ModelSpec(
            name=_SMP_UNET_RESNET34,
            input_channels=4,
            output_channels=1,
            pretrained=False,
            parameters={},
        ),
        ModelSpec(
            name=_SMP_UNET_RESNET50,
            input_channels=4,
            output_channels=1,
            pretrained=False,
            parameters={},
        ),
        ModelSpec(
            name=_SMP_UNET_RESNET101,
            input_channels=4,
            output_channels=1,
            pretrained=False,
            parameters={},
        ),
        ModelSpec(
            name=_SMP_UNET_RESNET152,
            input_channels=4,
            output_channels=1,
            pretrained=False,
            parameters={},
        ),
    ]


def create_model(spec: ModelSpec) -> ModelHandle:
    return _create_model(spec, initialize_pretrained=True)


def create_model_for_checkpoint(spec: ModelSpec) -> ModelHandle:
    """Создать архитектуру checkpoint без сетевой загрузки исходных весов."""

    return _create_model(spec, initialize_pretrained=False)


def _create_model(spec: ModelSpec, *, initialize_pretrained: bool) -> ModelHandle:
    if spec.name not in _SUPPORTED_NAMES:
        raise ModelsError(f"Неподдерживаемая архитектура модели: {spec.name}")
    if spec.name in _SMP_ENCODERS:
        model = _create_smp_segformer(spec)
        return _wrap_next_gen(spec, model)
    if spec.name == _SMP_DEEPLABV3PLUS_RESNET50:
        model = _create_smp_deeplabv3plus(spec)
        return _wrap_next_gen(spec, model)
    if spec.name in _SMP_UNET_ENCODERS:
        model = _create_smp_unet(spec)
        return _wrap_next_gen(spec, model)
    return _create_segformer(spec, initialize_pretrained=initialize_pretrained)


def _import_smp():
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ModelsError(
            "Для создания SMP-моделей требуется optional dependency segmentation_models_pytorch. "
            "Установите пакет через `pip install segmentation-models-pytorch`."
        ) from exc
    return smp


def _ensure_torch_for_smp() -> None:
    if torch is None:
        raise ModelsError(
            "Для создания SMP-моделей требуется optional dependency torch. "
            "Установите пакет через `pip install -e .[torch]`."
        )


def _create_smp_segformer(spec: ModelSpec):
    smp = _import_smp()
    _ensure_torch_for_smp()
    if spec.pretrained:
        raise ModelsError("SMP SegFormer в MLSystem2 поддерживает только encoder_weights=None.")
    return smp.Segformer(
        encoder_name=_SMP_ENCODERS[spec.name],
        encoder_weights=None,
        in_channels=spec.input_channels,
        classes=spec.output_channels,
        activation=None,
    )


def _create_smp_deeplabv3plus(spec: ModelSpec):
    smp = _import_smp()
    _ensure_torch_for_smp()
    if spec.pretrained:
        raise ModelsError("SMP DeepLabV3Plus в MLSystem2 поддерживает только encoder_weights=None.")
    return smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=spec.input_channels,
        classes=spec.output_channels,
        activation=None,
    )


def _create_smp_unet(spec: ModelSpec):
    smp = _import_smp()
    _ensure_torch_for_smp()
    if spec.pretrained:
        raise ModelsError("SMP UNet в MLSystem2 поддерживает только encoder_weights=None.")
    return smp.Unet(
        encoder_name=_SMP_UNET_ENCODERS[spec.name],
        encoder_weights=None,
        in_channels=spec.input_channels,
        classes=spec.output_channels,
        activation=None,
    )


def _create_segformer(spec: ModelSpec, *, initialize_pretrained: bool = True) -> ModelHandle:
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
    is_next_gen = spec.parameters.get("pipeline_variant") == "next_gen"
    if is_next_gen and spec.name != _SEGFORMER_B0:
        raise ModelsError("next_gen pretrained-путь реализован только для segformer_b0")
    if is_next_gen and spec.pretrained and initialize_pretrained:
        try:
            model = SegformerForSemanticSegmentation.from_pretrained(
                _PRETRAINED_B0,
                revision=_PRETRAINED_B0_REVISION,
            )
            _adapt_pretrained_segformer(model, spec.input_channels, spec.output_channels)
            parameters = dict(spec.parameters)
            parameters["pretrained_source"] = _PRETRAINED_B0
            parameters["pretrained_revision"] = _PRETRAINED_B0_REVISION
            parameters["input_adapter"] = "rgb_copy_plus_red_to_nir"
            parameters["hf_config"] = model.config.to_dict()
            resolved_spec = spec.model_copy(update={"parameters": parameters})
            return _wrap_next_gen(resolved_spec, model)
        except Exception as exc:
            raise ModelsError(
                "Не удалось загрузить pinned pretrained segformer_b0 из Hugging Face"
            ) from exc
    if is_next_gen and not initialize_pretrained:
        raw_config = spec.parameters.get("hf_config")
        if not isinstance(raw_config, dict):
            raise ModelsError("next_gen checkpoint не содержит сохранённую HF-конфигурацию")
        config = SegformerConfig(**raw_config)
        return _wrap_next_gen(spec, SegformerForSemanticSegmentation(config))
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
            return ModelHandle(spec=spec, model=_SegFormerRawInputWrapper(model))
        except Exception as exc:
            raise ModelsError(
                f"Не удалось загрузить pretrained {spec.name} из Hugging Face"
            ) from exc
    model = SegformerForSemanticSegmentation(config)
    if is_next_gen:
        parameters = dict(spec.parameters)
        parameters["hf_config"] = model.config.to_dict()
        resolved_spec = spec.model_copy(update={"parameters": parameters})
        return _wrap_next_gen(resolved_spec, model)
    return ModelHandle(spec=spec, model=_SegFormerRawInputWrapper(model))


def _adapt_pretrained_segformer(model, input_channels: int, output_channels: int) -> None:
    if input_channels != 4 or output_channels != 1:
        raise ModelsError("Pinned pretrained SegFormer next_gen требует вход 4 и выход 1")
    projection = _first_patch_projection(model)
    replacement = torch.nn.Conv2d(
        input_channels,
        projection.out_channels,
        kernel_size=projection.kernel_size,
        stride=projection.stride,
        padding=projection.padding,
        dilation=projection.dilation,
        groups=projection.groups,
        bias=projection.bias is not None,
        padding_mode=projection.padding_mode,
    )
    with torch.no_grad():
        replacement.weight[:, :3].copy_(projection.weight[:, :3])
        replacement.weight[:, 3].copy_(projection.weight[:, 0])
        if projection.bias is not None:
            replacement.bias.copy_(projection.bias)
    _set_first_patch_projection(model, replacement)
    classifier = model.decode_head.classifier
    model.decode_head.classifier = torch.nn.Conv2d(
        classifier.in_channels,
        output_channels,
        kernel_size=classifier.kernel_size,
        stride=classifier.stride,
        padding=classifier.padding,
        bias=classifier.bias is not None,
    )
    model.config.num_channels = input_channels
    model.config.num_labels = output_channels
    model.config.id2label = {0: "foreground"}
    model.config.label2id = {"foreground": 0}


def _first_patch_projection(model):
    stages = getattr(model.segformer, "stages", None)
    if stages is not None:
        return stages[0].patch_embeddings.proj
    return model.segformer.encoder.patch_embeddings[0].proj


def _set_first_patch_projection(model, projection) -> None:
    stages = getattr(model.segformer, "stages", None)
    if stages is not None:
        stages[0].patch_embeddings.proj = projection
    else:
        model.segformer.encoder.patch_embeddings[0].proj = projection


def _wrap_next_gen(spec: ModelSpec, model) -> ModelHandle:
    if spec.parameters.get("pipeline_variant") != "next_gen":
        return ModelHandle(spec=spec, model=model)
    preprocessing = spec.parameters.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ModelsError("next_gen ModelSpec не содержит preprocessing")
    return ModelHandle(
        spec=spec,
        model=_InputPreprocessingWrapper(model, preprocessing),
    )


if torch is not None:

    class _InputPreprocessingWrapper(torch.nn.Module):
        def __init__(self, model, preprocessing: dict[str, object]) -> None:
            super().__init__()
            self.model = model
            self.mode = str(preprocessing.get("mode") or "scale_255")
            self.nodata = float(preprocessing.get("nodata", 0.0))
            mean = preprocessing.get("mean") or [0.0, 0.0, 0.0, 0.0]
            std = preprocessing.get("std") or [1.0, 1.0, 1.0, 1.0]
            low = preprocessing.get("low") or [0.0, 0.0, 0.0, 0.0]
            high = preprocessing.get("high") or [255.0, 255.0, 255.0, 255.0]
            self.register_buffer("preprocess_mean", torch.tensor(mean).view(1, -1, 1, 1))
            self.register_buffer("preprocess_std", torch.tensor(std).view(1, -1, 1, 1))
            self.register_buffer("preprocess_low", torch.tensor(low).view(1, -1, 1, 1))
            self.register_buffer("preprocess_high", torch.tensor(high).view(1, -1, 1, 1))

        def forward(self, x):
            raw = x.float()
            valid = torch.any(raw != self.nodata, dim=1, keepdim=True)
            if self.mode == "robust_percentile":
                denominator = (self.preprocess_high - self.preprocess_low).clamp_min(1.0)
                normalized = ((raw - self.preprocess_low) / denominator).clamp(0.0, 1.0)
            else:
                normalized = raw / 255.0
                if self.mode == "imagenet_rgb_red_nir":
                    normalized = (normalized - self.preprocess_mean) / self.preprocess_std
            normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
            output = self.model(normalized)
            logits = output.logits if hasattr(output, "logits") else output
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.shape[-2:] != raw.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=raw.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            return torch.where(valid, logits, torch.full_like(logits, -1000.0))

    class _SegFormerRawInputWrapper(torch.nn.Module):
        def __init__(self, model, input_scale: float = 255.0) -> None:
            super().__init__()
            self.model = model
            self.input_scale = float(input_scale)

        def forward(self, x):
            return self.model(x.float() / self.input_scale)

else:

    class _InputPreprocessingWrapper:
        def __init__(self, model, preprocessing: dict[str, object]) -> None:
            raise ModelsError(
                "Для создания next_gen модели требуется optional dependency torch."
            )

    class _SegFormerRawInputWrapper:
        def __init__(self, model, input_scale: float = 255.0) -> None:
            raise ModelsError(
                "Для создания SegFormer требуется optional dependency torch. "
                "Установите пакет через `pip install -e .[torch]`."
            )
