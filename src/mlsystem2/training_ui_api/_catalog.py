"""Каталог моделей, доступных в UI."""

from __future__ import annotations

from mlsystem2.models.api import list_supported_models

from .contracts import ModelInfo


MODEL_DISPLAY_NAMES = {
    "smp_deeplabv3plus_resnet50": "deeplabV3+",
    "smp_segformer_b1": "segformer b1",
    "smp_segformer_b2": "segformer b2",
    "smp_segformer_b3": "segformer b3",
    "smp_unet_resnet34": "unet + resnet34",
    "smp_unet_resnet50": "unet + resnet50",
    "smp_unet_resnet101": "unet + resnet101",
    "smp_unet_resnet152": "unet + resnet152",
}
UI_ARCHITECTURES = tuple(MODEL_DISPLAY_NAMES)


def ui_model_infos() -> list[ModelInfo]:
    supported = {item.name: item for item in list_supported_models()}
    ui_models = []
    for architecture in UI_ARCHITECTURES:
        spec = supported.get(architecture)
        if spec is None:
            continue
        ui_models.append(
            ModelInfo(
                architecture=architecture,
                display_name=MODEL_DISPLAY_NAMES[architecture],
                input_channels=spec.input_channels,
                output_channels=spec.output_channels,
                pretrained=spec.pretrained,
            )
        )
    return ui_models
