"""Совместимые профили и полный перенос весов без преобразования головы или декодера."""

from __future__ import annotations

from copy import deepcopy

from .contracts import ModelHandle, ModelSpec, ModelsError


_SEMANTIC = (
    "next_gen2", "next_gen3", "next_gen4", "next_gen4_global_norm",
    "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1",
)
_CHANNEL_NORM = tuple(f"next_gen4_consistency_v{index}" for index in range(2, 8))
_SMP = {f"smp_segformer_b{index}" for index in range(4)}


def compatible_training_pipelines(spec: ModelSpec) -> list[str]:
    variant = str(spec.parameters.get("pipeline_variant") or "legacy")
    if spec.name not in _SMP | {"segformer_b0"} or spec.input_channels not in (3, 4):
        return [variant]
    if variant in {"object_f1", "object_f1_v2"}:
        # Одинаковое число каналов ещё не означает одинаковый смысл выходов.
        return [variant]
    if variant in _CHANNEL_NORM:
        return list(_CHANNEL_NORM) if spec.name in _SMP else [variant]
    if variant in _SEMANTIC:
        return (["legacy"] if spec.output_channels > 1 else []) + list(_SEMANTIC)
    if variant == "legacy" and spec.output_channels > 1:
        return ["legacy", *_SEMANTIC]
    if variant == "next_gen":
        return ["legacy", "next_gen"]
    return [variant]


def initialize_model(source: ModelHandle, target_spec: ModelSpec, *, preserve_global_preprocessing: bool = True) -> ModelHandle:
    from ._factory import (
        _InputPreprocessingWrapper, _SegFormerRawInputWrapper, create_model_for_checkpoint,
    )

    target_variant = str(target_spec.parameters.get("pipeline_variant") or "legacy")
    if ((source.spec.name, source.spec.input_channels, source.spec.output_channels)
            != (target_spec.name, target_spec.input_channels, target_spec.output_channels)
            or target_variant not in compatible_training_pipelines(source.spec)):
        raise ModelsError("Конвейер несовместим с архитектурой, каналами, выходной головой или нормализацией декодера исходной сети.")
    parameters = deepcopy(target_spec.parameters)
    previous = source.spec.parameters.get("preprocessing") or {}
    requested = parameters.get("preprocessing") or {}
    reuse_global = preserve_global_preprocessing and previous.get("mode") == requested.get("mode") == "global_minmax"
    if reuse_global:
        parameters["preprocessing"] = deepcopy(previous)
    if source.spec.parameters.get("hf_config") and target_spec.name.startswith("segformer_"):
        parameters["hf_config"] = source.spec.parameters["hf_config"]
    if target_variant != "legacy":
        parameters["weight_initialization"] = {
            "source_pipeline_variant": source.spec.parameters.get("pipeline_variant", "legacy"),
            "target_pipeline_variant": target_variant, "policy": "strict_core_state_dict",
            "preprocessing_policy": "preserve_source_global" if reuse_global else "target_profile",
        }
    target = create_model_for_checkpoint(target_spec.model_copy(update={"parameters": parameters}))

    def core(handle):
        return handle.model.model if isinstance(handle.model, (_InputPreprocessingWrapper, _SegFormerRawInputWrapper)) else handle.model

    source_state, target_state = core(source).state_dict(), core(target).state_dict()
    if source_state.keys() != target_state.keys() or any(
        source_state[key].shape != target_state[key].shape for key in source_state
    ):
        raise ModelsError("Структура весов исходной сети не совпадает с выбранным конвейером; частичный перенос запрещён.")
    core(target).load_state_dict(source_state, strict=True)
    # Буферы внешнего preprocessing намеренно принадлежат целевому профилю,
    # включая границы, посчитанные только по train нового этапа.
    return target
