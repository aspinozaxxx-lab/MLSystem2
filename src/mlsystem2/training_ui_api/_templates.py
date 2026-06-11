"""Исходные шаблоны обучения и инференса для миграций и reset."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CONFIG_SCHEMA: dict[str, Any] = {
    "fields": [
        {
            "key": "dataset.val_fraction",
            "label": "Доля валидации",
            "value_type": "number",
            "tooltip": "Какая часть тайлов уходит в validation split.",
            "min_value": 0.01,
            "max_value": 0.9,
        },
        {
            "key": "tile_preparation.tile_size",
            "label": "Размер тайла",
            "value_type": "integer",
            "tooltip": "Размер квадратного окна, которое модель получает на вход.",
            "min_value": 64,
        },
        {
            "key": "tile_preparation.stride",
            "label": "Шаг тайлинга",
            "value_type": "integer",
            "tooltip": "Расстояние между соседними тайлами; не должно превышать размер тайла.",
            "min_value": 1,
        },
        {
            "key": "tile_preparation.augmentation_level",
            "label": "Уровень аугментаций",
            "value_type": "integer",
            "tooltip": "0 выключает аугментации; 1 включает геометрию; 2 и 3 добавляют фотометрию.",
            "min_value": 0,
            "max_value": 3,
        },
        {
            "key": "tile_preparation.positive_factor",
            "label": "Доля positive тайлов",
            "value_type": "number",
            "tooltip": "Целевая доля foreground/positive samples в train epoch.",
            "min_value": 0.01,
            "max_value": 0.99,
        },
        {
            "key": "train.epochs",
            "label": "Эпохи",
            "value_type": "integer",
            "tooltip": "Максимальное число эпох обучения.",
            "min_value": 1,
        },
        {
            "key": "train.batch_size",
            "label": "Batch size",
            "value_type": "integer",
            "tooltip": "Количество тайлов в одном optimizer step.",
            "min_value": 1,
        },
        {
            "key": "train.learning_rate",
            "label": "Learning rate",
            "value_type": "number",
            "tooltip": "Скорость обучения AdamW.",
            "min_value": 0,
        },
        {
            "key": "train.weight_decay",
            "label": "Weight decay",
            "value_type": "number",
            "tooltip": "L2-регуляризация AdamW.",
            "min_value": 0,
        },
        {
            "key": "train.loss",
            "label": "Loss",
            "value_type": "select",
            "tooltip": "Функция потерь train loop.",
            "options": [
                "bce_dice",
                "focal_dice",
                "focal_tversky",
            ],
        },
        {
            "key": "train.focal_alpha",
            "label": "Focal alpha",
            "value_type": "number",
            "tooltip": "Баланс focal-компоненты для focal loss.",
            "min_value": 0,
            "max_value": 1,
        },
        {
            "key": "train.pos_weight",
            "label": "Positive weight",
            "value_type": "number",
            "tooltip": "Вес positive-класса для binary loss.",
            "min_value": 0,
        },
        {
            "key": "train.tversky_alpha",
            "label": "Tversky alpha",
            "value_type": "number",
            "tooltip": "Штраф false positive в Tversky-компоненте.",
            "min_value": 0,
        },
        {
            "key": "train.tversky_beta",
            "label": "Tversky beta",
            "value_type": "number",
            "tooltip": "Штраф false negative в Tversky-компоненте.",
            "min_value": 0,
        },
        {
            "key": "train.threshold",
            "label": "Порог",
            "value_type": "number",
            "tooltip": "Порог вероятности для binary validation и инференса.",
            "min_value": 0,
            "max_value": 1,
        },
        {
            "key": "train.early_stopping_patience",
            "label": "Early stopping patience",
            "value_type": "integer",
            "tooltip": "Сколько эпох без улучшения ждать до остановки.",
            "min_value": 1,
        },
        {
            "key": "train.max_train_batches_per_epoch",
            "label": "Train batch-ей на эпоху",
            "value_type": "integer-null",
            "tooltip": "Ограничивает размер train epoch. Пустое значение означает весь train split.",
            "min_value": 1,
        },
        {
            "key": "train.max_val_batches_per_epoch",
            "label": "Validation batch-ей на эпоху",
            "value_type": "integer-null",
            "tooltip": "Ограничивает размер validation epoch. Пустое значение означает весь validation split.",
            "min_value": 1,
        },
        {
            "key": "train.max_training_time_sec",
            "label": "Максимальное время обучения, сек",
            "value_type": "integer-null",
            "tooltip": "Wall-clock лимит train loop. Пустое значение означает обучение без лимита; проверяется после завершения эпохи.",
            "min_value": 1,
        },
    ]
}

BASE_DEFAULT_CONFIG: dict[str, Any] = {
    "dataset.val_fraction": 0.2,
    "tile_preparation.tile_size": 512,
    "tile_preparation.stride": 256,
    "tile_preparation.augmentation_level": 3,
    "tile_preparation.positive_factor": 0.8,
    "train.epochs": 80,
    "train.batch_size": 4,
    "train.learning_rate": 0.00001,
    "train.weight_decay": 0.0001,
    "train.loss": "focal_tversky",
    "train.focal_alpha": 0.6,
    "train.pos_weight": 1.0,
    "train.tversky_alpha": 0.4,
    "train.tversky_beta": 0.6,
    "train.threshold": 0.7,
    "train.early_stopping_patience": 12,
    "train.max_train_batches_per_epoch": 72,
    "train.max_val_batches_per_epoch": 1000,
    "train.max_training_time_sec": None,
}


CONFIG_KEYS = {str(field["key"]) for field in CONFIG_SCHEMA["fields"]}
CONFIG_FIELDS = {str(field["key"]): field for field in CONFIG_SCHEMA["fields"]}


INFERENCE_CONFIG_SCHEMA: dict[str, Any] = {
    "fields": [
        {
            "key": "postprocess.mask_min_object_pixels",
            "label": "Удалять объекты маски меньше, px",
            "value_type": "integer-null",
            "tooltip": "Аналог MaskMorphology remove_small_objects. Пусто сохраняет автоматический профиль.",
            "min_value": 1,
        },
        {
            "key": "postprocess.mask_min_hole_pixels",
            "label": "Заливать дырки маски меньше, px",
            "value_type": "integer-null",
            "tooltip": "Аналог MaskMorphology remove_small_holes. Пусто сохраняет автоматический профиль.",
            "min_value": 1,
        },
        {
            "key": "postprocess.binary_closing_radius",
            "label": "Радиус binary closing, px",
            "value_type": "integer-null",
            "tooltip": "Аналог MaskMorphology binary_closing disk. Пусто не меняет автоматический профиль.",
            "min_value": 1,
        },
        {
            "key": "postprocess.min_area_m2",
            "label": "Мин. площадь полигона, м²",
            "value_type": "number-null",
            "tooltip": "Аналог FilterSmallObjects min_area. Пусто сохраняет автоматический профиль.",
            "min_value": 0,
        },
        {
            "key": "postprocess.min_hole_area_m2",
            "label": "Мин. площадь дырки, м²",
            "value_type": "number-null",
            "tooltip": "Аналог RemoveSmallHoles min_hole_area. Пусто сохраняет автоматический профиль.",
            "min_value": 0,
        },
        {
            "key": "postprocess.simplify_m",
            "label": "Упрощение контура, м",
            "value_type": "number-null",
            "tooltip": "Аналог Simplify rate. Пусто сохраняет автоматический профиль.",
            "min_value": 0,
        },
        {
            "key": "postprocess.filter_compact_objects.enabled",
            "label": "Удалять компактные объекты",
            "value_type": "boolean",
            "tooltip": "Включает FilterCompactObjects для удаления объектов, похожих на озера/пруды.",
        },
        {
            "key": "postprocess.filter_compact_objects.min_isoperimetric_quotient",
            "label": "Порог компактности ISO",
            "value_type": "number",
            "tooltip": "Объект считается компактным при isoperimetric quotient не ниже этого значения.",
            "min_value": 0,
            "max_value": 1,
        },
        {
            "key": "postprocess.filter_compact_objects.max_bbox_ratio",
            "label": "Порог вытянутости bbox",
            "value_type": "number",
            "tooltip": "Компактный объект удаляется, если отношение сторон minimum rotated rectangle ниже этого значения.",
            "min_value": 1,
        },
    ]
}

INFERENCE_BASE_DEFAULT_CONFIG: dict[str, Any] = {
    "postprocess.mask_min_object_pixels": None,
    "postprocess.mask_min_hole_pixels": None,
    "postprocess.binary_closing_radius": None,
    "postprocess.min_area_m2": None,
    "postprocess.min_hole_area_m2": None,
    "postprocess.simplify_m": None,
    "postprocess.filter_compact_objects.enabled": False,
    "postprocess.filter_compact_objects.min_isoperimetric_quotient": 0.25,
    "postprocess.filter_compact_objects.max_bbox_ratio": 3.5,
}

RIVERS_INFERENCE_CONFIG: dict[str, Any] = {
    "postprocess.min_area_m2": 10000.0,
    "postprocess.min_hole_area_m2": 5000.0,
    "postprocess.simplify_m": 15.0,
    "postprocess.filter_compact_objects.enabled": True,
    "postprocess.filter_compact_objects.min_isoperimetric_quotient": 0.25,
    "postprocess.filter_compact_objects.max_bbox_ratio": 3.5,
}

INFERENCE_CONFIG_KEYS = {str(field["key"]) for field in INFERENCE_CONFIG_SCHEMA["fields"]}
INFERENCE_CONFIG_FIELDS = {
    str(field["key"]): field for field in INFERENCE_CONFIG_SCHEMA["fields"]
}


def sanitize_template_config(
    config: dict[str, Any] | None,
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        key: value
        for key, value in (fallback or BASE_DEFAULT_CONFIG).items()
        if key in CONFIG_KEYS
    }
    for key, value in (config or {}).items():
        if key in CONFIG_KEYS:
            options = CONFIG_FIELDS[key].get("options")
            if options is not None and value not in options:
                continue
            result[key] = value
    return result


def sanitize_inference_template_config(
    config: dict[str, Any] | None,
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        key: value
        for key, value in (fallback or INFERENCE_BASE_DEFAULT_CONFIG).items()
        if key in INFERENCE_CONFIG_KEYS
    }
    for key, value in (config or {}).items():
        if key not in INFERENCE_CONFIG_KEYS:
            continue
        field = INFERENCE_CONFIG_FIELDS[key]
        options = field.get("options")
        if options is not None and value not in options:
            continue
        result[key] = value
    return result


def initial_templates() -> list[dict[str, Any]]:
    rows = [
        _template(
            "smp_deeplabv3plus_resnet50",
            "deeplabV3+",
            source="analogy",
            overrides={"train.batch_size": 4, "train.threshold": 0.65},
        ),
        _template(
            "smp_segformer_b2",
            "segformer b2",
            source="hpo_best",
            source_mlflow_run_id="59b45400260c4e4da5d6f753244339b1",
            overrides={
                "train.epochs": 30,
                "train.max_train_batches_per_epoch": 72,
                "train.max_val_batches_per_epoch": 1000,
                "train.max_training_time_sec": 1800,
            },
        ),
        _template(
            "smp_segformer_b3",
            "segformer b3",
            source="analogy",
            overrides={"train.batch_size": 2, "train.epochs": 80},
        ),
        _template(
            "smp_unet_resnet34",
            "unet + resnet34",
            source="analogy",
            overrides={"train.batch_size": 8, "train.epochs": 60, "train.threshold": 0.65},
        ),
        _template(
            "smp_unet_resnet50",
            "unet + resnet50",
            source="analogy",
            overrides={"train.batch_size": 6, "train.epochs": 70, "train.threshold": 0.65},
        ),
        _template(
            "smp_unet_resnet101",
            "unet + resnet101",
            source="analogy",
            overrides={"train.batch_size": 4, "train.epochs": 70, "train.threshold": 0.65},
        ),
        _template(
            "smp_unet_resnet152",
            "unet + resnet152",
            source="analogy",
            overrides={"train.batch_size": 2, "train.epochs": 70, "train.threshold": 0.65},
        ),
    ]
    return rows


def initial_inference_templates() -> list[dict[str, Any]]:
    rows = [
        _inference_template(
            "smp_deeplabv3plus_resnet50",
            "deeplabV3+",
            source="analogy",
        ),
        _inference_template(
            "smp_segformer_b2",
            "segformer b2",
            source="analogy",
        ),
        _inference_template(
            "smp_segformer_b3",
            "segformer b3",
            source="analogy",
        ),
        _inference_template(
            "smp_unet_resnet34",
            "unet + resnet34",
            source="analogy",
        ),
        _inference_template(
            "smp_unet_resnet50",
            "unet + resnet50",
            source="analogy",
        ),
        _inference_template(
            "smp_unet_resnet101",
            "unet + resnet101",
            source="analogy",
        ),
        _inference_template(
            "smp_unet_resnet152",
            "unet + resnet152",
            source="analogy",
        ),
        _inference_template(
            "smp_segformer_b2",
            "segformer b2 / Реки\\main",
            source="analogy",
            dataset_key="Реки\\main",
            dataset_name="Реки\\main",
            overrides=RIVERS_INFERENCE_CONFIG,
        ),
    ]
    return rows


def _template(
    architecture: str,
    display_name: str,
    *,
    source: str,
    source_mlflow_run_id: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    default_config = deepcopy(BASE_DEFAULT_CONFIG)
    if overrides:
        default_config.update(overrides)
    return {
        "architecture": architecture,
        "display_name": display_name,
        "config_schema": deepcopy(CONFIG_SCHEMA),
        "default_config": default_config,
        "baseline_default_config": deepcopy(default_config),
        "source": source,
        "baseline_source": source,
        "source_mlflow_run_id": source_mlflow_run_id,
        "baseline_source_mlflow_run_id": source_mlflow_run_id,
        "is_active": True,
        "version": 1,
    }


def _inference_template(
    architecture: str,
    display_name: str,
    *,
    source: str,
    dataset_key: str | None = None,
    dataset_name: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    default_config = deepcopy(INFERENCE_BASE_DEFAULT_CONFIG)
    if overrides:
        default_config.update(overrides)
    return {
        "architecture": architecture,
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "parent_template_id": None,
        "display_name": display_name,
        "config_schema": deepcopy(INFERENCE_CONFIG_SCHEMA),
        "default_config": default_config,
        "baseline_default_config": deepcopy(default_config),
        "source": source,
        "baseline_source": source,
        "source_mlflow_run_id": None,
        "baseline_source_mlflow_run_id": None,
        "is_active": True,
        "version": 1,
    }
