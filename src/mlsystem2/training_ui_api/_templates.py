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
            "key": "tile_preparation.context",
            "label": "Контекст тайла",
            "value_type": "integer",
            "tooltip": "Рамка входного тайла, исключённая из loss и итогового предсказания.",
            "min_value": 0,
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
            "tooltip": "Доля positive samples внутри train epoch; hard negative дефицит добавляется сюда, сумма трех долей должна быть равна 1.",
            "min_value": 0,
            "max_value": 1,
        },
        {
            "key": "tile_preparation.hard_negative_factor",
            "label": "Доля hard negative тайлов",
            "value_type": "number",
            "tooltip": "Доля hard negative samples внутри marked budget; если их нет или мало, остаток берется из positive, сумма трех долей должна быть равна 1.",
            "min_value": 0,
            "max_value": 1,
        },
        {
            "key": "tile_preparation.background_factor",
            "label": "Доля background тайлов",
            "value_type": "number",
            "tooltip": "Доля обычных background samples в train epoch; не получает дефицит hard negative, сумма трех долей должна быть равна 1.",
            "min_value": 0,
            "max_value": 1,
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
                "cross_entropy",
                "cross_entropy_dice",
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
            "key": "train.hard_negative_weight",
            "label": "Hard negative weight",
            "value_type": "number",
            "tooltip": "Множитель штрафа только для пикселей внутри размеченных hard-negative зон. Остальной background имеет обычный вес 1.",
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
            "tooltip": "Ограничивает balanced validation subset до оценки RAM и чтения тайлов. Пустое значение означает весь subset.",
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
    "tile_preparation.context": 0,
    "tile_preparation.augmentation_level": 3,
    "tile_preparation.positive_factor": 0.8,
    "tile_preparation.hard_negative_factor": 0.0,
    "tile_preparation.background_factor": 0.2,
    "train.epochs": 80,
    "train.batch_size": 4,
    "train.learning_rate": 0.00001,
    "train.weight_decay": 0.0001,
    "train.loss": "focal_tversky",
    "train.focal_alpha": 0.6,
    "train.pos_weight": 1.0,
    "train.hard_negative_weight": 1.0,
    "train.tversky_alpha": 0.4,
    "train.tversky_beta": 0.6,
    "train.threshold": 0.7,
    "train.early_stopping_patience": 12,
    "train.max_train_batches_per_epoch": 72,
    "train.max_val_batches_per_epoch": 1000,
    "train.max_training_time_sec": None,
}


TILE_FACTOR_KEYS = {
    "tile_preparation.positive_factor",
    "tile_preparation.hard_negative_factor",
    "tile_preparation.background_factor",
}


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

_TRAIN_FIELD_HELP: dict[str, tuple[str, str]] = {
    "dataset.val_fraction": (
        "Доля подготовленных тайлов, которая уходит в validation. Увеличение делает оценку стабильнее, но оставляет меньше данных для обучения. Связано с общим числом сцен: на маленьких датасетах слишком большая доля может обеднить train.",
        "0.1..0.25 обычно, 0.2 как базовый выбор; ниже 0.1 только для очень маленьких smoke-запусков, выше 0.3 при большом датасете и шумной метрике.",
    ),
    "tile_preparation.tile_size": (
        "Размер квадратного окна в пикселях. Чем больше тайл, тем больше контекст и расход GPU/RAM; чем меньше, тем быстрее обучение, но хуже крупные объекты и границы.",
        "256..768; 512 обычно. Увеличивать для крупных объектов и контекстных признаков, уменьшать при OOM или мелких объектах.",
    ),
    "tile_preparation.stride": (
        "Шаг между соседними окнами. Меньший шаг увеличивает перекрытие и число тайлов, помогает не терять границы объектов, но замедляет подготовку и обучение.",
        "tile_size/2 обычно. Делать ближе к tile_size для быстрых запусков, меньше tile_size/2 для редких объектов и важных границ.",
    ),
    "tile_preparation.context": (
        "Ширина контекстной рамки вокруг полезного центра. Модель видит всю рамку, но loss, validation и инференс используют только центр.",
        "128 для входа 768 (полезный центр 512); 0 сохраняет прежнее поведение.",
    ),
    "tile_preparation.augmentation_level": (
        "Интенсивность train-аугментаций. Применяется к positive и hard-negative тайлам, обычный background не аугментируется. Чем выше уровень, тем лучше обобщение, но выше риск исказить слабые признаки.",
        "0 для диагностики, 1 для геометрии, 2..3 для рабочих запусков; 3 использовать, когда датасет небольшой или есть переобучение.",
    ),
    "tile_preparation.positive_factor": (
        "Целевая доля positive тайлов в train sampler. Вместе с hard-negative образует marked-бюджет; если hard-negative тайлов мало, остаток hard-negative доли переносится сюда.",
        "0.4..0.8; повышать при редких positive объектах или низком recall, снижать при большом числе false positive.",
    ),
    "tile_preparation.hard_negative_factor": (
        "Целевая доля hard-negative тайлов внутри train sampler. Это размеченные области, которые должны оставаться фоном. Не создает отдельный класс модели.",
        "0..0.4; начинать с 0.1..0.3 при известных ложных срабатываниях. Если hard-negative мало, sampler ограничит долю и отдаст остаток positive.",
    ),
    "tile_preparation.background_factor": (
        "Доля обычных фоновых тайлов. Background не получает дефицит marked-бюджета и нужен, чтобы модель видела разнообразный нормальный фон.",
        "0.1..0.4; снижать при очень редких объектах, повышать при false positive на обычном фоне. Сумма трех долей должна быть 1.",
    ),
    "train.epochs": (
        "Максимальное число эпох. Реальная остановка может наступить раньше по early stopping или wall-clock лимиту.",
        "30..100 для рабочих запусков; 1..5 для smoke. Увеличивать, если val F1 еще растет и нет переобучения.",
    ),
    "train.batch_size": (
        "Количество тайлов в одном optimizer step. Больше batch стабилизирует градиент, но требует больше GPU-памяти.",
        "2..8 для 512px на текущих моделях; уменьшать при OOM, повышать при свободной памяти и шумном loss.",
    ),
    "train.learning_rate": (
        "Скорость обновления AdamW. Слишком большая дает скачки loss и плохой checkpoint, слишком маленькая замедляет обучение.",
        "1e-5..3e-4; для fine-tune SegFormer обычно 1e-5..5e-5. Менять вместе с batch size и длительностью обучения.",
    ),
    "train.weight_decay": (
        "L2-регуляризация AdamW. Помогает против переобучения, но слишком большое значение мешает подстроиться под новый класс.",
        "0..1e-3; обычно 1e-4. Повышать при переобучении, снижать если train loss плохо падает.",
    ),
    "train.loss": (
        "Binary loss для обучения одного класса. bce_dice проще и стабильнее, focal_dice сильнее фокусируется на сложных пикселях, focal_tversky отдельно управляет FP/FN.",
        "focal_tversky для рабочих запусков с дисбалансом, bce_dice для диагностики, focal_dice если нужно усилить сложные пиксели без Tversky.",
    ),
    "train.focal_alpha": (
        "Баланс focal-компоненты между positive и background. Влияет на focal_dice/focal_tversky и связан с pos_weight.",
        "0.4..0.8; повышать при низком recall, снижать при избытке false positive.",
    ),
    "train.pos_weight": (
        "Вес positive пикселей в binary BCE/focal части loss. Background остается 1, hard negative регулируется отдельным hard_negative_weight.",
        "1..5; повышать при пропусках объектов и низком recall, снижать при жирных масках и false positive.",
    ),
    "train.hard_negative_weight": (
        "Множитель штрафа только для пикселей внутри размеченных hard-negative зон. Остальной background имеет обычный вес 1; positive pixels регулируются отдельно через positive weight и Dice/Tversky-компоненты.",
        "1..5; 1 выключает усиление. Повышать при false positive на hard-negative объектах, снижать если модель начинает терять похожие настоящие positive.",
    ),
    "train.tversky_alpha": (
        "Штраф false positive в Tversky-компоненте. Больше alpha делает модель осторожнее и уменьшает лишние выделения.",
        "0.3..0.7; повышать при false positive, снижать если модель недовыделяет объекты.",
    ),
    "train.tversky_beta": (
        "Штраф false negative в Tversky-компоненте. Больше beta сильнее наказывает пропуски positive пикселей.",
        "0.3..0.8; повышать при низком recall, снижать при разрастании масок. Часто alpha+beta держат около 1.",
    ),
    "train.threshold": (
        "Порог вероятности для binary validation и последующего инференса из checkpoint metadata. Не меняет logits, но меняет precision/recall trade-off.",
        "0.5..0.8; повышать при false positive, снижать при пропусках. Лучший threshold также оценивается на validation.",
    ),
    "train.early_stopping_patience": (
        "Сколько эпох ждать без улучшения validation F1 перед остановкой. Защищает от лишнего времени и переобучения.",
        "8..20; меньше для быстрых итераций, больше когда метрика шумная или датасет большой.",
    ),
    "train.max_train_batches_per_epoch": (
        "Ограничение длины train epoch в batch-ах. Это debug/smoke рычаг: он ускоряет итерацию, но меняет статистику обучения.",
        "Пусто для полного обучения; 50..200 для быстрых HPO/smoke, если нужно сравнить конфиги за ограниченное время.",
    ),
    "train.max_val_batches_per_epoch": (
        "Ограничение фиксированного balanced validation subset в batch-ах до оценки RAM и чтения тайлов. Меньший subset снижает расход памяти и ускоряет проверку, но делает метрику менее стабильной.",
        "256..1000 для крупных тайлов; пусто только когда полный val гарантированно помещается в RAM или допустим ленивый режим.",
    ),
    "train.max_training_time_sec": (
        "Wall-clock лимит обучения. Проверяется после завершения эпохи, поэтому процесс сохраняет final checkpoint штатно.",
        "Пусто без лимита; 1800 для короткого запуска, 7200..14400 для длинных рабочих обучений.",
    ),
}

_INFERENCE_FIELD_HELP: dict[str, tuple[str, str]] = {
    "postprocess.mask_min_object_pixels": (
        "Удаляет мелкие connected components прямо в raster mask до векторизации. Помогает убрать шум, но может потерять маленькие настоящие объекты.",
        "Пусто для авто-профиля; 16..128 px. Повышать при точечном шуме, снижать для мелких объектов.",
    ),
    "postprocess.mask_min_hole_pixels": (
        "Заливает маленькие дырки в raster mask. Укрупняет и стабилизирует полигоны, но может закрыть реальные внутренние просветы.",
        "Пусто для авто-профиля; 16..256 px. Повышать при рваных масках, снижать для объектов с настоящими отверстиями.",
    ),
    "postprocess.binary_closing_radius": (
        "Радиус morphological closing в пикселях. Соединяет близкие фрагменты и сглаживает разрывы до векторизации.",
        "Пусто для авто-профиля; 1..3 px. Использовать при разорванных масках, не повышать при близких разных объектах.",
    ),
    "postprocess.min_area_m2": (
        "Минимальная площадь итогового полигона. Фильтрует мелкие ложные объекты после перевода в геометрию.",
        "Пусто для авто-профиля; 100..10000 м² по масштабу класса. Повышать при мелком шуме, снижать для маленьких объектов.",
    ),
    "postprocess.min_hole_area_m2": (
        "Минимальная площадь дырки, которую оставляем в полигоне. Меньшие дырки удаляются из геометрии.",
        "Пусто для авто-профиля; 100..10000 м². Повышать для цельных объектов, снижать если внутренние пустоты важны.",
    ),
    "postprocess.simplify_m": (
        "Упрощение контура в метрах. Снижает число вершин и делает GeoJSON легче, но может съесть тонкие детали.",
        "Пусто для авто-профиля; 1..20 м. Повышать для тяжелых/шумных контуров, снижать для точной границы.",
    ),
    "postprocess.filter_compact_objects.enabled": (
        "Включает удаление компактных объектов по форме. Полезно для рек, когда нужно убрать озера и пруды, но опасно для классов, где объект сам компактный.",
        "Обычно выключено; включать для вытянутых классов вроде рек, где компактные полигоны являются ложными.",
    ),
    "postprocess.filter_compact_objects.min_isoperimetric_quotient": (
        "Порог компактности: чем выше, тем меньше объектов считаются достаточно компактными для удаления.",
        "0.2..0.5; повышать, если фильтр удаляет слишком много, снижать, если компактные ложные объекты остаются.",
    ),
    "postprocess.filter_compact_objects.max_bbox_ratio": (
        "Порог вытянутости minimum rotated rectangle. Компактный объект удаляется только если он не слишком вытянут.",
        "2..5; снижать для более строгого удаления круглых объектов, повышать чтобы захватывать слегка вытянутые ложные объекты.",
    ),
}

def _apply_schema_help(schema: dict[str, Any], help_by_key: dict[str, tuple[str, str]]) -> None:
    for field in schema["fields"]:
        key = str(field["key"])
        details = help_by_key.get(key)
        if details is None:
            continue
        field["tooltip"], field["recommended_range"] = details


_apply_schema_help(CONFIG_SCHEMA, _TRAIN_FIELD_HELP)
_apply_schema_help(INFERENCE_CONFIG_SCHEMA, _INFERENCE_FIELD_HELP)

CONFIG_KEYS = {str(field["key"]) for field in CONFIG_SCHEMA["fields"]}
CONFIG_FIELDS = {str(field["key"]): field for field in CONFIG_SCHEMA["fields"]}
INFERENCE_CONFIG_KEYS = {str(field["key"]) for field in INFERENCE_CONFIG_SCHEMA["fields"]}
INFERENCE_CONFIG_FIELDS = {
    str(field["key"]): field for field in INFERENCE_CONFIG_SCHEMA["fields"]
}


def sanitize_template_config(
    config: dict[str, Any] | None,
    *,
    fallback: dict[str, Any] | None = None,
    normalize_factors: bool = True,
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
    if "tile_preparation.context" not in (config or {}):
        tile_size = int(result.get("tile_preparation.tile_size") or 0)
        result["tile_preparation.context"] = 128 if tile_size == 768 else 0
    _resolve_legacy_tile_factors(result, config or {})
    if normalize_factors:
        normalize_tile_factors(result)
    return result


def normalize_tile_factors(config: dict[str, Any]) -> None:
    positive_factor = _float_or_default(config.get("tile_preparation.positive_factor"), 0.5)
    hard_negative_factor = _float_or_default(
        config.get("tile_preparation.hard_negative_factor"),
        0.0,
    )
    background_factor = _float_or_default(config.get("tile_preparation.background_factor"), 0.5)
    if abs(positive_factor + hard_negative_factor + background_factor - 1.0) <= 1e-6:
        config["tile_preparation.positive_factor"] = positive_factor
        config["tile_preparation.hard_negative_factor"] = hard_negative_factor
        config["tile_preparation.background_factor"] = background_factor
        return
    background_factor = min(max(background_factor, 0.0), 1.0)
    hard_negative_factor = min(max(hard_negative_factor, 0.0), 1.0 - background_factor)
    positive_factor = max(0.0, round(1.0 - hard_negative_factor - background_factor, 12))
    config["tile_preparation.positive_factor"] = positive_factor
    config["tile_preparation.hard_negative_factor"] = hard_negative_factor
    config["tile_preparation.background_factor"] = background_factor


def _resolve_legacy_tile_factors(result: dict[str, Any], config: dict[str, Any]) -> None:
    if "tile_preparation.background_factor" in config:
        return
    positive_factor = _float_or_default(result.get("tile_preparation.positive_factor"), 0.5)
    hard_negative_factor = _float_or_default(
        result.get("tile_preparation.hard_negative_factor"),
        0.0,
    )
    result["tile_preparation.background_factor"] = (
        1.0 - positive_factor - hard_negative_factor
    )


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
            "external_torchscript",
            "импортированная TorchScript-модель",
            source="manual",
        ),
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
