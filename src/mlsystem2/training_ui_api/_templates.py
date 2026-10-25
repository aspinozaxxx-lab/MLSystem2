"""Исходные шаблоны обучения и инференса для миграций и reset."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .contracts import TrainingUIAPIError

COMPACT_FILTER_REMOVE = "remove_compact"
COMPACT_FILTER_KEEP = "keep_compact"
COMPACT_FILTER_MODES = (COMPACT_FILTER_REMOVE, COMPACT_FILTER_KEEP)
NEXT_GEN2_INFERENCE_THRESHOLD = 0.9
NEXT_GEN2_INFERENCE_BATCH_SIZE = 32
NEXT_GEN2_TRAIN_BATCH_SIZES = {512: 16, 768: 8, 1024: 4, 1536: 2}
NEXT_GEN2_MODEL_BATCH_SIZES = {
    "segformer_b0": (16, 8, 4, 2), "smp_segformer_b0": (16, 8, 4, 2),
    "smp_segformer_b1": (8, 4, 2, 1), "smp_segformer_b2": (4, 2, 1, 1),
    "smp_segformer_b3": (4, 2, 1, 1),
}
OBJECT_F1_MODEL_NAMES = frozenset(
    name for name in NEXT_GEN2_MODEL_BATCH_SIZES if name.startswith("smp_segformer_")
)


def next_gen2_train_batch_size(tile_size: int, architecture: str = "segformer_b0") -> int:
    if architecture not in NEXT_GEN2_MODEL_BATCH_SIZES:
        raise TrainingUIAPIError("next-gen2 поддерживает только архитектуры SegFormer.")
    if tile_size not in NEXT_GEN2_TRAIN_BATCH_SIZES:
        raise TrainingUIAPIError("next-gen2: размер тайла должен быть 512, 768, 1024 или 1536")
    return NEXT_GEN2_MODEL_BATCH_SIZES[architecture][tuple(NEXT_GEN2_TRAIN_BATCH_SIZES).index(tile_size)]


def next_gen2_inference_batch_size(tile_size: int, architecture: str = "segformer_b0") -> int:
    """Ограничить объём пикселей в inference-пакете при увеличении окна."""
    if tile_size in NEXT_GEN2_TRAIN_BATCH_SIZES:
        return 2 * next_gen2_train_batch_size(tile_size, architecture)
    return max(1, min(NEXT_GEN2_INFERENCE_BATCH_SIZE, 32 * 512 ** 2 // max(1, tile_size) ** 2))


NEXT_GEN2_DEFAULT_CONFIG: dict[str, Any] = {
    "train.pipeline_variant": "next_gen2",
    "dataset.val_fraction": 0.2,
    "tile_preparation.tile_size": 512,
    "tile_preparation.stride": 256,
    "tile_preparation.context": 0,
    "tile_preparation.augmentation_level": 3,
    "tile_preparation.positive_factor": 0.5,
    "tile_preparation.hard_negative_factor": 0.0,
    "tile_preparation.background_factor": 0.5,
    "train.pretrained": True,
    "train.initial_checkpoint_uri": None,
    "train.epochs": 20,
    "train.batch_size": 16,
    "train.learning_rate": 1e-4,
    "train.weight_decay": 0.01,
    "train.loss": "cross_entropy_tversky",
    "train.tversky_alpha": 0.75,
    "train.tversky_beta": 0.25,
    "train.pos_weight": 1.0,
    "train.background_weight": 1.0,
    "train.hard_negative_weight": 1.0,
    "train.threshold": 0.5,
    "train.early_stopping_patience": 10,
    "train.max_train_batches_per_epoch": None,
    "train.max_val_batches_per_epoch": None,
    "train.max_training_time_sec": None,
}

NEXT_GEN2_EDITABLE_KEYS = {
    "train.pretrained", "tile_preparation.tile_size", "train.epochs", "train.early_stopping_patience", "train.max_training_time_sec",
    "train.diagnostic_checkpoint_interval_sec",
}
NEXT_GEN3_DEFAULT_CONFIG = {**NEXT_GEN2_DEFAULT_CONFIG, "train.pipeline_variant": "next_gen3"}
NEXT_GEN4_DEFAULT_CONFIG = {**NEXT_GEN3_DEFAULT_CONFIG, "train.pipeline_variant": "next_gen4", "tile_preparation.stride": 512}
NEXT_GEN4_GLOBAL_NORM_DEFAULT_CONFIG = {**NEXT_GEN4_DEFAULT_CONFIG, "train.pipeline_variant": "next_gen4_global_norm"}
NEXT_GEN4_AUGMENTATION_V3_DEFAULT_CONFIG = {**NEXT_GEN4_GLOBAL_NORM_DEFAULT_CONFIG, "train.pipeline_variant": "next_gen4_augmentation_v3"}
NEXT_GEN4_AUGMENTATION_V4_DEFAULT_CONFIG = {**NEXT_GEN4_AUGMENTATION_V3_DEFAULT_CONFIG, "train.pipeline_variant": "next_gen4_augmentation_v4"}
NEXT_GEN4_CONSISTENCY_V1_DEFAULT_CONFIG = {**NEXT_GEN4_AUGMENTATION_V4_DEFAULT_CONFIG, "train.pipeline_variant": "next_gen4_consistency_v1"}
NEXT_GEN4_CONSISTENCY_V2_DEFAULT_CONFIG = {**NEXT_GEN4_CONSISTENCY_V1_DEFAULT_CONFIG, "train.pipeline_variant": "next_gen4_consistency_v2"}
NEXT_GEN4_CONSISTENCY_V3_DEFAULT_CONFIG = {**NEXT_GEN4_CONSISTENCY_V2_DEFAULT_CONFIG, "train.pipeline_variant": "next_gen4_consistency_v3"}
NEXT_GEN4_CONSISTENCY_V4_DEFAULT_CONFIG = {**NEXT_GEN4_CONSISTENCY_V3_DEFAULT_CONFIG, "train.pipeline_variant": "next_gen4_consistency_v4"}
NEXT_GEN4_CONSISTENCY_V4_DESCRIPTION = (
    "Отдельный профиль согласованности v4 для SMP SegFormer B0/B1/B2/B3. "
    "Наследует модель, попиксельную LayerNorm и настройки v3. "
    "KL по правильному уверенному учителю сохраняется в обоих направлениях. "
    "На допустимых train-пикселях, где ни один учитель не подходит, оба контекста "
    "получают CrossEntropy по обучающему эталону вместо отсутствующего дополнительного градиента.\n\n"
    "KL и новый надзор образуют одну попиксельную функцию потерь каждого направления: "
    "равный вес присутствующих классов, среднее активных направлений, общий вес 0,25 "
    "и вероятность 0,5. Невалидные и неоднозначные пиксели исключены. "
    "Это не обучение по внешней тестовой маске и не усреднение ошибочных ответов.\n\n"
    "Выберите next-gen4-consistency-v4 в ручном запуске или шаблоне SMP SegFormer. "
    "В истории отдельно видны вклад KL, вклад CE неохваченных пикселей и счётчики. "
    "Сетка, split, данные, глобальная нормализация, основной loss, sampler, scheduler, "
    "выбор best и инференс прежние; старые профили и веса не преобразуются. "
    "Отсутствие кругов нужно подтвердить новым обучением и оценкой всей сцены."
)
NEXT_GEN4_CONSISTENCY_V3_DESCRIPTION = (
    "Отдельный профиль согласованности v3 для SMP SegFormer B0/B1/B2/B3. "
    "Сохраняет попиксельную LayerNorm декодера и остальные настройки v2. "
    "KL-штраф действует в обоих направлениях: правильный прогноз полного окна учит вырезку, "
    "а правильная вырезка может учить полное окно. Учитель всегда отсоединён от градиента, "
    "его уверенность должна быть строго выше 0,9 и класс должен совпадать с train-разметкой.\n\n"
    "Для каждого направления сохраняется равный вес присутствующих классов; "
    "активные направления усредняются, общий вес штрафа остаётся 0,25. "
    "Нет подходящего учителя в обоих контекстах — нет дополнительного градиента. "
    "Оба прохода выполняются в eval с восстановлением флагов; optimizer обновляется один раз. "
    "Данные, split, основной CE/Tversky, sampler, scheduler и выбор best по val loss прежние.\n\n"
    "Выберите next-gen4-consistency-v3 в ручном запуске или шаблоне SMP SegFormer. "
    "Период 900 секунд сохраняет текущие срезы на границах эпох без перезапуска optimizer. "
    "Внешний инференс, Gaussian, пороги и постобработка не меняются; "
    "старые веса не преобразуются. Новая обратная ветка устраняет односторонность штрафа, "
    "но качество всей сцены и отсутствие кругов необходимо подтвердить новым обучением."
)
NEXT_GEN4_CONSISTENCY_V2_DESCRIPTION = (
    "Отдельный профиль согласованности v2 для SMP SegFormer B0/B1/B2/B3. "
    "Единственная замена относительно v1 — нормализатор декодера: вместо BatchNorm "
    "используется LayerNorm по каналам каждого пикселя, без статистики соседних пикселей "
    "или других окон пакета. Формула одинакова при обучении и проверке; "
    "предобученный encoder и остальные слои сохраняются. Это не общая пространственная GroupNorm.\n\n"
    "Общая нормализация raw-входа только по train, защищённые аугментации и вырезки v4, "
    "KL-штраф 0,25 по правильному уверенному учителю, основной CE/Tversky, сэмплер, "
    "split, seed, scheduler и выбор лучших весов по validation loss не меняются. "
    "При инференсе нет дополнительной ветки; Gaussian, пороги и постобработка прежние.\n\n"
    "Выберите next-gen4-consistency-v2 в ручном запуске или шаблоне SMP SegFormer "
    "и обучите новую модель. Для временной диагностики задайте период срезов 900 секунд. "
    "Checkpoint сохраняет точную версию нормализатора; старые веса не преобразуются. "
    "Профиль устраняет зависимость этого слоя от состава batch, но исчезновение кругов "
    "и сохранение качества всей сцены нужно проверить на новых весах."
)
NEXT_GEN4_CONSISTENCY_V1_DESCRIPTION = (
    "Next-gen4-consistency-v1 наследует глобальную нормализацию и защиту valid-mask v4. "
    "Дополнительный KL-штраф с весом 0,25 учит одинаковые допустимые пиксели сохранять прогноз "
    "при изменении контекста. В половине train-batch сравниваются полное окно и его немасштабированная "
    "вырезка со стороной 50% либо 75%. Используются только уже выбранные train-пиксели; "
    "правильный по разметке учитель должен иметь вероятность более 0,9. "
    "Оба дополнительных прохода идут в eval-режиме без повторного обновления BatchNorm и без dropout; "
    "градиент учителя отключён. Классы балансируются, nodata и неоднозначные отрицательные области исключены.\n\n"
    "Основной CE/Tversky, split, автоматический фон, scheduler и best по validation loss прежние. "
    "Проверка и инференс не используют дополнительную ветку, Gaussian и пороги не меняются. "
    "Обучающий шаг дороже, поэтому за одинаковое время может пройти меньше эпох. "
    "Для временной диагностики можно задать период текущих срезов, например 900 секунд. "
    "Срез сохраняется после завершённой эпохи с фактическим временем; обучение не перезапускается. "
    "Выберите отдельный профиль в ручном запуске или шаблоне SegFormer; старые веса не изменяются. "
    "Устранение кругов нужно подтвердить проверкой, рост общей F1 сам по себе этого не доказывает."
)
NEXT_GEN4_AUGMENTATION_V4_DESCRIPTION = (
    "Next-gen4-augmentation-v4 — отдельный профиль на основе v3: глобальная нормализация train "
    "и защита valid-mask сохраняются. Перед прежней геометрией с вероятностью 50% выбирается "
    "квадратная вырезка площадью 25–100% текущего train-окна и масштабируется обратно. "
    "Снимок и все маски преобразуются совместно; соседние validation/test-окна не читаются. "
    "Если геометрия убрала все допустимые пиксели непустого окна, возвращается исходный пример без повтора. "
    "Это увеличивает разнообразие контекста и долей объекта, в том числе почти сплошной воды. "
    "Вырезки не применяются при проверке или инференсе.\n\n"
    "В отчётах MLflow отдельно сохраняются счётчики и F1/полнота плотных окон: "
    "не менее 99% допустимых пикселей и не менее 90% либо 99% площади объектов среди них. "
    "Validation оценивается каждую эпоху, test и неаугментированные плотные train-окна — на лучших весах. "
    "Диагностические пороги 0,5 и 0,9 фиксированы и не участвуют в выборе checkpoint; "
    "пустая группа имеет число окон 0 и неопределённые метрики.\n\n"
    "Выберите next-gen4-augmentation-v4 в ручном запуске или шаблоне SegFormer B0/B1/B2/B3 "
    "и обучите новую сеть. Старые профили, веса, исходные TIFF и разметка не меняются. "
    "Сетка, split, автоматический фон, loss, scheduler и Gaussian-инференс остаются прежними. "
    "Профиль уменьшает нехватку разнообразных контекстов, но устранение пропусков и рост F1 "
    "нужно подтвердить сравнением новых результатов, а не считать гарантированными."
)
NEXT_GEN4_GLOBAL_NORM_DESCRIPTION = (
    "Next-gen4-global-norm сохраняет неперекрывающиеся train-тайлы и автоматический фон next-gen4, "
    "но использует общую поканальную min-max нормализацию вместо независимой нормализации каждого окна. "
    "Минимумы и максимумы рассчитываются один раз только по валидным пикселям train-тайлов до аугментаций. "
    "Validation, test, nodata, alpha и padding в статистику не входят. "
    "Границы сохраняются в checkpoint и одинаково применяются при обучении, проверке, "
    "нативном инференсе и экспорте ONNX/Triton. Значения вне train-диапазона ограничиваются от 0 до 1. "
    "Постоянный во всём train канал даёт нули без деления на ноль; нулевое нормализованное значение "
    "само по себе не становится nodata — маска определяется по исходным пикселям.\n\n"
    "Размер окна 512/768/1024/1536 px, шаг подготовки равен размеру окна; split 60/20/20 с seed 42, "
    "Gaussian-инференс с перекрытием и полным покрытием краёв — как в next-gen4. "
    "Зоны одного TIFF используют общую сетку, но соседство тайлов и пересечения разных TIFF "
    "не обеспечивают независимость территорий. Предобучение, loss, scheduler и правила выбора весов не меняются.\n\n"
    "Выберите отдельный конвейер next-gen4-global-norm в ручном запуске или шаблоне SegFormer B0/B1/B2/B3 "
    "и обучите новую сеть. Старые конвейеры и модели не переключаются автоматически. "
    "Продолжение допускает только совместимый checkpoint с теми же параметрами нормализации. "
    "Общий масштаб устраняет зависимость яркости от состава окна, но не гарантирует исчезновение "
    "всех пропусков или рост F1: результат нужно проверить на независимой разметке."
)
NEXT_GEN4_AUGMENTATION_V3_DESCRIPTION = (
    "Next-gen4-augmentation-v3 — отдельный профиль на основе next-gen4-global-norm. "
    "Общие поканальные границы рассчитываются только на train и сохраняются в checkpoint. "
    "Геометрия совместно переносит изображение, разметку, отрицательные области и valid-mask. "
    "Яркость, контраст, шум и размытие не должны превращать валидный пиксель в нулевой nodata. "
    "При таком конфликте целиком отменяется только фотометрический блок этого примера; "
    "геометрия сохраняется, значения не подменяются единицами и повторного случайного подбора нет. "
    "Если сама интерполяция геометрии обнулила валидные пиксели, геометрический блок отменяется "
    "вместе со всеми его масками. За пределами valid-mask восстанавливается ноль.\n\n"
    "Доступны SegFormer B0/B1/B2/B3, RGB и RGB+NIR, окна 512/768/1024/1536. "
    "Сетка без перекрытий, split 60/20/20, автоматический фон, loss, scheduler, выбор весов "
    "и Gaussian-инференс наследуются от next-gen4-global-norm. Порог внешнего инференса остаётся 0,9. "
    "Политика validity_guard_v3 фиксируется в спецификации модели. Старые профили и веса не меняются; "
    "нужна новая тренировка. Защита устраняет конфликт аугментаций с nodata, но не гарантирует "
    "исчезновение всех пропусков и зависимостей прогноза от окна."
)
NEXT_GEN4_DESCRIPTION = (
    "Next-gen4 сохраняет автоматическую балансировку фона next-gen3, но устраняет пересечение обучающих и проверочных тайлов одного TIFF. "
    "Размер окна выбирается при запуске и в шаблоне: 512, 768, 1024 или 1536 px; шаг подготовки всегда равен размеру окна, контекст равен нулю. "
    "Сетка привязана к началу исходного снимка. Краевые тайлы дополняются nodata, не участвующим в loss и метриках. "
    "Зоны одного исходного TIFF используют общую сетку; один тайл не может попасть в разные выборки.\n\n"
    "Случайное разбиение исходных тайлов 60/20/20 с seed 42, без разделения по сценам. Пересечения разных снимков разрешены; "
    "соседство тайлов и повторная съёмка той же территории остаются ограничениями независимости оценки. "
    "Manifest сохраняет сетку, индексы и проверку отсутствия пересечений внутри исходного TIFF. "
    "Предобучение, архитектуры, loss, scheduler, полная validation и автоматический фон наследуются от next-gen3. "
    "При инференсе сохраняется Gaussian с шагом в половину окна; nodata исключается из результата."
)
NEXT_GEN3_DESCRIPTION = (
    "Next-gen3 — SegFormer с автоматической балансировкой объектов и фона. "
    "Профиль наследует архитектуры, min-max, аугментации, loss, optimizer и полную validation next-gen2. "
    "Любой допустимый пиксель объекта делает тайл положительным; nodata и неразмеченные зоны исключены из loss и метрик. "
    "Пустой тайл считается фоном только в пределах принятой зоны разметки: пропущенные объекты остаются ошибкой эталона.\n\n"
    "Число шагов фиксируется на весь запуск. Бюджет — все информативные тайлы плюс не больше восьми фонов на каждый, "
    "с резервом фона 256 тайлов или 32 на снимок, если столько доступно. "
    "При крайне редких объектах бюджет ограничен 20 примерами на информативный тайл; резерв охватывается ротацией. "
    "Суммарный бюджет не превышает полный train. Каждый информативный тайл показывается минимум один и максимум четыре раза за эпоху. "
    "Фон внутри эпохи уникален и сменяется по снимкам и пространственным зонам; до 25% квоты занимают сложные отрицательные примеры, "
    "найденные только на train, остальные выбираются ротацией. При малом числе фоновых тайлов используются все.\n\n"
    "После двух прогревочных эпох доля фона меняется шагом до пяти процентных пунктов по сглаженным ошибкам полной validation "
    "на фиксированном пороге 0,5. Рост ложных срабатываний увеличивает долю фона; падение полноты объектов защищает положительные примеры. "
    "Доля фона ограничена 25–90%, если доступно достаточно обоих типов; при дефиците берётся весь фон. "
    "Пределы квот и повторов имеют приоритет. Test не управляет балансировкой. "
    "Квоты, ошибки, причины решений, seed и состояние сохраняются в checkpoint и MLflow.\n\n"
    "Выберите next-gen3 в конвейере шаблона или ручного запуска; дополнительных коэффициентов не требуется. "
    "Лучший checkpoint выбирается по минимуму validation loss, test оценивается один раз. "
    "Оконное разбиение 60/20/20 с перекрытием наследуется от next-gen2: географическая независимость не обеспечивается. "
    "Качество подтверждайте на отдельной территории. Инференс сохраняет Gaussian-сборку и порог 0,9; "
    "порог обратной связи 0,5 не является метрикой внешнего инференса. "
    "При продолжении от весов новый этап заново запускает балансировщик, как optimizer и scheduler."
)
NEXT_GEN2_DESCRIPTION = (
    "Профиль на основе segFormer_train_hlam_main_v2.ipynb для выбранной архитектуры SegFormer. "
    "SegFormer B0/B1/B2/B3 использует соответствующий encoder MiT и новую голову. "
    "Галочка «Предобученные веса» загружает веса ImageNet для encoder; без неё вся сеть инициализируется случайно. "
    "Бинарная сеть различает фон и объект; мультиклассовая — фон и каждый тип объектов из схемы датасета. "
    "Каналы определяются датасетом: RED, GRN, BLU, NIR для Канопус или RED, GRN, BLU для ортофотопланов. "
    "При включённых предобученных весах для NIR копируются веса RED; у RGB-модели сохраняется исходная входная свёртка.\n\n"
    "Размер тайла на выбор: 512, 768, 1024 или 1536 px, по умолчанию 512. "
    "Шаг равен половине размера тайла: перекрытие 50%, без контекста и дополнения краёв. "
    "Каждый канал каждого тайла нормализуется min-max; постоянный канал становится нулевым. Nodata входит в расчёты. "
    "Случайное разбиение по тайлам: 60% train, 20% validation, 20% test, seed 42. "
    "Пространственные пересечения не исключаются: соседние тайлы могут содержать общие пиксели в разных выборках, "
    "поэтому такая оценка не заменяет проверку на независимых снимках.\n\n"
    "Сэмплирование с возвращением: вес тайла с объектами — 7 при доле объекта больше 0,1%; "
    "с hard negative — 8; обычного фона — 1. Это относительные веса тайлов, а не проценты выборки. "
    "Общий hard negative перекрывает положительную разметку и считается фоном; отрицательная область конкретного класса штрафует только этот класс. "
    "Веса классов CrossEntropy рассчитываются по обратным частотам только обучающих масок.\n\n"
    "Аугментации применяются ко всем обучающим тайлам до нормализации: горизонтальное и вертикальное отражения "
    "и поворот на 90° с вероятностью 0,5 каждый; affine с вероятностью 0,4 — масштаб 0,9–1,1, "
    "сдвиг до 32 px и поворот до 15° с зеркальными границами. С вероятностью 0,3 выбирается шум "
    "с дисперсией 5–30 или Gaussian blur с ядром 3–5; яркость и контраст ±15% — с вероятностью 0,3. "
    "Validation и test без аугментаций.\n\n"
    "Batch size выбирается автоматически для тайлов 512/768/1024/1536 px: "
    "B0 — 16/8/4/2, B1 — 8/4/2/1, B2 и B3 — 4/2/1/1, "
    "чтобы большие окна помещались в видеопамять. AdamW, learning rate 0,0001, weight decay 0,01; float32 без gradient clipping. "
    "Параллельная загрузка и аугментации используют серверные DataLoader workers. "
    "Loss: 25% CrossEntropy + 75% Tversky (alpha 0,75, beta 0,25). "
    "ReduceLROnPlateau следит за validation loss: patience 3, уменьшение LR вдвое. "
    "Полная validation — каждую эпоху; лучший checkpoint и early stopping — по минимуму validation loss. "
    "После обучения лучшие веса один раз оцениваются на test при пороге 0,5; метрики и LR по эпохам сохраняются в MLflow.\n\n"
    "По умолчанию: максимум 20 эпох, early stopping patience 10 эпох, без лимита времени. "
    "Изменять можно предобученные веса, размер тайла и эти три условия остановки. Лимит времени проверяется после завершения эпохи."
)


OBJECT_F1_DEFAULT_CONFIG = {**NEXT_GEN2_DEFAULT_CONFIG, "train.pipeline_variant": "object_f1"}
OBJECT_F1_V2_DEFAULT_CONFIG = {**NEXT_GEN2_DEFAULT_CONFIG, "train.pipeline_variant": "object_f1_v2"}
OBJECT_F1_DESCRIPTION = (
    "Конвейер для SegFormer B0/B1/B2/B3 на основе next-gen2 с соответствующим encoder MiT. "
    "Сеть предсказывает область и отдельную карту границ; "
    "после объединения окон watershed разделяет соприкасающиеся объекты на самостоятельные полигоны. "
    "Один Polygon — один объект, части MultiPolygon считаются отдельно. Исходная разметка не изменяется.\n\n"
    "RGB для ортофотопланов, RGB+NIR для Канопуса. Alpha используется только как valid mask. "
    "Предобученные веса — ImageNet encoder; галочку можно снять. Min-max нормализация каждого канала окна. "
    "Размеры тайла 512/768/1024/1536 px, шаг — половина тайла, полное покрытие краёв. "
    "Batch size выбирается автоматически: B0 — 16/8/4/2, B1 — 8/4/2/1, B2 и B3 — 4/2/1/1.\n\n"
    "Разбиение 60/20/20 по независимым группам снимков, seed 42. Пересекающиеся территории остаются в одной части. "
    "Sampler 7/8/1 и аугментации next-gen2; заполнение за пределами изображения исключено из loss. "
    "Границы шириной 2 px формируются по ID после совместных преобразований изображения и масок.\n\n"
    "FP32, AdamW, learning rate 0,0001, weight decay 0,01. Loss области: 25% weighted CrossEntropy + 75% Tversky "
    "(alpha 0,75; beta 0,25); добавляется половина loss границ (BCE и Dice поровну). "
    "ReduceLROnPlateau уменьшает LR вдвое после 3 эпох без улучшения validation loss.\n\n"
    "Полная validation каждую эпоху: сначала объединяются окна, затем выделяются объекты. "
    "Лучшие веса и early stopping — по максимальному object F1 при IoU ≥ 0,5. "
    "Пороги области и границ — 0,5; те же правила используются в псевдоразметке и Geoalert. "
    "Test оценивается один раз на лучшем checkpoint. По умолчанию 20 эпох, patience 10, без лимита времени."
)


OBJECT_F1_V2_DESCRIPTION = (
    "Конвейер object f1 v2 для бинарных датасетов и SegFormer B0/B1/B2/B3. "
    "Сеть различает три взаимоисключающих класса: фон, внутренность объекта и широкую разделительную полосу. "
    "Полные исходные полигоны и их идентификаторы сохраняются; обучающие маски строятся автоматически. "
    "Граница шириной до 8 px адаптируется к размеру объекта, чтобы у небольших участков оставалась внутренность. "
    "Небольшие нахлёсты разметки разрешаются при подготовке целей с сохранением общего стыка.\n\n"
    "RGB для ортофотопланов, RGB+NIR для Канопуса; alpha используется только как маска валидности. "
    "Предобученные веса ImageNet можно включить независимо. Нормализация min-max по каналам окна. "
    "Тайлы 512/768/1024/1536 px, шаг — половина тайла, полное покрытие краёв. "
    "Batch size: B0 — 16/8/4/2, B1 — 8/4/2/1, B2 и B3 — 4/2/1/1.\n\n"
    "Разбиение примерно 60/20/20 по независимым группам снимков, seed 42; необходимо минимум три группы. "
    "Пересекающиеся территории остаются в одной части. Nodata, alpha и заполнение после аугментаций "
    "исключены из обучения и метрик. Веса классов рассчитываются только по обучающей части.\n\n"
    "FP32, AdamW, learning rate 0,0001, weight decay 0,01. "
    "Взвешенная CrossEntropy и Tversky обучают три класса с усилением разделительных полос. "
    "При распознавании вероятности окон объединяются, внутренние маркеры очищаются от мелкого шума, "
    "короткие разрывы границ обрабатываются, затем watershed восстанавливает полную площадь объектов "
    "и сохраняет отдельные идентификаторы до GeoJSON.\n\n"
    "Полная validation каждую эпоху на собранных снимках; лучшие веса и ранняя остановка — по object F1 "
    "при IoU ≥ 0,5. Scheduler следит за validation loss. Test оценивается один раз на лучших весах. "
    "Доступны предобучение, размер тайла и условия остановки: по умолчанию 20 эпох, patience 10, без лимита времени. "
    "Профиль улучшает механизм обучения границ, но качество разделения необходимо проверить на независимых снимках."
)


def fixed_pipeline_defaults(variant: str) -> dict[str, Any]:
    if variant == "next_gen4_consistency_v4":
        return NEXT_GEN4_CONSISTENCY_V4_DEFAULT_CONFIG
    if variant == "next_gen4_consistency_v3":
        return NEXT_GEN4_CONSISTENCY_V3_DEFAULT_CONFIG
    if variant == "next_gen4_consistency_v2":
        return NEXT_GEN4_CONSISTENCY_V2_DEFAULT_CONFIG
    if variant == "next_gen4_consistency_v1":
        return NEXT_GEN4_CONSISTENCY_V1_DEFAULT_CONFIG
    if variant == "next_gen4_augmentation_v4":
        return NEXT_GEN4_AUGMENTATION_V4_DEFAULT_CONFIG
    if variant == "next_gen4_augmentation_v3":
        return NEXT_GEN4_AUGMENTATION_V3_DEFAULT_CONFIG
    if variant == "next_gen4_global_norm":
        return NEXT_GEN4_GLOBAL_NORM_DEFAULT_CONFIG
    if variant == "next_gen4":
        return NEXT_GEN4_DEFAULT_CONFIG
    if variant == "next_gen3":
        return NEXT_GEN3_DEFAULT_CONFIG
    if variant == "object_f1_v2":
        return OBJECT_F1_V2_DEFAULT_CONFIG
    return OBJECT_F1_DEFAULT_CONFIG if variant == "object_f1" else NEXT_GEN2_DEFAULT_CONFIG


CONFIG_SCHEMA: dict[str, Any] = {
    "pipeline_defaults": {"next_gen2": NEXT_GEN2_DEFAULT_CONFIG, "next_gen3": NEXT_GEN3_DEFAULT_CONFIG, "next_gen4": NEXT_GEN4_DEFAULT_CONFIG, "next_gen4_global_norm": NEXT_GEN4_GLOBAL_NORM_DEFAULT_CONFIG, "next_gen4_augmentation_v3": NEXT_GEN4_AUGMENTATION_V3_DEFAULT_CONFIG, "next_gen4_augmentation_v4": NEXT_GEN4_AUGMENTATION_V4_DEFAULT_CONFIG, "next_gen4_consistency_v1": NEXT_GEN4_CONSISTENCY_V1_DEFAULT_CONFIG, "next_gen4_consistency_v2": NEXT_GEN4_CONSISTENCY_V2_DEFAULT_CONFIG, "next_gen4_consistency_v3": NEXT_GEN4_CONSISTENCY_V3_DEFAULT_CONFIG, "next_gen4_consistency_v4": NEXT_GEN4_CONSISTENCY_V4_DEFAULT_CONFIG, "object_f1": OBJECT_F1_DEFAULT_CONFIG, "object_f1_v2": OBJECT_F1_V2_DEFAULT_CONFIG},
"pipeline_descriptions": {"next_gen2": NEXT_GEN2_DESCRIPTION, "next_gen3": NEXT_GEN3_DESCRIPTION, "next_gen4": NEXT_GEN4_DESCRIPTION, "next_gen4_global_norm": NEXT_GEN4_GLOBAL_NORM_DESCRIPTION, "next_gen4_augmentation_v3": NEXT_GEN4_AUGMENTATION_V3_DESCRIPTION, "next_gen4_augmentation_v4": NEXT_GEN4_AUGMENTATION_V4_DESCRIPTION, "next_gen4_consistency_v1": NEXT_GEN4_CONSISTENCY_V1_DESCRIPTION, "next_gen4_consistency_v2": NEXT_GEN4_CONSISTENCY_V2_DESCRIPTION, "next_gen4_consistency_v3": NEXT_GEN4_CONSISTENCY_V3_DESCRIPTION, "next_gen4_consistency_v4": NEXT_GEN4_CONSISTENCY_V4_DESCRIPTION, "object_f1": OBJECT_F1_DESCRIPTION, "object_f1_v2": OBJECT_F1_V2_DESCRIPTION},
    "fields": [
        {
            "key": "train.pipeline_variant",
            "label": "Вариант конвейера",
            "value_type": "select",
            "tooltip": "Сначала выберите архитектуру. legacy — прежнее обучение; next-gen2 — профиль ноутбука с выбором весов по validation loss. object f1 обучает области и границы, выбирая лучшие веса по объектной F1 полных проверочных снимков. object f1 v2 различает фон, внутренность и широкую границу, восстанавливает полные отдельные полигоны и фильтрует мелкий шум. Оба объектных профиля предназначены для бинарных датасетов и SegFormer B0/B1/B2/B3. Подробности внизу формы.",
            "options": ["legacy", "next_gen2", "next_gen3", "next_gen4", "next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "object_f1", "object_f1_v2"],
        },
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
            "key": "train.pretrained",
            "label": "Предобученные веса",
            "value_type": "boolean",
            "tooltip": "Загрузить веса ImageNet для encoder выбранной SegFormer. Без галочки сеть обучается со случайной инициализацией. Доступно в legacy, next-gen2, object f1 и object f1 v2.",
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
                "cross_entropy_tversky",
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
            "key": "train.background_weight",
            "label": "Вес фона",
            "value_type": "number",
            "tooltip": "Базовый множитель штрафа для пикселей фона.",
            "min_value": 0.000001,
        },
        {
            "key": "train.hard_negative_weight",
            "label": "Hard negative weight",
            "value_type": "number",
            "tooltip": "Дополнительный множитель штрафа для размеченных hard-negative зон; умножается на вес фона.",
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
        {
            "key": "train.diagnostic_checkpoint_interval_sec",
            "label": "Период диагностических срезов, сек",
            "value_type": "integer-null",
            "tooltip": "Необязательные текущие веса на границе эпох без остановки optimizer. 900 — примерно каждые 15 минут; пусто — не сохранять.",
            "min_value": 1,
        },
    ]
}

BASE_DEFAULT_CONFIG: dict[str, Any] = {
    "train.pipeline_variant": "legacy",
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
    "train.pretrained": False,
    "train.learning_rate": 0.00001,
    "train.weight_decay": 0.0001,
    "train.loss": "focal_tversky",
    "train.focal_alpha": 0.6,
    "train.pos_weight": 1.0,
    "train.background_weight": 1.0,
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
            "key": "postprocess.smooth.enabled",
            "label": "Сглаживать контур",
            "value_type": "boolean",
            "tooltip": "Включает мягкое сглаживание полигонов после фильтрации.",
        },
        {
            "key": "postprocess.smooth.iterations",
            "label": "Итерации сглаживания",
            "value_type": "integer",
            "tooltip": "Число проходов углового сглаживания Smooth.",
            "min_value": 1,
            "max_value": 5,
        },
        {
            "key": "postprocess.smooth.offset",
            "label": "Сила сглаживания",
            "value_type": "number",
            "tooltip": "Доля ребра, срезаемая Smooth на каждом проходе.",
            "min_value": 0,
            "max_value": 0.5,
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
            "label": "Фильтровать объекты по компактности",
            "value_type": "boolean",
            "tooltip": "Включает прямой или обратный фильтр формы объектов.",
        },
        {
            "key": "postprocess.filter_compact_objects.mode",
            "label": "Режим фильтра компактности",
            "value_type": "select",
            "tooltip": "remove_compact удаляет компактные объекты, keep_compact оставляет только компактные.",
            "options": list(COMPACT_FILTER_MODES),
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
    "postprocess.smooth.enabled": False,
    "postprocess.smooth.iterations": 1,
    "postprocess.smooth.offset": 0.125,
    "postprocess.simplify_m": None,
    "postprocess.filter_compact_objects.enabled": False,
    "postprocess.filter_compact_objects.mode": COMPACT_FILTER_REMOVE,
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
        "Вес positive пикселей в binary BCE/focal части loss. Фон регулируется отдельным background_weight, hard negative — hard_negative_weight.",
        "1..5; повышать при пропусках объектов и низком recall, снижать при жирных масках и false positive.",
    ),
    "train.background_weight": (
        "Базовый вес всех пикселей фона, включая nodata и padding. В binary применяется к BCE/focal и false-positive части Dice/Tversky; в multiclass — к background-классу 0. Для размеченного hard negative дополнительно действует hard_negative_weight.",
        "0.1..2; 1 сохраняет обычный вес. Снижать, если многочисленный фон подавляет обучение объектов, повышать при избытке false positive на обычном фоне.",
    ),
    "train.hard_negative_weight": (
        "Дополнительный множитель штрафа только для пикселей внутри размеченных hard-negative зон. Он умножается на background_weight; positive pixels регулируются отдельно через positive weight и Dice/Tversky-компоненты.",
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
    "train.diagnostic_checkpoint_interval_sec": (
        "Сохраняет текущие, не лучшие веса на границе завершённой эпохи; фактическое время может быть позже заданного рубежа. Обучение и optimizer не перезапускаются.",
        "Пусто — без дополнительных файлов; 900 секунд — диагностические срезы примерно каждые 15 минут.",
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
    "postprocess.smooth.enabled": (
        "Включает мягкое угловое сглаживание полигонов после фильтров формы и площади.",
        "Обычно выключено; включать для классов с заметно ступенчатыми контурами, например рек.",
    ),
    "postprocess.smooth.iterations": (
        "Число проходов Smooth. Каждый проход увеличивает число вершин до последующего Simplify.",
        "1 для мягкого сглаживания; увеличивать только после проверки геометрии и размера GeoJSON.",
    ),
    "postprocess.smooth.offset": (
        "Доля соседнего ребра, используемая для срезания каждого угла.",
        "0.125 для мягкого сглаживания; допустимый диапазон больше 0 и не более 0.5.",
    ),
    "postprocess.simplify_m": (
        "Упрощение контура в метрах. Снижает число вершин и делает GeoJSON легче, но может съесть тонкие детали.",
        "Пусто для авто-профиля; 1..20 м. Повышать для тяжелых/шумных контуров, снижать для точной границы.",
    ),
    "postprocess.filter_compact_objects.enabled": (
        "Включает фильтрацию объектов по форме в выбранном прямом или обратном режиме.",
        "Включать для рек в режиме remove_compact и для озёр в режиме keep_compact.",
    ),
    "postprocess.filter_compact_objects.mode": (
        "Определяет направление фильтра: remove_compact удаляет компактные объекты, keep_compact удаляет некомпактные.",
        "remove_compact для вытянутых классов вроде рек; keep_compact для компактных классов вроде озёр.",
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
INFERENCE_CONFIG_FIELDS = {str(field["key"]): field for field in INFERENCE_CONFIG_SCHEMA["fields"]}


def sanitize_template_config(
    config: dict[str, Any] | None,
    *,
    fallback: dict[str, Any] | None = None,
    normalize_factors: bool = True,
    architecture: str = "segformer_b0",
) -> dict[str, Any]:
    if (config or {}).get("train.pipeline_variant") == "next_gen":
        raise TrainingUIAPIError("Конвейер next-gen снят с запуска. Выберите legacy или next-gen2.")
    result = {
        key: value for key, value in (fallback or BASE_DEFAULT_CONFIG).items() if key in CONFIG_KEYS
    }
    if (config or {}).get("train.pipeline_variant") == "legacy" and result.get("train.pipeline_variant") in {"next_gen2", "next_gen3", "next_gen4", "next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "object_f1", "object_f1_v2"}:
        result.update(BASE_DEFAULT_CONFIG)
    if (
        (config or {}).get("train.pipeline_variant") in {"next_gen2", "next_gen3", "next_gen4", "next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "object_f1", "object_f1_v2"}
        and result.get("train.pipeline_variant") != (config or {}).get("train.pipeline_variant")
    ):
        result.update(fixed_pipeline_defaults(str((config or {}).get("train.pipeline_variant"))))
    for key, value in (config or {}).items():
        if key in CONFIG_KEYS:
            options = CONFIG_FIELDS[key].get("options")
            if options is not None and value not in options:
                continue
            result[key] = value
    if result.get("train.pipeline_variant") in {"next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4"} and architecture not in OBJECT_F1_MODEL_NAMES:
        raise TrainingUIAPIError("Профиль попиксельной согласованности поддерживает только SMP SegFormer B0/B1/B2/B3")
    if result.get("train.pipeline_variant") in {"object_f1", "object_f1_v2"} and architecture not in OBJECT_F1_MODEL_NAMES:
        label = "object f1 v2" if result.get("train.pipeline_variant") == "object_f1_v2" else "object f1"
        raise TrainingUIAPIError(f"{label} доступен для SegFormer B0/B1/B2/B3")
    if result.get("train.pipeline_variant") in {"next_gen2", "next_gen3", "next_gen4", "next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "object_f1", "object_f1_v2"}:
        result.update({key: value for key, value in fixed_pipeline_defaults(str(result["train.pipeline_variant"])).items() if key not in NEXT_GEN2_EDITABLE_KEYS})
        tile_size = result.get("tile_preparation.tile_size")
        if not isinstance(tile_size, (int, float)) or tile_size not in NEXT_GEN2_TRAIN_BATCH_SIZES:
            raise TrainingUIAPIError("next-gen2: размер тайла должен быть 512, 768, 1024 или 1536")
        result["tile_preparation.stride"] = tile_size if result["train.pipeline_variant"] in {"next_gen4", "next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4"} else tile_size // 2
        result["train.batch_size"] = next_gen2_train_batch_size(tile_size, architecture)
    elif "tile_preparation.context" not in (config or {}):
        tile_size = int(result.get("tile_preparation.tile_size") or 0)
        result["tile_preparation.context"] = 128 if tile_size == 768 else 0
    if result.get("train.pipeline_variant") == "legacy" and result.get("train.loss") == "cross_entropy_tversky":
        result["train.loss"] = BASE_DEFAULT_CONFIG["train.loss"]
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
    result["tile_preparation.background_factor"] = 1.0 - positive_factor - hard_negative_factor


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
            "smp_segformer_b0",
            "segformer b0",
            source="analogy",
        ),
        _template(
            "smp_segformer_b1",
            "segformer b1",
            source="analogy",
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
    schema = deepcopy(CONFIG_SCHEMA)
    legacy_defaults = deepcopy(BASE_DEFAULT_CONFIG)
    if default_config.get("train.pipeline_variant") == "legacy":
        legacy_defaults.update(default_config)
    schema["fields"][0]["options"] = ["legacy", "next_gen2", "next_gen3", "next_gen4", "next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1"]
    if architecture in OBJECT_F1_MODEL_NAMES:
        schema["fields"][0]["options"].extend(["next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "object_f1", "object_f1_v2"])
    schema["pipeline_defaults"] = {"legacy": legacy_defaults}
    if architecture in OBJECT_F1_MODEL_NAMES:
        schema["pipeline_defaults"]["next_gen4_consistency_v2"] = {
            **NEXT_GEN4_CONSISTENCY_V2_DEFAULT_CONFIG, "train.batch_size": next_gen2_train_batch_size(512, architecture),
        }
        schema["pipeline_defaults"]["next_gen4_consistency_v3"] = {
            **NEXT_GEN4_CONSISTENCY_V3_DEFAULT_CONFIG, "train.batch_size": next_gen2_train_batch_size(512, architecture),
        }
        schema["pipeline_defaults"]["next_gen4_consistency_v4"] = {
            **NEXT_GEN4_CONSISTENCY_V4_DEFAULT_CONFIG, "train.batch_size": next_gen2_train_batch_size(512, architecture),
        }
        schema["pipeline_defaults"]["object_f1"] = {
            **OBJECT_F1_DEFAULT_CONFIG, "train.batch_size": next_gen2_train_batch_size(512, architecture),
        }
        schema["pipeline_defaults"]["object_f1_v2"] = {
            **OBJECT_F1_V2_DEFAULT_CONFIG, "train.batch_size": next_gen2_train_batch_size(512, architecture),
        }
    if architecture in NEXT_GEN2_MODEL_BATCH_SIZES:
        schema["pipeline_defaults"]["next_gen2"] = {
            **NEXT_GEN2_DEFAULT_CONFIG, "train.batch_size": next_gen2_train_batch_size(512, architecture),
        }
        schema["pipeline_defaults"]["next_gen3"] = {
            **NEXT_GEN3_DEFAULT_CONFIG, "train.batch_size": next_gen2_train_batch_size(512, architecture),
        }
        schema["pipeline_defaults"]["next_gen4"] = {
            **NEXT_GEN4_DEFAULT_CONFIG, "train.batch_size": next_gen2_train_batch_size(512, architecture),
        }
        schema["pipeline_defaults"]["next_gen4_global_norm"] = {
            **NEXT_GEN4_GLOBAL_NORM_DEFAULT_CONFIG, "train.batch_size": next_gen2_train_batch_size(512, architecture),
        }
        schema["pipeline_defaults"]["next_gen4_augmentation_v3"] = {
            **NEXT_GEN4_AUGMENTATION_V3_DEFAULT_CONFIG, "train.batch_size": next_gen2_train_batch_size(512, architecture),
        }
        schema["pipeline_defaults"]["next_gen4_augmentation_v4"] = {
            **NEXT_GEN4_AUGMENTATION_V4_DEFAULT_CONFIG, "train.batch_size": next_gen2_train_batch_size(512, architecture),
        }
        schema["pipeline_defaults"]["next_gen4_consistency_v1"] = {
            **NEXT_GEN4_CONSISTENCY_V1_DEFAULT_CONFIG, "train.batch_size": next_gen2_train_batch_size(512, architecture),
        }
    else:
        schema["fields"][0]["options"] = ["legacy"]
        schema["pipeline_descriptions"] = {}
    return {
        "architecture": architecture,
        "display_name": display_name,
        "config_schema": schema,
        "default_config": default_config,
        "baseline_default_config": deepcopy(default_config),
        "source": source,
        "baseline_source": source,
        "source_mlflow_run_id": source_mlflow_run_id,
        "baseline_source_mlflow_run_id": source_mlflow_run_id,
        "is_active": True,
        "version": 1,
    }
