# Модуль models

## Назначение

`models` создает поддерживаемые segmentation-модели и загружает или сохраняет локальные checkpoint-файлы. Модуль принимает raw Geoalert-compatible tensors и не знает о DataLoader или task.

## Публичный интерфейс

- `list_supported_models() -> list[ModelSpec]` - возвращает `segformer_b0`, `segformer_b2`, `smp_segformer_b0`, `smp_segformer_b1`, `smp_segformer_b2`, `smp_segformer_b3`, `smp_deeplabv3plus_resnet50`, `smp_unet_resnet34`, `smp_unet_resnet50`, `smp_unet_resnet101`, `smp_unet_resnet152`.
- `create_model(spec: ModelSpec) -> ModelHandle` - создает модель по спецификации.
- `load_checkpoint(request: LoadCheckpointRequest) -> LoadedCheckpoint` - загружает локальный `.pt` checkpoint.
- `save_checkpoint(request: SaveCheckpointRequest) -> CheckpointArtifact` - сохраняет локальный `.pt` checkpoint.

## Публичные контракты

- `ModelsError` - ошибка модели или checkpoint.
- `ModelSpec` - поля `name`, `input_channels`, `output_channels`, `pretrained`, `parameters`.
- `ModelHandle` - поля `spec`, `model`.
- `LoadCheckpointRequest` - поля `checkpoint_uri`, `model_spec`, `map_location`.
- `SaveCheckpointRequest` - поля `model`, `checkpoint_uri`, `metadata`.
- `CheckpointArtifact` - поля `uri`, `format`, `metadata`.
- `LoadedCheckpoint` - поля `model`, `artifact`.

## Список используемых данным модулем модулей и с какой целью

Модуль не использует публичные API других модулей. `torch` подключается как optional dependency без падения при импорте модуля, `transformers` импортируется лениво при создании Hugging Face SegFormer, `segmentation_models_pytorch` импортируется лениво при создании SMP SegFormer и SMP DeepLabV3Plus.

## Алгоритм работы и его особенности

HF SegFormer строится через transformers, SMP — через Segformer с MiT B0/B1/B2/B3.
В legacy сохраняются raw-input ABI и прежние головы; HF делит вход на 255, SMP получает raw.
Next-gen2 поддерживает обе реализации: вход RGB или RGB+NIR, два внутренних logits, внешний выход —
разность foreground/background. Каналы каждого окна нормализуются min-max, постоянный канал даёт нули.
HF B0 сохраняет закреплённые веса ноутбука; SMP использует encoder ImageNet и новую голову.
Для четырёх каналов копируются RGB и RED→NIR, смещение новой свёртки случайное.
`return_two_class_logits=True` возвращает оба logits для CrossEntropy + Tversky.
Checkpoint хранит spec, веса и metadata; восстановление любой модели не загружает pretrained-веса.
Сохранённые next_gen совместимы. DeepLabV3Plus и UNet сохраняют прежнее поведение.
Подробные параметры и правила вариантов описаны в разделе конвейера обучения архитектуры.
