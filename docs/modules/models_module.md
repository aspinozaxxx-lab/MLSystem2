# Модуль models

## Назначение

`models` создает поддерживаемые SegFormer-модели и загружает или сохраняет локальные checkpoint-файлы. Модуль не нормализует входные tensors и не знает о DataLoader.

## Публичный интерфейс

- `list_supported_models() -> list[ModelSpec]` - возвращает только `segformer_b0` и `segformer_b2`.
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

Модуль не использует публичные API других модулей. `torch` и `transformers` импортируются лениво внутри реализации.

## Алгоритм работы и его особенности

Поддерживаются только `segformer_b0` и `segformer_b2`. `create_model` строит Hugging Face `SegformerForSemanticSegmentation` с `num_channels=spec.input_channels` и `num_labels=spec.output_channels`.

Конфигурация `segformer_b0`: `depths=[2, 2, 2, 2]`, `hidden_sizes=[32, 64, 160, 256]`, `decoder_hidden_size=256`, pretrained источник `nvidia/segformer-b0-finetuned-ade-512-512`.

Конфигурация `segformer_b2`: `depths=[3, 4, 6, 3]`, `hidden_sizes=[64, 128, 320, 512]`, `decoder_hidden_size=768`, pretrained источник `nvidia/segformer-b2-finetuned-ade-512-512`.

Pretrained загрузка выполняется только при `spec.pretrained=true` с `ignore_mismatched_sizes=True`. Checkpoint `.pt` хранит `model_state_dict`, `model_spec`, `metadata`. `load_checkpoint` работает с локальными путями; S3 и MLflow URI здесь не добавляются.
