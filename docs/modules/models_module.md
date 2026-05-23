# Модуль models

## Назначение

`models` создает SegFormer B2, загружает чекпойнты и сохраняет чекпойнты.

## Публичный интерфейс

- `list_supported_models() -> list[ModelSpec]` — возвращает только `segformer_b2`.
- `create_model(spec: ModelSpec) -> ModelHandle` — создает модель по спецификации.
- `load_checkpoint(request: LoadCheckpointRequest) -> LoadedCheckpoint` — загружает локальный `.pt` checkpoint.
- `save_checkpoint(request: SaveCheckpointRequest) -> CheckpointArtifact` — сохраняет локальный `.pt` checkpoint.

## Публичные контракты

- `ModelsError` — ошибка модели или checkpoint.
- `ModelSpec` — поля `name`, `input_channels`, `output_channels`, `pretrained`, `parameters`.
- `ModelHandle` — поля `spec`, `model`.
- `LoadCheckpointRequest` — поля `checkpoint_uri`, `model_spec`, `map_location`.
- `SaveCheckpointRequest` — поля `model`, `checkpoint_uri`, `metadata`.
- `CheckpointArtifact` — поля `uri`, `format`, `metadata`.
- `LoadedCheckpoint` — поля `model`, `artifact`.

## Список используемых данным модулем модулей и с какой целью

Модуль не использует публичные API других модулей.

## Алгоритм работы и его особенности

Поддерживается только `segformer_b2`. `torch` и `transformers` импортируются лениво внутри реализации. `create_model` строит Hugging Face `SegformerForSemanticSegmentation` с B2 config: `depths=[3,4,6,3]`, `hidden_sizes=[64,128,320,512]`, `decoder_hidden_size=768`, `num_channels=input_channels`, `num_labels=output_channels`; при `pretrained=true` пробует `nvidia/segformer-b2-finetuned-ade-512-512` с `ignore_mismatched_sizes=True`. Checkpoint `.pt` хранит `model_state_dict`, `model_spec`, `metadata`.
