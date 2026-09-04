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

`ModelSpec.output_channels` задает число каналов logits. Для binary segmentation это `1`; для multiclass segmentation это `len(dataset.classes)+1`, где нулевой канал соответствует background.

В `legacy` поддерживаются две ветки SegFormer. `segformer_b0` и `segformer_b2` строятся через Hugging Face `SegformerForSemanticSegmentation` с `num_channels=spec.input_channels` и `num_labels=spec.output_channels`, затем оборачиваются приватным wrapper. Legacy-wrapper сохраняет внешний raw Geoalert ABI и внутри `forward` выполняет фиксированное scaling `x.float() / 255.0` перед SegFormer.

`smp_segformer_b0`, `smp_segformer_b1`, `smp_segformer_b2` и `smp_segformer_b3` строятся через `segmentation_models_pytorch.Segformer` с `encoder_name="mit_b0"`, `"mit_b1"`, `"mit_b2"` или `"mit_b3"`, `encoder_weights=None`, `in_channels=spec.input_channels`, `classes=spec.output_channels`, `activation=None`. UI использует все варианты B0/B1/B2/B3. Для legacy SMP wrapper `x / 255.0` не применяется; next-gen B0 получает общий preprocessing-wrapper согласно `ModelSpec`.

`smp_deeplabv3plus_resnet50` добавлен как один необходимый вариант DeepLabV3Plus для проверки старого MLSystem-compatible train path. Модель строится через `segmentation_models_pytorch.DeepLabV3Plus` с `encoder_name="resnet50"`, `encoder_weights=None`, `in_channels=spec.input_channels`, `classes=spec.output_channels`, `activation=None`. Input tensor остается `[B,C,H,W]` в raw Geoalert-compatible диапазоне, где `C` берётся из `ModelSpec`. Output - logits `[B,output_channels,H,W]`; activation внутри модели не применяется, а `train` сам выполняет sigmoid/cross entropy, loss и расчет метрик.

`smp_unet_resnet34`, `smp_unet_resnet50`, `smp_unet_resnet101` и `smp_unet_resnet152` добавлены как необходимые архитектуры для UI запуска обучения. Они строятся через `segmentation_models_pytorch.Unet` с соответствующим `encoder_name`, `encoder_weights=None`, `in_channels=spec.input_channels`, `classes=spec.output_channels`, `activation=None`. Эти варианты не добавляют отдельной preprocessing-логики и используют тот же raw Geoalert-compatible tensor ABI, что и другие SMP-модели.

Конфигурация `segformer_b0`: `depths=[2, 2, 2, 2]`, `hidden_sizes=[32, 64, 160, 256]`, `decoder_hidden_size=256`, pretrained источник `nvidia/segformer-b0-finetuned-ade-512-512`.

Конфигурация `segformer_b2`: `depths=[3, 4, 6, 3]`, `hidden_sizes=[64, 128, 320, 512]`, `decoder_hidden_size=768`, pretrained источник `nvidia/segformer-b2-finetuned-ade-512-512`.

В `legacy` Hugging Face pretrained сохраняет прежний путь с `ignore_mismatched_sizes=True`. Для `next_gen` HF B0 используется только закреплённый `nvidia/segformer-b0-finetuned-ade-512-512@489d5cd81a0b59fab9b7ea758d3548ebe99677da` без `ignore_mismatched_sizes`: первый convolution расширяется `3→4`, RGB копируется, NIR получает RED, head заменяется на один logit. Wrapper `next_gen` содержит выбранный preprocessing и обнуляет nodata уже после нормализации; поэтому одна формула входит в PyTorch checkpoint и ONNX. Полная HF-конфигурация и provenance хранятся в `ModelSpec.parameters`. `load_checkpoint` строит HF-архитектуру из сохранённой конфигурации и не обращается к Hub; `pretrained=true` после сохранения является только provenance. Checkpoint `.pt` хранит `model_state_dict`, `model_spec`, `metadata`.
