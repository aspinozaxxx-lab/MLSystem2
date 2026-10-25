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

`ModelSpec.parameters.pipeline_variant=next_gen3` сохраняет ABI next-gen2: два внутренних binary logits либо N+1 multiclass, min-max и автономное восстановление без загрузки pretrained. `parameters.sampler` хранит версию автоматической политики; состояние контроллера хранится в metadata, не является командой инференсу или повторной загрузки весов.

`next_gen4` сохраняет тот же ABI и требует `preprocessing.mask_output_nodata=true`: нулевые во всех входных каналах пиксели дают фон внутри обёртки PyTorch/ONNX. Флаг сохраняется в спецификации checkpoint; восстановление автономное. Исторические спецификации других вариантов не меняются.

`next_gen4_global_norm` сохраняет этот ABI с `preprocessing.mode=global_minmax`: `low/high` по 3/4 каналам, `source=retained_train_windows`, `valid_pixel_count`, `window_count`, `epsilon=1e-6`, `nodata=0`, `mask_output_nodata=true`. Общая обёртка применяет сохранённые train-границы и в PyTorch, и в ONNX. Маска определяется до нормализации: нормализованный ноль валидного пикселя не подавляет прогноз. Конфигурация и границы проверяются при создании и автономной загрузке; старый оконный профиль не меняется.

`next_gen4_augmentation_v3` наследует этот ABI и нормализацию, требует `parameters.augmentation_policy=validity_guard_v3` при создании и автономной загрузке. Маркер не меняет forward инференса, а фиксирует защищённую подготовку train-примеров; исторические спецификации не дополняются автоматически.

`next_gen4_augmentation_v4` сохраняет тот же ABI и требует `augmentation_policy=validity_guard_v4_context_crop_v1` и точный `context_crop` из `docs/architecture.md`: площадь 0,25–1,0, квадрат, вероятность 0,5, размер 512/768/1024/1536, источник внутри train-окна. Загрузка и продолжение не подменяют эту политику; forward и нормализация не меняются.

`next_gen4_consistency_v1` наследует ABI, `preprocessing`, `augmentation_policy` и
`context_crop` v4. Обязательный `parameters.context_consistency` строго фиксирует
политику `confident_teacher_kl_v1` из архитектуры. Создание и загрузка отвергают другую
политику; она сохраняется как происхождение обучения и не меняет forward инференса.

`next_gen4_consistency_v2` наследует эту политику, но только для SMP B0/B1/B2/B3 заменяет
единственный `decoder.fuse_stage.1` на попиксельную LayerNorm по каналам, без batch/пространственной
статистики. Обязательный `decoder_normalization={version:per_pixel_channel_layer_norm_v1,
epsilon:1e-5,affine:true}` проверяется до создания/загрузки. Исторические веса не преобразуются;
при неожиданной структуре декодера создание завершается ошибкой. ABI и raw-нормализация прежние.

`next_gen4_consistency_v3` сохраняет модель v2 и требует точную политику
`bidirectional_confident_teacher_kl_v1` с `direction_reduction=mean_of_active_directions`.
Остальные поля политики прежние. Несовместимая версия отвергается до создания/загрузки;
forward не содержит дополнительных веток. Старые checkpoint не переинтерпретируются.

`next_gen4_consistency_v4` сохраняет ту же модель и требует точную политику
`bidirectional_confident_teacher_kl_with_gt_fallback_v1` с прежними полями v3 и
`uncovered_supervision=ground_truth_cross_entropy_v1`. Проверки выполняются до
создания/загрузки; маркер LayerNorm обязателен. Изменение loss принадлежит train,
forward и инференс новых веток не содержат; старые checkpoint не преобразуются.

`next_gen4_consistency_v5` наследует модель и нормализатор v4, но требует политику
`full_query_detached_crop_kv_gt_v1` из архитектуры. Дополнительный аргумент forward
`context_reference` допустим только для v5 в eval: та же сеть без градиента получает
raw-вырезку и временно передаёт её отсоединённые K/V всех attention-блоков полному
query-проходу с графом. Batch/каналы/device совпадают; квадратная вырезка не больше
query, стороны кратны32. Hooks удаляются при любом исходе, параметры/буферы не меняются.
Без аргумента формула PyTorch/ONNX и ABI прежние; чужая политика/структура отвергается.

## Список используемых данным модулем модулей и с какой целью

Модуль не использует публичные API других модулей. `torch` подключается как optional dependency без падения при импорте модуля, `transformers` импортируется лениво при создании Hugging Face SegFormer, `segmentation_models_pytorch` импортируется лениво при создании SMP SegFormer и SMP DeepLabV3Plus.

## Алгоритм работы и его особенности

`object_f1_v2` использует три взаимоисключающих logits фон/внутренность/граница и обязательный `output_layout=background_interior_boundary`; несовместимая схема отклоняется. Логический выход один, полные параметры разделения и подготовки целей хранятся в ModelSpec. Внешняя нормализация и offline-восстановление общие с объектным профилем; старые веса сохраняют прежнее значение каналов.

`object_f1` поддерживает smp_segformer_b0/b1/b2/b3 с соответствующим encoder MiT: общий decoder и три сырых logits (фон, объект, граница). Внешний forward сохраняет все три канала; логическое output_channels=1 описывает единственный семантический класс. Нормализация и предобучение как next-gen2; параметры layout/разделения входят в checkpoint. Восстановление всех четырёх архитектур выполняется без сетевой загрузки предобученных весов.

HF использует transformers, SMP — Segformer с MiT B0/B1/B2/B3.
В legacy HF делит raw на 255; SMP без предобучения получает raw, с предобучением —
raw/255 и ImageNet mean/std (NIR как RED), без маски nodata. Режим задан ModelSpec.pretrained.
Next-gen2 принимает RGB/RGB+NIR: для binary два внутренних logits, снаружи разность foreground/background;
для multiclass все N+1 logits по схеме, включая фон. При восстановлении проверяется размер головы.
Нормализация каналов окна — min-max; постоянный канал даёт нули.
HF B0 сохраняет веса ноутбука; SMP с pretrained=true — encoder ImageNet и новую голову,
с false — случайные веса. При предобучении копируются RGB и RED→NIR, смещение новой свёртки случайное.
`return_two_class_logits=True` возвращает оба logits для CrossEntropy + Tversky.
Checkpoint хранит spec, веса и metadata; восстановление не загружает pretrained-веса.
Сохранённые next_gen совместимы. DeepLabV3Plus и UNet не меняются.
