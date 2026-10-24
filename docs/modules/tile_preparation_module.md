# Модуль tile_preparation

## Назначение

`tile_preparation` создаёт train/val DataLoader по списку независимых TIFF. Тайлы каждого снимка строятся отдельно; перекрывающиеся снимки не объединяются, файлы тайлов на диск не записываются.

## Публичный интерфейс

- `create_tile_dataloader(request: TileDataloaderRequest) -> object` — загрузить `settings.tile_preparation`, создать Dataset и PyTorch DataLoader.

Batch содержит `images: float32[B,C,H,W]`, binary `masks: float32[B,1,H,W]` либо multiclass `masks: long[B,H,W]` и `batch_meta` со счётчиками категорий/аугментаций. Значения mask: `-1` — hard negative, `0` — фон, `1` либо `1..N` — positive/class id.

## Публичные контракты

- `TilePreparationError` — ошибка подготовки loader.
- `HARD_NEGATIVE_LABEL=-1` — служебная метка hard-negative пикселя.
- `TileClassAnnotation` — `class_id`, `slug`, `name`, `annotation_file`, optional `hard_negative_annotation_file`, `priority`.
- `TileClassDefinition` — `class_id`, `slug`, `name`, `color`, `priority` для class-filtered чтения одного per-image GeoJSON.
- `TileSceneSource` — `scene_id`, `image_path`, optional per-image `annotation_file`, `footprint_file`, `parent_scene_id`, `zone_id`, `region_geometry` в CRS TIFF, `region_window=(x,y,width,height)` в пикселях TIFF.
- `TileSplitRequest` — `val_fraction`, `test_fraction` (default 0), `seed`, `strategy=window_random|scene_fold|scene_groups`, `validation_fold`, `spatial_purge`.
- `TileDataloaderRequest` — непустой `scenes`, optional общие binary-файлы, legacy `class_annotations` либо per-image `classes`, `batch_size`, `mode=train|val|test`, optional `tile_split`, `max_batches_per_epoch`, `include_object_instances`, `pipeline_variant=legacy|next_gen|next_gen2|next_gen3|next_gen4|next_gen4_global_norm|next_gen4_augmentation_v3|next_gen4_augmentation_v4|object_f1|object_f1_v2`, optional `input_channels` (число модельных каналов во всех профилях; 3/4 для объектных профилей), optional сбор histogram. Для обоих профилей глобальной нормализации сбор разрешён только при `mode=train`.
- `SamplingFeedback` — только агрегаты полной validation: `empty_tiles`, `false_positive_tiles`, `empty_pixels`, `false_positive_pixels`, `positive_pixels`, `true_positive_pixels`; фиксированный порог 0,5.
- `AdaptiveSamplingController` — протокол train-сэмплера: `set_epoch(epoch)`, `observe_train(indices,scores)`, `update_feedback(feedback)`, `snapshot(include_state=False)`. Состояние и история включаются явно; validation/test не передают индексы. Batch next-gen3/4 содержит `sample_indices` и обязательный `valid_pixels`. Политика `adaptive_background_v1`, квоты, ротация и защита полноты описаны в `docs/architecture.md`; прежние сэмплеры не меняются.

`next_gen4` сохраняет адаптивный сэмплер, но строит общую для зон TIFF сетку от (0,0), stride=tile_size и padding краёв. Все представления исходного тайла назначаются одной части 60/20/20. Manifest фиксирует политику сетки и нулевое пересечение тайлов одного TIFF между частями; пересечения разных TIFF разрешены. Размер остаётся выбираемым.

`next_gen4_global_norm` сохраняет эту сетку и дополнительно собирает 256-bin histogram модельных каналов только выбранных валидных train-окон до аугментаций. Счётчики суммируются в существующем проходе точной статистики, включая несколько процессов; held-out части того же TIFF не читаются для нормализации. Источник — `retained_train_windows`, числа окон и валидных пикселей сохраняются; вход под невалидной маской получает ноль независимо от исходного значения nodata. Штатная маска TIFF сохраняется; alpha и nodata всех входных каналов проверяются дополнительно, исходный ноль и padding исключаются также внутри размеченных зон. Границы рассчитывает `train_pipeline`, а применяет `models`.

`next_gen4_augmentation_v3` сохраняет эту сетку и статистику. Геометрия синхронно переносит все маски; если она обнулила валидный пиксель, целиком отменяется с масками. Фотометрия проверяется относительно принятой valid-mask по каждому пикселю: полностью нулевой или нечисловой вход отменяет только фотометрический блок. Невалидная область восстанавливается нулём; исходное противоречие вызывает ошибку. Искусственных значений и повторного подбора нет; seed эпохи/draw и spawn сохраняются. Подробные правила — в `docs/architecture.md`.

`TileDataloaderRequest.pipeline_variant` также принимает `next_gen4_augmentation_v4`: до прежней геометрии добавлена квадратная вырезка 25–100% площади с вероятностью 0,5 только внутри train-тайла, с возвратом к выбранному размеру и общей геометрией масок. Защита v3 сохраняется. Train manifest содержит `dense_windows` с числом исходных окон ≥90%/99% объекта при valid ≥99%. Только v4 возвращаемый train-loader имеет `dense_window_loader` (либо `None`, если подходящих окон нет): последовательный неповторяющийся аудит плотных train-окон на копии Dataset без аугментаций, с теми же train-индексами и общей функцией сборки batch. Рабочий Dataset не изменяется. Validation/test и train-histogram не аугментируются.

`TileDataloaderRequest.pipeline_variant=next_gen4_consistency_v1` полностью наследует v4:
тот же split, статистика, protected crop и dense-loader. Согласованную пару формирует train
только из готового batch; tile_preparation не читает для неё соседние val/test-пиксели.

`next_gen4_consistency_v2` наследует ту же подготовку без изменения сетки, данных,
масок, статистики, аугментаций или сэмплера; нормализатор декодера здесь не создаётся.

## Список используемых данным модулем модулей и с какой целью

- `inference.api` — единая сетка окон полного покрытия для object_f1.
- `settings.api` — параметры сетки, workers, sampling, augmentation и val-cache.
- `rasterio` — ленивое чтение окон отдельных TIFF с `boundless=True`.
- `shapely`, `rasterio.features` — пространственный индекс и rasterize разметки.
- `torch.utils.data` — sampler, DataLoader и фиксированный val subset.

## Алгоритм работы и его особенности

Зональные сцены используют локальную сетку с полным покрытием и дополнением nodata; чтение и растеризация выполняются в координатах исходного TIFF. Вход, контекст и supervision ограничены зоной и raster mask; полностью пустые окна исключены. Для всех профилей batch с зонами содержит valid_pixels, синхронные аугментации сохраняют nodata. В смешанных batch незональные legacy/next-gen2 получают единичную маску совместимости. Статистика и sampling учитывают только допустимую часть. LRU разделяется по пути TIFF; metadata и split manifest сохраняют зону и исходную сцену, поснимочный split не разносит зоны одного TIFF.

Во всех профилях явный `input_channels=3` допускает совместное чтение RGB и RGB+alpha: модель получает только первые три канала, четвёртый при явном ColorInterp.alpha ограничивает valid pixels. Прозрачные и полностью пустые окна исключаются; аугментации синхронно переносят маску, а loss, метрики и notebook sampling не считают прозрачность фоном. При наличии RGBA в наборе валидная маска передаётся также для RGB-сцен, чтобы смешанные batch сохраняли единый контракт. Четырёхканальный вход требует RGB+NIR и отклоняет alpha. Прежнее незональное поведение RGB+NIR сохраняется.

`object_f1` добавляет `input_channels`, постоянные instance ID во всех loaders, boundary_target, boundary_valid, valid_pixels и координаты окон в batch_meta. `scene_groups` делит независимые группы 60/20/20; общая сетка полного покрытия получена через `inference.api`. RGBA читается как RGB + valid mask. Части MultiPolygon разделяются; конфликтные пиксели исключаются из границ. При аугментации все маски преобразуются совместно; sampler и workers наследуются из next-gen2.

`object_f1_v2` сохраняет этот split, полные ID, площадь, RGB/alpha и синхронные аугментации. После геометрии строится внутренняя полоса до 8 пикселей; для узкого связного компонента ширина ограничивается половиной его глубины, сохраняя сердцевину. Края окна и переходы через nodata не создают границ; из контроля исключаются только невалидные пиксели и соседняя полоса в один пиксель. Пересечения разрешает прежний стабильный порядок растеризации, но их швы обучаются. Диагностика сохраняет число дублей и существенных нахлёстов (от 10% меньшего полигона) с предупреждениями. Только train рассчитывает количества и веса трёх классов: фон, сердцевина, граница; обратные частоты присутствующих классов нормированы к среднему 1, вес границы умножен на 3. Количества, веса и ширина сохранены в split manifest. Старый `object_f1` не меняется.

Подготовка точной статистики `object_f1_v2` использует число процессов из `tile_preparation.num_workers`; при 0/1 работает последовательно. Процессы запускаются через `spawn`, самостоятельно открывают TIFF и индексы разметки; очередь ограничена двумя заданиями на процесс, блок содержит не более 16 окон. TIFF явно закрываются финализатором процесса до разрушения интерпретатора. Между процессами передаются только счётчики и доли, порядок окон и веса сэмплера сохраняются побитово. Дополнительная растеризация наложений для этой статистики не выполняется, поскольку её результат не используется. Сетка, категории, разбиение и статистика сопровождаются журналом этапов; статистика сообщает число обработанных окон и время каждые 15 секунд. Эпохи и прежние профили не меняются.

`next_gen2` строит полные окна 512/768/1024/1536 с шагом в половину окна без контекста; исторические незональные наборы без RGB+alpha сохраняют отсутствие фильтрации nodata. Случайное разбиение
60/20/20 с seed 42 воспроизводит два train_test_split ноутбука; пересечения не исключаются.
Manifest содержит все координаты и индексы train/val/test. По сырым train-маскам вычисляются
обратные частоты всех классов и фона и sampler 7/8/1 для любого positive/hard negative/фона; порог positive 0.001.
Multiclass сохраняет ID 1..N, а class-specific отрицательные области исключены из частот; общий negative — фон.
Albumentations синхронно переносит семантическую маску и все class-specific negative каналы.
Hard negative перекрывает positive. Для аффинных преобразований class-specific negative маски
временно передаются как uint8 0/1, затем возвращаются в boolean; multiclass target сохраняет int64.
Ко всем train-тайлам применяются геометрия и фотометрия
Albumentations до нормализации в модели. Val/test не балансируются и не аугментируются.
Workers берутся из серверных settings (8); prefetch_factor=2, persistent_workers=False, при CUDA — pinned memory.
Во всех профилях дочерние загрузчики запускаются через `spawn`, чтобы не наследовать блокировки потоков MLflow, OpenSSL и GDAL. Ожидание batch ограничено 180 секундами; зависший загрузчик завершает задание с ошибкой и не удерживает очередь бессрочно. Однопроцессное чтение не имеет этого таймаута. Способ запуска и таймаут входят в диагностику next-gen2/object f1.
Каждый worker получает отдельный RNG Albumentations через worker_seed, OpenCV работает в один поток без OpenCL.
Воспроизводимость требует одинаковых seed и числа workers. Параметры загрузчика записываются в журнал.
Глобальные и per-image источники остаются штатными.

Для каждой сцены строится сетка полезных центров `0,stride,...`. Полное входное окно начинается в `-context` и имеет размер `tile_size`; его полезный центр имеет размер `tile_size - 2 × context`. Поэтому первые и последние пиксели TIFF попадают в полезную область, а внешняя рамка читается через `boundless=True` как nodata. `context=0` полностью сохраняет прежнюю сетку. Sampling-категория определяется только по геометрии полезного центра. Настройки обязаны удовлетворять `tile_size > 2 × context`.

Coarse/sparse valid-footprint фильтр заранее удаляет полностью black/nodata окна без полного чтения всех тайлов, а частично невалидные окна остаются. Дескрипторы TIFF открываются лениво и удерживаются ограниченным LRU на worker. Разметка сцены индексируется отдельно; positive перекрывает hard negative. В multiclass слои растеризуются в порядке приоритета и дают `int64` mask `-1/0/1..N`. Для управляемого датасета class-specific hard negative дополнительно растеризуется в синхронный boolean-тензор `[N,H,W]`: канал задаёт только тот тип, для которого область является отрицательной, а hard negative без класса остаётся общим `-1`. Nodata по значению, нулевая `dataset_mask` и boundless-padding объединяются в одну маску: изображение на ней получает исходное значение nodata, а supervision и class-specific masks принудительно получают фон `0`, подрезая разметку по фактическому снимку. Геометрические аугментации преобразуют все маски синхронно с изображением и target, после фотометрических изменений nodata восстанавливается. Split вычисляется хешем `seed+scene_id+x+y`, поэтому не зависит от порядка сцен. Train использует category-aware sampling и аугментации positive/hard-negative; при `class_balance=true` positive-веса выравниваются по типам, а дефицит отражается предупреждением. Val выбирает фиксированный balanced subset и кэширует его в RAM при безопасном лимите, иначе лениво читает те же индексы. Диагностика содержит разрешение и числа candidate/valid/positive/hard-negative/background окон по каждому TIFF. Пересекающиеся TIFF дают независимые тайлы.

В `next_gen` scene-fold упорядочивает сцены по SHA-256 от seed и scene id. Validation получает целые held-out
сцены и все их окна в естественном отношении. Spatial purge удаляет train-окна, полный вход которых пересекает
расширенный на один пиксель footprint held-out сцены; split manifest фиксирует fold, сцены и удалённые окна.
Batch дополнительно содержит `valid_pixels`, `scene_ids`, координаты окон и размеры сцен. Epoch/draw-aware
sampler передаёт номер эпохи и draw в seed аугментации; все категории получают синхронную геометрию, а level 3 —
gamma и независимый gain каналов. Для robust preprocessing 256-bin histogram читается потоково один раз только
по retained train-сценам. Эта ветка не меняет алгоритмы `legacy`.

`prefetch_epochs` задаёт целевой объём готовых batch, а `prefetch_factor` рассчитывается обратно
пропорционально `num_workers`; уменьшение числа процессов не сокращает заданный префетч. В серверном профиле
используется восемь workers и один epoch префетча. Worker-процессы получают признак
`MLSYSTEM2_TILE_WORKER=1`; они ждут возобновления до открытия TIFF и подготовки следующего тайла
только после появления `paused` с тем же token, что у `pause.request` в `MLSYSTEM2_TRAINING_CONTROL_DIR`.
До подтверждения паузы чтение продолжается, чтобы train loop мог получить batch или сбросить итератор
DataLoader и освободить GPU. Уже готовая очередь не очищается.
