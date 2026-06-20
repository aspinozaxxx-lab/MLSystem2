# Модуль tile_preparation

## Назначение

`tile_preparation` создает train loader и val loader по одному VRT XML и binary или multiclass GeoJSON-разметке. Модуль отвечает только за формирование loader для уже подготовленных VRT и не сохраняет tiles на диск. Val tiles кэшируются только в RAM текущего training process. По умолчанию split не выполняется; если в request передан `tile_split`, модуль делит список окон общего VRT на непересекающиеся train/val subsets.

Для `mode=train` чтение raster data, определение nodata pixels и rasterize mask выполняются лениво в `Dataset.__getitem__`, то есть в основном процессе или в PyTorch DataLoader workers. Для `mode=val` модуль перед обучением выбирает фиксированный balanced subset без replacement, один раз читает tiles, собирает CPU batch tensors и дальше возвращает одни и те же batch-и из RAM на каждой эпохе.

## Публичный интерфейс

- `create_tile_dataloader(request: TileDataloaderRequest) -> torch.utils.data.DataLoader` - загружает текущие настройки `settings.tile_preparation`, создает `Dataset`, опционально применяет `tile_split` и возвращает DataLoader.

Batch DataLoader:
- `images: torch.float32 [B, C, tile_size, tile_size]`;
- binary `masks: torch.float32 [B, 1, tile_size, tile_size]`, supervision values `-1=hard negative`, `0=background`, `1=positive`;
- multiclass `masks: torch.long [B, tile_size, tile_size]`, supervision values `-1=hard negative`, `0=background`, `1..N=class id`;
- `batch_meta: dict` с полями `positive_tile_count`, `hard_negative_tile_count`, `background_tile_count`, `augmented_tile_count`, `augmented_positive_tile_count`, `augmented_hard_negative_tile_count`, `tile_augmented`, `tile_positive`, `tile_hard_negative`, `tile_background`, `tile_category`.

## Публичные контракты

- `TilePreparationError` - ошибка подготовки tile DataLoader.
- `HARD_NEGATIVE_LABEL = -1` - служебное значение hard-negative пикселя в supervision mask; это не class id модели.
- `TileClassAnnotation` - поля `class_id`, `slug`, `name`, `annotation_file`, optional `hard_negative_annotation_file`, `priority`.
- `TileSplitRequest` - поля `val_fraction`, `seed`; задает deterministic split окон общего VRT.
- `TileDataloaderRequest` - поля `vrt_xml`, `annotation_file`, optional `hard_negative_annotation_file`, `class_annotations`, `batch_size`, `mode`, `tile_split`, `max_batches_per_epoch`. Валидация: либо задан binary `annotation_file` и `class_annotations=[]`, либо задан непустой `class_annotations` и `annotation_file=None`; top-level `hard_negative_annotation_file` используется только в binary режиме.

## Список используемых данным модулем модулей и с какой целью

- `settings.api` - получить `tile_size`, `stride`, `num_workers`, `prefetch_epochs`, `seed`, `augmentation_level`, `positive_factor`, `hard_negative_factor`, `background_factor`, `val_positive_factor`, `class_balance`.
- `rasterio` - открыть VRT и лениво читать image windows с `boundless=True`.
- `shapely` и `rasterio.features` - загрузить GeoJSON и rasterize mask в окно tile.
- `torch.utils.data` - создать train `DataLoader`, sampler и batch tensors для cached val loader.

Train DataLoader получает effective `prefetch_factor`, рассчитанный от размера train split, `batch_size`, числа workers, `prefetch_epochs` и optional `max_batches_per_epoch`. Если consumer ограничивает train-эпоху, расчет использует `min(full_batches_per_epoch, max_batches_per_epoch)`, чтобы prefetch в эпохах соответствовал фактически читаемым batch-ам. Например, для 163 batches/epoch, `num_workers=16` и `prefetch_epochs=2` effective factor будет `21`; если эпоха ограничена 72 batch-ами, factor будет `9`. Для val `prefetch_epochs` не применяется: val loader работает из RAM cache.

## Алгоритм работы и его особенности

При создании Dataset разрешено:
- открыть VRT;
- прочитать metadata: `width`, `height`, `count`, CRS, nodata;
- построить список окон по VRT/source rects;
- построить coarse valid-data footprint одним низкоразрешенным чтением VRT и отфильтровать black/nodata-only окна.

При создании Dataset запрещено:
- читать raster data по всем окнам;
- rasterize masks по всем окнам;
- заранее готовить все tiles или batches.

Окна строятся регулярной Geoalert-compatible сеткой: `0, stride, 2*stride, ...` до границы source rect или raster. Shifted last tile не добавляется. Окно всегда имеет размер `tile_size x tile_size`; выход за bounds закрывается `rasterio.read(boundless=True, fill_value=nodata)`.

После построения candidate windows модуль строит внутренний coarse valid-data footprint с фиксированным шагом `64` пикселя: сначала читает masks VRT в низком разрешении, затем низкоразрешенные raw values и считает valid cell только там, где mask valid и хотя бы один канал не равен нулю с eps `1e-6`. Candidate window должен пересекать valid cell. Затем выполняется точная sparse-проверка только по глобальным пикселям, которые соответствуют диагностической сетке tile (`0, 64, ..., center, last`). Для очень больших VRT, где полный footprint был бы дороже самого обучения, модуль пропускает чтение всей мозаики и выполняет sparse-проверку сразу по candidate windows, читая только узкие диапазоны строк. Это не читает каждый tile целиком и убирает black/nodata-only окна до DataLoader.

В `__getitem__` image читается через rasterio с `out_shape=(count, tile_size, tile_size)`, приводится только к `float32` и не нормализуется. Channel order сохраняет порядок каналов raster/VRT.

Dataset строит единую supervision mask одним helper: сначала background `0`, затем hard-negative geometries со служебным значением `-1`, затем positive layers. Positive всегда перезаписывает hard negative в пересечении, а nodata pixels в конце становятся background `0`. Binary mask возвращает shape `1,H,W`, dtype `float32`, значения `-1/0/1`; multiclass mask возвращает `int64 [H,W]`, значения `-1/0/1..N`. Классы применяются по `(priority, class_id)`: меньший priority записывается раньше, больший priority перекрывает его; при равном priority более поздний `class_id` перекрывает более ранний. `-1` является внутренним служебным значением и не является классом модели.

Аугментация сохраняет raw Geoalert tensor ABI подготовленных снимков: после photometric значения image зажимаются в диапазон `0..255`. Geometric flips/rotations применяются к image и единой supervision mask, поэтому значения `-1`, `0` и positive labels преобразуются вместе; photometric применяется только к image. Модуль не зануляет прямоугольные patch-и в image или mask. Категория tile определяется до аугментации и после нее не меняется. При `mode=train` и `augmentation_level > 0` аугментация применяется к positive и hard_negative tiles; обычный background не аугментируется. Sample meta содержит `augmented`, `category`, `positive`, `hard_negative`, `background`; collate собирает batch `(images, masks, batch_meta)` с category counters и per-tile flags, без отдельной pixel mask в meta.

Dataset всегда строит cheap-index для `train` и `val`: по bounds окна проверяет пересечение с GeoJSON geometry без чтения raster data и без rasterize. Для sample meta категория определяется по supervision mask: `positive`, если есть pixels `>0`; `hard_negative`, если positive нет, но есть pixels `-1`; иначе `background`. При пересечении positive и hard-negative разметки категория всегда `positive`, но hard-negative pixels вне positive объекта сохраняются. В multiclass режиме positive hint означает пересечение с геометрией любого класса, а hard-negative hint строится по объединению hard-negative геометрий всех классов. Если задан `tile_split`, positive и non-positive hints делятся на train/val по `val_fraction` и `seed`; пересечения между subsets запрещены, редкие positive windows не дублируются, а предупреждения доступны в tile report. Поля `estimated_positive_tiles`, `estimated_hard_negative_tiles` и `estimated_background_tiles` являются geometry-intersection hint, а не точным rasterized mask count. Для `train` DataLoader использует `WeightedRandomSampler` с replacement. `positive_factor + hard_negative_factor` образуют общий marked-бюджет, `background_factor` остается отдельным budget обычного background. Effective hard-negative budget не превышает заданный `hard_negative_factor` и дополнительно ограничен долей hard-negative tiles внутри marked-пула: `marked_factor * hard_negative_tile_count / (positive_tile_count + hard_negative_tile_count)`. Если hard-negative tiles нет или их мало относительно marked tiles, остаток marked-бюджета переносится в positive. Внутри effective категории вес одного tile равен бюджету категории, деленному на число tiles категории; при `class_balance=true` только effective positive-бюджет распределяется между positive-классами. Если `background_factor > 0`, а background tiles нет, или если marked-бюджет некуда перенести из-за отсутствия positive tiles, создание sampler завершается `TilePreparationError`; нулевой фактор отключает соответствующий budget. Фактические доли доступны как `positive_factor_used`, `hard_negative_factor_used`, `background_factor_used`, а перенос hard-negative budget добавляет warning. Для `val` train factors не применяются: выбирается фиксированный balanced subset по positive/non-positive hints, порядок детерминирован от `seed`, replacement не используется. Shuffle и augmentation для val выключены.

Полностью black/nodata-only tiles фильтруются заранее через coarse valid-data footprint или через sparse-проверку candidate windows для больших VRT и не попадают в DataLoader. Диагностика Dataset доступна как внутренние attributes: `candidate_window_count_before_valid_filter`, `black_filtered_window_count`, `valid_footprint_stride`, `valid_footprint_valid_cells`, `valid_footprint_total_cells`.
