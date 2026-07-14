# Модуль metrics

## Назначение

`metrics` считает пиксельную и объектовую F1 без принятия решений о checkpoint или ранней остановке.

## Публичный интерфейс

- `compute_pixel_f1(request: PixelF1Request) -> PixelF1Result` - считает бинарную pixel F1.
- `compute_object_f1(request: ObjectF1Request) -> ObjectF1Result` - считает объектовую F1 одного тайла.
- `summarize_epoch_metrics(history: list[EpochMetrics]) -> MetricsSummary` - сводит историю pixel F1.

## Публичные контракты

- `MetricsError` - ошибка входных масок.
- `PixelF1Request`, `PixelF1Result` - порог, precision, recall, F1 и pixel TP/FP/FN.
- `ObjectF1Request`, `ObjectF1Result` - instance mask эталона, бинарная маска прогноза, IoU threshold, precision, recall, F1 и object TP/FP/FN.
- `EpochMetrics`, `MetricsSummary` - совместимые DTO сводки pixel-истории.

## Список используемых данным модулем модулей и с какой целью

- `numpy` и `scipy` - connected components, IoU-граф и максимальное двудольное сопоставление.

## Алгоритм работы и его особенности

`compute_object_f1` выделяет в прогнозе компоненты связности по восьми соседям. Для каждой пары эталонного
instance id и предсказанной компоненты строится ребро при `IoU ≥ iou_threshold`, после чего максимальное
двудольное сопоставление обеспечивает соответствие один к одному. Несопоставленные прогнозы являются FP,
несопоставленные эталоны — FN. Функция работает на одном тайле; агрегацию тайлов выполняет вызывающий модуль,
поэтому фрагменты одного исходного объекта на разных тайлах считаются разными объектами.

