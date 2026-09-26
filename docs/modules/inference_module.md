# Модуль inference

## Назначение

Общий CPU-код сборки окон и выделения объектов для обучения, PyTorch и Geoalert. Прежний `run_inference` остаётся незавершённым CLI-путём и не используется новым профилем.

## Публичный интерфейс

- `run_inference(request: InferenceRequest) -> InferenceResult` — существующий CLI-контракт.
- `create_object_scene(request: ObjectSceneRequest) -> ObjectSceneAccumulator` — создать дисковый накопитель сцены.
- `separate_objects(foreground, boundary, valid_pixels, config: ObjectSeparationConfig | None = None)` — вернуть двумерную карту `int32` ID из вероятностей и valid mask.
- `object_window_origins(length: int, tile_size: int) -> list[int]` — координаты полных окон с половинным шагом и покрытием края; маленький снимок дополняется до окна.

## Публичные контракты

- `InferenceError` — ошибка обработки.
- `InferenceConfig` — checkpoint_uri, threshold, batch_size, device.
- `InferenceArtifact` — uri, kind, metadata.
- `InferenceRequest` — config, images_dir, output_uri, optional model_spec.
- `InferenceResult` — status, artifacts, report.
- `ObjectSeparationConfig` — foreground_threshold=0.5, boundary_threshold=0.5, min_marker_pixels=4.
- `ObjectSceneRequest` — width, height, tile_size, optional work_dir, separation.
- `ObjectWindowPrediction` — x, y, probabilities[2,H,W], valid_pixels[H,W].
- `ObjectSceneResult` — probabilities, valid_pixels, instances, object_count. Дисковые массивы действуют до close.
- `ObjectSceneAccumulator` — протокол add_window(window), finish() → ObjectSceneResult, close(); закрытие обязательно, в том числе при ошибке.

## Список используемых данным модулем модулей и с какой целью

- `models.api/contracts` — только прежний CLI-путь и его DTO; модели загружаются лениво.
- NumPy, SciPy, scikit-image — Gaussian, связные области и watershed; PyTorch для CPU API не требуется.

## Алгоритм работы и его особенности

Окна накапливаются в дисковых массивах с Gaussian sigma=tile_size/4 и valid mask. Вероятности нормализуются полосами. Область порогуется, компоненты маркируются с 4-связностью. В bbox каждой компоненты маркеры образуются удалением границ, мелкие маркеры отсеиваются; при отсутствии маркера выбирается максимум расстояния до края. Watershed по вероятности границы возвращает ID без фоновых щелей. Соседние ID не объединяются. RAM зависит от активного bbox, а не всех карт сцены. Невалидные пиксели получают ID 0. CPU API используется одинаково в validation/test и инференсе. Временные массивы удаляются при close.
