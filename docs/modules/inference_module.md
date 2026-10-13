# Модуль inference

## Назначение

Общий CPU-код сборки окон и выделения объектов для обучения, PyTorch и Geoalert. Прежний `run_inference` остаётся незавершённым CLI-путём и не используется новым профилем.

## Публичный интерфейс

- `run_inference(request: InferenceRequest) -> InferenceResult` — существующий CLI-контракт.
- `create_object_scene(request: ObjectSceneRequest) -> ObjectSceneAccumulator` — создать дисковый накопитель сцены.
- `separate_objects(foreground, boundary, valid_pixels, config: ObjectSeparationConfig | None = None)` — вернуть двумерную карту `int32` ID из вероятностей и valid mask.
- `object_window_origins(length: int, tile_size: int) -> list[int]` — координаты полных окон с половинным шагом и покрытием края; маленький снимок дополняется до окна.
- `object_separation_config(model_spec: ModelSpec) -> ObjectSeparationConfig` — прочитать сохранённые параметры и проверить соответствие профилю checkpoint; для v2 отсутствие параметров является ошибкой.

## Публичные контракты

- `InferenceError` — ошибка обработки.
- `InferenceConfig` — checkpoint_uri, threshold, batch_size, device.
- `InferenceArtifact` — uri, kind, metadata.
- `InferenceRequest` — config, images_dir, output_uri, optional model_spec.
- `InferenceResult` — status, artifacts, report.
- `ObjectSeparationConfig` — profile=`object_f1|object_f1_v2`, foreground_threshold=0.5, boundary_threshold=0.5, min_marker_pixels=4, min_object_pixels=1, boundary_closing_radius=0, fallback_peak_prominence=2. Для v2 значения по умолчанию: min_marker_pixels=16, min_object_pixels=16, boundary_closing_radius=1; явно сохранённые значения имеют приоритет.
- `ObjectSceneRequest` — width, height, tile_size, optional work_dir, separation.
- `ObjectWindowPrediction` — x, y, probabilities[2,H,W], valid_pixels[H,W].
- `ObjectSceneResult` — probabilities, valid_pixels, instances, object_count. Дисковые массивы действуют до close; resegment перезаписывает instances предыдущего результата, поэтому показатели надо рассчитать до следующего вызова.
- `ObjectSceneAccumulator` — протокол add_window(window), finish() → ObjectSceneResult, resegment(config: ObjectSeparationConfig) → ObjectSceneResult, close(). Повторное разделение доступно только после finish и до close; использует те же дисковые карты без повторной нормализации вероятностей. Закрытие обязательно, в том числе при ошибке.

## Список используемых данным модулем модулей и с какой целью

- `models.api/contracts` — прежний CLI-путь и его DTO, а также `ModelSpec` для чтения сохранённого профиля разделения; модели загружаются лениво.
- NumPy, SciPy, scikit-image — Gaussian, связные области и watershed; PyTorch для CPU API не требуется.

## Алгоритм работы и его особенности

Окна накапливаются на диске с Gaussian sigma=tile_size/4 и valid mask. Вероятности нормализуются полосами; компоненты имеют 4-связность. Исторический профиль сохраняет алгоритм. V2 сначала исключает мелкие компоненты, замыкает короткие разрывы границ и фильтрует внутренние маркеры. Без маркеров берутся значимые максимумы расстояния до края; тонкий объект получает один центр. Watershed восстанавливает полную область без фоновых щелей. Малые итоговые центры удаляются с перераспределением их пикселей, соседние ID не объединяются. Дыры и nodata остаются фоном. Временные карты обрабатываются по bbox; накопитель закрывается через close. Validation/test, PyTorch и Geoalert используют один CPU API и сохранённый профиль checkpoint. Порог границы допускает калибровку на validation; inference не меняет его по шаблону класса.
