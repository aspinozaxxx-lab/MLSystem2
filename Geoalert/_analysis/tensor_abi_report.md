# Geoalert Tensor ABI Report

## Status
- Частично подтверждено: route, pipeline parser, tiling, input tensor перед Triton, запись raster mask и vectorization подтверждены исполняемым кодом.
- Python 3.12.10 установлен в user scope; `py_compile` для измененного `adapters.py` проходит.
- Заблокировано для полного runtime-проверки: Docker/WSL отсутствуют, часть Workflow Engine/inference конфигурации санитизирована.

## Route
Mapflow API -> Workflow Engine -> action `inference` -> RabbitMQ worker `inference` -> `DataProcessor` -> `urban.Compose` -> `SplitRaster` -> `Segmentation` -> `TritonAdapter` -> `VectorizeMasks`.

Кодовые ссылки: `ProcessingResource.scala:86-90`, `ProductionWorkflowEngine.scala:87-93`, `Action.java:54-57`, `InferenceTaskCreator.java:102-144`, `inference/data_processor.py:37-64`.

## Input tensor formation
- source code path: `Geoalert/Workflow Engine/inference-v1.5.5/modules/urban/urban/bricks/model_bricks/segmentation.py:154-158` читает `input_rasters` через `io.read_bc`, затем запускает `self._predictor.process`.
- input raster source: `SplitRaster` создает отдельные GeoTIFF-файлы каналов; `raster_ops.py:43-47` вызывает `split`, `functional/raster_ops/split.py:102-155` читает band из `input.tif` и пишет `RED.tif`, `GRN.tif`, `BLU.tif`, `NIR.tif` по выбранным именам.
- tiling code: `aeronet_raster/collectionprocessor.py:130-151` строит blocks, `collectionprocessor.py:52-57` читает sample.
- tile size: для Kanopus workflow выбран `sample_size: [512, 512]` в YAML; код использует `self.sample_size` в `Segmentation`: `segmentation.py:57`, `segmentation.py:78`, и в `CollectionProcessor`: `collectionprocessor.py:437`, `collectionprocessor.py:494`.
- stride: шаг равен `sample_size` по `y` и `x`; `collectionprocessor.py:137-138` использует `range(..., h)` и `range(..., w)`.
- edge behavior: окна создаются от `-bound` до размеров растра; для `bounds=0` размер окна равен `sample_size`, крайние окна читаются `boundless=True` с fill nodata: `band.py:196-224`.
- padding: `padding='none'` не делает mirror padding; mirror выполняется только при `self.padding == 'mirror'`: `collectionprocessor.py:71-77`.
- nodata behavior: если весь sample равен nodata, модель не вызывается и пишется empty block: `collectionprocessor.py:477-481`. При `nodata_mask_mode=True` создается mask по всем каналам `sample == nodata`: `collectionprocessor.py:68-69`; для Kanopus YAML выбран `false`, значит этот mask не создается.
- channel order: порядок ровно порядок `input_rasters`, потому что `BandCollection.ordered(*self.channels)` вызывается перед sample: `collectionprocessor.py:54-57`, а `BandCollection.ordered` собирает bands в порядке имен: `bandcollection.py:136-147`.
- array layout before adapter: `BandCollectionSample.numpy(axis=0)` возвращает `np.stack(..., axis=0)`, то есть `C,H,W`: `bandcollectionsample.py:132-133`. Комментарий адаптера фиксирует входной формат `(C, H, W)`: `adapters.py:25`.
- shape before adapter: для Kanopus при `bounds=0`, `sample_size=[512,512]`, `input_rasters=[RED,GRN,BLU,NIR]` форма перед adapter: `(4,512,512)`. Это следует из `Segmentation` -> `CollectionProcessor` -> `BandCollectionSample.numpy`; сами имена и размер выбраны YAML, поведение формы подтверждено кодом.
- shape sent to Triton: `TritonAdapter.predict` оборачивает numpy input в список и передает `x[i].shape` в `InferInput`: `adapters.py:490-505`. Для Kanopus без `input_ndim`/`input_transpose` в `TritonAdapter` это та же `(4,512,512)`.
- dtype before adapter: dtype приходит из rasterio read без нормализации: `band.py:221-224`.
- dtype sent to Triton: `ModelAdapter` вызывает preprocess, где `input_dtype` делает `x.astype(input_dtype)`: `adapters.py:122-134`, `adapters.py:236-239`. Для Kanopus YAML выбран `input_dtype: int16`, значит в Triton уходит `np.int16`.
- normalization/scaling: в найденном path нет деления на 255, mean/std, clipping или scaling перед Triton; preprocess делает только expand dims, transpose и astype: `adapters.py:128-135`.
- batch dimension: `use_batch_dim` не используется исполняемым кодом. Batch dimension добавляется только если задан `input_ndim` и `x.ndim < input_ndim`: `adapters.py:128-130`. У `TritonAdapter` нет собственных default `input_ndim`/`input_transpose`: `adapters.py:440-452`; значит для Kanopus batch dimension не добавляется.
- contiguous: код не вызывает `np.ascontiguousarray` перед `set_data_from_numpy`; отправляется массив после `astype`, если dtype задан: `adapters.py:133-134`, `adapters.py:504-505`. `astype` обычно возвращает новый массив, но contiguous ABI как требование кодом не зафиксирован.

## Output tensor handling
- source code path: `TritonAdapter.predict` читает все outputs через `response.as_numpy(outp['name'])`: `adapters.py:514-525`.
- output names: берутся из Triton model metadata `model_metadata['outputs']`: `adapters.py:476-484`.
- output shape/dtype: raw shape/dtype не преобразуются внутри `TritonAdapter.predict`; если один output, возвращается массив `results[0]`: `adapters.py:514-525`.
- threshold/argmax: в `Segmentation.processing_fn` нет threshold/argmax; он только вызывает adapter, затем optional postprocessors: `segmentation.py:90-99`. В Kanopus YAML postprocessors не выбраны, значит threshold/argmax не подтверждены кодом для этого workflow.
- output dtype before write: `CollectionProcessor` пишет raster в `dst_dtype=self.adapter.output_dtype or 'uint8'`: `segmentation.py:81`. У `ModelAdapter.output_dtype` default `uint8`: `adapters.py:199-200`; postprocess `astype(output_dtype)` выполняется, если `output_dtype` задан: `adapters.py:140-154`.
- stitching: `SampleCollectionWindowWriter.write` записывает каждый output channel отдельным writer: `collectionprocessor.py:360-369`. При `bound_mode='drop'` `weight_mtrx=None`: `collectionprocessor.py:452-461`; writer режет bounds и пишет окно: `collectionprocessor.py:293-301`. При `bounds=0` резки границ нет.
- nodata in output: если весь input sample nodata, пишется empty block с `dst_nodata`: `collectionprocessor.py:477-481`; если `nodata_mask` передан, raster pixels по mask заполняются nodata: `collectionprocessor.py:244-245`.
- vectorization: `VectorizeMasks` читает raster masks через `io.read_bc` и вызывает `polygonize(band)`: `raster_ops.py:449-458`.
- foreground condition: polygonize трактует ненулевые pixels как объект, нули как background: `functional/polygonize.py:13-16`.
- polygonization: используется `cv2.findContours` по squeezed 2D mask: `functional/polygonize.py:84-98`, координаты переводятся через raster transform: `functional/polygonize.py:121-128`.

## Available bricks
- Raster preprocessing: `SplitRaster`, `MergeRaster`, `BrightnessNormalization`, `RoundRaster`, `AddConstant`, `ReplacePixelValue`, `ApplyMask`, `MultiThresholding`, `MaskMorphology`, `ZonalStats` в `urban/bricks/raster_ops.py`.
- Model adapters: `MockAdapter`, `TorchJitModelAdapter`, `TFServingModelAdapter`, `TorchServingModelAdapter`, `TritonAdapter` в `urban/bricks/adapters.py`.
- Model inference: `Segmentation`, `InstanceRegression`, `MetaAnglesRegression`, `EmbeddingEstimator`, `NSPDParcels`, `SAMAutoMaskGenerator`, `SAMPromptMaskGenerator`, `Text2Box` в `urban/bricks/model_bricks`.
- Postprocessing: `VectorizeMasks` в `raster_ops.py`, `LabelsToOnehot` и `SeparateInstances` в `model_bricks/postprocess.py`, vector bricks в `urban/bricks/vector_ops.py`, `buildings_postprocessing.py`, `roads_postprocessing.py`, `fields_postprocessing.py`.

## Debug instrumentation
- Добавлен временный hook в `Geoalert/Workflow Engine/inference-v1.5.5/modules/urban/urban/bricks/adapters.py:71-110`, `adapters.py:496-504`, `adapters.py:517-524`.
- Включение: `GEOALERT_DEBUG_TENSORS=1`.
- Папка по умолчанию: `Geoalert/_analysis/debug_tensors`, можно переопределить `GEOALERT_DEBUG_TENSORS_DIR`.
- Сохраняет `.npy` и `.json` для входов Triton до `InferInput.set_data_from_numpy` и outputs после `response.as_numpy`.

## Blockers
- Python 3.12.10 установлен через `winget`; синтаксическая проверка `python -m py_compile Geoalert\Workflow Engine\inference-v1.5.5\modules\urban\urban\bricks\adapters.py` прошла.
- pytest inference не запускался: для полноценного запуска нужны зависимости worker, включая приватный `we-queue-client[minio]==1.5.2`.
- Docker Desktop и WSL отсутствуют.
- `Geoalert/Workflow Engine/inference-v1.5.5/Dockerfile:14-16` и `inference/message_handler.py:14-17` содержат санитизированные секреты, запуск worker как есть заблокирован.
- `we-queue-client[minio]==1.5.2` ставится из приватного GitLab PyPI в Dockerfile: `Dockerfile:14-16`; без доступа к этому пакету очередь worker не поднимется.

## What MLSystem2 must match
- Для совместимости с этим Kanopus inference путь MLSystem2 должен уметь формировать raw `int16` без деления на `255`/`np.iinfo(dtype).max`; текущий MLSystem2 нормализует integer в `_dataset.py:143-150`.
- Layout для Geoalert перед Triton - `C,H,W`; MLSystem2 уже возвращает image как channel-first, но затем batch collate дает `[B,C,H,W]` в `_dataloader.py:72-79`. Для Geoalert Kanopus batch dimension не нужен.
- Edge behavior Geoalert при `bounds=0` читает крайние окна `boundless=True` и заполняет nodata: `band.py:221-224`; MLSystem2 строит shifted last tile через `_windows.py:98-99`, это отличается.
- Geoalert пропускает полностью nodata sample без вызова модели: `collectionprocessor.py:477-481`; MLSystem2 сейчас зануляет invalid mask pixels и возвращает tile: `_dataset.py:60-64`.
