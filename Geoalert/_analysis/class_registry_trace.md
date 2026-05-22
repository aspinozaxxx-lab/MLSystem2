# Class registry trace

## Pipeline YAML parsing
- `Geoalert/Workflow Engine/inference-v1.5.5/inference/data_processor.py:37-45` - worker декодирует base64 pipeline в файл и вызывает `urban.Compose.load`.
- `Geoalert/Workflow Engine/inference-v1.5.5/modules/urban/urban/base/compose.py:187-203` - `Compose.load` вызывает `parse_config`, проверяет версию и создает `Compose.from_config`.
- `Geoalert/Workflow Engine/inference-v1.5.5/modules/urban/urban/base/parser.py:76-86` - рекурсивно заменяет ключ `_class` на `brick_class`.

## Registry
- `Geoalert/Workflow Engine/inference-v1.5.5/modules/urban/urban/base/registry_object.py:8-13` - metaclass регистрирует каждый класс-наследник `RegistryObject` в `CLASS_REGISTRY` по имени класса.
- `Geoalert/Workflow Engine/inference-v1.5.5/modules/urban/urban/base/brick.py:22-33` - `Brick.from_config` берет `brick_class`, ищет класс в `CLASS_REGISTRY` и создает объект `cls(**config)`.
- `Geoalert/Workflow Engine/inference-v1.5.5/modules/urban/urban/bricks/__init__.py:1-94` - импортирует bricks, чтобы классы попали в registry.
- `Geoalert/Workflow Engine/inference-v1.5.5/modules/urban/urban/__init__.py:4-5` - импортирует `bricks` и `base`.

## Конкретные классы
- `_class: Compose` -> `Geoalert/Workflow Engine/inference-v1.5.5/modules/urban/urban/base/compose.py:28`.
- `_class: SplitRaster` -> `Geoalert/Workflow Engine/inference-v1.5.5/modules/urban/urban/bricks/raster_ops.py:28`.
- `_class: Segmentation` -> `Geoalert/Workflow Engine/inference-v1.5.5/modules/urban/urban/bricks/model_bricks/segmentation.py:16`.
- `_class: TritonAdapter` -> `Geoalert/Workflow Engine/inference-v1.5.5/modules/urban/urban/bricks/adapters.py:440`.
- `_class: VectorizeMasks` -> `Geoalert/Workflow Engine/inference-v1.5.5/modules/urban/urban/bricks/raster_ops.py:409`.

## Важное ограничение
- `use_batch_dim` найден только в YAML и тестовых YAML, но не найден в исполняемом Python-коде Geoalert. Подтвержденная логика batch dimension находится в `input_ndim`/`output_ndim` адаптера: `adapters.py:122-134` и `adapters.py:140-154`.
