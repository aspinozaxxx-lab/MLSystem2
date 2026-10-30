"""Расширения Geoalert для оконного инференса next-gen2 через штатный adapter."""

from contextlib import ExitStack
import json
from pathlib import Path
import time

import numpy as np
import rasterio
from rasterio.windows import Window
from mlsystem2.training_ui_api._raster_valid_data import valid_data_mask


_OBJECT_PERFORMANCE_FILENAME = "object_inference_performance.json"


def register_notebook_bricks() -> None:
    """Зарегистрировать брики в том же процессе, который загружает Compose."""
    from pydantic import Field
    from urban import SplitRaster as OriginalSplitRaster
    from urban.bricks.model_bricks.modelbrick import ModelBrick
    from urban.base.brick import Brick
    from mlsystem2.inference.contracts import ObjectSeparationConfig

    class SplitRaster(OriginalSplitRaster):
        apply_mask: bool = True

        def __call__(self, path):
            if self.apply_mask:
                return super().__call__(path)
            _split_unmasked(Path(path) / f"{self.input}.{self.input_ext}", Path(path), self.output)

    class SlidingWindowSegmentation(ModelBrick):
        input_rasters: list[str] = Field(min_length=1)
        output_labels: list[str] = Field(min_length=1)
        window_size: int = Field(gt=0)
        stride: int = Field(gt=0)
        sigma_scale: float = Field(default=0.25, gt=0)
        pad_edges: bool = False
        shift_edges: bool = False

        def __call__(self, path):
            _sliding_window_probabilities(
                Path(path), self.input_rasters, self.output_labels, self.adapter,
                self.window_size, self.stride, self.sigma_scale,
                pad_edges=self.pad_edges,
                shift_edges=self.shift_edges,
            )

    class ObjectF1Segmentation(ModelBrick):
        input_raster: str = "input"
        input_channels: int = Field(default=3, ge=3, le=4)
        output_raster: str = "object_instances"
        window_size: int = Field(gt=0)
        threshold: float = Field(default=0.5, gt=0, lt=1)
        separation: ObjectSeparationConfig | None = None
        batch_size: int = Field(default=1, ge=1, le=32)

        def __call__(self, path):
            _object_instance_segmentation(
                Path(path), self.input_raster, self.output_raster, self.adapter,
                input_channels=self.input_channels, window_size=self.window_size,
                threshold=self.threshold, separation=self.separation, batch_size=self.batch_size,
            )

    class MulticlassProbabilityMasks(Brick):
        input_rasters: list[str] = Field(min_length=2)
        output_masks: list[str] = Field(min_length=1)
        threshold: float = Field(default=0.9, ge=0, le=1)

        def __call__(self, path):
            _multiclass_probability_masks(Path(path), self.input_rasters, self.output_masks, self.threshold)

    from mlsystem2.training_ui_api._geoalert_vector import register_vector_bricks

    register_vector_bricks()


def _object_instance_segmentation(directory, input_raster, output_raster, adapter, *,
                                  input_channels, window_size, threshold, separation, batch_size):
    """V2 отправляет полный NCHW batch; исторический профиль остаётся последовательным."""
    from mlsystem2.training_ui_api._object_inference import predict_instances
    from mlsystem2.inference.contracts import ObjectSeparationConfig

    config = ObjectSeparationConfig.model_validate(separation) if separation is not None else None
    batched = config is not None and config.profile == "object_f1_v2"
    effective_batch_size = batch_size if batched else 1
    performance = {"configured_batch_size": batch_size, "effective_batch_size": effective_batch_size}
    started = time.perf_counter()
    accumulator = None
    try:
        with rasterio.open(directory / f"{input_raster}.tif") as source:
            predict = adapter if batched else lambda images: np.stack([adapter(image) for image in images])
            accumulator, result = predict_instances(
                source, predict, input_indexes=tuple(range(1, input_channels + 1)),
                tile_size=window_size, batch_size=effective_batch_size,
                threshold=threshold, separation=config, metrics=performance,
            )
            performance["configured_batch_size"] = batch_size
            write_started = time.perf_counter()
            profile = {**_raster_profile(source, "int32"), "nodata": 0}
            with rasterio.open(directory / f"{output_raster}.tif", "w", **profile) as target:
                for _, window in target.block_windows(1):
                    x, y, w, h = map(int, (window.col_off, window.row_off, window.width, window.height))
                    target.write(result.instances[y:y+h, x:x+w], 1, window=window)
            performance["instance_write_sec"] = time.perf_counter() - write_started
    finally:
        if accumulator is not None:
            cleanup_started = time.perf_counter()
            accumulator.close()
            performance["cleanup_sec"] = time.perf_counter() - cleanup_started
        performance["elapsed_sec"] = time.perf_counter() - started
        _save_object_performance(directory, performance)


def _save_object_performance(directory, performance):
    """Служебный замер не заменяет ошибку инференса и не отменяет готовый растр."""
    try:
        serialized = json.dumps(performance, ensure_ascii=False)
        (directory / _OBJECT_PERFORMANCE_FILENAME).write_text(serialized, encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        try:
            print(f"Не удалось сохранить замер скорости объектного инференса: {exc}", flush=True)
        except (OSError, UnicodeError):
            pass
        return
    try:
        print("Этапы объектного инференса: " + serialized, flush=True)
    except (OSError, UnicodeError):
        pass


def _raster_profile(source, dtype):
    return {
        "driver": "GTiff", "width": source.width, "height": source.height,
        "crs": source.crs, "transform": source.transform, "count": 1, "dtype": dtype,
        "nodata": None, "tiled": True, "blockxsize": 256, "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }


def _split_unmasked(source_path: Path, directory: Path, labels: list[str]) -> None:
    """Сохранить исходные значения каналов, включая пиксели под маской TIFF."""
    with rasterio.open(source_path) as source, ExitStack() as stack:
        if source.count < len(labels):
            raise ValueError("В снимке недостаточно каналов для next-gen2.")
        outputs = [stack.enter_context(rasterio.open(
            directory / f"{label}.tif", "w", **_raster_profile(source, source.dtypes[index]),
        )) for index, label in enumerate(labels)]
        for _, window in source.block_windows(1):
            valid = valid_data_mask(source, window, tuple(range(1, len(labels) + 1)))
            for index, output in enumerate(outputs, start=1):
                output.write(source.read(index, window=window, masked=False), 1, window=window)
                output.write_mask(valid.astype("uint8") * 255, window=window)


def _probability_window_origins(length, size, stride, *, pad_edges=False, shift_edges=False):
    """Полный реальный контекст у края; нули нужны только снимку меньше окна."""
    if shift_edges:
        if not pad_edges:
            raise ValueError("Сдвиг краевых окон требует полного покрытия снимка")
        if length <= size:
            return [0]
        positions = list(range(0, length - size + 1, stride))
        if positions[-1] != length - size:
            positions.append(length - size)
        return positions
    return range(0, length if pad_edges else length - size + 1, stride)


def _sliding_window_probabilities(directory, inputs, outputs, adapter, size, stride, sigma_scale, *, pad_edges=False, shift_edges=False):
    """Объединить полные окна как в ноутбуке, храня в RAM только одну полосу."""
    if not 0 < stride <= size or not np.isfinite(sigma_scale) or sigma_scale <= 0:
        raise ValueError("Некорректные шаг или Gaussian sigma оконного инференса.")
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(directory / f"{name}.tif")) for name in inputs]
        source = sources[0]
        height, width = source.height, source.width
        if any((s.height, s.width, s.transform, s.crs) !=
               (height, width, source.transform, source.crs) for s in sources):
            raise ValueError("Каналы next-gen2 должны иметь одинаковую растровую сетку.")
        targets = [stack.enter_context(rasterio.open(
            directory / f"{name}.tif", "w", **_raster_profile(source, "float32"),
        )) for name in outputs]
        axis = np.arange(size)
        gaussian = np.exp(-((axis - size / 2) ** 2) / (2 * (size * sigma_scale) ** 2))
        weight = np.outer(gaussian, gaussian).astype(np.float32)
        sums = np.zeros((len(outputs), size, width), dtype=np.float32)
        weights = np.zeros((size, width), dtype=np.float32)
        rows = _probability_window_origins(height, size, stride, pad_edges=pad_edges, shift_edges=shift_edges) if width >= size or pad_edges else range(0)
        columns = _probability_window_origins(width, size, stride, pad_edges=pad_edges, shift_edges=shift_edges)
        written = 0
        for row_index, y in enumerate(rows):
            for x in columns:
                w, h = min(size, width-x), min(size, height-y)
                window = Window(x, y, w, h)
                image = np.zeros((len(sources), size, size), dtype=np.float32)
                image[:, :h, :w] = np.stack([s.read(1, window=window, out_dtype="float32", masked=False) for s in sources])
                valid = np.logical_and.reduce([s.dataset_mask(window=window) > 0 for s in sources])
                valid &= np.any(image[:, :h, :w] != 0, axis=0)
                if not valid.any():
                    continue
                if pad_edges:
                    image[:, :h, :w] *= valid
                probabilities = np.asarray(adapter(image), dtype=np.float32)
                if probabilities.shape != (len(outputs), size, size) or not np.isfinite(probabilities).all():
                    raise ValueError("Adapter next-gen2 вернул некорректную карту вероятностей.")
                current_weight = weight[:h, :w] * valid
                sums[:, :h, x:x+w] += probabilities[:, :h, :w] * current_weight
                weights[:h, x:x+w] += current_weight
            step = rows[row_index + 1] - y if row_index + 1 < len(rows) else size
            count = min(step, height-y)
            denominator = np.maximum(weights[:count], 1e-6)
            window = Window(0, y, width, count)
            valid_output = weights[:count] > 0
            for target, values in zip(targets, sums, strict=True):
                target.write(values[:count] / denominator, 1, window=window)
                target.write_mask(valid_output.astype("uint8") * 255, window=window)
            written = y + count
            if y != rows[-1]:
                sums[:, :size - step] = sums[:, step:].copy()
                sums[:, size - step:] = 0
                weights[:size - step] = weights[step:].copy()
                weights[size - step:] = 0
        # Непокрытые края и снимки меньше окна остаются нулевыми, без padding.
        for y in range(written, height, size):
            count = min(size, height - y)
            zeros = np.zeros((count, width), dtype=np.float32)
            for target in targets:
                target.write(zeros, 1, window=Window(0, y, width, count))


def _multiclass_probability_masks(directory, inputs, outputs, threshold):
    """Выбрать один класс после объединения всех вероятностей, включая фон."""
    if len(inputs) != len(outputs) + 1:
        raise ValueError("Число вероятностей должно соответствовать типам объектов и фону.")
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(directory / f"{name}.tif")) for name in inputs]
        source = sources[0]
        if any((s.width, s.height, s.transform, s.crs) != (source.width, source.height, source.transform, source.crs) for s in sources):
            raise ValueError("Вероятности классов должны иметь одинаковую растровую сетку.")
        targets = [stack.enter_context(rasterio.open(directory / f"{name}.tif", "w", **_raster_profile(source, "uint8"))) for name in outputs]
        for _, window in source.block_windows(1):
            probabilities = np.stack([s.read(1, window=window) for s in sources])
            if not np.isfinite(probabilities).all():
                raise ValueError("Вероятности классов содержат некорректные значения.")
            labels = probabilities.argmax(axis=0)
            labels[probabilities.max(axis=0) <= threshold] = 0
            for class_id, target in enumerate(targets, start=1):
                target.write((labels == class_id).astype(np.uint8), 1, window=window)
