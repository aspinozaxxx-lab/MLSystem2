"""Расширения Geoalert для оконного инференса next-gen2 через штатный adapter."""

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def register_notebook_bricks() -> None:
    """Зарегистрировать брики в том же процессе, который загружает Compose."""
    from pydantic import Field
    from urban import SplitRaster as OriginalSplitRaster
    from urban.bricks.model_bricks.modelbrick import ModelBrick

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

        def __call__(self, path):
            _sliding_window_probabilities(
                Path(path), self.input_rasters, self.output_labels, self.adapter,
                self.window_size, self.stride, self.sigma_scale,
            )

    class ObjectF1Segmentation(ModelBrick):
        input_raster: str = "input"
        input_channels: int = Field(default=3, ge=3, le=4)
        output_raster: str = "object_instances"
        window_size: int = Field(gt=0)
        threshold: float = Field(default=0.5, gt=0, lt=1)

        def __call__(self, path):
            from mlsystem2.training_ui_api._object_inference import predict_instances

            with rasterio.open(Path(path) / f"{self.input_raster}.tif") as source:
                accumulator, result = predict_instances(source,
                    lambda images: np.stack([self.adapter(image) for image in images]),
                    input_indexes=tuple(range(1, self.input_channels + 1)), tile_size=self.window_size, threshold=self.threshold)
                try:
                    profile = {**_raster_profile(source, "int32"), "nodata": 0}
                    with rasterio.open(Path(path) / f"{self.output_raster}.tif", "w", **profile) as target:
                        for _, window in target.block_windows(1):
                            x, y, w, h = map(int, (window.col_off, window.row_off, window.width, window.height))
                            target.write(result.instances[y:y+h, x:x+w], 1, window=window)
                finally:
                    accumulator.close()


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
            for index, output in enumerate(outputs, start=1):
                output.write(source.read(index, window=window, masked=False), 1, window=window)


def _sliding_window_probabilities(directory, inputs, outputs, adapter, size, stride, sigma_scale):
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
        rows = range(0, height - size + 1, stride) if width >= size else range(0)
        written = 0
        for y in rows:
            for x in range(0, width - size + 1, stride):
                window = Window(x, y, size, size)
                image = np.stack([s.read(1, window=window, out_dtype="float32", masked=False) for s in sources])
                probabilities = np.asarray(adapter(image), dtype=np.float32)
                if probabilities.shape != (len(outputs), size, size) or not np.isfinite(probabilities).all():
                    raise ValueError("Adapter next-gen2 вернул некорректную карту вероятностей.")
                sums[:, :, x:x + size] += probabilities * weight
                weights[:, x:x + size] += weight
            count = size if y == rows[-1] else stride
            denominator = np.maximum(weights[:count], 1e-6)
            window = Window(0, y, width, count)
            for target, values in zip(targets, sums, strict=True):
                target.write(values[:count] / denominator, 1, window=window)
            written = y + count
            if y != rows[-1]:
                sums[:, :size - stride] = sums[:, stride:].copy()
                sums[:, size - stride:] = 0
                weights[:size - stride] = weights[stride:].copy()
                weights[size - stride:] = 0
        # Непокрытые края и снимки меньше окна остаются нулевыми, без padding.
        for y in range(written, height, size):
            count = min(size, height - y)
            zeros = np.zeros((count, width), dtype=np.float32)
            for target in targets:
                target.write(zeros, 1, window=Window(0, y, width, count))
