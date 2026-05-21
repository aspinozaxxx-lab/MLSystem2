"""Dataset тайлов по одному VRT XML и GeoJSON."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rasterio.io import DatasetReader, MemoryFile
from rasterio.windows import Window

from ._annotations import AnnotationIndex, load_annotation_index
from ._augmentations import apply_augmentations
from ._mask import rasterize_window_mask
from ._vrt import open_vrt_reader, open_vrt_xml
from ._windows import build_tile_windows


class TileDataset:
    def __init__(
        self,
        *,
        vrt_xml: str,
        annotation_file: str | Path,
        tile_size: int,
        stride: int,
        mode: str,
        seed: int,
        augmentation_level: int,
    ) -> None:
        self._vrt_xml = vrt_xml
        self._annotation_file = Path(annotation_file)
        self._tile_size = tile_size
        self._mode = mode
        self._seed = seed
        self._augmentation_level = augmentation_level
        self._memory_file: MemoryFile | None = None
        self._dataset: DatasetReader | None = None
        self._annotation_index: AnnotationIndex | None = None

        with open_vrt_xml(vrt_xml) as dataset:
            self._count = dataset.count
            self._dtypes = tuple(dataset.dtypes)
            self._vrt_crs = dataset.crs.to_string() if dataset.crs is not None else None
            self._windows = build_tile_windows(dataset.width, dataset.height, tile_size, stride)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        tile_window = self._windows[index]
        dataset = self._open_dataset()
        window = Window(tile_window.x, tile_window.y, tile_window.width, tile_window.height)

        image = self._read_image(dataset, window)
        valid_mask = self._read_valid_mask(dataset, window)
        image[:, ~valid_mask] = 0.0

        mask = self._read_annotation_mask(dataset, window, valid_mask)
        if self._mode == "train" and self._augmentation_level > 0:
            image, mask = apply_augmentations(
                image,
                mask,
                level=self._augmentation_level,
                seed=self._seed,
                sample_index=index,
            )

        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def close(self) -> None:
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None
        if self._memory_file is not None:
            self._memory_file.close()
            self._memory_file = None

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_memory_file"] = None
        state["_dataset"] = None
        state["_annotation_index"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _open_dataset(self) -> DatasetReader:
        if self._dataset is None:
            self._memory_file, self._dataset = open_vrt_reader(self._vrt_xml)
        return self._dataset

    def _annotation_index_or_load(self) -> AnnotationIndex:
        if self._annotation_index is None:
            self._annotation_index = load_annotation_index(self._annotation_file, self._vrt_crs)
        return self._annotation_index

    def _read_image(self, dataset: DatasetReader, window: Window) -> np.ndarray:
        image = dataset.read(
            window=window,
            boundless=True,
            fill_value=0,
            out_shape=(self._count, self._tile_size, self._tile_size),
            masked=False,
        )
        return _to_float32(image, self._dtypes)

    def _read_valid_mask(self, dataset: DatasetReader, window: Window) -> np.ndarray:
        masks = dataset.read_masks(
            window=window,
            boundless=True,
            out_shape=(self._count, self._tile_size, self._tile_size),
        )
        if masks.ndim == 2:
            return masks > 0
        return np.all(masks > 0, axis=0)

    def _read_annotation_mask(
        self,
        dataset: DatasetReader,
        window: Window,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        geometries = self._annotation_index_or_load().query_bounds(dataset.window_bounds(window))
        mask = rasterize_window_mask(
            geometries,
            out_shape=(self._tile_size, self._tile_size),
            transform=dataset.window_transform(window),
        )
        mask[~valid_mask] = 0
        return mask.astype(np.float32, copy=False)[None, :, :]


def _to_float32(image: np.ndarray, dtypes: tuple[str, ...]) -> np.ndarray:
    result = image.astype(np.float32, copy=False)
    for channel, dtype_name in enumerate(dtypes):
        dtype = np.dtype(dtype_name)
        if dtype == np.dtype("uint8"):
            result[channel] /= 255.0
        elif np.issubdtype(dtype, np.integer):
            result[channel] /= float(np.iinfo(dtype).max)
    return result
