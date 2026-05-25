"""Dataset тайлов по одному VRT XML и GeoJSON."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rasterio.io import DatasetReader, MemoryFile
from rasterio.windows import Window

from ._annotations import AnnotationIndex, load_annotation_index
from ._augmentations import apply_augmentations
from ._mask import rasterize_window_mask
from ._valid_footprint import filter_valid_windows
from ._vrt import open_vrt_reader, open_vrt_xml
from ._windows import build_vrt_source_windows_with_diagnostics


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
        smart_tiling: bool,
        positive_factor: float = 0.5,
    ) -> None:
        self._vrt_xml = vrt_xml
        self._annotation_file = Path(annotation_file)
        self._tile_size = tile_size
        self._mode = mode
        self._seed = seed
        self._augmentation_level = augmentation_level
        self._smart_tiling = smart_tiling
        self._positive_factor = positive_factor
        self._memory_file: MemoryFile | None = None
        self._dataset: DatasetReader | None = None
        self._annotation_index: AnnotationIndex | None = None
        self._positive_hint_by_index: list[bool] | None = None

        with open_vrt_xml(vrt_xml) as dataset:
            self._count = dataset.count
            self._nodata = _resolve_nodata(dataset)
            self._vrt_crs = dataset.crs.to_string() if dataset.crs is not None else None
            candidate_windows, diagnostics = build_vrt_source_windows_with_diagnostics(
                vrt_xml,
                dataset.width,
                dataset.height,
                tile_size,
                stride,
            )
            valid_windows, valid_diagnostics = filter_valid_windows(
                dataset,
                candidate_windows,
                nodata=self._nodata,
            )
            self._windows = valid_windows
            self._source_rect_count = diagnostics.source_rect_count
            self._candidate_window_count = valid_diagnostics.valid_window_count
            self._uses_vrt_source_rects = diagnostics.uses_vrt_source_rects
            self._candidate_window_count_before_valid_filter = (
                valid_diagnostics.candidate_window_count_before_valid_filter
            )
            self._black_filtered_window_count = valid_diagnostics.black_filtered_window_count
            self._valid_footprint_stride = valid_diagnostics.valid_footprint_stride
            self._valid_footprint_valid_cells = valid_diagnostics.valid_footprint_valid_cells
            self._valid_footprint_total_cells = valid_diagnostics.valid_footprint_total_cells
            if self._smart_tiling and self._mode in {"train", "val"}:
                self._positive_hint_by_index = self._build_positive_hints(dataset)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, dict[str, bool]]:
        tile_window = self._windows[index]
        dataset = self._open_dataset()
        window = Window(tile_window.x, tile_window.y, tile_window.width, tile_window.height)

        image_raw = self._read_image_raw(dataset, window)
        nodata_pixels = _nodata_pixels(image_raw, self._nodata)
        image = image_raw.astype(np.float32, copy=False)

        mask = self._read_annotation_mask(dataset, window, nodata_pixels)
        positive = bool(np.count_nonzero(mask) > 0)
        augmented = False
        should_augment = self._mode == "train" and self._augmentation_level > 0
        if should_augment and (not self._smart_tiling or positive):
            image, mask, augmented = apply_augmentations(
                image,
                mask,
                level=self._augmentation_level,
                seed=self._seed,
                sample_index=index,
            )

        return (
            np.ascontiguousarray(image),
            np.ascontiguousarray(mask),
            {"augmented": augmented, "positive": positive},
        )

    @property
    def source_rect_count(self) -> int:
        return self._source_rect_count

    @property
    def candidate_window_count(self) -> int:
        return self._candidate_window_count

    @property
    def candidate_window_count_before_valid_filter(self) -> int:
        return self._candidate_window_count_before_valid_filter

    @property
    def black_filtered_window_count(self) -> int:
        return self._black_filtered_window_count

    @property
    def valid_footprint_stride(self) -> int:
        return self._valid_footprint_stride

    @property
    def valid_footprint_valid_cells(self) -> int:
        return self._valid_footprint_valid_cells

    @property
    def valid_footprint_total_cells(self) -> int:
        return self._valid_footprint_total_cells

    @property
    def uses_vrt_source_rects(self) -> bool:
        return self._uses_vrt_source_rects

    @property
    def smart_tiling_enabled(self) -> bool:
        return self._smart_tiling

    @property
    def estimated_positive_tiles(self) -> int | None:
        if self._positive_hint_by_index is None:
            return None
        return sum(1 for item in self._positive_hint_by_index if item)

    @property
    def estimated_negative_tiles(self) -> int | None:
        if self._positive_hint_by_index is None:
            return None
        return sum(1 for item in self._positive_hint_by_index if not item)

    def sampling_weights(self, positive_factor: float | None = None) -> list[float] | None:
        if self._positive_hint_by_index is None:
            return None
        positive_count = self.estimated_positive_tiles or 0
        negative_count = self.estimated_negative_tiles or 0
        if positive_count == 0 or negative_count == 0:
            return None
        factor = self._positive_factor if positive_factor is None else positive_factor
        positive_weight = factor / positive_count
        negative_weight = (1.0 - factor) / negative_count
        return [
            positive_weight if positive else negative_weight
            for positive in self._positive_hint_by_index
        ]

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

    def _read_image_raw(self, dataset: DatasetReader, window: Window) -> np.ndarray:
        return dataset.read(
            window=window,
            boundless=True,
            fill_value=self._nodata,
            out_shape=(self._count, self._tile_size, self._tile_size),
            masked=False,
        )

    def _read_annotation_mask(
        self,
        dataset: DatasetReader,
        window: Window,
        nodata_pixels: np.ndarray,
    ) -> np.ndarray:
        geometries = self._annotation_index_or_load().query_bounds(dataset.window_bounds(window))
        mask = rasterize_window_mask(
            geometries,
            out_shape=(self._tile_size, self._tile_size),
            transform=dataset.window_transform(window),
        )
        mask[nodata_pixels] = 0
        return mask.astype(np.float32, copy=False)[None, :, :]

    def _build_positive_hints(self, dataset: DatasetReader) -> list[bool]:
        annotation_index = self._annotation_index_or_load()
        hints: list[bool] = []
        for tile_window in self._windows:
            window = Window(tile_window.x, tile_window.y, tile_window.width, tile_window.height)
            hints.append(bool(annotation_index.query_bounds(dataset.window_bounds(window))))
        return hints


def _resolve_nodata(dataset: DatasetReader) -> object:
    if dataset.nodata is not None:
        return dataset.nodata
    for nodata in dataset.nodatavals:
        if nodata is not None:
            return nodata
    return 0


def _nodata_pixels(image: np.ndarray, nodata: object) -> np.ndarray:
    if _is_nan(nodata):
        return np.all(np.isnan(image), axis=0)
    return np.all(image == nodata, axis=0)


def _is_nan(value: object) -> bool:
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False
