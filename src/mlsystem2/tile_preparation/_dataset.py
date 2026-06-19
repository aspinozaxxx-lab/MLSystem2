"""Dataset тайлов по одному VRT XML и GeoJSON."""

from __future__ import annotations

import random
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
from .contracts import TileClassAnnotation, TilePreparationError, TileSplitRequest


TILE_CATEGORY_POSITIVE = "positive"
TILE_CATEGORY_HARD_NEGATIVE = "hard_negative"
TILE_CATEGORY_BACKGROUND = "background"


class TileDataset:
    def __init__(
        self,
        *,
        vrt_xml: str,
        annotation_file: str | Path | None = None,
        hard_negative_annotation_file: str | Path | None = None,
        class_annotations: list[TileClassAnnotation] | None = None,
        tile_size: int,
        stride: int,
        mode: str,
        seed: int,
        augmentation_level: int,
        positive_factor: float = 0.5,
        hard_negative_factor: float = 0.0,
        background_factor: float = 0.5,
        class_balance: bool = False,
        tile_split: TileSplitRequest | None = None,
    ) -> None:
        self._vrt_xml = vrt_xml
        self._annotation_file = Path(annotation_file) if annotation_file is not None else None
        self._hard_negative_annotation_file = (
            Path(hard_negative_annotation_file)
            if hard_negative_annotation_file is not None
            else None
        )
        self._class_annotations = sorted(
            list(class_annotations or []),
            key=lambda item: (item.priority, item.class_id),
        )
        self._tile_size = tile_size
        self._mode = mode
        self._seed = seed
        self._augmentation_level = augmentation_level
        self._positive_factor = positive_factor
        self._hard_negative_factor = hard_negative_factor
        self._background_factor = background_factor
        self._sampling_factor_used: dict[str, float] | None = None
        self._sampling_warnings: list[str] = []
        self._class_balance = class_balance
        self._tile_split = tile_split
        self._tile_split_warnings: list[str] = []
        self._pool_window_count = 0
        self._split_window_count = 0
        self._memory_file: MemoryFile | None = None
        self._dataset: DatasetReader | None = None
        self._annotation_index: AnnotationIndex | None = None
        self._class_annotation_indexes: dict[int, AnnotationIndex] = {}
        self._hard_negative_annotation_indexes: dict[Path, AnnotationIndex] = {}
        self._positive_hint_by_index: list[bool] | None = None
        self._hard_negative_hint_by_index: list[bool] | None = None
        self._category_hint_by_index: list[str] | None = None
        self._class_hints_by_index: list[frozenset[int]] | None = None

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
            should_build_hints = self._mode in {"train", "val"} or self._tile_split is not None
            if should_build_hints:
                if self._class_annotations:
                    self._class_hints_by_index = self._build_class_hints(dataset)
                    self._positive_hint_by_index = [
                        bool(hints) for hints in self._class_hints_by_index
                    ]
                else:
                    self._positive_hint_by_index = self._build_positive_hints(dataset)
                self._hard_negative_hint_by_index = self._build_hard_negative_hints(dataset)
                self._category_hint_by_index = _tile_categories(
                    self._positive_hint_by_index,
                    self._hard_negative_hint_by_index,
                )
            self._pool_window_count = len(self._windows)
            if self._tile_split is not None:
                self._apply_tile_split(self._tile_split)
            self._split_window_count = len(self._windows)

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
        positive_before_augmentation = bool(np.count_nonzero(mask) > 0)
        category = self._sample_category(index, positive_before_augmentation)
        augmented = False
        should_augment = self._mode == "train" and self._augmentation_level > 0
        if should_augment and category in {
            TILE_CATEGORY_POSITIVE,
            TILE_CATEGORY_HARD_NEGATIVE,
        }:
            image, mask, augmented = apply_augmentations(
                image,
                mask,
                level=self._augmentation_level,
                seed=self._seed,
                sample_index=index,
            )

        meta = self._sample_meta(category, augmented)
        return (
            np.ascontiguousarray(image),
            np.ascontiguousarray(mask),
            meta,
        )

    @property
    def channel_count(self) -> int:
        return self._count

    @property
    def tile_size(self) -> int:
        return self._tile_size

    @property
    def positive_hints(self) -> list[bool] | None:
        if self._positive_hint_by_index is None:
            return None
        return list(self._positive_hint_by_index)

    @property
    def hard_negative_hints(self) -> list[bool] | None:
        if self._hard_negative_hint_by_index is None:
            return None
        return list(self._hard_negative_hint_by_index)

    @property
    def tile_categories(self) -> list[str] | None:
        categories = self._category_hints()
        if categories is None:
            return None
        return list(categories)

    @property
    def uses_multiclass_masks(self) -> bool:
        return bool(self._class_annotations)

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
    def pool_window_count(self) -> int:
        return self._pool_window_count

    @property
    def split_window_count(self) -> int:
        return self._split_window_count

    @property
    def tile_split_warnings(self) -> list[str]:
        return list(self._tile_split_warnings)

    @property
    def estimated_positive_tiles(self) -> int | None:
        if self._positive_hint_by_index is None:
            return None
        return sum(1 for item in self._positive_hint_by_index if item)

    @property
    def estimated_hard_negative_tiles(self) -> int | None:
        categories = self._category_hints()
        if categories is None:
            return None
        return sum(1 for item in categories if item == TILE_CATEGORY_HARD_NEGATIVE)

    @property
    def estimated_background_tiles(self) -> int | None:
        categories = self._category_hints()
        if categories is None:
            return None
        return sum(1 for item in categories if item == TILE_CATEGORY_BACKGROUND)

    @property
    def positive_factor_used(self) -> float | None:
        factors = self._sampling_factor_used
        if factors is None:
            return None
        return factors[TILE_CATEGORY_POSITIVE]

    @property
    def hard_negative_factor_used(self) -> float | None:
        factors = self._sampling_factor_used
        if factors is None:
            return None
        return factors[TILE_CATEGORY_HARD_NEGATIVE]

    @property
    def background_factor_used(self) -> float | None:
        factors = self._sampling_factor_used
        if factors is None:
            return None
        return factors[TILE_CATEGORY_BACKGROUND]

    @property
    def sampling_warnings(self) -> list[str]:
        return list(self._sampling_warnings)

    @property
    def class_balance_enabled(self) -> bool:
        return bool(self._class_balance and self._class_annotations)

    @property
    def class_balance_warnings(self) -> list[str]:
        if not self.class_balance_enabled:
            return []
        counts = self._class_positive_tile_counts()
        if counts is None:
            return []
        warnings: list[str] = []
        for slug, count in counts.items():
            if count == 0:
                warnings.append(f"class_balance: для класса {slug} нет positive windows.")
            elif count < 3:
                warnings.append(f"class_balance: для класса {slug} найдено мало positive windows: {count}.")
        return warnings

    def sampling_weights(
        self,
        positive_factor: float | None = None,
        hard_negative_factor: float | None = None,
        background_factor: float | None = None,
    ) -> list[float] | None:
        categories = self._category_hints()
        if categories is None:
            return None
        factors = {
            TILE_CATEGORY_POSITIVE: (
                getattr(self, "_positive_factor", 0.5)
                if positive_factor is None
                else positive_factor
            ),
            TILE_CATEGORY_HARD_NEGATIVE: (
                getattr(self, "_hard_negative_factor", 0.0)
                if hard_negative_factor is None
                else hard_negative_factor
            ),
            TILE_CATEGORY_BACKGROUND: (
                getattr(self, "_background_factor", 0.5)
                if background_factor is None
                else background_factor
            ),
        }
        counts = _category_counts(categories)
        factors, warnings = _effective_sampling_factors(
            factors,
            counts,
            dataset_size=len(categories),
        )
        self._sampling_factor_used = factors
        self._sampling_warnings = warnings
        positive_count = counts[TILE_CATEGORY_POSITIVE]
        hard_negative_count = counts[TILE_CATEGORY_HARD_NEGATIVE]
        background_count = counts[TILE_CATEGORY_BACKGROUND]
        positive_factor_value = factors[TILE_CATEGORY_POSITIVE]
        hard_negative_weight = (
            factors[TILE_CATEGORY_HARD_NEGATIVE] / hard_negative_count
            if hard_negative_count > 0
            else 0.0
        )
        background_weight = (
            factors[TILE_CATEGORY_BACKGROUND] / background_count
            if background_count > 0
            else 0.0
        )
        positive_weights: list[float] | None = None
        if self.class_balance_enabled and self._class_hints_by_index is not None:
            positive_weights = self._class_balanced_positive_weights(positive_factor_value)
        positive_weight = (
            positive_factor_value / positive_count
            if positive_count > 0
            else 0.0
        )
        weights: list[float] = []
        for index, category in enumerate(categories):
            if category == TILE_CATEGORY_POSITIVE:
                weights.append(
                    positive_weights[index] if positive_weights is not None else positive_weight
                )
            elif category == TILE_CATEGORY_HARD_NEGATIVE:
                weights.append(hard_negative_weight)
            else:
                weights.append(background_weight)
        return weights

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
        state["_class_annotation_indexes"] = {}
        state["_hard_negative_annotation_indexes"] = {}
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
        if self._annotation_file is None:
            raise TilePreparationError("Для binary mask не задан annotation_file.")
        if self._annotation_index is None:
            self._annotation_index = load_annotation_index(self._annotation_file, self._vrt_crs)
        return self._annotation_index

    def _class_annotation_index_or_load(
        self,
        annotation: TileClassAnnotation,
    ) -> AnnotationIndex:
        index = self._class_annotation_indexes.get(annotation.class_id)
        if index is None:
            index = load_annotation_index(annotation.annotation_file, self._vrt_crs)
            self._class_annotation_indexes[annotation.class_id] = index
        return index

    def _hard_negative_annotation_index_or_load(
        self,
        annotation_file: Path,
    ) -> AnnotationIndex:
        index = self._hard_negative_annotation_indexes.get(annotation_file)
        if index is None:
            index = load_annotation_index(annotation_file, self._vrt_crs)
            self._hard_negative_annotation_indexes[annotation_file] = index
        return index

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
        if self._class_annotations:
            return self._read_multiclass_annotation_mask(dataset, window, nodata_pixels)
        geometries = self._annotation_index_or_load().query_bounds(dataset.window_bounds(window))
        mask = rasterize_window_mask(
            geometries,
            out_shape=(self._tile_size, self._tile_size),
            transform=dataset.window_transform(window),
        )
        mask[nodata_pixels] = 0
        return mask.astype(np.float32, copy=False)[None, :, :]

    def _read_multiclass_annotation_mask(
        self,
        dataset: DatasetReader,
        window: Window,
        nodata_pixels: np.ndarray,
    ) -> np.ndarray:
        mask = np.zeros((self._tile_size, self._tile_size), dtype=np.int64)
        for annotation in self._class_annotations:
            geometries = self._class_annotation_index_or_load(annotation).query_bounds(
                dataset.window_bounds(window)
            )
            class_mask = rasterize_window_mask(
                geometries,
                out_shape=(self._tile_size, self._tile_size),
                transform=dataset.window_transform(window),
            )
            class_mask[nodata_pixels] = 0
            mask[class_mask == 1] = annotation.class_id
        return mask

    def _apply_tile_split(self, tile_split: TileSplitRequest) -> None:
        if self._positive_hint_by_index is None:
            return
        train_indices, val_indices, warnings = _split_tile_indices(
            self._positive_hint_by_index,
            val_fraction=tile_split.val_fraction,
            seed=tile_split.seed,
        )
        selected_indices = train_indices if self._mode == "train" else val_indices
        if not any(self._positive_hint_by_index[index] for index in selected_indices):
            warnings.append(f"tile_split: subset {self._mode} не содержит positive windows.")
        if not selected_indices:
            warnings.append(f"tile_split: subset {self._mode} пуст.")

        self._windows = [self._windows[index] for index in selected_indices]
        self._positive_hint_by_index = [
            self._positive_hint_by_index[index] for index in selected_indices
        ]
        if self._hard_negative_hint_by_index is not None:
            self._hard_negative_hint_by_index = [
                self._hard_negative_hint_by_index[index] for index in selected_indices
            ]
        if self._category_hint_by_index is not None:
            self._category_hint_by_index = [
                self._category_hint_by_index[index] for index in selected_indices
            ]
        if self._class_hints_by_index is not None:
            self._class_hints_by_index = [
                self._class_hints_by_index[index] for index in selected_indices
            ]
        self._tile_split_warnings = warnings

    def _build_positive_hints(self, dataset: DatasetReader) -> list[bool]:
        hints: list[bool] = []
        for tile_window in self._windows:
            window = Window(tile_window.x, tile_window.y, tile_window.width, tile_window.height)
            bounds = dataset.window_bounds(window)
            if self._class_annotations:
                hints.append(
                    any(
                        self._class_annotation_index_or_load(annotation).query_bounds(bounds)
                        for annotation in self._class_annotations
                    )
                )
            else:
                annotation_index = self._annotation_index_or_load()
                hints.append(bool(annotation_index.query_bounds(bounds)))
        return hints

    def _build_class_hints(self, dataset: DatasetReader) -> list[frozenset[int]]:
        hints: list[frozenset[int]] = []
        for tile_window in self._windows:
            window = Window(tile_window.x, tile_window.y, tile_window.width, tile_window.height)
            bounds = dataset.window_bounds(window)
            class_ids = {
                annotation.class_id
                for annotation in self._class_annotations
                if self._class_annotation_index_or_load(annotation).query_bounds(bounds)
            }
            hints.append(frozenset(class_ids))
        return hints

    def _build_hard_negative_hints(self, dataset: DatasetReader) -> list[bool]:
        annotation_files = self._hard_negative_annotation_files()
        if not annotation_files:
            return [False for _ in self._windows]
        hints: list[bool] = []
        for tile_window in self._windows:
            window = Window(tile_window.x, tile_window.y, tile_window.width, tile_window.height)
            bounds = dataset.window_bounds(window)
            hints.append(
                any(
                    self._hard_negative_annotation_index_or_load(annotation_file).query_bounds(bounds)
                    for annotation_file in annotation_files
                )
            )
        return hints

    def _hard_negative_annotation_files(self) -> list[Path]:
        paths: list[Path] = []
        if self._hard_negative_annotation_file is not None:
            paths.append(self._hard_negative_annotation_file)
        for annotation in self._class_annotations:
            if annotation.hard_negative_annotation_file is not None:
                paths.append(Path(annotation.hard_negative_annotation_file))
        return _unique_paths(paths)

    def _category_hints(self) -> list[str] | None:
        category_hints = getattr(self, "_category_hint_by_index", None)
        if category_hints is not None:
            return category_hints
        positive_hints = getattr(self, "_positive_hint_by_index", None)
        if positive_hints is None:
            return None
        hard_negative_hints = getattr(self, "_hard_negative_hint_by_index", None) or [
            False for _ in positive_hints
        ]
        return _tile_categories(positive_hints, hard_negative_hints)

    def _sample_category(self, index: int, positive: bool) -> str:
        if positive:
            return TILE_CATEGORY_POSITIVE
        hard_negative_hints = self._hard_negative_hint_by_index
        if hard_negative_hints is not None and hard_negative_hints[index]:
            return TILE_CATEGORY_HARD_NEGATIVE
        return TILE_CATEGORY_BACKGROUND

    def _class_balanced_positive_weights(
        self,
        factor: float,
    ) -> list[float] | None:
        if self._class_hints_by_index is None:
            return None
        if factor == 0.0:
            return [0.0 for _ in self._class_hints_by_index]
        counts_by_id = {annotation.class_id: 0 for annotation in self._class_annotations}
        for hints in self._class_hints_by_index:
            for class_id in hints:
                if class_id in counts_by_id:
                    counts_by_id[class_id] += 1
        positive_class_ids = [
            class_id for class_id, count in counts_by_id.items() if count > 0
        ]
        if not positive_class_ids:
            return None
        class_budget = factor / len(positive_class_ids)
        weights: list[float] = []
        for hints in self._class_hints_by_index:
            if not hints:
                weights.append(0.0)
                continue
            weight = sum(
                class_budget / counts_by_id[class_id]
                for class_id in hints
                if counts_by_id.get(class_id, 0) > 0
            )
            weights.append(weight)
        return weights

    def _class_positive_tile_counts(self) -> dict[str, int] | None:
        if self._class_hints_by_index is None:
            return None
        counts = {annotation.slug: 0 for annotation in self._class_annotations}
        slug_by_id = {annotation.class_id: annotation.slug for annotation in self._class_annotations}
        for hints in self._class_hints_by_index:
            for class_id in hints:
                slug = slug_by_id.get(class_id)
                if slug is not None:
                    counts[slug] += 1
        return counts

    def _sample_meta(self, category: str, augmented: bool) -> dict[str, object]:
        return {
            "augmented": augmented,
            "category": category,
            "positive": category == TILE_CATEGORY_POSITIVE,
            "hard_negative": category == TILE_CATEGORY_HARD_NEGATIVE,
            "background": category == TILE_CATEGORY_BACKGROUND,
        }


def _split_tile_indices(
    positive_hints: list[bool],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[str]]:
    positive_indices = [index for index, positive in enumerate(positive_hints) if positive]
    negative_indices = [index for index, positive in enumerate(positive_hints) if not positive]
    rng = random.Random(seed)
    train_positive, val_positive, warnings = _split_index_group(
        positive_indices,
        val_fraction=val_fraction,
        rng=rng,
        group_name="positive",
    )
    train_negative, val_negative, negative_warnings = _split_index_group(
        negative_indices,
        val_fraction=val_fraction,
        rng=rng,
        group_name="negative",
    )
    train_indices = sorted([*train_positive, *train_negative])
    val_indices = sorted([*val_positive, *val_negative])
    return train_indices, val_indices, [*warnings, *negative_warnings]


def _split_index_group(
    indices: list[int],
    *,
    val_fraction: float,
    rng: random.Random,
    group_name: str,
) -> tuple[list[int], list[int], list[str]]:
    if not indices:
        warning = (
            ["tile_split: в общем пуле нет positive windows."]
            if group_name == "positive"
            else []
        )
        return [], [], warning
    if len(indices) == 1:
        warning = (
            ["tile_split: positive windows меньше 2, val positive остается пустым."]
            if group_name == "positive"
            else []
        )
        return list(indices), [], warning

    shuffled = list(indices)
    rng.shuffle(shuffled)
    val_count = int(round(len(indices) * val_fraction))
    val_count = min(len(indices) - 1, max(1, val_count))
    val_set = set(shuffled[:val_count])
    train = [index for index in indices if index not in val_set]
    val = [index for index in indices if index in val_set]
    return train, val, []


def _tile_categories(
    positive_hints: list[bool],
    hard_negative_hints: list[bool],
) -> list[str]:
    categories: list[str] = []
    for positive, hard_negative in zip(positive_hints, hard_negative_hints):
        if positive:
            categories.append(TILE_CATEGORY_POSITIVE)
        elif hard_negative:
            categories.append(TILE_CATEGORY_HARD_NEGATIVE)
        else:
            categories.append(TILE_CATEGORY_BACKGROUND)
    return categories


def _category_counts(categories: list[str]) -> dict[str, int]:
    return {
        TILE_CATEGORY_POSITIVE: sum(1 for item in categories if item == TILE_CATEGORY_POSITIVE),
        TILE_CATEGORY_HARD_NEGATIVE: sum(
            1 for item in categories if item == TILE_CATEGORY_HARD_NEGATIVE
        ),
        TILE_CATEGORY_BACKGROUND: sum(
            1 for item in categories if item == TILE_CATEGORY_BACKGROUND
        ),
    }


def _effective_sampling_factors(
    factors: dict[str, float],
    counts: dict[str, int],
    *,
    dataset_size: int,
) -> tuple[dict[str, float], list[str]]:
    positive_factor = factors[TILE_CATEGORY_POSITIVE]
    hard_negative_factor = factors[TILE_CATEGORY_HARD_NEGATIVE]
    background_factor = factors[TILE_CATEGORY_BACKGROUND]
    positive_count = counts[TILE_CATEGORY_POSITIVE]
    hard_negative_count = counts[TILE_CATEGORY_HARD_NEGATIVE]
    background_count = counts[TILE_CATEGORY_BACKGROUND]
    marked_factor = positive_factor + hard_negative_factor
    if background_factor > 0.0 and background_count == 0:
        raise TilePreparationError(
            "tile_preparation.background_factor > 0, но background tiles не найдены."
        )
    hard_negative_cap = (
        hard_negative_count / dataset_size
        if dataset_size > 0 and hard_negative_count > 0
        else 0.0
    )
    effective_hard_negative_factor = min(hard_negative_factor, hard_negative_cap)
    effective_positive_factor = marked_factor - effective_hard_negative_factor
    if effective_positive_factor > 0.0 and positive_count == 0:
        raise TilePreparationError(
            "tile_preparation positive+hard_negative budget > 0, "
            "но positive tiles для marked budget не найдены."
        )
    warnings: list[str] = []
    if hard_negative_factor > effective_hard_negative_factor:
        warnings.append(
            "hard_negative_factor_used уменьшен из-за отсутствия или малого числа "
            "hard_negative tiles; недостающий marked budget перенесен в positive."
        )
    return (
        {
            TILE_CATEGORY_POSITIVE: effective_positive_factor,
            TILE_CATEGORY_HARD_NEGATIVE: effective_hard_negative_factor,
            TILE_CATEGORY_BACKGROUND: background_factor,
        },
        warnings,
    )


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


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
