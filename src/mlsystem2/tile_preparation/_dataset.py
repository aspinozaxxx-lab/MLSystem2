"""Dataset тайлов по независимым TIFF и GeoJSON."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.windows import Window

from ._annotations import AnnotationIndex, load_annotation_index
from ._augmentations import apply_augmentations
from ._mask import HARD_NEGATIVE_LABEL, build_supervision_mask, rasterize_instance_mask
from ._valid_footprint import filter_valid_windows
from ._windows import TileWindow, build_tile_windows
from .contracts import (
    TileClassAnnotation,
    TilePreparationError,
    TileSceneSource,
    TileSplitRequest,
)


TILE_CATEGORY_POSITIVE = "positive"
TILE_CATEGORY_HARD_NEGATIVE = "hard_negative"
TILE_CATEGORY_BACKGROUND = "background"
_MAX_OPEN_RASTERS = 8


@dataclass(frozen=True, slots=True)
class _SceneTileWindow:
    scene_index: int
    scene_id: str
    window: TileWindow


class TileDataset:
    def __init__(
        self,
        *,
        scenes: list[TileSceneSource],
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
        include_object_instances: bool = False,
    ) -> None:
        if not scenes:
            raise TilePreparationError("Не задано ни одного TIFF для нарезки тайлов.")
        self._scenes = list(scenes)
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
        self._include_object_instances = include_object_instances
        self._tile_split_warnings: list[str] = []
        self._pool_window_count = 0
        self._split_window_count = 0
        self._datasets: OrderedDict[int, DatasetReader] = OrderedDict()
        self._annotation_indexes: dict[tuple[Path, str | None, str | None], AnnotationIndex] = {}
        self._positive_hint_by_index: list[bool] | None = None
        self._hard_negative_hint_by_index: list[bool] | None = None
        self._category_hint_by_index: list[str] | None = None
        self._class_hints_by_index: list[frozenset[int]] | None = None
        self._scene_nodata: list[object] = []
        self._scene_crs: list[str | None] = []
        self._windows: list[_SceneTileWindow] = []
        self._candidate_window_count_before_valid_filter = 0
        self._black_filtered_window_count = 0
        self._valid_footprint_stride = 64
        self._valid_footprint_valid_cells = 0
        self._valid_footprint_total_cells = 0
        self._count: int | None = None

        for scene_index, scene in enumerate(self._scenes):
            image_path = Path(scene.image_path)
            try:
                with rasterio.open(image_path) as dataset:
                    if self._count is None:
                        self._count = dataset.count
                    elif dataset.count != self._count:
                        raise TilePreparationError(
                            "Количество каналов TIFF различается: "
                            f"ожидалось {self._count}, получено {dataset.count}: {image_path}"
                        )
                    nodata = _resolve_nodata(dataset)
                    self._scene_nodata.append(nodata)
                    self._scene_crs.append(
                        dataset.crs.to_string() if dataset.crs is not None else None
                    )
                    candidate_windows = build_tile_windows(
                        dataset.width,
                        dataset.height,
                        tile_size,
                        stride,
                    )
                    valid_windows, diagnostics = filter_valid_windows(
                        dataset,
                        candidate_windows,
                        nodata=nodata,
                    )
            except TilePreparationError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise TilePreparationError(f"Не удалось открыть TIFF: {image_path}") from exc

            self._windows.extend(
                _SceneTileWindow(
                    scene_index=scene_index,
                    scene_id=scene.scene_id,
                    window=window,
                )
                for window in valid_windows
            )
            self._candidate_window_count_before_valid_filter += (
                diagnostics.candidate_window_count_before_valid_filter
            )
            self._black_filtered_window_count += diagnostics.black_filtered_window_count
            self._valid_footprint_valid_cells += diagnostics.valid_footprint_valid_cells
            self._valid_footprint_total_cells += diagnostics.valid_footprint_total_cells

        if self._count is None:
            raise TilePreparationError("Не удалось определить число каналов TIFF.")
        self._candidate_window_count = len(self._windows)
        should_build_hints = self._mode in {"train", "val"} or self._tile_split is not None
        if should_build_hints:
            self._build_hints()
        self._pool_window_count = len(self._windows)
        if self._tile_split is not None:
            self._apply_tile_split(self._tile_split)
        self._split_window_count = len(self._windows)
        self._close_datasets()

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        scene_window = self._windows[index]
        dataset = self._open_dataset(scene_window.scene_index)
        tile_window = scene_window.window
        window = Window(tile_window.x, tile_window.y, tile_window.width, tile_window.height)
        nodata = self._scene_nodata[scene_window.scene_index]

        image_raw = self._read_image_raw(dataset, window, nodata)
        nodata_pixels = np.logical_or(
            _nodata_pixels(image_raw, nodata),
            self._read_invalid_data_pixels(dataset, window),
        )
        image = image_raw.astype(np.float32, copy=False)
        image[:, nodata_pixels] = nodata
        mask = self._read_supervision_mask(
            scene_window.scene_index,
            dataset,
            window,
            nodata_pixels,
        )
        category = _tile_category_from_supervision_mask(mask)
        augmented = False
        should_augment = self._mode == "train" and self._augmentation_level > 0
        if should_augment and category in {
            TILE_CATEGORY_POSITIVE,
            TILE_CATEGORY_HARD_NEGATIVE,
        }:
            image, mask, augmented = apply_augmentations(
                image,
                mask,
                nodata_pixels=nodata_pixels,
                nodata=nodata,
                level=self._augmentation_level,
                seed=self._seed,
                sample_index=index,
            )

        object_instances = (
            self._read_object_instances(
                scene_window.scene_index,
                dataset,
                window,
                nodata_pixels,
            )
            if self._include_object_instances
            else None
        )
        meta = self._sample_meta(category, augmented, object_instances)
        return np.ascontiguousarray(image), np.ascontiguousarray(mask), meta

    @property
    def channel_count(self) -> int:
        return int(self._count)

    @property
    def tile_size(self) -> int:
        return self._tile_size

    @property
    def scene_count(self) -> int:
        return len(self._scenes)

    @property
    def positive_hints(self) -> list[bool] | None:
        return None if self._positive_hint_by_index is None else list(self._positive_hint_by_index)

    @property
    def hard_negative_hints(self) -> list[bool] | None:
        return (
            None
            if self._hard_negative_hint_by_index is None
            else list(self._hard_negative_hint_by_index)
        )

    @property
    def tile_categories(self) -> list[str] | None:
        categories = self._category_hints()
        return None if categories is None else list(categories)

    @property
    def uses_multiclass_masks(self) -> bool:
        return bool(self._class_annotations)

    @property
    def includes_object_instances(self) -> bool:
        return self._include_object_instances

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
        return self._factor_used(TILE_CATEGORY_POSITIVE)

    @property
    def hard_negative_factor_used(self) -> float | None:
        return self._factor_used(TILE_CATEGORY_HARD_NEGATIVE)

    @property
    def background_factor_used(self) -> float | None:
        return self._factor_used(TILE_CATEGORY_BACKGROUND)

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
                warnings.append(
                    f"class_balance: для класса {slug} найдено мало positive windows: {count}."
                )
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
                self._positive_factor if positive_factor is None else positive_factor
            ),
            TILE_CATEGORY_HARD_NEGATIVE: (
                self._hard_negative_factor
                if hard_negative_factor is None
                else hard_negative_factor
            ),
            TILE_CATEGORY_BACKGROUND: (
                self._background_factor if background_factor is None else background_factor
            ),
        }
        counts = _category_counts(categories)
        factors, warnings = _effective_sampling_factors(factors, counts)
        self._sampling_factor_used = factors
        self._sampling_warnings = warnings
        positive_count = counts[TILE_CATEGORY_POSITIVE]
        hard_negative_count = counts[TILE_CATEGORY_HARD_NEGATIVE]
        background_count = counts[TILE_CATEGORY_BACKGROUND]
        positive_weights: list[float] | None = None
        if self.class_balance_enabled and self._class_hints_by_index is not None:
            positive_weights = self._class_balanced_positive_weights(
                factors[TILE_CATEGORY_POSITIVE]
            )
        positive_weight = (
            factors[TILE_CATEGORY_POSITIVE] / positive_count if positive_count else 0.0
        )
        hard_negative_weight = (
            factors[TILE_CATEGORY_HARD_NEGATIVE] / hard_negative_count
            if hard_negative_count
            else 0.0
        )
        background_weight = (
            factors[TILE_CATEGORY_BACKGROUND] / background_count if background_count else 0.0
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
        self._close_datasets()

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_datasets"] = OrderedDict()
        state["_annotation_indexes"] = {}
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _open_dataset(self, scene_index: int) -> DatasetReader:
        dataset = self._datasets.pop(scene_index, None)
        if dataset is None:
            dataset = rasterio.open(Path(self._scenes[scene_index].image_path))
        self._datasets[scene_index] = dataset
        while len(self._datasets) > _MAX_OPEN_RASTERS:
            _, stale = self._datasets.popitem(last=False)
            stale.close()
        return dataset

    def _close_datasets(self) -> None:
        for dataset in self._datasets.values():
            dataset.close()
        self._datasets.clear()

    def _annotation_index(
        self,
        annotation_file: str | Path,
        scene_index: int,
        *,
        role: str | None = None,
    ) -> AnnotationIndex:
        path = Path(annotation_file)
        crs = self._scene_crs[scene_index]
        key = (path, crs, role)
        index = self._annotation_indexes.get(key)
        if index is None:
            if role == "positive":
                index = load_annotation_index(path, crs, role="positive")
            elif role == "hard_negative":
                index = load_annotation_index(path, crs, role="hard_negative")
            else:
                index = load_annotation_index(path, crs)
            self._annotation_indexes[key] = index
        return index

    def _read_image_raw(
        self,
        dataset: DatasetReader,
        window: Window,
        nodata: object,
    ) -> np.ndarray:
        return dataset.read(
            window=window,
            boundless=True,
            fill_value=nodata,
            out_shape=(self.channel_count, self._tile_size, self._tile_size),
            masked=False,
        )

    def _read_invalid_data_pixels(
        self,
        dataset: DatasetReader,
        window: Window,
    ) -> np.ndarray:
        valid_mask = dataset.dataset_mask(
            window=window,
            boundless=True,
            out_shape=(self._tile_size, self._tile_size),
        )
        return valid_mask == 0

    def _read_supervision_mask(
        self,
        scene_index: int,
        dataset: DatasetReader,
        window: Window,
        nodata_pixels: np.ndarray,
    ) -> np.ndarray:
        bounds = dataset.window_bounds(window)
        mask = build_supervision_mask(
            positive_layers=self._positive_layers(scene_index, bounds),
            hard_negative_geometries=self._hard_negative_geometries(
                scene_index,
                bounds,
            ),
            out_shape=(self._tile_size, self._tile_size),
            transform=dataset.window_transform(window),
            nodata_pixels=nodata_pixels,
        )
        if self._class_annotations:
            return mask.astype(np.int64, copy=False)
        return mask.astype(np.float32, copy=False)[None, :, :]

    def _read_object_instances(
        self,
        scene_index: int,
        dataset: DatasetReader,
        window: Window,
        nodata_pixels: np.ndarray,
    ) -> np.ndarray:
        bounds = dataset.window_bounds(window)
        geometries = self._positive_index(scene_index).query_bounds(bounds)
        return rasterize_instance_mask(
            geometries,
            out_shape=(self._tile_size, self._tile_size),
            transform=dataset.window_transform(window),
            nodata_pixels=nodata_pixels,
        )

    def _positive_index(self, scene_index: int) -> AnnotationIndex:
        scene = self._scenes[scene_index]
        if scene.annotation_file is not None:
            return self._annotation_index(
                scene.annotation_file,
                scene_index,
                role="positive",
            )
        if self._annotation_file is None:
            raise TilePreparationError("Для binary mask не задан annotation_file.")
        return self._annotation_index(self._annotation_file, scene_index)

    def _positive_layers(
        self,
        scene_index: int,
        bounds: tuple[float, float, float, float],
    ) -> list[tuple[int, list[object]]]:
        if self._class_annotations:
            return [
                (
                    annotation.class_id,
                    self._annotation_index(
                        annotation.annotation_file,
                        scene_index,
                    ).query_bounds(bounds),
                )
                for annotation in self._class_annotations
            ]
        return [(1, self._positive_index(scene_index).query_bounds(bounds))]

    def _hard_negative_geometries(
        self,
        scene_index: int,
        bounds: tuple[float, float, float, float],
    ) -> list[object]:
        scene = self._scenes[scene_index]
        if scene.annotation_file is not None:
            return self._annotation_index(
                scene.annotation_file,
                scene_index,
                role="hard_negative",
            ).query_bounds(bounds)
        geometries: list[object] = []
        if self._hard_negative_annotation_file is not None:
            geometries.extend(
                self._annotation_index(
                    self._hard_negative_annotation_file,
                    scene_index,
                ).query_bounds(bounds)
            )
        for annotation in self._class_annotations:
            if annotation.hard_negative_annotation_file is None:
                continue
            geometries.extend(
                self._annotation_index(
                    annotation.hard_negative_annotation_file,
                    scene_index,
                ).query_bounds(bounds)
            )
        return geometries

    def _build_hints(self) -> None:
        positive_hints: list[bool] = []
        hard_negative_hints: list[bool] = []
        class_hints: list[frozenset[int]] | None = [] if self._class_annotations else None
        for scene_window in self._windows:
            scene_index = scene_window.scene_index
            dataset = self._open_dataset(scene_index)
            item = scene_window.window
            window = Window(item.x, item.y, item.width, item.height)
            bounds = dataset.window_bounds(window)
            if self._class_annotations:
                class_ids = {
                    annotation.class_id
                    for annotation in self._class_annotations
                    if self._annotation_index(
                        annotation.annotation_file,
                        scene_index,
                    ).query_bounds(bounds)
                }
                assert class_hints is not None
                class_hints.append(frozenset(class_ids))
                positive_hints.append(bool(class_ids))
            else:
                positive_hints.append(
                    bool(self._positive_index(scene_index).query_bounds(bounds))
                )
            hard_negative_hints.append(
                bool(self._hard_negative_geometries(scene_index, bounds))
            )
        self._positive_hint_by_index = positive_hints
        self._hard_negative_hint_by_index = hard_negative_hints
        self._category_hint_by_index = _tile_categories(
            positive_hints,
            hard_negative_hints,
        )
        self._class_hints_by_index = class_hints

    def _apply_tile_split(self, tile_split: TileSplitRequest) -> None:
        if self._positive_hint_by_index is None:
            return
        train_indices, val_indices, warnings = _split_tile_indices(
            self._positive_hint_by_index,
            self._windows,
            val_fraction=tile_split.val_fraction,
            seed=tile_split.seed,
        )
        selected_indices = train_indices if self._mode == "train" else val_indices
        if not any(self._positive_hint_by_index[index] for index in selected_indices):
            warnings.append(f"tile_split: subset {self._mode} не содержит positive windows.")
        if not selected_indices:
            warnings.append(f"tile_split: subset {self._mode} пуст.")
        self._select_indices(selected_indices)
        self._tile_split_warnings = warnings

    def _select_indices(self, selected_indices: list[int]) -> None:
        self._windows = [self._windows[index] for index in selected_indices]
        if self._positive_hint_by_index is not None:
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

    def _category_hints(self) -> list[str] | None:
        cached = getattr(self, "_category_hint_by_index", None)
        if cached is not None:
            return cached
        if self._positive_hint_by_index is None:
            return None
        hard_negative = getattr(self, "_hard_negative_hint_by_index", None) or [
            False for _ in self._positive_hint_by_index
        ]
        return _tile_categories(self._positive_hint_by_index, hard_negative)

    def _factor_used(self, category: str) -> float | None:
        if self._sampling_factor_used is None:
            return None
        return self._sampling_factor_used[category]

    def _class_balanced_positive_weights(self, factor: float) -> list[float] | None:
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
            weights.append(
                sum(
                    class_budget / counts_by_id[class_id]
                    for class_id in hints
                    if counts_by_id.get(class_id, 0) > 0
                )
            )
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

    def _sample_meta(
        self,
        category: str,
        augmented: bool,
        object_instances: np.ndarray | None = None,
    ) -> dict[str, object]:
        meta: dict[str, object] = {
            "augmented": augmented,
            "category": category,
            "positive": category == TILE_CATEGORY_POSITIVE,
            "hard_negative": category == TILE_CATEGORY_HARD_NEGATIVE,
            "background": category == TILE_CATEGORY_BACKGROUND,
        }
        if object_instances is not None:
            meta["object_instances"] = np.ascontiguousarray(object_instances)
        return meta


def _split_tile_indices(
    positive_hints: list[bool],
    windows: list[_SceneTileWindow],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[str]]:
    positive_indices = [index for index, positive in enumerate(positive_hints) if positive]
    negative_indices = [index for index, positive in enumerate(positive_hints) if not positive]
    train_positive, val_positive, warnings = _split_index_group(
        positive_indices,
        windows,
        val_fraction=val_fraction,
        seed=seed,
        group_name="positive",
    )
    train_negative, val_negative, negative_warnings = _split_index_group(
        negative_indices,
        windows,
        val_fraction=val_fraction,
        seed=seed,
        group_name="negative",
    )
    return (
        sorted([*train_positive, *train_negative]),
        sorted([*val_positive, *val_negative]),
        [*warnings, *negative_warnings],
    )


def _split_index_group(
    indices: list[int],
    windows: list[_SceneTileWindow],
    *,
    val_fraction: float,
    seed: int,
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

    ordered = sorted(indices, key=lambda index: _window_split_key(windows[index], seed))
    val_count = min(len(indices) - 1, max(1, int(round(len(indices) * val_fraction))))
    val_set = set(ordered[:val_count])
    return (
        [index for index in indices if index not in val_set],
        [index for index in indices if index in val_set],
        [],
    )


def _window_split_key(window: _SceneTileWindow, seed: int) -> bytes:
    value = f"{seed}\0{window.scene_id}\0{window.window.x}\0{window.window.y}"
    return hashlib.sha256(value.encode("utf-8")).digest()


def _tile_categories(
    positive_hints: list[bool],
    hard_negative_hints: list[bool],
) -> list[str]:
    categories: list[str] = []
    for positive, hard_negative in zip(positive_hints, hard_negative_hints, strict=True):
        if positive:
            categories.append(TILE_CATEGORY_POSITIVE)
        elif hard_negative:
            categories.append(TILE_CATEGORY_HARD_NEGATIVE)
        else:
            categories.append(TILE_CATEGORY_BACKGROUND)
    return categories


def _tile_category_from_supervision_mask(mask: np.ndarray) -> str:
    if bool(np.any(mask > 0)):
        return TILE_CATEGORY_POSITIVE
    if bool(np.any(mask == HARD_NEGATIVE_LABEL)):
        return TILE_CATEGORY_HARD_NEGATIVE
    return TILE_CATEGORY_BACKGROUND


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
    marked_count = positive_count + hard_negative_count
    hard_negative_cap = (
        marked_factor * hard_negative_count / marked_count if marked_count > 0 else 0.0
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
            "hard_negative tiles относительно marked tiles; недостающий marked budget "
            "перенесен в positive."
        )
    return (
        {
            TILE_CATEGORY_POSITIVE: effective_positive_factor,
            TILE_CATEGORY_HARD_NEGATIVE: effective_hard_negative_factor,
            TILE_CATEGORY_BACKGROUND: background_factor,
        },
        warnings,
    )


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
