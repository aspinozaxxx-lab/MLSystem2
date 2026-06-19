"""Создание torch DataLoader для тайлов."""

from __future__ import annotations

import math
import os
import random

import numpy as np

from mlsystem2.settings.api import get_settings

from ._dataset import (
    TILE_CATEGORY_BACKGROUND,
    TILE_CATEGORY_HARD_NEGATIVE,
    TILE_CATEGORY_POSITIVE,
    TileDataset,
)
from .contracts import TileDataloaderRequest, TilePreparationError


VAL_CACHE_AVAILABLE_MEMORY_FRACTION = 0.5
VAL_CACHE_ESTIMATE_OVERHEAD = 1.25


def create_tile_dataloader(
    request: TileDataloaderRequest,
) -> object:
    try:
        import torch
        from torch.utils.data import DataLoader
        from torch.utils.data import WeightedRandomSampler
    except ImportError as exc:
        raise TilePreparationError(
            "Для создания tile DataLoader требуется установленный PyTorch."
        ) from exc

    tile_settings = get_settings().tile_preparation
    try:
        dataset = TileDataset(
            vrt_xml=request.vrt_xml,
            annotation_file=request.annotation_file,
            hard_negative_annotation_file=request.hard_negative_annotation_file,
            class_annotations=request.class_annotations,
            tile_size=tile_settings.tile_size,
            stride=tile_settings.stride,
            mode=request.mode,
            seed=tile_settings.seed,
            augmentation_level=tile_settings.augmentation_level,
            positive_factor=tile_settings.positive_factor,
            hard_negative_factor=tile_settings.hard_negative_factor,
            background_factor=tile_settings.background_factor,
            class_balance=tile_settings.class_balance,
            tile_split=request.tile_split,
        )
    except TilePreparationError:
        raise
    except Exception as exc:
        raise TilePreparationError("Не удалось подготовить Dataset тайлов") from exc

    if request.mode == "val":
        return _create_cached_val_loader(
            torch=torch,
            dataset=dataset,
            batch_size=request.batch_size,
            seed=tile_settings.seed,
        )

    generator = torch.Generator()
    generator.manual_seed(tile_settings.seed)

    sampler = None
    weights = dataset.sampling_weights(
        tile_settings.positive_factor,
        tile_settings.hard_negative_factor,
        tile_settings.background_factor,
    )
    if weights is not None:
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )

    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": request.batch_size,
        "shuffle": request.mode == "train" and sampler is None,
        "num_workers": tile_settings.num_workers,
        "collate_fn": _collate_tile_batch,
        "generator": generator,
        "worker_init_fn": _seed_tile_worker,
    }
    if sampler is not None:
        dataloader_kwargs["sampler"] = sampler
    if tile_settings.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = _effective_prefetch_factor(
            prefetch_epochs=tile_settings.prefetch_epochs,
            dataset_size=len(dataset),
            batch_size=request.batch_size,
            num_workers=tile_settings.num_workers,
            max_batches_per_epoch=request.max_batches_per_epoch,
        )
        dataloader_kwargs["persistent_workers"] = True

    return DataLoader(**dataloader_kwargs)


class _CachedValLoader:
    def __init__(
        self,
        *,
        torch,
        dataset: TileDataset,
        batch_size: int,
        indices: list[int],
    ) -> None:
        self.dataset = dataset
        self.sampler = None
        self.batch_size = batch_size
        self.cache_mode = "memory"
        self.cached_tiles = len(indices)
        self._indices = list(indices)
        try:
            self._batches = self._build_batches(torch)
            self.cached_batches = len(self._batches)
        finally:
            self.dataset.close()

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)

    def _build_batches(self, torch) -> list[tuple[object, object, dict[str, object]]]:
        del torch
        batches: list[tuple[object, object, dict[str, object]]] = []
        samples: list[tuple[np.ndarray, np.ndarray, dict[str, object]]] = []
        try:
            for index in self._indices:
                samples.append(self.dataset[index])
                if len(samples) == self.batch_size:
                    batches.append(_collate_tile_batch(samples))
                    samples = []
            if samples:
                batches.append(_collate_tile_batch(samples))
        except MemoryError as exc:
            raise TilePreparationError("Val tile cache не поместился в RAM.") from exc
        return batches


def _create_cached_val_loader(
    *,
    torch,
    dataset: TileDataset,
    batch_size: int,
    seed: int,
) -> _CachedValLoader:
    indices = _balanced_val_indices(dataset, seed=seed)
    _ensure_val_cache_fits_memory(dataset, tile_count=len(indices))
    return _CachedValLoader(
        torch=torch,
        dataset=dataset,
        batch_size=batch_size,
        indices=indices,
    )


def _balanced_val_indices(dataset: TileDataset, *, seed: int) -> list[int]:
    hints = dataset.positive_hints
    if hints is None:
        raise TilePreparationError("Val balanced cache требует positive/negative hints.")
    positive_indices = [index for index, positive in enumerate(hints) if positive]
    negative_indices = [index for index, positive in enumerate(hints) if not positive]
    if not positive_indices or not negative_indices:
        raise TilePreparationError(
            "Val balanced cache требует и positive, и negative tiles "
            f"после tile_split: positive={len(positive_indices)}, negative={len(negative_indices)}."
        )

    rng = random.Random(seed)
    rng.shuffle(positive_indices)
    rng.shuffle(negative_indices)
    group_size = min(len(positive_indices), len(negative_indices))
    balanced_indices: list[int] = []
    for positive_index, negative_index in zip(
        positive_indices[:group_size],
        negative_indices[:group_size],
    ):
        balanced_indices.append(positive_index)
        balanced_indices.append(negative_index)
    return balanced_indices


def _ensure_val_cache_fits_memory(dataset: TileDataset, *, tile_count: int) -> None:
    required_bytes = _estimate_val_cache_bytes(dataset, tile_count=tile_count)
    available_bytes = _available_memory_bytes()
    if available_bytes is None:
        return
    allowed_bytes = int(available_bytes * VAL_CACHE_AVAILABLE_MEMORY_FRACTION)
    if required_bytes > allowed_bytes:
        raise TilePreparationError(
            "Val tile cache не помещается в RAM: "
            f"требуется около {_format_bytes(required_bytes)}, "
            f"доступный безопасный лимит {_format_bytes(allowed_bytes)}."
        )


def _estimate_val_cache_bytes(dataset: TileDataset, *, tile_count: int) -> int:
    tile_pixels = dataset.tile_size * dataset.tile_size
    image_bytes = tile_count * dataset.channel_count * tile_pixels * np.dtype(np.float32).itemsize
    mask_dtype = np.dtype(np.int64) if dataset.uses_multiclass_masks else np.dtype(np.float32)
    mask_bytes = tile_count * tile_pixels * mask_dtype.itemsize
    return int((image_bytes + mask_bytes) * VAL_CACHE_ESTIMATE_OVERHEAD)


def _available_memory_bytes() -> int | None:
    if os.name == "nt":
        return _windows_available_memory_bytes()
    mem_available_bytes = _linux_mem_available_bytes()
    if mem_available_bytes is not None:
        return mem_available_bytes
    return _sysconf_available_memory_bytes()


def _linux_mem_available_bytes(meminfo_path: str = "/proc/meminfo") -> int | None:
    try:
        with open(meminfo_path, encoding="utf-8") as meminfo:
            for line in meminfo:
                if not line.startswith("MemAvailable:"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    return None
                return int(parts[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _sysconf_available_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return int(page_size * available_pages)


def _windows_available_memory_bytes() -> int | None:
    try:
        import ctypes
    except ImportError:
        return None

    class _MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullAvailPhys)


def _format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} TiB"


def _effective_prefetch_factor(
    *,
    prefetch_epochs: float,
    dataset_size: int,
    batch_size: int,
    num_workers: int,
    max_batches_per_epoch: int | None = None,
) -> int:
    if num_workers <= 0 or dataset_size <= 0:
        return 1
    batches_per_epoch = math.ceil(dataset_size / batch_size)
    if max_batches_per_epoch is not None:
        batches_per_epoch = min(batches_per_epoch, max_batches_per_epoch)
    target_prefetch_batches = math.ceil(batches_per_epoch * prefetch_epochs)
    return max(1, math.ceil(target_prefetch_batches / num_workers))


def _collate_tile_batch(samples: list[tuple[np.ndarray, np.ndarray, dict[str, object]]]):
    try:
        import torch
    except ImportError as exc:
        raise TilePreparationError("Для сборки batch требуется установленный PyTorch.") from exc

    images = torch.stack(
        [torch.as_tensor(sample[0], dtype=torch.float32) for sample in samples],
        dim=0,
    )
    masks = _collate_masks(torch, samples)
    metas = [sample[2] if len(sample) > 2 else {} for sample in samples]
    tile_augmented = [bool(meta.get("augmented", False)) for meta in metas]
    tile_categories = [
        str(meta.get("category") or _legacy_category(meta))
        for meta in metas
    ]
    tile_positive = [category == TILE_CATEGORY_POSITIVE for category in tile_categories]
    tile_hard_negative = [
        category == TILE_CATEGORY_HARD_NEGATIVE for category in tile_categories
    ]
    tile_background = [category == TILE_CATEGORY_BACKGROUND for category in tile_categories]
    augmented_positive = [
        augmented and positive
        for augmented, positive in zip(tile_augmented, tile_positive)
    ]
    augmented_hard_negative = [
        augmented and hard_negative
        for augmented, hard_negative in zip(tile_augmented, tile_hard_negative)
    ]
    batch_meta = {
        "augmented_tile_count": sum(1 for item in tile_augmented if item),
        "positive_tile_count": sum(1 for item in tile_positive if item),
        "hard_negative_tile_count": sum(1 for item in tile_hard_negative if item),
        "background_tile_count": sum(1 for item in tile_background if item),
        "augmented_positive_tile_count": sum(1 for item in augmented_positive if item),
        "augmented_hard_negative_tile_count": sum(
            1 for item in augmented_hard_negative if item
        ),
        "tile_augmented": tile_augmented,
        "tile_positive": tile_positive,
        "tile_hard_negative": tile_hard_negative,
        "tile_background": tile_background,
        "tile_category": tile_categories,
    }
    return images, masks, batch_meta


def _collate_masks(torch, samples: list[tuple[np.ndarray, np.ndarray, dict[str, object]]]):
    first_mask = samples[0][1]
    dtype = torch.long if first_mask.ndim == 2 else torch.float32
    return torch.stack(
        [torch.as_tensor(sample[1], dtype=dtype) for sample in samples],
        dim=0,
    )


def _legacy_category(meta: dict[str, object]) -> str:
    if bool(meta.get("positive", False)):
        return TILE_CATEGORY_POSITIVE
    if bool(meta.get("hard_negative", False)):
        return TILE_CATEGORY_HARD_NEGATIVE
    return TILE_CATEGORY_BACKGROUND


def _seed_tile_worker(worker_id: int) -> None:
    del worker_id
    import torch

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None and hasattr(worker_info.dataset, "close"):
        worker_info.dataset.close()
