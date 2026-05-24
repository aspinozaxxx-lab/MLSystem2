"""Создание torch DataLoader для тайлов."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

from mlsystem2.settings.api import get_settings

from ._dataset import TileDataset
from .contracts import TileDataloaderRequest, TilePreparationError

if TYPE_CHECKING:
    import torch


def create_tile_dataloader(
    request: TileDataloaderRequest,
) -> "torch.utils.data.DataLoader":
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
            tile_size=tile_settings.tile_size,
            stride=tile_settings.stride,
            mode=request.mode,
            seed=tile_settings.seed,
            augmentation_level=tile_settings.augmentation_level,
            smart_tiling=tile_settings.smart_tiling,
        )
    except TilePreparationError:
        raise
    except Exception as exc:
        raise TilePreparationError("Не удалось подготовить Dataset тайлов") from exc

    generator = torch.Generator()
    generator.manual_seed(tile_settings.seed)

    sampler = None
    if request.mode == "train" and tile_settings.smart_tiling:
        weights = dataset.sampling_weights()
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
        dataloader_kwargs["prefetch_factor"] = tile_settings.prefetch_factor
        dataloader_kwargs["persistent_workers"] = True

    return DataLoader(**dataloader_kwargs)


def _collate_tile_batch(samples: list[tuple[np.ndarray, np.ndarray, dict[str, bool]]]):
    try:
        import torch
    except ImportError as exc:
        raise TilePreparationError("Для сборки batch требуется установленный PyTorch.") from exc

    images = torch.stack(
        [torch.as_tensor(sample[0], dtype=torch.float32) for sample in samples],
        dim=0,
    )
    masks = torch.stack(
        [torch.as_tensor(sample[1], dtype=torch.float32) for sample in samples],
        dim=0,
    )
    metas = [sample[2] if len(sample) > 2 else {} for sample in samples]
    tile_augmented = [bool(meta.get("augmented", False)) for meta in metas]
    tile_positive = [bool(meta.get("positive", False)) for meta in metas]
    batch_meta = {
        "augmented_tile_count": sum(1 for item in tile_augmented if item),
        "positive_tile_count": sum(1 for item in tile_positive if item),
        "tile_augmented": tile_augmented,
        "tile_positive": tile_positive,
    }
    return images, masks, batch_meta


def _seed_tile_worker(worker_id: int) -> None:
    del worker_id
    import torch

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None and hasattr(worker_info.dataset, "close"):
        worker_info.dataset.close()
