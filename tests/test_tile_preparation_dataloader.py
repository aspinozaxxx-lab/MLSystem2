from __future__ import annotations

import json
import re
import sys
import builtins
from time import perf_counter
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_origin

from mlsystem2.settings.api import load_settings
from mlsystem2.tile_preparation.api import create_tile_dataloader
from mlsystem2.tile_preparation._augmentations import _geometric
from mlsystem2.tile_preparation import _dataloader as dataloader_impl
from mlsystem2.tile_preparation._dataloader import (
    _available_memory_bytes,
    _effective_prefetch_factor,
    _ensure_val_cache_fits_memory,
    _linux_mem_available_bytes,
)
from mlsystem2.tile_preparation._dataset import (
    TILE_CATEGORY_BACKGROUND,
    TILE_CATEGORY_HARD_NEGATIVE,
    TILE_CATEGORY_POSITIVE,
    TileDataset,
)
from mlsystem2.tile_preparation.contracts import (
    TileClassAnnotation,
    TileDataloaderRequest,
    TilePreparationError,
    TileSplitRequest,
)


def test_linux_mem_available_bytes_reads_memavailable(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "\n".join(
            [
                "MemTotal:       129414060 kB",
                "MemFree:          9700572 kB",
                "MemAvailable:   104127980 kB",
            ]
        ),
        encoding="utf-8",
    )

    assert _linux_mem_available_bytes(str(meminfo)) == 104127980 * 1024


def test_available_memory_bytes_prefers_linux_memavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dataloader_impl.os, "name", "posix", raising=False)
    monkeypatch.setattr(dataloader_impl, "_linux_mem_available_bytes", lambda: 99)
    monkeypatch.setattr(dataloader_impl, "_sysconf_available_memory_bytes", lambda: 9)

    assert _available_memory_bytes() == 99


def test_available_memory_bytes_falls_back_to_sysconf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dataloader_impl.os, "name", "posix", raising=False)
    monkeypatch.setattr(dataloader_impl, "_linux_mem_available_bytes", lambda: None)
    monkeypatch.setattr(dataloader_impl, "_sysconf_available_memory_bytes", lambda: 9)

    assert _available_memory_bytes() == 9


def test_val_cache_memory_check_uses_memavailable_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Dataset:
        tile_size = 768
        channel_count = 4
        uses_multiclass_masks = False

    monkeypatch.setattr(dataloader_impl, "_available_memory_bytes", lambda: 99 * 1024**3)

    _ensure_val_cache_fits_memory(Dataset(), tile_count=1004)


def test_create_tile_dataloader_reports_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch заблокирован тестом")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(TilePreparationError, match="PyTorch"):
        create_tile_dataloader(
            TileDataloaderRequest(
                vrt_xml="",
                annotation_file="annotations.geojson",
                batch_size=1,
                mode="val",
            )
        )


def test_create_tile_dataloader_returns_image_mask_meta_tuple(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    raster_path = tmp_path / "image.tif"
    data = np.full((1, 4, 8), 1000, dtype=np.uint16)
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation_height4(annotation_file)
    load_settings(_write_config(tmp_path, tile_size=4, stride=4, batch_size=2, input_channels=1))

    loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            annotation_file=annotation_file,
            batch_size=2,
            mode="val",
        )
    )

    batch = next(iter(loader))
    assert isinstance(batch, tuple)
    assert len(batch) == 3
    images, masks, batch_meta = batch
    assert images.shape == (2, 1, 4, 4)
    assert masks.shape == (2, 1, 4, 4)
    assert batch_meta["augmented_tile_count"] == 0
    assert batch_meta["augmented_positive_tile_count"] == 0
    assert batch_meta["augmented_hard_negative_tile_count"] == 0
    assert batch_meta["tile_augmented"] == [False, False]
    assert len(batch_meta["tile_positive"]) == 2
    assert batch_meta["positive_tile_count"] == sum(batch_meta["tile_positive"])
    assert batch_meta["positive_tile_count"] == 1
    assert batch_meta["hard_negative_tile_count"] == 0
    assert batch_meta["background_tile_count"] == 1
    assert batch_meta["tile_category"] == [TILE_CATEGORY_POSITIVE, TILE_CATEGORY_BACKGROUND]
    assert images.dtype == torch.float32
    assert masks.dtype == torch.float32
    assert set(torch.unique(masks).tolist()) <= {0.0, 1.0}
    positive_by_mask = torch.count_nonzero(masks.flatten(1).sum(dim=1) > 0).item()
    assert positive_by_mask == batch_meta["positive_tile_count"]

    loader.dataset.close()


def test_create_tile_dataloader_keeps_raw_integer_values_and_chw_layout(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    raster_path = tmp_path / "uint16.tif"
    data = np.stack(
        [
            np.full((4, 4), 1000, dtype=np.uint16),
            np.full((4, 4), 2000, dtype=np.uint16),
        ]
    )
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "empty.geojson"
    _write_empty_annotation(annotation_file)
    load_settings(
        _write_config(
            tmp_path,
            tile_size=4,
            stride=4,
            batch_size=1,
            input_channels=2,
            positive_factor=0.0,
            background_factor=1.0,
        )
    )

    loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            annotation_file=annotation_file,
            batch_size=1,
            mode="train",
        )
    )

    images, masks, batch_meta = next(iter(loader))
    assert images.shape == (1, 2, 4, 4)
    assert masks.shape == (1, 1, 4, 4)
    assert batch_meta == {
        "augmented_tile_count": 0,
        "positive_tile_count": 0,
        "hard_negative_tile_count": 0,
        "background_tile_count": 1,
        "augmented_positive_tile_count": 0,
        "augmented_hard_negative_tile_count": 0,
        "tile_augmented": [False],
        "tile_positive": [False],
        "tile_hard_negative": [False],
        "tile_background": [True],
        "tile_category": [TILE_CATEGORY_BACKGROUND],
    }
    assert images.dtype == torch.float32
    assert torch.equal(images[0], torch.as_tensor(data.astype(np.float32)))
    assert float(images.max().item()) == 2000.0

    loader.dataset.close()


def test_train_photometric_augmentation_keeps_raw_value_scale(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    raster_path = tmp_path / "raw_aug.tif"
    data = np.full((1, 4, 4), 1000, dtype=np.uint16)
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation_height4(annotation_file)
    load_settings(
        _write_config(
            tmp_path,
            tile_size=4,
            stride=4,
            batch_size=1,
            input_channels=1,
            augmentation_level=2,
            positive_factor=1.0,
            background_factor=0.0,
        )
    )

    loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            annotation_file=annotation_file,
            batch_size=1,
            mode="train",
        )
    )

    images, _masks, batch_meta = next(iter(loader))
    assert float(images.max().item()) > 1.0
    assert float(images.min().item()) >= 0.0
    assert float(images.max().item()) <= 255.0
    assert batch_meta == {
        "augmented_tile_count": 1,
        "positive_tile_count": 1,
        "hard_negative_tile_count": 0,
        "background_tile_count": 0,
        "augmented_positive_tile_count": 1,
        "augmented_hard_negative_tile_count": 0,
        "tile_augmented": [True],
        "tile_positive": [True],
        "tile_hard_negative": [False],
        "tile_background": [False],
        "tile_category": [TILE_CATEGORY_POSITIVE],
    }

    loader.dataset.close()


def test_multiclass_geometric_augmentation_keeps_labels() -> None:
    image = np.ones((1, 4, 4), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.int64)
    mask[0:2, 0:2] = 1
    mask[2:4, 2:4] = 2

    _image, augmented_mask, augmented = _geometric(image, mask, _DeterministicRng())

    assert augmented is True
    assert augmented_mask.shape == (4, 4)
    assert set(np.unique(augmented_mask).tolist()) == {0, 1, 2}


def test_effective_prefetch_factor_targets_requested_epochs() -> None:
    assert (
        _effective_prefetch_factor(
            prefetch_epochs=2.0,
            dataset_size=1301,
            batch_size=8,
            num_workers=16,
        )
        == 21
    )


def test_effective_prefetch_factor_uses_requested_epochs_only() -> None:
    assert (
        _effective_prefetch_factor(
            prefetch_epochs=0.25,
            dataset_size=32,
            batch_size=8,
            num_workers=16,
        )
        == 1
    )


def test_effective_prefetch_factor_respects_epoch_batch_limit() -> None:
    assert (
        _effective_prefetch_factor(
            prefetch_epochs=2.0,
            dataset_size=4549,
            batch_size=4,
            num_workers=16,
            max_batches_per_epoch=128,
        )
        == 16
    )


def test_create_tile_dataloader_returns_multiclass_long_mask(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    raster_path = tmp_path / "multiclass.tif"
    data = np.full((1, 4, 4), 1000, dtype=np.uint16)
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    class_a = tmp_path / "class_a.geojson"
    class_b = tmp_path / "class_b.geojson"
    _write_annotation_polygon(class_a, [[0, 3], [1, 3], [1, 4], [0, 4], [0, 3]])
    _write_annotation_polygon(class_b, [[2, 1], [3, 1], [3, 2], [2, 2], [2, 1]])
    load_settings(
        _write_config(
            tmp_path,
            tile_size=4,
            stride=4,
            batch_size=1,
            input_channels=1,
            positive_factor=1.0,
            background_factor=0.0,
        )
    )

    loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            class_annotations=[
                TileClassAnnotation(
                    class_id=1,
                    slug="class_a",
                    name="Класс А",
                    annotation_file=class_a,
                ),
                TileClassAnnotation(
                    class_id=2,
                    slug="class_b",
                    name="Класс Б",
                    annotation_file=class_b,
                ),
            ],
            batch_size=1,
            mode="train",
        )
    )

    images, masks, batch_meta = next(iter(loader))
    assert images.shape == (1, 1, 4, 4)
    assert masks.shape == (1, 4, 4)
    assert masks.dtype == torch.long
    assert set(torch.unique(masks).tolist()) == {0, 1, 2}
    assert batch_meta["positive_tile_count"] == 1
    assert batch_meta["hard_negative_tile_count"] == 0
    assert batch_meta["background_tile_count"] == 0
    assert batch_meta["tile_category"] == [TILE_CATEGORY_POSITIVE]
    loader.dataset.close()


def test_create_tile_dataloader_resolves_multiclass_overlap_by_priority(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    raster_path = tmp_path / "overlap.tif"
    data = np.full((1, 4, 4), 1000, dtype=np.uint16)
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    class_a = tmp_path / "overlap_a.geojson"
    class_b = tmp_path / "overlap_b.geojson"
    polygon = [[0, 3], [1, 3], [1, 4], [0, 4], [0, 3]]
    _write_annotation_polygon(class_a, polygon)
    _write_annotation_polygon(class_b, polygon)
    load_settings(
        _write_config(
            tmp_path,
            tile_size=4,
            stride=4,
            batch_size=1,
            input_channels=1,
            positive_factor=1.0,
            background_factor=0.0,
        )
    )

    loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            class_annotations=[
                TileClassAnnotation(
                    class_id=1,
                    slug="class_a",
                    name="Класс А",
                    annotation_file=class_a,
                    priority=10,
                ),
                TileClassAnnotation(
                    class_id=2,
                    slug="class_b",
                    name="Класс Б",
                    annotation_file=class_b,
                    priority=-10,
                ),
            ],
            batch_size=1,
            mode="train",
        )
    )

    _images, masks, _meta = next(iter(loader))
    assert torch.equal(torch.unique(masks), torch.tensor([0, 1]))
    loader.dataset.close()


def test_create_tile_dataloader_reads_edge_tile_as_regular_grid_with_nodata_fill(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    raster_path = tmp_path / "edge.tif"
    data = np.arange(25, dtype=np.int16).reshape(1, 5, 5) + 1000
    _write_raster_data(raster_path, data, nodata=-1)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "empty.geojson"
    _write_empty_annotation(annotation_file)
    load_settings(_write_config(tmp_path, tile_size=4, stride=4, batch_size=4, input_channels=1))

    dataset = TileDataset(
        vrt_xml=vrt_xml,
        annotation_file=annotation_file,
        tile_size=4,
        stride=4,
        mode="val",
        seed=42,
        augmentation_level=0,
    )

    image, mask, sample_meta = dataset[1]
    edge_tile = image[0]
    assert len(dataset) == 4
    assert image.shape == (1, 4, 4)
    assert mask.shape == (1, 4, 4)
    assert sample_meta == {
        "augmented": False,
        "category": TILE_CATEGORY_BACKGROUND,
        "positive": False,
        "hard_negative": False,
        "background": True,
    }
    assert torch.equal(
        torch.as_tensor(edge_tile[:, 0]),
        torch.as_tensor(data[0, 0:4, 4].astype(np.float32)),
    )
    assert torch.all(torch.as_tensor(edge_tile[:, 1:]) == -1.0)

    dataset.close()


def test_create_tile_dataloader_filters_fully_nodata_tiles(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    raster_path = tmp_path / "nodata.tif"
    data = np.zeros((1, 4, 4), dtype=np.uint16)
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation_height4(annotation_file)
    load_settings(_write_config(tmp_path, tile_size=4, stride=4, batch_size=1, input_channels=1))

    dataset = TileDataset(
        vrt_xml=vrt_xml,
        annotation_file=annotation_file,
        tile_size=4,
        stride=4,
        mode="val",
        seed=42,
        augmentation_level=0,
    )

    assert len(dataset) == 0
    assert dataset.candidate_window_count_before_valid_filter == 1
    assert dataset.black_filtered_window_count == 1

    dataset.close()


def test_valid_footprint_filter_removes_zero_window_and_keeps_nonzero_window(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    raster_path = tmp_path / "mixed.tif"
    data = np.zeros((1, 64, 128), dtype=np.uint16)
    data[:, :, 64:] = 1000
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "empty.geojson"
    _write_empty_annotation(annotation_file)
    load_settings(_write_config(tmp_path, tile_size=64, stride=64, batch_size=2, input_channels=1))

    dataset = TileDataset(
        vrt_xml=vrt_xml,
        annotation_file=annotation_file,
        tile_size=64,
        stride=64,
        mode="val",
        seed=42,
        augmentation_level=0,
    )

    assert len(dataset) == 1
    assert dataset.candidate_window_count_before_valid_filter == 2
    assert dataset.candidate_window_count == 1
    assert dataset.black_filtered_window_count == 1
    assert dataset.valid_footprint_stride == 64
    image, mask, sample_meta = dataset[0]
    assert image.shape == (1, 64, 64)
    assert torch.all(torch.as_tensor(image) == 1000.0)
    assert torch.all(torch.as_tensor(mask) == 0.0)
    assert sample_meta == {
        "augmented": False,
        "category": TILE_CATEGORY_BACKGROUND,
        "positive": False,
        "hard_negative": False,
        "background": True,
    }

    dataset.close()


def test_large_vrt_valid_filter_skips_full_footprint_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    raster_path = tmp_path / "large_path.tif"
    data = np.zeros((1, 64, 128), dtype=np.uint16)
    data[:, :, 64:] = 1000
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "empty.geojson"
    _write_empty_annotation(annotation_file)
    load_settings(_write_config(tmp_path, tile_size=64, stride=64, batch_size=2, input_channels=1))

    import mlsystem2.tile_preparation._valid_footprint as valid_footprint

    monkeypatch.setattr(valid_footprint, "_MAX_FULL_FOOTPRINT_CELLS", 1)

    def fail_full_footprint(*args, **kwargs):
        raise AssertionError("full footprint read should be skipped")

    monkeypatch.setattr(valid_footprint, "_read_valid_footprint", fail_full_footprint)

    dataset = TileDataset(
        vrt_xml=vrt_xml,
        annotation_file=annotation_file,
        tile_size=64,
        stride=64,
        mode="val",
        seed=42,
        augmentation_level=0,
    )

    assert len(dataset) == 1
    assert dataset.candidate_window_count_before_valid_filter == 2
    assert dataset.candidate_window_count == 1
    image, _mask, _sample_meta = dataset[0]
    assert torch.all(torch.as_tensor(image) == 1000.0)

    dataset.close()


def test_tile_dataset_does_not_read_windows_during_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raster_path = tmp_path / "lazy.tif"
    data = np.ones((1, 8, 8), dtype=np.uint16)
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "empty.geojson"
    _write_empty_annotation(annotation_file)
    read_calls = 0

    def fake_read_image_raw(self, dataset, window):
        nonlocal read_calls
        read_calls += 1
        return np.ones((1, 4, 4), dtype=np.uint16)

    monkeypatch.setattr(TileDataset, "_read_image_raw", fake_read_image_raw)

    dataset = TileDataset(
        vrt_xml=vrt_xml,
        annotation_file=annotation_file,
        tile_size=4,
        stride=4,
        mode="val",
        seed=42,
        augmentation_level=0,
    )

    assert len(dataset) == 4
    assert read_calls == 0
    image, mask, sample_meta = dataset[0]
    assert read_calls == 1
    assert image.shape == (1, 4, 4)
    assert mask.shape == (1, 4, 4)
    assert sample_meta == {
        "augmented": False,
        "category": TILE_CATEGORY_BACKGROUND,
        "positive": False,
        "hard_negative": False,
        "background": True,
    }
    dataset.close()


def test_create_tile_dataloader_is_fast_on_synthetic_data(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    raster_path = tmp_path / "fast.tif"
    data = np.ones((1, 16, 16), dtype=np.uint16)
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation_height4(annotation_file)
    load_settings(_write_config(tmp_path, tile_size=4, stride=4, batch_size=1, input_channels=1))

    started = perf_counter()
    loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            annotation_file=annotation_file,
            batch_size=1,
            mode="val",
        )
    )

    assert perf_counter() - started < 2.0
    assert len(loader.dataset) == 16
    assert loader.cache_mode == "memory"
    assert loader.cached_tiles == 2
    loader.dataset.close()


def test_train_loader_is_stable_with_same_seed_when_augmentation_is_disabled(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    raster_path = tmp_path / "image.tif"
    _write_raster(raster_path)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation(annotation_file)
    load_settings(_write_config(tmp_path, tile_size=4, stride=4, batch_size=2))

    request = TileDataloaderRequest(
        vrt_xml=vrt_xml,
        annotation_file=annotation_file,
        batch_size=2,
        mode="train",
    )
    first_images, first_masks, first_meta = next(iter(create_tile_dataloader(request)))
    second_images, second_masks, second_meta = next(iter(create_tile_dataloader(request)))

    assert torch.equal(first_images, second_images)
    assert torch.equal(first_masks, second_masks)
    assert first_meta == second_meta
    assert first_meta["augmented_tile_count"] == 0
    assert len(first_meta["tile_augmented"]) == 2
    assert len(first_meta["tile_positive"]) == 2


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="multiprocessing DataLoader на Windows нестабилен для unit-теста",
)
def test_create_tile_dataloader_with_worker_prefetch(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    raster_path = tmp_path / "image.tif"
    _write_raster(raster_path)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation(annotation_file)
    load_settings(_write_config(tmp_path, tile_size=4, stride=4, batch_size=2, num_workers=1))

    loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            annotation_file=annotation_file,
            batch_size=2,
            mode="val",
        )
    )

    images, masks, batch_meta = next(iter(loader))
    assert images.shape == (2, 3, 4, 4)
    assert masks.shape == (2, 1, 4, 4)
    assert batch_meta["augmented_tile_count"] == 0


def test_tile_category_precedence_keeps_hard_negative_masks_empty(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    raster_path = tmp_path / "category.tif"
    data = np.full((1, 4, 12), 1000, dtype=np.uint16)
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "annotations.geojson"
    hard_negative_file = tmp_path / "hard_negative.geojson"
    _write_annotation_polygon(
        annotation_file,
        [[0.5, 2.5], [1.5, 2.5], [1.5, 3.5], [0.5, 3.5], [0.5, 2.5]],
    )
    _write_annotation_polygon(
        hard_negative_file,
        [[0.5, 2.5], [5.5, 2.5], [5.5, 3.5], [0.5, 3.5], [0.5, 2.5]],
    )

    dataset = TileDataset(
        vrt_xml=vrt_xml,
        annotation_file=annotation_file,
        hard_negative_annotation_file=hard_negative_file,
        tile_size=4,
        stride=4,
        mode="train",
        seed=42,
        augmentation_level=0,
    )

    positive_image, positive_mask, positive_meta = dataset[0]
    hard_image, hard_mask, hard_meta = dataset[1]
    background_image, background_mask, background_meta = dataset[2]

    assert positive_image.shape == hard_image.shape == background_image.shape == (1, 4, 4)
    assert dataset.tile_categories == [
        TILE_CATEGORY_POSITIVE,
        TILE_CATEGORY_HARD_NEGATIVE,
        TILE_CATEGORY_BACKGROUND,
    ]
    assert positive_meta["category"] == TILE_CATEGORY_POSITIVE
    assert hard_meta["category"] == TILE_CATEGORY_HARD_NEGATIVE
    assert background_meta["category"] == TILE_CATEGORY_BACKGROUND
    assert np.count_nonzero(positive_mask) > 0
    assert np.count_nonzero(hard_mask) == 0
    assert np.count_nonzero(background_mask) == 0
    dataset.close()


def test_train_augmentation_applies_to_positive_and_hard_negative_tiles(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    raster_path = tmp_path / "smart_aug.tif"
    data = np.full((1, 4, 12), 1000, dtype=np.uint16)
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "annotations.geojson"
    hard_negative_file = tmp_path / "hard_negative.geojson"
    _write_annotation_polygon(
        annotation_file,
        [[0.5, 2.5], [1.5, 2.5], [1.5, 3.5], [0.5, 3.5], [0.5, 2.5]],
    )
    _write_annotation_polygon(
        hard_negative_file,
        [[4.5, 2.5], [5.5, 2.5], [5.5, 3.5], [4.5, 3.5], [4.5, 2.5]],
    )

    dataset = TileDataset(
        vrt_xml=vrt_xml,
        annotation_file=annotation_file,
        hard_negative_annotation_file=hard_negative_file,
        tile_size=4,
        stride=4,
        mode="train",
        seed=42,
        augmentation_level=2,
    )

    _positive_image, _positive_mask, positive_meta = dataset[0]
    _hard_image, hard_mask, hard_meta = dataset[1]
    _background_image, _background_mask, background_meta = dataset[2]

    assert positive_meta["augmented"] is True
    assert positive_meta["category"] == TILE_CATEGORY_POSITIVE
    assert hard_meta["augmented"] is True
    assert hard_meta["category"] == TILE_CATEGORY_HARD_NEGATIVE
    assert background_meta["augmented"] is False
    assert background_meta["category"] == TILE_CATEGORY_BACKGROUND
    assert np.count_nonzero(hard_mask) == 0
    dataset.close()


def test_collate_batch_reports_category_and_augmentation_counters() -> None:
    pytest.importorskip("torch")
    image = np.ones((1, 2, 2), dtype=np.float32)
    empty_mask = np.zeros((1, 2, 2), dtype=np.float32)
    positive_mask = np.ones((1, 2, 2), dtype=np.float32)

    _images, _masks, batch_meta = dataloader_impl._collate_tile_batch(
        [
            (image, positive_mask, {"category": TILE_CATEGORY_POSITIVE, "augmented": True}),
            (image, empty_mask, {"category": TILE_CATEGORY_HARD_NEGATIVE, "augmented": True}),
            (image, empty_mask, {"category": TILE_CATEGORY_BACKGROUND, "augmented": False}),
        ]
    )

    assert batch_meta["positive_tile_count"] == 1
    assert batch_meta["hard_negative_tile_count"] == 1
    assert batch_meta["background_tile_count"] == 1
    assert batch_meta["augmented_tile_count"] == 2
    assert batch_meta["augmented_positive_tile_count"] == 1
    assert batch_meta["augmented_hard_negative_tile_count"] == 1
    assert batch_meta["tile_category"] == [
        TILE_CATEGORY_POSITIVE,
        TILE_CATEGORY_HARD_NEGATIVE,
        TILE_CATEGORY_BACKGROUND,
    ]


def test_weighted_sampler_is_used_for_train_and_val_is_cached(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    raster_path = tmp_path / "smart_sampler.tif"
    data = np.full((1, 4, 8), 1000, dtype=np.uint16)
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation_height4(annotation_file)
    load_settings(
        _write_config(
            tmp_path,
            tile_size=4,
            stride=4,
            batch_size=1,
            input_channels=1,
        )
    )

    train_loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            annotation_file=annotation_file,
            batch_size=1,
            mode="train",
        )
    )
    val_loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            annotation_file=annotation_file,
            batch_size=1,
            mode="val",
        )
    )

    assert isinstance(train_loader.sampler, torch.utils.data.WeightedRandomSampler)
    assert val_loader.sampler is None
    assert val_loader.cache_mode == "memory"
    assert val_loader.cached_tiles == 2
    assert val_loader.cached_batches == 2
    assert train_loader.dataset.estimated_positive_tiles == 1
    assert train_loader.dataset.estimated_hard_negative_tiles == 0
    assert train_loader.dataset.estimated_background_tiles == 1
    assert val_loader.dataset.estimated_positive_tiles == 1
    assert val_loader.dataset.estimated_hard_negative_tiles == 0
    assert val_loader.dataset.estimated_background_tiles == 1
    train_loader.dataset.close()
    val_loader.dataset.close()


def test_val_cached_loader_returns_same_batches_on_each_iteration(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    raster_path = tmp_path / "cached_val_stable.tif"
    data = np.full((1, 4, 8), 1000, dtype=np.uint16)
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation_height4(annotation_file)
    load_settings(
        _write_config(
            tmp_path,
            tile_size=4,
            stride=4,
            batch_size=2,
            input_channels=1,
            val_positive_factor=0.5,
        )
    )

    val_loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            annotation_file=annotation_file,
            batch_size=2,
            mode="val",
        )
    )

    first_batches = list(val_loader)
    second_batches = list(val_loader)

    assert len(first_batches) == len(second_batches) == 1
    first_images, first_masks, first_meta = first_batches[0]
    second_images, second_masks, second_meta = second_batches[0]
    assert torch.equal(first_images, second_images)
    assert torch.equal(first_masks, second_masks)
    assert first_meta == second_meta
    assert first_meta["positive_tile_count"] == 1
    val_loader.dataset.close()


def test_val_cached_loader_reads_tiles_only_during_cache_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    raster_path = tmp_path / "cached_val_reads.tif"
    data = np.full((1, 4, 8), 1000, dtype=np.uint16)
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation_height4(annotation_file)
    load_settings(_write_config(tmp_path, tile_size=4, stride=4, batch_size=1, input_channels=1))

    read_calls = 0
    mask_calls = 0
    original_read = TileDataset._read_image_raw
    original_mask = TileDataset._read_annotation_mask

    def counted_read(self, dataset, window):
        nonlocal read_calls
        read_calls += 1
        return original_read(self, dataset, window)

    def counted_mask(self, dataset, window, nodata_pixels):
        nonlocal mask_calls
        mask_calls += 1
        return original_mask(self, dataset, window, nodata_pixels)

    monkeypatch.setattr(TileDataset, "_read_image_raw", counted_read)
    monkeypatch.setattr(TileDataset, "_read_annotation_mask", counted_mask)

    val_loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            annotation_file=annotation_file,
            batch_size=1,
            mode="val",
        )
    )

    assert val_loader.cached_tiles == 2
    assert read_calls == 2
    assert mask_calls == 2
    list(val_loader)
    list(val_loader)
    assert read_calls == 2
    assert mask_calls == 2
    val_loader.dataset.close()


def test_val_cached_loader_uses_min_group_without_replacement(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    raster_path = tmp_path / "cached_val_imbalance.tif"
    data = np.full((1, 4, 52), 1000, dtype=np.uint16)
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation_polygon(
        annotation_file,
        [[0.5, 2.5], [11.5, 2.5], [11.5, 3.5], [0.5, 3.5], [0.5, 2.5]],
    )
    load_settings(_write_config(tmp_path, tile_size=4, stride=4, batch_size=2, input_channels=1))

    val_loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            annotation_file=annotation_file,
            batch_size=2,
            mode="val",
        )
    )

    assert val_loader.dataset.estimated_positive_tiles == 3
    assert val_loader.dataset.estimated_hard_negative_tiles == 0
    assert val_loader.dataset.estimated_background_tiles == 10
    assert val_loader.cached_tiles == 6
    positive_tiles = 0
    total_tiles = 0
    for _images, _masks, batch_meta in val_loader:
        positive_tiles += batch_meta["positive_tile_count"]
        total_tiles += len(batch_meta["tile_positive"])
    assert positive_tiles == 3
    assert total_tiles == 6
    val_loader.dataset.close()


def test_val_cached_loader_requires_positive_and_negative_tiles(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    raster_path = tmp_path / "cached_val_empty_positive.tif"
    data = np.full((1, 4, 8), 1000, dtype=np.uint16)
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "empty.geojson"
    _write_empty_annotation(annotation_file)
    load_settings(_write_config(tmp_path, tile_size=4, stride=4, batch_size=1, input_channels=1))

    with pytest.raises(TilePreparationError, match="positive=0"):
        create_tile_dataloader(
            TileDataloaderRequest(
                vrt_xml=vrt_xml,
                annotation_file=annotation_file,
                batch_size=1,
                mode="val",
            )
        )


def test_tile_split_divides_common_pool_without_overlap(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    raster_path = tmp_path / "tile_split.tif"
    data = np.full((1, 4, 16), 1000, dtype=np.uint16)
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation_polygon(
        annotation_file,
        [[0.5, 2.5], [5.5, 2.5], [5.5, 3.5], [0.5, 3.5], [0.5, 2.5]],
    )
    load_settings(_write_config(tmp_path, tile_size=4, stride=4, batch_size=1, input_channels=1))
    tile_split = TileSplitRequest(val_fraction=0.5, seed=7)

    train_loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            annotation_file=annotation_file,
            batch_size=1,
            mode="train",
            tile_split=tile_split,
        )
    )
    val_loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            annotation_file=annotation_file,
            batch_size=1,
            mode="val",
            tile_split=tile_split,
        )
    )

    train_keys = _window_keys(train_loader.dataset)
    val_keys = _window_keys(val_loader.dataset)
    assert len(train_keys) == 2
    assert len(val_keys) == 2
    assert set(train_keys).isdisjoint(val_keys)
    assert len(set(train_keys) | set(val_keys)) == 4
    assert train_loader.dataset.pool_window_count == 4
    assert val_loader.dataset.pool_window_count == 4
    assert train_loader.dataset.split_window_count == 2
    assert val_loader.dataset.split_window_count == 2
    assert train_loader.dataset.estimated_positive_tiles == 1
    assert val_loader.dataset.estimated_positive_tiles == 1
    train_loader.dataset.close()
    val_loader.dataset.close()


def test_tile_split_is_stable_with_same_seed(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    raster_path = tmp_path / "tile_split_stable.tif"
    data = np.full((1, 4, 16), 1000, dtype=np.uint16)
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation_polygon(
        annotation_file,
        [[0.5, 2.5], [5.5, 2.5], [5.5, 3.5], [0.5, 3.5], [0.5, 2.5]],
    )
    load_settings(_write_config(tmp_path, tile_size=4, stride=4, batch_size=1, input_channels=1))
    request = TileDataloaderRequest(
        vrt_xml=vrt_xml,
        annotation_file=annotation_file,
        batch_size=1,
        mode="val",
        tile_split=TileSplitRequest(val_fraction=0.5, seed=11),
    )

    first_loader = create_tile_dataloader(request)
    second_loader = create_tile_dataloader(request)

    assert _window_keys(first_loader.dataset) == _window_keys(second_loader.dataset)
    first_loader.dataset.close()
    second_loader.dataset.close()


def test_tile_split_reports_warning_for_tiny_positive_pool(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    raster_path = tmp_path / "tile_split_tiny.tif"
    data = np.full((1, 4, 8), 1000, dtype=np.uint16)
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation_height4(annotation_file)
    load_settings(_write_config(tmp_path, tile_size=4, stride=4, batch_size=1, input_channels=1))

    dataset = TileDataset(
        vrt_xml=vrt_xml,
        annotation_file=annotation_file,
        tile_size=4,
        stride=4,
        mode="val",
        seed=42,
        augmentation_level=0,
        tile_split=TileSplitRequest(val_fraction=0.5, seed=42),
    )

    assert len(dataset) == 0
    assert any("positive windows меньше 2" in item for item in dataset.tile_split_warnings)
    assert any("subset val пуст" in item for item in dataset.tile_split_warnings)
    dataset.close()


def test_sampling_weights_distribute_three_category_budgets() -> None:
    dataset = TileDataset.__new__(TileDataset)
    dataset._positive_hint_by_index = [True, True, False, False, False, False]
    dataset._hard_negative_hint_by_index = [False, False, True, True, False, False]
    dataset._positive_factor = 0.5
    dataset._hard_negative_factor = 0.3
    dataset._background_factor = 0.2
    dataset._class_balance = False
    dataset._class_annotations = []
    dataset._class_hints_by_index = None

    weights = dataset.sampling_weights()

    assert weights is not None
    assert sum(weights[:2]) == pytest.approx(0.5)
    assert sum(weights[2:4]) == pytest.approx(0.3)
    assert sum(weights[4:]) == pytest.approx(0.2)
    assert dataset.positive_factor_used == pytest.approx(0.5)
    assert dataset.hard_negative_factor_used == pytest.approx(0.3)
    assert dataset.background_factor_used == pytest.approx(0.2)
    assert dataset.sampling_warnings == []


def test_sampling_weights_move_missing_hard_negative_budget_to_positive() -> None:
    dataset = TileDataset.__new__(TileDataset)
    dataset._positive_hint_by_index = [True, True, False, False]
    dataset._hard_negative_hint_by_index = [False, False, False, False]
    dataset._positive_factor = 0.5
    dataset._hard_negative_factor = 0.3
    dataset._background_factor = 0.2
    dataset._class_balance = False
    dataset._class_annotations = []
    dataset._class_hints_by_index = None

    weights = dataset.sampling_weights()

    assert weights is not None
    assert sum(weights[:2]) == pytest.approx(0.8)
    assert sum(weights[2:]) == pytest.approx(0.2)
    assert dataset.positive_factor_used == pytest.approx(0.8)
    assert dataset.hard_negative_factor_used == pytest.approx(0.0)
    assert dataset.background_factor_used == pytest.approx(0.2)
    assert any("hard_negative_factor_used" in item for item in dataset.sampling_warnings)


def test_sampling_weights_cap_small_hard_negative_budget_and_fill_positive() -> None:
    dataset = TileDataset.__new__(TileDataset)
    dataset._positive_hint_by_index = [True, True, False, False, False]
    dataset._hard_negative_hint_by_index = [False, False, True, False, False]
    dataset._positive_factor = 0.5
    dataset._hard_negative_factor = 0.3
    dataset._background_factor = 0.2
    dataset._class_balance = False
    dataset._class_annotations = []
    dataset._class_hints_by_index = None

    weights = dataset.sampling_weights()

    assert weights is not None
    assert sum(weights[:2]) == pytest.approx(0.8 * 2 / 3)
    assert weights[2] == pytest.approx(0.8 * 1 / 3)
    assert sum(weights[3:]) == pytest.approx(0.2)
    assert dataset.positive_factor_used == pytest.approx(0.8 * 2 / 3)
    assert dataset.hard_negative_factor_used == pytest.approx(0.8 * 1 / 3)
    assert dataset.background_factor_used == pytest.approx(0.2)
    assert any("marked tiles" in item for item in dataset.sampling_warnings)


def test_sampling_weights_cap_hard_negative_by_marked_pool_ratio_regression() -> None:
    positive_count = 1834
    hard_negative_count = 283
    background_count = 17608
    dataset = TileDataset.__new__(TileDataset)
    dataset._positive_hint_by_index = (
        [True] * positive_count
        + [False] * hard_negative_count
        + [False] * background_count
    )
    dataset._hard_negative_hint_by_index = (
        [False] * positive_count
        + [True] * hard_negative_count
        + [False] * background_count
    )
    dataset._positive_factor = 0.5
    dataset._hard_negative_factor = 0.3
    dataset._background_factor = 0.2
    dataset._class_balance = False
    dataset._class_annotations = []
    dataset._class_hints_by_index = None

    weights = dataset.sampling_weights()

    expected_hard_factor = 0.8 * hard_negative_count / (positive_count + hard_negative_count)
    assert weights is not None
    assert sum(weights[:positive_count]) == pytest.approx(0.8 - expected_hard_factor)
    assert sum(weights[positive_count : positive_count + hard_negative_count]) == pytest.approx(
        expected_hard_factor
    )
    assert sum(weights[positive_count + hard_negative_count :]) == pytest.approx(0.2)
    assert dataset.hard_negative_factor_used == pytest.approx(expected_hard_factor)
    assert dataset.hard_negative_factor_used != pytest.approx(
        hard_negative_count / (positive_count + hard_negative_count + background_count)
    )
    assert dataset.positive_factor_used == pytest.approx(0.8 - expected_hard_factor)
    assert dataset.background_factor_used == pytest.approx(0.2)


def test_sampling_weights_allow_only_one_category_budget() -> None:
    dataset = TileDataset.__new__(TileDataset)
    dataset._positive_hint_by_index = [False, False, False]
    dataset._hard_negative_hint_by_index = [True, True, True]
    dataset._positive_factor = 0.0
    dataset._hard_negative_factor = 1.0
    dataset._background_factor = 0.0
    dataset._class_balance = False
    dataset._class_annotations = []
    dataset._class_hints_by_index = None

    weights = dataset.sampling_weights()

    assert weights == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_sampling_weights_reject_missing_background_factor_category() -> None:
    dataset = TileDataset.__new__(TileDataset)
    dataset._positive_hint_by_index = [True, False]
    dataset._hard_negative_hint_by_index = [False, True]
    dataset._positive_factor = 0.5
    dataset._hard_negative_factor = 0.2
    dataset._background_factor = 0.3
    dataset._class_balance = False
    dataset._class_annotations = []
    dataset._class_hints_by_index = None

    with pytest.raises(TilePreparationError, match="background_factor"):
        dataset.sampling_weights()


def test_sampling_weights_reject_marked_budget_without_positive_fill() -> None:
    dataset = TileDataset.__new__(TileDataset)
    dataset._positive_hint_by_index = [False, False, False]
    dataset._hard_negative_hint_by_index = [True, False, False]
    dataset._positive_factor = 0.5
    dataset._hard_negative_factor = 0.3
    dataset._background_factor = 0.2
    dataset._class_balance = False
    dataset._class_annotations = []
    dataset._class_hints_by_index = None

    with pytest.raises(TilePreparationError, match="positive tiles"):
        dataset.sampling_weights()


def test_multiclass_class_balance_sampling_weights_boost_rare_classes() -> None:
    dataset = TileDataset.__new__(TileDataset)
    dataset._positive_hint_by_index = [True, True, True, False, False]
    dataset._class_hints_by_index = [
        frozenset({1}),
        frozenset({1}),
        frozenset({2}),
        frozenset(),
        frozenset(),
    ]
    dataset._positive_factor = 0.8
    dataset._class_balance = True
    dataset._class_annotations = [
        TileClassAnnotation(class_id=1, slug="common", name="Common", annotation_file="a.geojson"),
        TileClassAnnotation(class_id=2, slug="rare", name="Rare", annotation_file="b.geojson"),
    ]

    weights = dataset.sampling_weights(0.8, 0.0, 0.2)

    assert weights is not None
    assert weights[2] > weights[0]
    assert sum(weights[:3]) == pytest.approx(0.8)
    assert sum(weights[3:]) == pytest.approx(0.2)


def test_class_balance_uses_effective_positive_budget_with_hard_negative_deficit() -> None:
    dataset = TileDataset.__new__(TileDataset)
    dataset._positive_hint_by_index = [True, True, True, False, False, False]
    dataset._hard_negative_hint_by_index = [False, False, False, True, False, False]
    dataset._class_hints_by_index = [
        frozenset({1}),
        frozenset({1}),
        frozenset({2}),
        frozenset(),
        frozenset(),
        frozenset(),
    ]
    dataset._positive_factor = 0.5
    dataset._hard_negative_factor = 0.3
    dataset._background_factor = 0.2
    dataset._class_balance = True
    dataset._class_annotations = [
        TileClassAnnotation(class_id=1, slug="common", name="Common", annotation_file="a.geojson"),
        TileClassAnnotation(class_id=2, slug="rare", name="Rare", annotation_file="b.geojson"),
    ]

    weights = dataset.sampling_weights()

    assert weights is not None
    assert weights[2] > weights[0]
    assert sum(weights[:3]) == pytest.approx(0.6)
    assert weights[3] == pytest.approx(0.2)
    assert sum(weights[4:]) == pytest.approx(0.2)
    assert dataset.positive_factor_used == pytest.approx(0.6)
    assert dataset.hard_negative_factor_used == pytest.approx(0.2)
    assert dataset.background_factor_used == pytest.approx(0.2)


def _window_keys(dataset) -> list[tuple[int, int]]:
    return [(window.x, window.y) for window in dataset._windows]


def _write_raster(path: Path) -> None:
    base = np.arange(36, dtype=np.uint8).reshape(6, 6)
    data = np.stack([base + 10, base + 40, base + 70])
    _write_raster_data(path, data, nodata=0)


def _write_raster_data(path: Path, data: np.ndarray, *, nodata: int | float | None) -> None:
    if data.ndim != 3:
        raise ValueError("Тестовый raster должен иметь форму [C, H, W].")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        dtype=str(data.dtype),
        crs="EPSG:3857",
        transform=from_origin(0, data.shape[1], 1, 1),
        nodata=nodata,
    ) as dataset:
        dataset.write(data)


def _write_vrt_xml(raster_path: Path) -> str:
    vrt_path = raster_path.with_suffix(".vrt")
    rio_copy(raster_path.as_posix(), vrt_path.as_posix(), driver="VRT")
    vrt_xml = vrt_path.read_text(encoding="utf-8")
    source = re.escape(raster_path.name)
    return re.sub(
        rf'<SourceFilename relativeToVRT="1">{source}</SourceFilename>',
        f'<SourceFilename relativeToVRT="0">{raster_path.as_posix()}</SourceFilename>',
        vrt_xml,
    )


def _write_annotation(path: Path) -> None:
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [0.5, 4.5],
                        [1.5, 4.5],
                        [1.5, 5.5],
                        [0.5, 5.5],
                        [0.5, 4.5],
                    ]],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_annotation_height4(path: Path) -> None:
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [0.5, 2.5],
                        [1.5, 2.5],
                        [1.5, 3.5],
                        [0.5, 3.5],
                        [0.5, 2.5],
                    ]],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_annotation_polygon(path: Path, coordinates: list[list[float]]) -> None:
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_empty_annotation(path: Path) -> None:
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": []}),
        encoding="utf-8",
    )


def _write_config(
    tmp_path: Path,
    *,
    tile_size: int,
    stride: int,
    batch_size: int,
    num_workers: int = 0,
    input_channels: int = 3,
    augmentation_level: int = 0,
    positive_factor: float = 0.5,
    hard_negative_factor: float = 0.0,
    background_factor: float = 0.5,
    val_positive_factor: float = 0.5,
) -> Path:
    settings_path = tmp_path / "config.yaml"
    settings_path.write_text(
        f"""
runtime:
  project_root: {tmp_path.as_posix()}
  scratch_root: {tmp_path.as_posix()}/scratch
  logs_root: {tmp_path.as_posix()}/logs
  cleanup_scratch_after_mlflow_log: false

dataset:
  images_dir: {tmp_path.as_posix()}/images
  scenes_file: {tmp_path.as_posix()}/scenes.txt
  annotation_file: {tmp_path.as_posix()}/annotations.geojson
  val_fraction: 0.2

tile_preparation:
  tile_size: {tile_size}
  stride: {stride}
  num_workers: {num_workers}
  prefetch_epochs: 2
  seed: 42
  augmentation_level: {augmentation_level}
  positive_factor: {positive_factor}
  hard_negative_factor: {hard_negative_factor}
  background_factor: {background_factor}
  val_positive_factor: {val_positive_factor}

train:
  model_name: segformer_b2
  input_channels: {input_channels}
  output_channels: 1
  pretrained: false
  initial_checkpoint_uri: null
  epochs: 1
  batch_size: {batch_size}
  device: cpu
  learning_rate: 0.00001
  weight_decay: 0.0001
  loss: bce_dice
  focal_alpha: 0.6
  pos_weight: 1.0
  tversky_alpha: 0.4
  tversky_beta: 0.6
  threshold: 0.5
  early_stopping_patience: 2

inference:
  checkpoint_uri: {tmp_path.as_posix()}/latest.pt
  threshold: 0.5
  batch_size: {batch_size}
  device: cpu

mlflow:
  enabled: false
  tracking_uri: {tmp_path.as_posix()}/mlruns
  experiment_name: MLSystem2-test
""",
        encoding="utf-8",
    )
    return settings_path

class _DeterministicRng:
    def random(self) -> float:
        return 0.0

    def integers(self, low: int, high: int | None = None) -> int:
        del low, high
        return 1
