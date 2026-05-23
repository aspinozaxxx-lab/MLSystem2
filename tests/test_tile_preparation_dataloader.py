from __future__ import annotations

import json
import re
import sys
import builtins
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_origin

from mlsystem2.settings.api import load_settings
from mlsystem2.tile_preparation.api import create_tile_dataloader
from mlsystem2.tile_preparation.contracts import TileDataloaderRequest, TilePreparationError


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


def test_create_tile_dataloader_returns_image_mask_tuple(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    raster_path = tmp_path / "image.tif"
    _write_raster(raster_path)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "annotations.geojson"
    _write_annotation(annotation_file)
    load_settings(_write_config(tmp_path, tile_size=4, stride=4, batch_size=4))

    loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            annotation_file=annotation_file,
            batch_size=4,
            mode="val",
        )
    )

    batch = next(iter(loader))
    assert isinstance(batch, tuple)
    assert len(batch) == 2
    images, masks = batch
    assert images.shape == (4, 3, 4, 4)
    assert masks.shape == (4, 1, 4, 4)
    assert images.dtype == torch.float32
    assert masks.dtype == torch.float32
    assert set(torch.unique(masks).tolist()) <= {0.0, 1.0}
    assert masks[0].sum() > 0
    assert masks[1:].sum() == 0

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
    load_settings(_write_config(tmp_path, tile_size=4, stride=4, batch_size=1, input_channels=2))

    loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            annotation_file=annotation_file,
            batch_size=1,
            mode="val",
        )
    )

    images, masks = next(iter(loader))
    assert images.shape == (1, 2, 4, 4)
    assert masks.shape == (1, 1, 4, 4)
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
    annotation_file = tmp_path / "empty.geojson"
    _write_empty_annotation(annotation_file)
    load_settings(
        _write_config(
            tmp_path,
            tile_size=4,
            stride=4,
            batch_size=1,
            input_channels=1,
            augmentation_level=2,
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

    images, _masks = next(iter(loader))
    assert float(images.max().item()) > 1.0

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

    loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            annotation_file=annotation_file,
            batch_size=4,
            mode="val",
        )
    )

    images, masks = next(iter(loader))
    edge_tile = images[1, 0]
    assert images.shape == (4, 1, 4, 4)
    assert masks.shape == (4, 1, 4, 4)
    assert torch.equal(edge_tile[:, 0], torch.as_tensor(data[0, 0:4, 4].astype(np.float32)))
    assert torch.all(edge_tile[:, 1:] == -1.0)

    loader.dataset.close()


def test_create_tile_dataloader_skips_fully_nodata_tiles(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    raster_path = tmp_path / "nodata.tif"
    data = np.zeros((1, 4, 4), dtype=np.uint16)
    _write_raster_data(raster_path, data, nodata=0)
    vrt_xml = _write_vrt_xml(raster_path)
    annotation_file = tmp_path / "empty.geojson"
    _write_empty_annotation(annotation_file)
    load_settings(_write_config(tmp_path, tile_size=4, stride=4, batch_size=1, input_channels=1))

    loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            annotation_file=annotation_file,
            batch_size=1,
            mode="val",
        )
    )

    assert len(loader.dataset) == 0

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
    first_images, first_masks = next(iter(create_tile_dataloader(request)))
    second_images, second_masks = next(iter(create_tile_dataloader(request)))

    assert torch.equal(first_images, second_images)
    assert torch.equal(first_masks, second_masks)


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

    images, masks = next(iter(loader))
    assert images.shape == (2, 3, 4, 4)
    assert masks.shape == (2, 1, 4, 4)


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
  prefetch_factor: 2
  seed: 42
  augmentation_level: {augmentation_level}

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
