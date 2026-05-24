"""Локальная диагностика связки dataset_preparing и tile_preparation."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from mlsystem2.dataset_preparing.api import prepare_dataset
from mlsystem2.dataset_preparing.contracts import DatasetPreparationRequest
from mlsystem2.settings.api import get_settings, load_settings
from mlsystem2.tile_preparation.api import create_tile_dataloader
from mlsystem2.tile_preparation.contracts import TileDataloaderRequest


IMAGES_DIR = Path(r"D:\Projects\ImagesDeforestationPrepared3857")
DATASET_DIR = Path(r"D:\Projects\MLMarkup\Вырубки")
OUT_DIR = Path(r"D:\Projects\test")

SCENES_FILE = DATASET_DIR / "deforestation.txt"
ANNOTATION_FILE = DATASET_DIR / "deforestation.geojson"

VAL_FRACTION = 0.2
TILE_SIZE = 1024
STRIDE = 512
NUM_WORKERS = 16
PREFETCH_FACTOR = 2
SEED = 42
AUGMENTATION_LEVEL = 3
BATCH_SIZE = 4
REQUESTED_BATCHES = 100
MODE = "train"


def main() -> int:
    total_started = perf_counter()
    paths = _paths()
    params = _params(saved_batches=0)
    timings: dict[str, float] = {
        "dataset_preparation_sec": 0.0,
        "dataloader_creation_sec": 0.0,
        "all_batches_sec": 0.0,
        "total_sec": 0.0,
    }
    batches: list[dict[str, Any]] = []

    try:
        _prepare_output_dir()
        _write_local_settings(paths["settings_file"])
        load_settings(paths["settings_file"])

        dataset_started = perf_counter()
        result = prepare_dataset(
            DatasetPreparationRequest(
                images_dir=str(IMAGES_DIR),
                scenes_file=str(SCENES_FILE),
                annotation_file=str(ANNOTATION_FILE),
                val_fraction=VAL_FRACTION,
            )
        )
        timings["dataset_preparation_sec"] = perf_counter() - dataset_started

        _write_json(OUT_DIR / "preparation_report.json", result.report.model_dump(mode="json"))

        if result.dataset is None or result.report.status != "ok":
            error = "dataset_preparing не вернул готовый датасет."
            _write_timing_report(
                status="error",
                error=error,
                paths=paths,
                params=params,
                timings=_finish_timings(timings, total_started),
                batches=batches,
            )
            print(error, file=sys.stderr)
            return 1

        Path(paths["train_vrt"]).write_text(result.dataset.train_vrt_xml, encoding="utf-8")
        Path(paths["val_vrt"]).write_text(result.dataset.val_vrt_xml, encoding="utf-8")

        dataloader_started = perf_counter()
        loader = create_tile_dataloader(
            TileDataloaderRequest(
                vrt_xml=result.dataset.train_vrt_xml,
                annotation_file=result.dataset.annotation_file,
                batch_size=get_settings().train.batch_size,
                mode=MODE,
            )
        )
        timings["dataloader_creation_sec"] = perf_counter() - dataloader_started

        batches_started = perf_counter()
        batches = _save_batches(loader)
        timings["all_batches_sec"] = perf_counter() - batches_started
        params = _params(saved_batches=len(batches))

        if not batches:
            error = "DataLoader не вернул ни одного batch."
            _write_timing_report(
                status="error",
                error=error,
                paths=paths,
                params=params,
                timings=_finish_timings(timings, total_started),
                batches=batches,
            )
            print(error, file=sys.stderr)
            return 1

        _write_timing_report(
            status="ok",
            error=None,
            paths=paths,
            params=params,
            timings=_finish_timings(timings, total_started),
            batches=batches,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        timings = _finish_timings(timings, total_started)
        params = _params(saved_batches=len(batches))
        error = str(exc)
        try:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            _write_timing_report(
                status="error",
                error=error,
                paths=paths,
                params=params,
                timings=timings,
                batches=batches,
            )
        except Exception as report_exc:  # noqa: BLE001
            error = f"{error}; не удалось записать timing report: {report_exc}"
        print(error, file=sys.stderr)
        return 1


def _prepare_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    batches_dir = OUT_DIR / "tile_batches"
    if batches_dir.exists():
        shutil.rmtree(batches_dir)
    batches_dir.mkdir(parents=True, exist_ok=True)


def _write_local_settings(path: str) -> None:
    settings_text = f"""
runtime:
  project_root: .
  scratch_root: '{OUT_DIR / "scratch"}'
  logs_root: '{OUT_DIR / "logs"}'
  cleanup_scratch_after_mlflow_log: false

dataset:
  images_dir: '{IMAGES_DIR}'
  scenes_file: '{SCENES_FILE}'
  annotation_file: '{ANNOTATION_FILE}'
  val_fraction: {VAL_FRACTION}

tile_preparation:
  tile_size: {TILE_SIZE}
  stride: {STRIDE}
  num_workers: {NUM_WORKERS}
  prefetch_factor: {PREFETCH_FACTOR}
  seed: {SEED}
  augmentation_level: {AUGMENTATION_LEVEL}

train:
  model_name: segformer_b2
  input_channels: 4
  output_channels: 1
  pretrained: false
  initial_checkpoint_uri: null
  epochs: 1
  batch_size: {BATCH_SIZE}
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
  checkpoint_uri: ./checkpoints/latest.pt
  threshold: 0.5
  batch_size: {BATCH_SIZE}
  device: cpu

mlflow:
  enabled: false
  tracking_uri: ./mlruns
  experiment_name: MLSystem2-modules-test
"""
    Path(path).write_text(settings_text.lstrip(), encoding="utf-8")


def _save_batches(loader: object) -> list[dict[str, Any]]:
    iterator = iter(loader)
    batches: list[dict[str, Any]] = []
    try:
        for batch_index in range(REQUESTED_BATCHES):
            fetch_started = perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            fetch_sec = perf_counter() - fetch_started

            images, masks = _expect_batch(batch)
            save_started = perf_counter()
            batch_dir = OUT_DIR / "tile_batches" / f"batch_{batch_index:04d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            tile_files = _save_tiles_as_png(images, masks, batch_dir)

            batch_meta = _batch_meta(
                batch_index=batch_index,
                images=images,
                masks=masks,
                tile_files=tile_files,
                fetch_sec=fetch_sec,
                save_sec=0.0,
            )
            _write_json(batch_dir / "batch_meta.json", batch_meta)
            save_sec = perf_counter() - save_started
            batch_meta["save_sec"] = save_sec
            batch_meta["batch_total_sec"] = fetch_sec + save_sec
            _write_json(batch_dir / "batch_meta.json", batch_meta)
            batches.append(_batch_report_item(batch_meta))
        return batches
    finally:
        _shutdown_loader_iterator(iterator)
        _close_loader_dataset(loader)


def _expect_batch(batch: object) -> tuple[object, object]:
    if not isinstance(batch, tuple) or len(batch) not in {2, 3}:
        raise RuntimeError("DataLoader batch должен быть tuple(images, masks[, batch_meta]).")
    return batch[0], batch[1]


def _save_tiles_as_png(images: object, masks: object, batch_dir: Path) -> list[dict[str, Any]]:
    image_array = _to_numpy(images)
    mask_array = _to_numpy(masks)
    if image_array.ndim != 4 or mask_array.ndim != 4:
        raise RuntimeError("Batch должен иметь формы images [B, C, H, W] и masks [B, 1, H, W].")
    if image_array.shape[0] != mask_array.shape[0]:
        raise RuntimeError("Количество images и masks в batch не совпадает.")

    tile_files: list[dict[str, Any]] = []
    for tile_index in range(image_array.shape[0]):
        preview_path = batch_dir / f"tile_{tile_index:04d}_preview_rgb.png"
        mask_path = batch_dir / f"tile_{tile_index:04d}_mask.png"
        channel_files = _save_image_channels_png(image_array[tile_index], batch_dir, tile_index)
        _save_preview_png(image_array[tile_index], preview_path)
        _save_mask_png(mask_array[tile_index], mask_path)
        tile_files.append(
            {
                "preview_rgb": preview_path.name,
                "mask": mask_path.name,
                "channels": channel_files,
            }
        )
    return tile_files


def _to_numpy(tensor: object):
    if not hasattr(tensor, "detach"):
        raise RuntimeError("Элемент batch не похож на torch.Tensor: нет метода detach().")
    return tensor.detach().cpu().numpy()


def _save_preview_png(image_chw, path: Path) -> None:
    import numpy as np
    from PIL import Image

    channels = image_chw.shape[0]
    if channels >= 3:
        image_hwc = image_chw[:3].transpose(1, 2, 0)
    elif channels == 1:
        image_hwc = np.repeat(image_chw[0:1], 3, axis=0).transpose(1, 2, 0)
    else:
        raise RuntimeError("Image tile должен содержать хотя бы один канал.")

    image_u8 = _stretch_image_to_uint8(image_hwc)
    Image.fromarray(image_u8, mode="RGB").save(path)


def _save_image_channels_png(image_chw, batch_dir: Path, tile_index: int) -> list[str]:
    from PIL import Image

    channel_files: list[str] = []
    for channel_index in range(image_chw.shape[0]):
        channel_path = batch_dir / f"tile_{tile_index:04d}_channel_{channel_index:02d}.png"
        channel_u8 = _stretch_channel_to_uint8(image_chw[channel_index])
        Image.fromarray(channel_u8, mode="L").save(channel_path)
        channel_files.append(channel_path.name)
    return channel_files


def _save_mask_png(mask_chw, path: Path) -> None:
    import numpy as np
    from PIL import Image

    if mask_chw.shape[0] != 1:
        raise RuntimeError("Mask tile должен иметь один канал.")
    mask_u8 = (np.clip(mask_chw[0], 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(mask_u8, mode="L").save(path)


def _stretch_image_to_uint8(image_hwc):
    import numpy as np

    image = image_hwc.astype(np.float32, copy=False)
    channels = [
        _stretch_channel_to_uint8(image[:, :, channel_index])
        for channel_index in range(image.shape[2])
    ]
    return np.stack(channels, axis=2)


def _stretch_channel_to_uint8(channel):
    import numpy as np

    image = channel.astype(np.float32, copy=False)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    nonzero = finite[finite > 0]
    values = nonzero if nonzero.size else finite
    min_value = float(np.percentile(values, 1))
    max_value = float(np.percentile(values, 99))
    if max_value <= min_value:
        min_value = float(values.min())
        max_value = float(values.max())
    if max_value <= min_value:
        return np.zeros(image.shape, dtype=np.uint8)

    scaled = (image - min_value) / (max_value - min_value) * 255.0
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _shutdown_loader_iterator(iterator: object) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _close_loader_dataset(loader: object) -> None:
    dataset = getattr(loader, "dataset", None)
    close = getattr(dataset, "close", None)
    if callable(close):
        close()


def _batch_meta(
    *,
    batch_index: int,
    images: object,
    masks: object,
    tile_files: list[dict[str, Any]],
    fetch_sec: float,
    save_sec: float,
) -> dict[str, Any]:
    image_min = float(images.min().item())
    image_max = float(images.max().item())
    per_channel_min = [
        float(value) for value in images.amin(dim=(0, 2, 3)).detach().cpu().tolist()
    ]
    per_channel_max = [
        float(value) for value in images.amax(dim=(0, 2, 3)).detach().cpu().tolist()
    ]
    per_channel_mean = [
        float(value) for value in images.mean(dim=(0, 2, 3)).detach().cpu().tolist()
    ]
    mask_min = float(masks.min().item())
    mask_max = float(masks.max().item())
    positive_mask_pixels = int((masks > 0).sum().item())
    return {
        "batch_index": batch_index,
        "images_shape": list(images.shape),
        "masks_shape": list(masks.shape),
        "images_dtype": str(images.dtype),
        "masks_dtype": str(masks.dtype),
        "image_min": image_min,
        "image_max": image_max,
        "per_channel_min": per_channel_min,
        "per_channel_max": per_channel_max,
        "per_channel_mean": per_channel_mean,
        "mask_min": mask_min,
        "mask_max": mask_max,
        "positive_mask_pixels": positive_mask_pixels,
        "tile_count": len(tile_files),
        "tiles": tile_files,
        "fetch_sec": fetch_sec,
        "save_sec": save_sec,
        "batch_total_sec": fetch_sec + save_sec,
    }


def _batch_report_item(batch_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_index": batch_meta["batch_index"],
        "fetch_sec": batch_meta["fetch_sec"],
        "save_sec": batch_meta["save_sec"],
        "batch_total_sec": batch_meta["batch_total_sec"],
        "images_shape": batch_meta["images_shape"],
        "masks_shape": batch_meta["masks_shape"],
        "images_dtype": batch_meta["images_dtype"],
        "image_min": batch_meta["image_min"],
        "image_max": batch_meta["image_max"],
        "per_channel_min": batch_meta["per_channel_min"],
        "per_channel_max": batch_meta["per_channel_max"],
        "per_channel_mean": batch_meta["per_channel_mean"],
        "positive_mask_pixels": batch_meta["positive_mask_pixels"],
        "tile_count": batch_meta["tile_count"],
    }


def _write_timing_report(
    *,
    status: str,
    error: str | None,
    paths: dict[str, str],
    params: dict[str, Any],
    timings: dict[str, float],
    batches: list[dict[str, Any]],
) -> None:
    _write_json(
        OUT_DIR / "modules_test_timing_report.json",
        {
            "status": status,
            "error": error,
            "paths": paths,
            "params": params,
            "timings": timings,
            "batches": batches,
        },
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _finish_timings(timings: dict[str, float], total_started: float) -> dict[str, float]:
    finished = dict(timings)
    finished["total_sec"] = perf_counter() - total_started
    return finished


def _paths() -> dict[str, str]:
    return {
        "images_dir": str(IMAGES_DIR),
        "dataset_dir": str(DATASET_DIR),
        "out_dir": str(OUT_DIR),
        "scenes_file": str(SCENES_FILE),
        "annotation_file": str(ANNOTATION_FILE),
        "settings_file": str(OUT_DIR / "modules_test.local.yaml"),
        "train_vrt": str(OUT_DIR / "train.vrt"),
        "val_vrt": str(OUT_DIR / "val.vrt"),
        "tile_batches_dir": str(OUT_DIR / "tile_batches"),
    }


def _params(*, saved_batches: int) -> dict[str, Any]:
    return {
        "val_fraction": VAL_FRACTION,
        "tile_size": TILE_SIZE,
        "stride": STRIDE,
        "num_workers": NUM_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "seed": SEED,
        "augmentation_level": AUGMENTATION_LEVEL,
        "batch_size": BATCH_SIZE,
        "requested_batches": REQUESTED_BATCHES,
        "saved_batches": saved_batches,
        "mode": MODE,
    }


if __name__ == "__main__":
    raise SystemExit(main())
