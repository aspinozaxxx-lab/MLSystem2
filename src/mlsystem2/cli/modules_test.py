"""Локальная диагностика связки dataset_preparing и tile_preparation."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

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
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
SEED = 42
AUGMENTATION_LEVEL = 3
SMART_TILING = True
BATCH_SIZE = 4
REQUESTED_BATCHES = 100
MODE = "train"

BLACK_SAMPLE_STEP = 64
BLACK_VALUE_EPS = 1e-6
MAX_BLACK_EXAMPLES = 20
PROGRESS_EVERY_BATCHES = 100


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
    tile_scan: dict[str, Any] | None = None

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
                tile_scan=tile_scan,
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
        batches, tile_scan = _scan_and_save_batches(loader)
        timings["all_batches_sec"] = perf_counter() - batches_started
        params = _params(saved_batches=len(batches))

        if tile_scan["total_batches"] == 0:
            error = "DataLoader не вернул ни одного batch."
            _write_timing_report(
                status="error",
                error=error,
                paths=paths,
                params=params,
                timings=_finish_timings(timings, total_started),
                batches=batches,
                tile_scan=tile_scan,
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
            tile_scan=tile_scan,
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
                tile_scan=tile_scan,
            )
        except Exception as report_exc:  # noqa: BLE001
            error = f"{error}; не удалось записать timing report: {report_exc}"
        print(error, file=sys.stderr)
        return 1


def _prepare_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for child in (OUT_DIR / "tile_batches", OUT_DIR / "black_tile_examples"):
        if child.exists():
            shutil.rmtree(child)
        child.mkdir(parents=True, exist_ok=True)


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
  smart_tiling: {str(SMART_TILING).lower()}

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


def _scan_and_save_batches(loader: object) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    iterator = iter(loader)
    batches: list[dict[str, Any]] = []
    scan_summary = _initial_tile_scan(loader)
    warnings: list[str] = []
    black_examples: list[dict[str, Any]] = []

    try:
        batch_index = 0
        while True:
            fetch_started = perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            fetch_sec = perf_counter() - fetch_started

            images, masks, source_batch_meta = _expect_batch(batch)
            image_array = _to_numpy(images)
            mask_array = _to_numpy(masks)
            _validate_batch_arrays(image_array, mask_array)

            tile_details = _tile_details(
                image_array=image_array,
                mask_array=mask_array,
                batch_meta=source_batch_meta,
                batch_index=batch_index,
                warnings=warnings,
            )
            _update_scan_summary(scan_summary, tile_details)

            tile_files: list[dict[str, Any]] = []
            save_sec = 0.0
            if batch_index < REQUESTED_BATCHES:
                save_started = perf_counter()
                batch_dir = OUT_DIR / "tile_batches" / f"batch_{batch_index:04d}"
                batch_dir.mkdir(parents=True, exist_ok=True)
                tile_files = _save_tiles_as_png(images, masks, batch_dir)
                batch_meta = _batch_meta(
                    batch_index=batch_index,
                    images=images,
                    masks=masks,
                    tile_files=tile_files,
                    tile_details=tile_details,
                    fetch_sec=fetch_sec,
                    save_sec=0.0,
                )
                save_sec = perf_counter() - save_started
                batch_meta["save_sec"] = save_sec
                batch_meta["batch_total_sec"] = fetch_sec + save_sec
                _write_json(batch_dir / "batch_meta.json", batch_meta)
                batches.append(_batch_report_item(batch_meta))

            _save_black_examples(
                image_array=image_array,
                mask_array=mask_array,
                tile_details=tile_details,
                black_examples=black_examples,
            )

            batch_index += 1
            if batch_index % PROGRESS_EVERY_BATCHES == 0:
                print(
                    "scanned "
                    f"batches={batch_index}, "
                    f"tiles={scan_summary['total_tiles']}, "
                    f"black={scan_summary['black_tiles']}, "
                    f"positive={scan_summary['positive_tiles']}"
                )

        scan_summary["warnings"] = warnings
        scan_summary["black_examples"] = black_examples
        return batches, scan_summary
    finally:
        _shutdown_loader_iterator(iterator)
        _close_loader_dataset(loader)


def _expect_batch(batch: object) -> tuple[object, object, dict[str, Any]]:
    if not isinstance(batch, tuple) or len(batch) not in {2, 3}:
        raise RuntimeError("DataLoader batch должен быть tuple(images, masks[, batch_meta]).")
    if len(batch) == 2:
        return batch[0], batch[1], {}
    return batch[0], batch[1], dict(batch[2] or {})


def _initial_tile_scan(loader: object) -> dict[str, Any]:
    dataset = getattr(loader, "dataset", None)
    return {
        "total_batches": 0,
        "total_tiles": 0,
        "black_tiles": 0,
        "non_black_tiles": 0,
        "nonfinite_image_tiles": 0,
        "positive_tiles": 0,
        "negative_tiles": 0,
        "real_tiles": 0,
        "augmented_tiles": 0,
        "positive_real_tiles": 0,
        "positive_augmented_tiles": 0,
        "negative_real_tiles": 0,
        "negative_augmented_tiles": 0,
        "positive_black_tiles": 0,
        "negative_black_tiles": 0,
        "positive_mask_pixels": 0,
        "mask_positive_pixels_total": 0,
        "source_rect_count": _dataset_attr(dataset, "source_rect_count"),
        "candidate_window_count": _dataset_attr(dataset, "candidate_window_count"),
        "uses_vrt_source_rects": _dataset_attr(dataset, "uses_vrt_source_rects"),
        "dataset_len": _safe_len(dataset),
        "loader_len": _safe_len(loader),
        "augmentation_level": AUGMENTATION_LEVEL,
        "smart_tiling": SMART_TILING,
        "black_detector": {
            "method": "sparse_grid_all_channels",
            "step": BLACK_SAMPLE_STEP,
            "eps": BLACK_VALUE_EPS,
            "definition": "all sampled grid pixels across all channels are <= eps",
        },
        "warnings": [],
        "black_examples": [],
    }


def _tile_details(
    *,
    image_array,
    mask_array,
    batch_meta: dict[str, Any],
    batch_index: int,
    warnings: list[str],
) -> list[dict[str, Any]]:
    tile_count = int(image_array.shape[0])
    tile_augmented = _bool_flags(batch_meta.get("tile_augmented"), tile_count)
    tile_positive = _bool_flags(batch_meta.get("tile_positive"), tile_count)
    details: list[dict[str, Any]] = []

    for tile_index in range(tile_count):
        stats = _tile_stats(
            image_array[tile_index],
            mask_array[tile_index],
            augmented=tile_augmented[tile_index],
        )
        if tile_positive[tile_index] != stats["positive"]:
            warnings.append(
                "batch_meta tile_positive отличается от mask-derived positive: "
                f"batch={batch_index}, tile={tile_index}, "
                f"meta={tile_positive[tile_index]}, mask={stats['positive']}"
            )
        details.append({"tile_index": tile_index, **stats})

    meta_positive_count = int(batch_meta.get("positive_tile_count", 0) or 0)
    mask_positive_count = sum(1 for item in details if item["positive"])
    if "positive_tile_count" in batch_meta and meta_positive_count != mask_positive_count:
        warnings.append(
            "batch_meta positive_tile_count отличается от mask-derived count: "
            f"batch={batch_index}, meta={meta_positive_count}, mask={mask_positive_count}"
        )

    meta_augmented_count = int(batch_meta.get("augmented_tile_count", 0) or 0)
    flag_augmented_count = sum(1 for item in details if item["augmented"])
    if "augmented_tile_count" in batch_meta and meta_augmented_count != flag_augmented_count:
        warnings.append(
            "batch_meta augmented_tile_count отличается от tile_augmented count: "
            f"batch={batch_index}, meta={meta_augmented_count}, flags={flag_augmented_count}"
        )

    return details


def _bool_flags(value: object, tile_count: int) -> list[bool]:
    if isinstance(value, list) and len(value) == tile_count:
        return [bool(item) for item in value]
    return [False for _ in range(tile_count)]


def _tile_stats(image_chw, mask_chw, augmented: bool | None) -> dict[str, Any]:
    nonfinite_image_pixels, image_min, image_max = _image_finite_stats(image_chw)
    positive_mask_pixels = int(np.count_nonzero(mask_chw > 0.5))
    return {
        "black": _is_black_tile(image_chw),
        "positive": positive_mask_pixels > 0,
        "augmented": bool(augmented),
        "nonfinite_image_pixels": nonfinite_image_pixels,
        "positive_mask_pixels": positive_mask_pixels,
        "image_min": image_min,
        "image_max": image_max,
    }


def _image_finite_stats(image_chw) -> tuple[int, float | None, float | None]:
    nonfinite_pixels = 0
    image_min: float | None = None
    image_max: float | None = None
    for channel in image_chw:
        finite = np.isfinite(channel)
        finite_count = int(np.count_nonzero(finite))
        nonfinite_pixels += int(channel.size - finite_count)
        if finite_count == 0:
            continue
        channel_min = float(np.min(channel, where=finite, initial=np.inf))
        channel_max = float(np.max(channel, where=finite, initial=-np.inf))
        image_min = channel_min if image_min is None else min(image_min, channel_min)
        image_max = channel_max if image_max is None else max(image_max, channel_max)
    return nonfinite_pixels, image_min, image_max


def _is_black_tile(image_chw) -> bool:
    if image_chw.ndim != 3:
        raise RuntimeError("Image tile должен иметь форму [C, H, W].")
    _channels, height, width = image_chw.shape
    y_indices = _sparse_indices(height)
    x_indices = _sparse_indices(width)
    sample = image_chw[:, y_indices][:, :, x_indices]
    if not np.isfinite(sample).all():
        return False
    return bool(np.all(np.abs(sample) <= BLACK_VALUE_EPS))


def _sparse_indices(size: int) -> list[int]:
    if size <= 0:
        return []
    indices = set(range(0, size, BLACK_SAMPLE_STEP))
    indices.add(size // 2)
    indices.add(size - 1)
    return sorted(indices)


def _update_scan_summary(scan_summary: dict[str, Any], tile_details: list[dict[str, Any]]) -> None:
    scan_summary["total_batches"] += 1
    scan_summary["total_tiles"] += len(tile_details)
    for detail in tile_details:
        positive = bool(detail["positive"])
        augmented = bool(detail["augmented"])
        black = bool(detail["black"])
        scan_summary["positive_mask_pixels"] += int(detail["positive_mask_pixels"])
        scan_summary["mask_positive_pixels_total"] += int(detail["positive_mask_pixels"])
        if int(detail["nonfinite_image_pixels"]) > 0:
            scan_summary["nonfinite_image_tiles"] += 1
        if black:
            scan_summary["black_tiles"] += 1
            if positive:
                scan_summary["positive_black_tiles"] += 1
            else:
                scan_summary["negative_black_tiles"] += 1
        else:
            scan_summary["non_black_tiles"] += 1
        if positive:
            scan_summary["positive_tiles"] += 1
        else:
            scan_summary["negative_tiles"] += 1
        if augmented:
            scan_summary["augmented_tiles"] += 1
            if positive:
                scan_summary["positive_augmented_tiles"] += 1
            else:
                scan_summary["negative_augmented_tiles"] += 1
        else:
            scan_summary["real_tiles"] += 1
            if positive:
                scan_summary["positive_real_tiles"] += 1
            else:
                scan_summary["negative_real_tiles"] += 1


def _save_black_examples(
    *,
    image_array,
    mask_array,
    tile_details: list[dict[str, Any]],
    black_examples: list[dict[str, Any]],
) -> None:
    if len(black_examples) >= MAX_BLACK_EXAMPLES:
        return
    examples_dir = OUT_DIR / "black_tile_examples"
    for detail in tile_details:
        if len(black_examples) >= MAX_BLACK_EXAMPLES:
            return
        if not detail["black"]:
            continue
        example_index = len(black_examples)
        tile_index = int(detail["tile_index"])
        prefix = f"black_{example_index:04d}_batch_tile_{tile_index:04d}"
        preview_path = examples_dir / f"{prefix}_preview_rgb.png"
        mask_path = examples_dir / f"{prefix}_mask.png"
        _save_preview_png(image_array[tile_index], preview_path)
        _save_mask_png(mask_array[tile_index], mask_path)
        black_examples.append(
            {
                "preview_rgb": str(preview_path),
                "mask": str(mask_path),
                "tile_detail": detail,
            }
        )


def _validate_batch_arrays(image_array, mask_array) -> None:
    if image_array.ndim != 4 or mask_array.ndim != 4:
        raise RuntimeError("Batch должен иметь формы images [B, C, H, W] и masks [B, 1, H, W].")
    if image_array.shape[0] != mask_array.shape[0]:
        raise RuntimeError("Количество images и masks в batch не совпадает.")


def _save_tiles_as_png(images: object, masks: object, batch_dir: Path) -> list[dict[str, Any]]:
    image_array = _to_numpy(images)
    mask_array = _to_numpy(masks)
    _validate_batch_arrays(image_array, mask_array)

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
    from PIL import Image

    if mask_chw.shape[0] != 1:
        raise RuntimeError("Mask tile должен иметь один канал.")
    mask_u8 = (np.clip(mask_chw[0], 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(mask_u8, mode="L").save(path)


def _stretch_image_to_uint8(image_hwc):
    image = image_hwc.astype(np.float32, copy=False)
    channels = [
        _stretch_channel_to_uint8(image[:, :, channel_index])
        for channel_index in range(image.shape[2])
    ]
    return np.stack(channels, axis=2)


def _stretch_channel_to_uint8(channel):
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
    tile_details: list[dict[str, Any]],
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
    counters = _batch_tile_counters(tile_details)
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
        **counters,
        "tile_details": tile_details,
        "tiles": tile_files,
        "fetch_sec": fetch_sec,
        "save_sec": save_sec,
        "batch_total_sec": fetch_sec + save_sec,
    }


def _batch_tile_counters(tile_details: list[dict[str, Any]]) -> dict[str, int]:
    black = sum(1 for item in tile_details if item["black"])
    positive = sum(1 for item in tile_details if item["positive"])
    augmented = sum(1 for item in tile_details if item["augmented"])
    positive_augmented = sum(1 for item in tile_details if item["positive"] and item["augmented"])
    negative_augmented = sum(1 for item in tile_details if not item["positive"] and item["augmented"])
    positive_real = sum(1 for item in tile_details if item["positive"] and not item["augmented"])
    negative_real = sum(1 for item in tile_details if not item["positive"] and not item["augmented"])
    positive_black = sum(1 for item in tile_details if item["positive"] and item["black"])
    negative_black = sum(1 for item in tile_details if not item["positive"] and item["black"])
    return {
        "black_tile_count": black,
        "positive_tile_count": positive,
        "negative_tile_count": len(tile_details) - positive,
        "augmented_tile_count": augmented,
        "real_tile_count": len(tile_details) - augmented,
        "positive_real_tile_count": positive_real,
        "positive_augmented_tile_count": positive_augmented,
        "negative_real_tile_count": negative_real,
        "negative_augmented_tile_count": negative_augmented,
        "black_positive_tile_count": positive_black,
        "black_negative_tile_count": negative_black,
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
        "black_tile_count": batch_meta["black_tile_count"],
        "positive_tile_count": batch_meta["positive_tile_count"],
        "negative_tile_count": batch_meta["negative_tile_count"],
        "augmented_tile_count": batch_meta["augmented_tile_count"],
        "real_tile_count": batch_meta["real_tile_count"],
        "positive_real_tile_count": batch_meta["positive_real_tile_count"],
        "positive_augmented_tile_count": batch_meta["positive_augmented_tile_count"],
        "negative_real_tile_count": batch_meta["negative_real_tile_count"],
        "negative_augmented_tile_count": batch_meta["negative_augmented_tile_count"],
        "black_positive_tile_count": batch_meta["black_positive_tile_count"],
        "black_negative_tile_count": batch_meta["black_negative_tile_count"],
    }


def _write_timing_report(
    *,
    status: str,
    error: str | None,
    paths: dict[str, str],
    params: dict[str, Any],
    timings: dict[str, float],
    batches: list[dict[str, Any]],
    tile_scan: dict[str, Any] | None,
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
            "tile_scan": tile_scan,
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
        "black_examples_dir": str(OUT_DIR / "black_tile_examples"),
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
        "smart_tiling": SMART_TILING,
        "batch_size": BATCH_SIZE,
        "requested_batches": REQUESTED_BATCHES,
        "saved_batches": saved_batches,
        "mode": MODE,
        "black_sample_step": BLACK_SAMPLE_STEP,
        "black_value_eps": BLACK_VALUE_EPS,
    }


def _dataset_attr(dataset: object, name: str) -> object:
    if dataset is None:
        return None
    return getattr(dataset, name, None)


def _safe_len(value: object) -> int | None:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
