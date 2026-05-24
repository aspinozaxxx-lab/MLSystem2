"""Локальный golden-test совместимости tensor ABI с Geoalert inference path."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


INPUT_DIR = Path(r"D:\Projects\TestDataset")
RESULT_DIR = INPUT_DIR / "MLSystem-Geolert-tensor-test-RESULT"


def main() -> int:
    input_raster = _find_input_raster(INPUT_DIR)
    if input_raster is None:
        print(f"В папке {INPUT_DIR} не найдено файлов *.tif или *.tiff.", file=sys.stderr)
        return 1

    _prepare_result_dir(RESULT_DIR)
    report_path = RESULT_DIR / "tensor_test_report.json"
    try:
        report = _run_test(input_raster, RESULT_DIR)
        _write_json(report_path, report)
        if report["status"] == "ok":
            print(f"Golden-test завершен успешно: {report_path}")
            return 0
        print(f"Golden-test завершен с ошибкой: {report['error']}", file=sys.stderr)
        print(f"Отчет: {report_path}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        report = _error_report(input_raster, RESULT_DIR, str(exc))
        _write_json(report_path, report)
        print(f"Golden-test завершен с ошибкой: {exc}", file=sys.stderr)
        print(f"Отчет: {report_path}", file=sys.stderr)
        return 1


def _run_test(input_raster: Path, result_dir: Path) -> dict[str, Any]:
    from mlsystem2.settings.api import load_settings
    from mlsystem2.tile_preparation.api import create_tile_dataloader
    from mlsystem2.tile_preparation.contracts import TileDataloaderRequest

    tile_size = _env_positive_int("MLSYSTEM_GEOALERT_TENSOR_TEST_TILE_SIZE", 512)
    stride = _env_positive_int("MLSYSTEM_GEOALERT_TENSOR_TEST_STRIDE", tile_size)
    samples_requested = _env_positive_int("MLSYSTEM_GEOALERT_TENSOR_TEST_SAMPLES", 10)

    empty_geojson_path = result_dir / "empty.geojson"
    _write_empty_geojson(empty_geojson_path)

    vrt_path = result_dir / "input.vrt"
    _build_vrt(input_raster, vrt_path)
    vrt_xml = vrt_path.read_text(encoding="utf-8")

    with rasterio.open(input_raster) as dataset:
        raster_info = {
            "width": dataset.width,
            "height": dataset.height,
            "count": dataset.count,
            "dtypes": list(dataset.dtypes),
            "nodata": _json_value(_resolve_nodata(dataset)),
        }
        settings_path = result_dir / "MLSystem-Geolert-tensor-test.local.yaml"
        _write_local_settings(
            settings_path,
            result_dir=result_dir,
            input_dir=INPUT_DIR,
            empty_geojson_path=empty_geojson_path,
            tile_size=tile_size,
            stride=stride,
            input_channels=dataset.count,
        )

    load_settings(settings_path)
    loader = create_tile_dataloader(
        TileDataloaderRequest(
            vrt_xml=vrt_xml,
            annotation_file=empty_geojson_path,
            batch_size=1,
            mode="val",
        )
    )

    samples: list[dict[str, Any]] = []
    max_abs_diff_overall = 0.0
    all_samples_match = True
    error: str | None = None
    iterator = iter(loader)
    try:
        expected_iterator = _expected_samples(input_raster, tile_size=tile_size, stride=stride)
        for sample_index in range(samples_requested):
            try:
                window_meta, expected_image, nodata = next(expected_iterator)
            except StopIteration:
                break

            try:
                mlsystem_image = _next_mlsystem_image(iterator)
            except StopIteration:
                error = "DataLoader закончился раньше expected sampler."
                all_samples_match = False
                break

            sample_dir = result_dir / "samples" / f"sample_{sample_index:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_meta = _compare_and_save_sample(
                sample_dir=sample_dir,
                sample_index=sample_index,
                window_meta=window_meta,
                mlsystem_image=mlsystem_image,
                expected_image=expected_image,
                nodata=nodata,
            )
            samples.append(sample_meta)

            max_abs_diff = sample_meta["max_abs_diff"]
            if max_abs_diff is not None:
                max_abs_diff_overall = max(max_abs_diff_overall, float(max_abs_diff))
            if not sample_meta["shape_match"]:
                all_samples_match = False
                error = "Обнаружено несовпадение shape."
            elif not sample_meta["dtype_match"]:
                all_samples_match = False
                error = "MLSystem2 image tensor имеет dtype не float32."
            elif not sample_meta["exact_match"]:
                all_samples_match = False
                error = "Обнаружено несовпадение значений tensor."
            else:
                if float(max_abs_diff) != 0.0:
                    all_samples_match = False
                    error = "Обнаружено несовпадение значений tensor."

        samples_checked = len(samples)
        if samples_checked == 0 and error is None:
            error = "Нет samples для проверки."
            all_samples_match = False
        status = "ok" if samples_checked > 0 and all_samples_match else "error"
        return {
            "status": status,
            "error": None if status == "ok" else error,
            "input_raster": str(input_raster),
            "result_dir": str(result_dir),
            "tile_size": tile_size,
            "stride": stride,
            "batch_size": 1,
            "samples_requested": samples_requested,
            "samples_checked": samples_checked,
            "all_samples_match": all_samples_match,
            "max_abs_diff_overall": max_abs_diff_overall,
            "raster": raster_info,
            "samples": samples,
        }
    finally:
        _shutdown_loader_iterator(iterator)
        _close_loader_dataset(loader)


def _expected_samples(
    input_raster: Path,
    *,
    tile_size: int,
    stride: int,
) -> Iterator[tuple[dict[str, int], np.ndarray, object]]:
    with rasterio.open(input_raster) as dataset:
        nodata = _resolve_nodata(dataset)
        for y in range(0, dataset.height, stride):
            for x in range(0, dataset.width, stride):
                window = Window(x, y, tile_size, tile_size)
                sample = dataset.read(
                    window=window,
                    boundless=True,
                    fill_value=nodata,
                    out_shape=(dataset.count, tile_size, tile_size),
                    masked=False,
                )
                if _is_fully_nodata(sample, nodata):
                    continue
                yield (
                    {"x": x, "y": y, "width": tile_size, "height": tile_size},
                    sample.astype(np.float32, copy=False),
                    nodata,
                )


def _compare_and_save_sample(
    *,
    sample_dir: Path,
    sample_index: int,
    window_meta: dict[str, int],
    mlsystem_image: np.ndarray,
    expected_image: np.ndarray,
    nodata: object,
) -> dict[str, Any]:
    np.save(sample_dir / "mlsystem_image.npy", mlsystem_image)
    np.save(sample_dir / "geoalert_expected_image.npy", expected_image)

    shape_match = mlsystem_image.shape == expected_image.shape
    dtype_match = mlsystem_image.dtype == np.dtype("float32")
    if shape_match:
        diff = mlsystem_image - expected_image
        max_abs_diff = float(np.max(np.abs(diff))) if diff.size else 0.0
        exact_match = bool(np.allclose(mlsystem_image, expected_image, atol=0.0, rtol=0.0))
    else:
        diff = np.array([], dtype=np.float32)
        max_abs_diff = None
        exact_match = False
    np.save(sample_dir / "diff.npy", diff)

    meta = {
        "sample_index": sample_index,
        "window": window_meta,
        "shape": list(mlsystem_image.shape),
        "expected_shape": list(expected_image.shape),
        "mlsystem_dtype": str(mlsystem_image.dtype),
        "expected_dtype": str(expected_image.dtype),
        "dtype_match": dtype_match,
        "shape_match": shape_match,
        "exact_match": exact_match,
        "max_abs_diff": max_abs_diff,
        "per_channel_min": _per_channel_stat(mlsystem_image, np.min),
        "per_channel_max": _per_channel_stat(mlsystem_image, np.max),
        "per_channel_mean": _per_channel_stat(mlsystem_image, np.mean),
        "nodata": _json_value(nodata),
        "nodata_pixels_all_channels": int(
            np.count_nonzero(_nodata_pixels(expected_image, nodata))
        ),
    }
    _write_json(sample_dir / "meta.json", meta)
    return meta


def _next_mlsystem_image(iterator: object) -> np.ndarray:
    batch = next(iterator)
    if not isinstance(batch, tuple) or len(batch) not in {2, 3}:
        raise RuntimeError("DataLoader batch должен быть tuple(images, masks[, batch_meta]).")
    images = batch[0]
    if list(images.shape[:1]) != [1]:
        raise RuntimeError("Golden-test ожидает batch_size=1.")
    return images[0].detach().cpu().numpy()


def _write_local_settings(
    path: Path,
    *,
    result_dir: Path,
    input_dir: Path,
    empty_geojson_path: Path,
    tile_size: int,
    stride: int,
    input_channels: int,
) -> None:
    settings_text = f"""
runtime:
  project_root: {_yaml_string(result_dir)}
  scratch_root: {_yaml_string(result_dir / "scratch")}
  logs_root: {_yaml_string(result_dir / "logs")}
  cleanup_scratch_after_mlflow_log: false

dataset:
  images_dir: {_yaml_string(input_dir)}
  scenes_file: {_yaml_string(result_dir / "scenes.txt")}
  annotation_file: {_yaml_string(empty_geojson_path)}
  val_fraction: 0.2

tile_preparation:
  tile_size: {tile_size}
  stride: {stride}
  num_workers: 0
  prefetch_factor: 2
  seed: 42
  augmentation_level: 0

train:
  model_name: segformer_b2
  input_channels: {input_channels}
  output_channels: 1
  pretrained: false
  initial_checkpoint_uri: null
  epochs: 1
  batch_size: 1
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
  checkpoint_uri: {_yaml_string(result_dir / "latest.pt")}
  threshold: 0.5
  batch_size: 1
  device: cpu

mlflow:
  enabled: false
  tracking_uri: {_yaml_string(result_dir / "mlruns")}
  experiment_name: MLSystem-Geolert-tensor-test
"""
    path.write_text(settings_text.lstrip(), encoding="utf-8")


def _build_vrt(input_raster: Path, vrt_path: Path) -> None:
    gdalbuildvrt = _find_gdalbuildvrt()
    if gdalbuildvrt is None:
        raise RuntimeError(
            "gdalbuildvrt не найден. Установите GDAL/QGIS или добавьте gdalbuildvrt в PATH."
        )
    command = [
        gdalbuildvrt,
        "-overwrite",
        vrt_path.as_posix(),
        input_raster.resolve().as_posix(),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"gdalbuildvrt завершился с ошибкой: {message}")
    if not vrt_path.is_file():
        raise RuntimeError(f"gdalbuildvrt не создал VRT: {vrt_path}")


def _find_gdalbuildvrt() -> str | None:
    executable = shutil.which("gdalbuildvrt")
    if executable is not None:
        return executable

    candidates = [
        Path(r"C:\Program Files\QGIS 3.44.10\bin\gdalbuildvrt.exe"),
        Path(r"C:\Program Files\QGIS 3.42.0\bin\gdalbuildvrt.exe"),
        Path(r"C:\Program Files\QGIS 3.40.0\bin\gdalbuildvrt.exe"),
    ]
    candidates.extend(
        sorted(Path(r"C:\Program Files").glob("QGIS */bin/gdalbuildvrt.exe"))
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _find_input_raster(input_dir: Path) -> Path | None:
    if not input_dir.is_dir():
        return None
    rasters = [*input_dir.glob("*.tif"), *input_dir.glob("*.tiff")]
    if not rasters:
        return None
    return sorted(rasters, key=lambda path: path.name.lower())[0]


def _prepare_result_dir(result_dir: Path) -> None:
    result_dir = result_dir.resolve()
    expected_parent = INPUT_DIR.resolve()
    if result_dir.parent != expected_parent:
        raise RuntimeError(f"Небезопасный путь результата: {result_dir}")
    if result_dir.exists():
        shutil.rmtree(result_dir)
    (result_dir / "samples").mkdir(parents=True, exist_ok=True)


def _write_empty_geojson(path: Path) -> None:
    _write_json(path, {"type": "FeatureCollection", "features": []})


def _resolve_nodata(dataset: rasterio.io.DatasetReader) -> object:
    if dataset.nodata is not None:
        return dataset.nodata
    for nodata in dataset.nodatavals:
        if nodata is not None:
            return nodata
    return 0


def _is_fully_nodata(image: np.ndarray, nodata: object) -> bool:
    if _is_nan(nodata):
        return bool(np.all(np.isnan(image)))
    return bool(np.all(image == nodata))


def _nodata_pixels(image: np.ndarray, nodata: object) -> np.ndarray:
    if _is_nan(nodata):
        return np.all(np.isnan(image), axis=0)
    return np.all(image == nodata, axis=0)


def _is_nan(value: object) -> bool:
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def _per_channel_stat(image: np.ndarray, stat_func: Any) -> list[float]:
    if image.ndim != 3:
        return []
    return [float(stat_func(image[channel_index])) for channel_index in range(image.shape[0])]


def _env_positive_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} должен быть целым числом.") from exc
    if value <= 0:
        raise RuntimeError(f"{name} должен быть больше 0.")
    return value


def _shutdown_loader_iterator(iterator: object) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _close_loader_dataset(loader: object) -> None:
    dataset = getattr(loader, "dataset", None)
    close = getattr(dataset, "close", None)
    if callable(close):
        close()


def _error_report(input_raster: Path, result_dir: Path, error: str) -> dict[str, Any]:
    return {
        "status": "error",
        "error": error,
        "input_raster": str(input_raster),
        "result_dir": str(result_dir),
        "tile_size": None,
        "stride": None,
        "batch_size": 1,
        "samples_requested": None,
        "samples_checked": 0,
        "all_samples_match": False,
        "max_abs_diff_overall": None,
        "raster": None,
        "samples": [],
    }


def _yaml_string(value: Path) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _json_value(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    return value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
