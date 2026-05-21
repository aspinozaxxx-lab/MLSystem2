"""Одноразовая подготовка GeoTIFF для VRT-мозаик."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


RAW_IMAGES_DIR = Path(r"D:\Projects\ImagesDeforestation")
PREPARED_IMAGES_DIR = Path(r"D:\Projects\ImagesDeforestationPrepared3857")
REPORT_PATH = Path(r"D:\Projects\test\prepare_images_for_vrt_report.json")
TARGET_CRS = "EPSG:3857"


def main() -> int:
    report = prepare_images_for_vrt(RAW_IMAGES_DIR, PREPARED_IMAGES_DIR, REPORT_PATH)
    return 0 if report["status"] == "ok" else 1


def prepare_images_for_vrt(
    raw_images_dir: Path,
    prepared_images_dir: Path,
    report_path: Path,
) -> dict[str, object]:
    files = sorted(
        [
            path
            for path in raw_images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ],
        key=lambda item: str(item).casefold(),
    )
    report_files: list[dict[str, object]] = []
    for input_path in files:
        output_path = prepared_images_dir / input_path.relative_to(raw_images_dir)
        record = {
            "input_path": input_path.resolve().as_posix(),
            "output_path": output_path.resolve().as_posix(),
            "status": "ok",
            "mask_source": None,
            "error": None,
        }
        try:
            record["mask_source"] = _prepare_one(input_path, output_path)
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = str(exc)
        report_files.append(record)

    error_count = sum(1 for item in report_files if item["status"] == "error")
    output_count = sum(1 for item in report_files if item["status"] == "ok")
    report = {
        "status": "error" if error_count else "ok",
        "input_count": len(files),
        "output_count": output_count,
        "error_count": error_count,
        "files": report_files,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _prepare_one(input_path: Path, output_path: Path) -> str:
    with rasterio.open(input_path) as src:
        if src.crs is None:
            raise ValueError("У исходного снимка нет CRS")
        if len(set(src.dtypes)) != 1:
            raise ValueError("Скрипт не меняет dtype, а снимок содержит разные dtype по каналам")

        source_mask, mask_source = _valid_mask(src)
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,
            TARGET_CRS,
            src.width,
            src.height,
            *src.bounds,
        )
        dst_mask = np.zeros((dst_height, dst_width), dtype=np.uint8)
        reproject(
            source=source_mask,
            destination=dst_mask,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            src_nodata=0,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
        dst_mask = np.where(dst_mask > 0, 255, 0).astype(np.uint8)

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            width=dst_width,
            height=dst_height,
            count=src.count,
            dtype=src.dtypes[0],
            crs=TARGET_CRS,
            transform=dst_transform,
            nodata=None,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="deflate",
            BIGTIFF="IF_SAFER",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK="YES"):
            with rasterio.open(output_path, "w", **profile) as dst:
                for band_index in range(1, src.count + 1):
                    dst_data = np.zeros((dst_height, dst_width), dtype=src.dtypes[band_index - 1])
                    reproject(
                        source=rasterio.band(src, band_index),
                        destination=dst_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=TARGET_CRS,
                        resampling=Resampling.nearest,
                    )
                    dst_data[dst_mask == 0] = 0
                    dst.write(dst_data, band_index)
                dst.write_mask(dst_mask)
    return mask_source


def _valid_mask(src: rasterio.io.DatasetReader) -> tuple[np.ndarray, str]:
    mask = src.dataset_mask()
    if mask.max() == 0:
        raise ValueError("dataset_mask полностью пустая")
    if mask.min() < mask.max():
        return mask.astype(np.uint8), "dataset_mask"

    empty_tuple = _detect_empty_tuple(src)
    if empty_tuple is None:
        raise ValueError("Не удалось определить пустую область по углам снимка")
    derived_mask = _mask_from_empty_tuple(src, empty_tuple)
    if derived_mask.max() == 0 or derived_mask.min() == derived_mask.max():
        raise ValueError("Маска по угловому tuple не отделяет пустые пиксели")
    return derived_mask, "corner_tuple"


def _detect_empty_tuple(src: rasterio.io.DatasetReader) -> tuple[Any, ...] | None:
    corner_pixels = [
        _read_pixel_tuple(src, 0, 0),
        _read_pixel_tuple(src, 0, src.width - 1),
        _read_pixel_tuple(src, src.height - 1, 0),
        _read_pixel_tuple(src, src.height - 1, src.width - 1),
    ]
    common_tuple, common_count = Counter(corner_pixels).most_common(1)[0]
    if common_count < 3:
        return None

    border_width = min(max(1, min(src.width, src.height) // 20), 128)
    border_total = 0
    border_matches = 0
    windows = [
        ((0, border_width), (0, src.width)),
        ((src.height - border_width, src.height), (0, src.width)),
        ((0, src.height), (0, border_width)),
        ((0, src.height), (src.width - border_width, src.width)),
    ]
    for window in windows:
        data = src.read(window=window)
        matches = _tuple_matches(data, common_tuple)
        border_matches += int(matches.sum())
        border_total += matches.size
    if border_total == 0 or border_matches / border_total < 0.10:
        return None
    return common_tuple


def _read_pixel_tuple(src: rasterio.io.DatasetReader, row: int, col: int) -> tuple[Any, ...]:
    data = src.read(window=((row, row + 1), (col, col + 1)))
    return tuple(item.item() for item in data[:, 0, 0])


def _mask_from_empty_tuple(src: rasterio.io.DatasetReader, empty_tuple: tuple[Any, ...]) -> np.ndarray:
    mask = np.zeros((src.height, src.width), dtype=np.uint8)
    for _, window in src.block_windows(1):
        data = src.read(window=window)
        empty_pixels = _tuple_matches(data, empty_tuple)
        row_start = int(window.row_off)
        row_stop = row_start + int(window.height)
        col_start = int(window.col_off)
        col_stop = col_start + int(window.width)
        mask[row_start:row_stop, col_start:col_stop] = np.where(empty_pixels, 0, 255)
    return mask


def _tuple_matches(data: np.ndarray, expected: tuple[Any, ...]) -> np.ndarray:
    matches = np.ones(data.shape[1:], dtype=bool)
    for band_index, expected_value in enumerate(expected):
        matches &= data[band_index] == expected_value
    return matches


if __name__ == "__main__":
    raise SystemExit(main())
