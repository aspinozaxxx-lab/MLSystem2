"""Одноразовая подготовка GeoTIFF для VRT-мозаик."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import boto3
import numpy as np
import rasterio
from rasterio.enums import ColorInterp, MaskFlags, Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import calculate_default_transform, reproject


RAW_IMAGES_DIR = Path(r"D:\Projects\ImagesDeforestation")
PREPARED_IMAGES_DIR = Path(r"D:\Projects\ImagesDeforestationPrepared3857")
REPORT_PATH = Path(r"D:\Projects\test\prepare_images_for_vrt_report.json")
TARGET_CRS = "EPSG:3857"
WORKERS = 8
SERVER_SOURCE_URI = "s3://mlsystems/images/kanopus/"
SERVER_PREPARED_IMAGES_DIR = Path("/data/mlsystem2/prepared_images/kanopus")
SERVER_REPORT_PATH = Path("/data/mlsystem2/prepared_images/report/prepare_images_for_vrt_report.json")
SERVER_WORKERS = 32


@dataclass(frozen=True)
class RunConfig:
    mode: str
    source_uri: str
    raw_images_dir: Path | None
    prepared_images_dir: Path
    report_path: Path
    workers: int


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _config_for_mode(args.mode)
    if config.mode == "server":
        report = prepare_images_for_vrt_from_s3(
            config.source_uri,
            config.prepared_images_dir,
            config.report_path,
            workers=config.workers,
            mode=config.mode,
        )
    else:
        if config.raw_images_dir is None:
            raise ValueError("Для local режима нужен локальный путь к исходным снимкам")
        report = prepare_images_for_vrt(
            config.raw_images_dir,
            config.prepared_images_dir,
            config.report_path,
            workers=config.workers,
            mode=config.mode,
            source_uri=config.source_uri,
        )
    return 0 if report["status"] == "ok" else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("local", "server"), default="local")
    return parser.parse_args(argv)


def _config_for_mode(mode: str) -> RunConfig:
    if mode == "local":
        return RunConfig(
            mode="local",
            source_uri=str(RAW_IMAGES_DIR),
            raw_images_dir=RAW_IMAGES_DIR,
            prepared_images_dir=PREPARED_IMAGES_DIR,
            report_path=REPORT_PATH,
            workers=WORKERS,
        )
    if mode == "server":
        return RunConfig(
            mode="server",
            source_uri=SERVER_SOURCE_URI,
            raw_images_dir=None,
            prepared_images_dir=SERVER_PREPARED_IMAGES_DIR,
            report_path=SERVER_REPORT_PATH,
            workers=SERVER_WORKERS,
        )
    raise ValueError(f"Неизвестный режим подготовки снимков: {mode}")


def prepare_images_for_vrt(
    raw_images_dir: Path,
    prepared_images_dir: Path,
    report_path: Path,
    workers: int = WORKERS,
    mode: str = "local",
    source_uri: str | None = None,
) -> dict[str, object]:
    files = _select_input_files(raw_images_dir)
    _log_run_start(mode, source_uri or raw_images_dir.resolve().as_posix(), len(files), workers)
    report_files: list[dict[str, object] | None] = [None] * len(files)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_prepare_record, raw_images_dir, prepared_images_dir, input_path): index
            for index, input_path in enumerate(files)
        }
        for future in as_completed(futures):
            record = future.result()
            report_files[futures[future]] = record
            _log_file_result(record)

    files_report = [item for item in report_files if item is not None]
    error_count = sum(1 for item in files_report if item["status"] == "error")
    output_count = sum(1 for item in files_report if item["status"] == "ok")
    report = {
        "status": "error" if error_count else "ok",
        "mode": mode,
        "source_uri": source_uri or raw_images_dir.resolve().as_posix(),
        "input_count": len(files),
        "output_count": output_count,
        "error_count": error_count,
        "workers": workers,
        "files": files_report,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def prepare_images_for_vrt_from_s3(
    source_uri: str,
    prepared_images_dir: Path,
    report_path: Path,
    workers: int = SERVER_WORKERS,
    mode: str = "server",
) -> dict[str, object]:
    rasters = _list_s3_rasters(source_uri)
    _log_run_start(mode, source_uri, len(rasters), workers)
    report_files: list[dict[str, object] | None] = [None] * len(rasters)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_prepare_s3_record, source_uri, prepared_images_dir, raster): index
            for index, raster in enumerate(rasters)
        }
        for future in as_completed(futures):
            record = future.result()
            report_files[futures[future]] = record
            _log_file_result(record)

    files_report = [item for item in report_files if item is not None]
    error_count = sum(1 for item in files_report if item["status"] == "error")
    output_count = sum(1 for item in files_report if item["status"] == "ok")
    report = {
        "status": "error" if error_count else "ok",
        "mode": mode,
        "source_uri": source_uri,
        "input_count": len(rasters),
        "output_count": output_count,
        "error_count": error_count,
        "workers": workers,
        "files": files_report,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _select_input_files(raw_images_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in raw_images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ],
        key=lambda item: str(item).casefold(),
    )


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    key: str


@dataclass(frozen=True)
class S3RasterObject:
    bucket: str
    key: str
    source_uri: str
    relative_path: PurePosixPath


def _prepare_record(raw_images_dir: Path, prepared_images_dir: Path, input_path: Path) -> dict[str, object]:
    output_path = prepared_images_dir / input_path.relative_to(raw_images_dir)
    record = _empty_file_record(input_path.resolve().as_posix(), output_path)
    try:
        record.update(_prepare_one(input_path, output_path))
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = str(exc)
    return record


def _prepare_s3_record(
    source_uri: str,
    prepared_images_dir: Path,
    raster: S3RasterObject,
) -> dict[str, object]:
    output_path = _output_path_for_s3_key(prepared_images_dir, source_uri, raster.key)
    record = _empty_file_record(raster.source_uri, output_path)
    try:
        with tempfile.TemporaryDirectory(prefix="mlsystem2_s3_image_") as temp_dir:
            temp_input = _download_s3_object_to_temp(raster, Path(temp_dir))
            record.update(_prepare_one(temp_input, output_path))
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = str(exc)
    return record


def _log_run_start(mode: str, source_uri: str, input_count: int, workers: int) -> None:
    print(
        f"START mode={mode} source_uri={source_uri} input_count={input_count} workers={workers}",
        flush=True,
    )


def _log_file_result(record: dict[str, object]) -> None:
    input_path = record.get("input_path")
    output_path = record.get("output_path")
    if record.get("status") == "error":
        print(
            f"ERROR input_path={input_path} output_path={output_path} error={record.get('error')}",
            flush=True,
        )
        return
    print(f"OK input_path={input_path} output_path={output_path}", flush=True)


def _empty_file_record(input_path: str, output_path: Path) -> dict[str, object]:
    return {
        "input_path": input_path,
        "output_path": output_path.resolve().as_posix(),
        "status": "ok",
        "source_count": None,
        "output_count": None,
        "source_nodata": None,
        "output_nodata": None,
        "source_dtypes": [],
        "output_dtypes": [],
        "source_colorinterp": [],
        "output_colorinterp": [],
        "source_descriptions": [],
        "output_descriptions": [],
        "source_had_alpha_colorinterp": False,
        "colorinterp_source_invalid": False,
        "has_internal_mask": False,
        "has_alpha": False,
        "has_sidecar_msk": False,
        "is_cog_check": False,
        "error": None,
    }


def _parse_s3_uri(uri: str) -> S3Uri:
    if not uri.startswith("s3://"):
        raise ValueError(f"Ожидался S3 URI: {uri}")
    without_scheme = uri.removeprefix("s3://")
    bucket, separator, key = without_scheme.partition("/")
    if not bucket or not separator:
        raise ValueError(f"Некорректный S3 URI: {uri}")
    return S3Uri(bucket=bucket, key=key)


def _list_s3_rasters(source_uri: str) -> list[S3RasterObject]:
    parsed = _parse_s3_uri(source_uri)
    prefix = _normalized_s3_prefix(parsed.key)
    client = _s3_client()
    rasters: list[S3RasterObject] = []
    continuation_token: str | None = None
    while True:
        request: dict[str, object] = {"Bucket": parsed.bucket, "Prefix": prefix}
        if continuation_token is not None:
            request["ContinuationToken"] = continuation_token
        response = client.list_objects_v2(**request)
        for item in response.get("Contents", []):
            key = str(item["Key"])
            if key.endswith("/") or PurePosixPath(key).suffix.lower() not in {".tif", ".tiff"}:
                continue
            relative_key = _relative_s3_key(prefix, key)
            rasters.append(
                S3RasterObject(
                    bucket=parsed.bucket,
                    key=key,
                    source_uri=f"s3://{parsed.bucket}/{key}",
                    relative_path=PurePosixPath(relative_key),
                )
            )
        if not response.get("IsTruncated"):
            break
        continuation_token = str(response["NextContinuationToken"])
    return sorted(rasters, key=lambda item: item.key.casefold())


def _download_s3_object_to_temp(raster: S3RasterObject, temp_dir: Path) -> Path:
    input_path = temp_dir / "input" / Path(*raster.relative_path.parts)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    _s3_client().download_file(raster.bucket, raster.key, str(input_path))
    return input_path


def _output_path_for_s3_key(prepared_images_dir: Path, source_uri: str, key: str) -> Path:
    parsed = _parse_s3_uri(source_uri)
    prefix = _normalized_s3_prefix(parsed.key)
    return prepared_images_dir.joinpath(*PurePosixPath(_relative_s3_key(prefix, key)).parts)


def _relative_s3_key(prefix: str, key: str) -> str:
    if not key.startswith(prefix):
        raise ValueError(f"S3 object не принадлежит prefix {prefix}: {key}")
    relative_key = key[len(prefix) :]
    if not relative_key:
        raise ValueError(f"S3 object не содержит относительный путь: {key}")
    return relative_key


def _normalized_s3_prefix(key: str) -> str:
    if not key:
        return ""
    return key if key.endswith("/") else f"{key}/"


def _s3_client() -> object:
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("MLFLOW_S3_ENDPOINT_URL")
    if endpoint_url:
        return boto3.client("s3", endpoint_url=endpoint_url)
    return boto3.client("s3")


def _prepare_one(input_path: Path, output_path: Path) -> dict[str, object]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        with rasterio.open(input_path) as src:
            return _prepare_open_dataset(src, output_path)


def _prepare_open_dataset(src: rasterio.io.DatasetReader, output_path: Path) -> dict[str, object]:
    if src.crs is None:
        raise ValueError("У исходного снимка нет CRS")
    if _has_invalid_geotransform(src):
        raise ValueError("У исходного снимка некорректный geotransform")
    if len(set(src.dtypes)) != 1:
        raise ValueError("Скрипт не меняет dtype, а снимок содержит разные dtype по каналам")

    source_nodata = _source_nodata(src)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src.crs,
        TARGET_CRS,
        src.width,
        src.height,
        *src.bounds,
    )

    source_count = src.count
    source_dtypes = tuple(src.dtypes)
    source_colorinterp = tuple(src.colorinterp)
    source_descriptions = tuple(src.descriptions)
    source_tags = _filtered_dataset_tags(src.tags())
    source_band_tags = [src.tags(band_index) for band_index in range(1, src.count + 1)]
    colorinterp_source_invalid = len(source_colorinterp) != src.count
    source_had_alpha_colorinterp = ColorInterp.alpha in source_colorinterp
    colorinterp = _sanitized_source_colorinterp(source_colorinterp, src.count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mlsystem2_prepare_image_") as temp_dir:
        temp_tif = Path(temp_dir) / "prepared_tmp.tif"
        _write_temp_geotiff(
            src=src,
            output_path=temp_tif,
            dst_width=dst_width,
            dst_height=dst_height,
            dst_transform=dst_transform,
            nodata=source_nodata,
            colorinterp=colorinterp,
            source_descriptions=source_descriptions,
            source_tags=source_tags,
            source_band_tags=source_band_tags,
        )
        _translate_to_cog(temp_tif, output_path, colorinterp, source_nodata)

    validation = _validate_output(
        output_path=output_path,
        source_count=source_count,
        source_dtypes=source_dtypes,
        source_nodata=source_nodata,
        expected_colorinterp=colorinterp,
        source_descriptions=source_descriptions,
        source_band_tags=source_band_tags,
    )
    return {
        "source_count": source_count,
        "source_nodata": _json_scalar(source_nodata),
        "source_dtypes": list(source_dtypes),
        "source_colorinterp": [item.name for item in source_colorinterp],
        "source_descriptions": list(source_descriptions),
        "source_had_alpha_colorinterp": source_had_alpha_colorinterp,
        "colorinterp_source_invalid": colorinterp_source_invalid,
        **validation,
    }


def _has_invalid_geotransform(src: rasterio.io.DatasetReader) -> bool:
    return bool(src.transform.is_identity or src.transform.a == 0 or src.transform.e == 0)


def _write_temp_geotiff(
    *,
    src: rasterio.io.DatasetReader,
    output_path: Path,
    dst_width: int,
    dst_height: int,
    dst_transform: object,
    nodata: float | int,
    colorinterp: tuple[ColorInterp, ...],
    source_descriptions: tuple[str | None, ...],
    source_tags: dict[str, str],
    source_band_tags: list[dict[str, str]],
) -> None:
    profile = {
        "driver": "GTiff",
        "width": dst_width,
        "height": dst_height,
        "count": src.count,
        "dtype": src.dtypes[0],
        "crs": TARGET_CRS,
        "transform": dst_transform,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "deflate",
        "BIGTIFF": "IF_SAFER",
        "nodata": nodata,
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        for band_index in range(1, src.count + 1):
            dst_data = np.full((dst_height, dst_width), nodata, dtype=src.dtypes[band_index - 1])
            reproject(
                source=rasterio.band(src, band_index),
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=nodata,
                dst_transform=dst_transform,
                dst_crs=TARGET_CRS,
                dst_nodata=nodata,
                resampling=Resampling.nearest,
            )
            dst.write(dst_data, band_index)
        dst.colorinterp = colorinterp
        if source_tags:
            dst.update_tags(**source_tags)
        for band_index, description in enumerate(source_descriptions, start=1):
            if description:
                dst.set_band_description(band_index, description)
        for band_index, tags in enumerate(source_band_tags, start=1):
            if tags:
                dst.update_tags(band_index, **tags)


def _translate_to_cog(
    input_path: Path,
    output_path: Path,
    colorinterp: tuple[ColorInterp, ...],
    nodata: float | int,
) -> None:
    gdal_translate = _find_gdal_translate()
    if gdal_translate is None:
        raise RuntimeError("gdal_translate не найден, COG driver недоступен")
    if output_path.exists():
        output_path.unlink()
    command = [
        gdal_translate,
        "-of",
        "COG",
        "-co",
        "COMPRESS=DEFLATE",
        "-co",
        "BIGTIFF=IF_SAFER",
        "-co",
        "RESAMPLING=NEAREST",
        "-colorinterp",
        ",".join(item.name for item in colorinterp),
        "-a_nodata",
        _format_nodata(nodata),
        input_path.as_posix(),
        output_path.as_posix(),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"gdal_translate COG завершился с ошибкой: {message}")

def _validate_output(
    *,
    output_path: Path,
    source_count: int,
    source_dtypes: tuple[str, ...],
    source_nodata: float | int,
    expected_colorinterp: tuple[ColorInterp, ...],
    source_descriptions: tuple[str | None, ...],
    source_band_tags: list[dict[str, str]],
) -> dict[str, object]:
    with rasterio.open(output_path) as ds:
        output_colorinterp = [item.name for item in ds.colorinterp]
        output_dtypes = list(ds.dtypes)
        output_descriptions = list(ds.descriptions)
        has_internal_mask = any(MaskFlags.per_dataset in flags for flags in ds.mask_flag_enums)
        has_alpha = ColorInterp.alpha in ds.colorinterp
        has_sidecar_msk = Path(str(output_path) + ".msk").exists()
        if ds.count != source_count:
            raise ValueError(f"COG output count отличается: {ds.count} != {source_count}")
        if has_alpha:
            raise ValueError("COG output содержит ColorInterp.alpha")
        if ds.crs != rasterio.crs.CRS.from_string(TARGET_CRS):
            raise ValueError(f"COG output CRS отличается от {TARGET_CRS}")
        if tuple(ds.dtypes) != source_dtypes:
            raise ValueError(f"COG output dtype отличается: {ds.dtypes} != {source_dtypes}")
        if not _nodata_equal(ds.nodata, source_nodata):
            raise ValueError(f"COG output nodata отличается: {ds.nodata} != {source_nodata}")
        if tuple(ds.colorinterp) != expected_colorinterp:
            raise ValueError(
                f"COG output colorinterp отличается: {ds.colorinterp} != {expected_colorinterp}"
            )
        if tuple(ds.descriptions) != source_descriptions:
            raise ValueError(
                f"COG output descriptions отличаются: {ds.descriptions} != {source_descriptions}"
            )
        for band_index, expected_tags in enumerate(source_band_tags, start=1):
            output_tags = ds.tags(band_index)
            missing_tags = {
                key: value
                for key, value in expected_tags.items()
                if output_tags.get(key) != value
            }
            if missing_tags:
                raise ValueError(f"COG output потерял band tags {band_index}: {missing_tags}")
        if has_internal_mask:
            raise ValueError("COG output содержит internal mask")
        if has_sidecar_msk:
            raise ValueError("COG output содержит sidecar .msk")

    is_cog_check = _check_cog_layout(output_path)
    if not is_cog_check:
        raise ValueError("gdalinfo не подтвердил COG layout")
    return {
        "output_count": source_count,
        "output_nodata": _json_scalar(source_nodata),
        "output_dtypes": output_dtypes,
        "output_colorinterp": output_colorinterp,
        "output_descriptions": output_descriptions,
        "has_internal_mask": has_internal_mask,
        "has_alpha": has_alpha,
        "has_sidecar_msk": has_sidecar_msk,
        "is_cog_check": is_cog_check,
    }


def _check_cog_layout(output_path: Path) -> bool:
    gdalinfo = _find_gdalinfo()
    if gdalinfo is None:
        raise RuntimeError("gdalinfo не найден, COG layout не проверен")
    result = subprocess.run(
        [gdalinfo, output_path.as_posix()],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"gdalinfo завершился с ошибкой: {message}")
    return "LAYOUT=COG" in result.stdout


def _sanitized_source_colorinterp(
    src_colorinterp: tuple[ColorInterp, ...],
    count: int,
) -> tuple[ColorInterp, ...]:
    if len(src_colorinterp) != count:
        return tuple(ColorInterp.undefined for _ in range(count))
    return tuple(
        ColorInterp.undefined if item == ColorInterp.alpha else item
        for item in src_colorinterp
    )


def _filtered_dataset_tags(tags: dict[str, str]) -> dict[str, str]:
    blocked_fragments = ("alpha", "mask")
    return {
        key: value
        for key, value in tags.items()
        if not any(fragment in key.casefold() for fragment in blocked_fragments)
    }


def _source_nodata(src: rasterio.io.DatasetReader) -> float | int:
    if src.nodata is not None:
        nodata = src.nodata
    else:
        nodata_values = [value for value in src.nodatavals if value is not None]
        if not nodata_values:
            raise ValueError("У исходного снимка нет nodata")
        nodata = nodata_values[0]

    for value in src.nodatavals:
        if value is not None and not _nodata_equal(value, nodata):
            raise ValueError("На каналах исходного снимка разный nodata")
    return nodata


def _nodata_equal(left: float | int | None, right: float | int | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))


def _format_nodata(nodata: float | int) -> str:
    value = float(nodata)
    if value.is_integer():
        return str(int(value))
    return str(nodata)


def _json_scalar(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    value_float = float(value)
    if value_float.is_integer():
        return int(value_float)
    return value_float


def _find_gdal_translate() -> str | None:
    return _find_gdal_executable("gdal_translate")


def _find_gdalinfo() -> str | None:
    return _find_gdal_executable("gdalinfo")


def _find_gdal_executable(name: str) -> str | None:
    executable = shutil.which(name)
    if executable is not None:
        return executable
    for candidate in (
        Path(rf"C:\Program Files\QGIS 3.44.10\bin\{name}.exe"),
        Path(rf"C:\Program Files\QGIS 3.42.0\bin\{name}.exe"),
        Path(rf"C:\Program Files\QGIS 3.40.0\bin\{name}.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
