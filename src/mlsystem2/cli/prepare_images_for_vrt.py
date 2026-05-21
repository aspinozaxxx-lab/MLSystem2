"""Одноразовая подготовка GeoTIFF для VRT-мозаик."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, MaskFlags, Resampling
from rasterio.warp import calculate_default_transform, reproject


RAW_IMAGES_DIR = Path(r"D:\Projects\ImagesDeforestation")
PREPARED_IMAGES_DIR = Path(r"D:\Projects\ImagesDeforestationPrepared3857")
REPORT_PATH = Path(r"D:\Projects\test\prepare_images_for_vrt_report.json")
TARGET_CRS = "EPSG:3857"
DEBUG_ONE_IMAGE = Path(
    r"D:\Projects\ImagesDeforestation\irkutsk\KV5_24818_25736-01_KANOPUS_20230618_035304_20.L2.PMS.SCN03.tif"
)


def main() -> int:
    report = prepare_images_for_vrt(RAW_IMAGES_DIR, PREPARED_IMAGES_DIR, REPORT_PATH)
    return 0 if report["status"] == "ok" else 1


def prepare_images_for_vrt(
    raw_images_dir: Path,
    prepared_images_dir: Path,
    report_path: Path,
) -> dict[str, object]:
    files = _select_input_files(raw_images_dir)
    report_files: list[dict[str, object]] = []
    for input_path in files:
        output_path = prepared_images_dir / input_path.relative_to(raw_images_dir)
        record = {
            "input_path": input_path.resolve().as_posix(),
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
        try:
            record.update(_prepare_one(input_path, output_path))
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


def _select_input_files(raw_images_dir: Path) -> list[Path]:
    if DEBUG_ONE_IMAGE.is_file() and _is_relative_to(DEBUG_ONE_IMAGE, raw_images_dir):
        return [DEBUG_ONE_IMAGE]
    files = sorted(
        [
            path
            for path in raw_images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ],
        key=lambda item: str(item).casefold(),
    )
    return files[:1]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _prepare_one(input_path: Path, output_path: Path) -> dict[str, object]:
    with rasterio.open(input_path) as src:
        if src.crs is None:
            raise ValueError("У исходного снимка нет CRS")
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
