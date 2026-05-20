"""Сборка warped VRT XML в памяти."""

from __future__ import annotations

import copy
import math
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from rasterio.warp import calculate_default_transform

from ._raster_validation import RasterInfo

TARGET_CRS = "EPSG:3857"


@dataclass(frozen=True)
class _WarpedVrt:
    xml_root: ET.Element
    width: int
    height: int
    geotransform: tuple[float, float, float, float, float, float]
    data_types: tuple[str, ...]
    srs: str
    nodata: str


def build_vrt_xml(
    rasters: list[RasterInfo],
    *,
    target_resolution: tuple[float, float] | None = None,
) -> str:
    if not rasters:
        raise ValueError("Для построения VRT нужен хотя бы один снимок")

    gdalwarp = _find_gdalwarp()
    if gdalwarp is None:
        raise RuntimeError("gdalwarp не найден")

    xres, yres = target_resolution or calculate_target_resolution(rasters)
    nodata = _format_number(rasters[0].nodata)
    with tempfile.TemporaryDirectory(prefix="mlsystem2_vrt_") as temp_dir:
        warped_sources = [
            _build_source_warped_vrt(
                gdalwarp=gdalwarp,
                raster=raster,
                output_path=Path(temp_dir) / f"source_{index}.vrt",
                xres=xres,
                yres=yres,
                nodata=nodata,
            )
            for index, raster in enumerate(rasters)
        ]
        output_path = Path(temp_dir) / "mosaic.vrt"
        output_path.write_text(_build_mosaic_vrt_xml(warped_sources, xres, yres), encoding="utf-8")
        return output_path.read_text(encoding="utf-8")


def calculate_target_resolution(rasters: list[RasterInfo]) -> tuple[float, float]:
    resolutions: list[tuple[float, float]] = []
    for raster in rasters:
        transform, _, _ = calculate_default_transform(
            raster.crs,
            TARGET_CRS,
            raster.width,
            raster.height,
            *raster.bounds,
        )
        xres = abs(float(transform.a))
        yres = abs(float(transform.e))
        if math.isfinite(xres) and math.isfinite(yres) and xres > 0 and yres > 0:
            resolutions.append((xres, yres))
    if not resolutions:
        raise ValueError("Не удалось вычислить target resolution для VRT")
    return min(xres for xres, _ in resolutions), min(yres for _, yres in resolutions)


def _build_source_warped_vrt(
    *,
    gdalwarp: str,
    raster: RasterInfo,
    output_path: Path,
    xres: float,
    yres: float,
    nodata: str,
) -> _WarpedVrt:
    command = [
        gdalwarp,
        "-of",
        "VRT",
        "-t_srs",
        TARGET_CRS,
        "-r",
        "near",
        "-srcnodata",
        nodata,
        "-dstnodata",
        nodata,
        "-tr",
        _format_number(xres),
        _format_number(yres),
        "-tap",
        raster.path.resolve().as_posix(),
        output_path.as_posix(),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"gdalwarp завершился с ошибкой: {message}")
    return _parse_warped_vrt(output_path.read_text(encoding="utf-8"))


def _parse_warped_vrt(xml: str) -> _WarpedVrt:
    root = ET.fromstring(xml)
    geotransform_text = root.findtext("GeoTransform")
    srs = root.findtext("SRS")
    if geotransform_text is None or srs is None:
        raise ValueError("gdalwarp вернул VRT без SRS или GeoTransform")
    bands = root.findall("VRTRasterBand")
    if not bands:
        raise ValueError("gdalwarp вернул VRT без каналов")
    nodata = bands[0].findtext("NoDataValue") or "0"
    return _WarpedVrt(
        xml_root=root,
        width=int(root.attrib["rasterXSize"]),
        height=int(root.attrib["rasterYSize"]),
        geotransform=tuple(float(value) for value in geotransform_text.split(",")),
        data_types=tuple(band.attrib["dataType"] for band in bands),
        srs=srs,
        nodata=nodata,
    )


def _build_mosaic_vrt_xml(sources: list[_WarpedVrt], xres: float, yres: float) -> str:
    left = min(source.geotransform[0] for source in sources)
    top = max(source.geotransform[3] for source in sources)
    right = max(source.geotransform[0] + source.width * xres for source in sources)
    bottom = min(source.geotransform[3] - source.height * yres for source in sources)
    width = int(round((right - left) / xres))
    height = int(round((top - bottom) / yres))

    root = ET.Element("VRTDataset", rasterXSize=str(width), rasterYSize=str(height))
    ET.SubElement(root, "SRS").text = sources[0].srs
    ET.SubElement(root, "GeoTransform").text = ", ".join(
        _format_number(value)
        for value in (left, xres, 0.0, top, 0.0, -yres)
    )

    for band_index, data_type in enumerate(sources[0].data_types, start=1):
        band = ET.SubElement(root, "VRTRasterBand", dataType=data_type, band=str(band_index))
        ET.SubElement(band, "NoDataValue").text = sources[0].nodata
        for source in sources:
            simple_source = ET.SubElement(band, "SimpleSource")
            simple_source.append(copy.deepcopy(source.xml_root))
            ET.SubElement(simple_source, "SourceBand").text = str(band_index)
            ET.SubElement(
                simple_source,
                "SourceProperties",
                RasterXSize=str(source.width),
                RasterYSize=str(source.height),
                DataType=source.data_types[band_index - 1],
                BlockXSize=str(min(source.width, 512)),
                BlockYSize=str(min(source.height, 128)),
            )
            ET.SubElement(
                simple_source,
                "SrcRect",
                xOff="0",
                yOff="0",
                xSize=str(source.width),
                ySize=str(source.height),
            )
            ET.SubElement(
                simple_source,
                "DstRect",
                xOff=str(int(round((source.geotransform[0] - left) / xres))),
                yOff=str(int(round((top - source.geotransform[3]) / yres))),
                xSize=str(source.width),
                ySize=str(source.height),
            )
    return ET.tostring(root, encoding="unicode")


def _find_gdalwarp() -> str | None:
    executable = shutil.which("gdalwarp")
    if executable is not None:
        return executable

    for candidate in (
        Path(r"C:\Program Files\QGIS 3.44.10\bin\gdalwarp.exe"),
        Path(r"C:\Program Files\QGIS 3.42.0\bin\gdalwarp.exe"),
        Path(r"C:\Program Files\QGIS 3.40.0\bin\gdalwarp.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _format_number(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.16g}"
    return str(value)
