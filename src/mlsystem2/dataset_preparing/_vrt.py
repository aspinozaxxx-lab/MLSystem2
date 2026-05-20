"""Сборка VRT XML в памяти."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from ._raster_validation import RasterInfo

GDAL_DTYPE_BY_RASTERIO = {
    "uint8": "Byte",
    "int8": "Int8",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "uint64": "UInt64",
    "int64": "Int64",
    "float32": "Float32",
    "float64": "Float64",
    "complex64": "CFloat32",
    "complex128": "CFloat64",
}


def build_vrt_xml(rasters: list[RasterInfo]) -> str:
    if not rasters:
        raise ValueError("Для построения VRT нужен хотя бы один снимок")

    left = min(raster.bounds.left for raster in rasters)
    right = max(raster.bounds.right for raster in rasters)
    top = max(raster.bounds.top for raster in rasters)
    bottom = min(raster.bounds.bottom for raster in rasters)
    resolution_x, resolution_y = rasters[0].resolution
    width = int(round((right - left) / resolution_x))
    height = int(round((top - bottom) / resolution_y))

    root = ET.Element(
        "VRTDataset",
        rasterXSize=str(width),
        rasterYSize=str(height),
    )
    ET.SubElement(root, "SRS").text = rasters[0].crs_wkt
    ET.SubElement(root, "GeoTransform").text = ", ".join(
        _format_number(value)
        for value in (left, resolution_x, 0.0, top, 0.0, -resolution_y)
    )

    for band_index in range(1, rasters[0].band_count + 1):
        band = ET.SubElement(
            root,
            "VRTRasterBand",
            dataType=_gdal_dtype(rasters[0].dtypes[band_index - 1]),
            band=str(band_index),
        )
        ET.SubElement(band, "NoDataValue").text = _format_number(rasters[0].nodata)
        for raster in rasters:
            source = ET.SubElement(band, "SimpleSource")
            filename = ET.SubElement(source, "SourceFilename", relativeToVRT="0")
            filename.text = raster.path.resolve().as_posix()
            ET.SubElement(source, "SourceBand").text = str(band_index)
            ET.SubElement(
                source,
                "SourceProperties",
                RasterXSize=str(raster.width),
                RasterYSize=str(raster.height),
                DataType=_gdal_dtype(raster.dtypes[band_index - 1]),
                BlockXSize=str(min(raster.width, 128)),
                BlockYSize=str(min(raster.height, 128)),
            )
            ET.SubElement(
                source,
                "SrcRect",
                xOff="0",
                yOff="0",
                xSize=str(raster.width),
                ySize=str(raster.height),
            )
            ET.SubElement(
                source,
                "DstRect",
                xOff=str(int(round((raster.bounds.left - left) / resolution_x))),
                yOff=str(int(round((top - raster.bounds.top) / resolution_y))),
                xSize=str(raster.width),
                ySize=str(raster.height),
            )

    return ET.tostring(root, encoding="unicode")


def _gdal_dtype(dtype: str) -> str:
    try:
        return GDAL_DTYPE_BY_RASTERIO[dtype]
    except KeyError as exc:
        raise ValueError(f"Неподдерживаемый dtype для VRT: {dtype}") from exc


def _format_number(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.16g}"
    return str(value)
