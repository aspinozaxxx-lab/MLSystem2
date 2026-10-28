"""Вырезка тестовой области: нативная сетка TIFF, точный контур и маска nodata."""

from copy import deepcopy
from math import ceil, floor

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.features import geometry_mask
from rasterio.windows import Window
from shapely.affinity import affine_transform
from shapely.geometry import mapping, shape

from ._raster_valid_data import _polygonal_geometry, clip_geometries_to_valid_data, valid_data_mask
from .contracts import TrainingUIAPIError


def crop_test_region(source, target, region, payload, imagery):
    """Записать только исходные пиксели, без изменения разрешения и интерполяции."""
    count = 4 if imagery.value == "kanopus" else 3
    polygon = shape(region)
    if polygon.geom_type not in {"Polygon", "MultiPolygon"} or not polygon.is_valid or polygon.is_empty:
        raise TrainingUIAPIError("Тестовая область должна быть непустым корректным полигоном.")
    with rasterio.open(source) as src:
        if src.crs is None or src.crs.to_epsg() != 3857 or any(dtype != "uint8" for dtype in src.dtypes):
            raise TrainingUIAPIError("Для тестовой вырезки нужен TIFF uint8 в EPSG:3857.")
        if (count == 4 and (src.count != 4 or ColorInterp.alpha in src.colorinterp)) or (
            count == 3 and not (src.count == 3 or (src.count == 4 and src.colorinterp[3] == ColorInterp.alpha))
        ):
            raise TrainingUIAPIError("Канопус должен содержать RGB+NIR, орто — RGB с необязательной альфой.")
        inverse = ~src.transform
        pixels = affine_transform(polygon, [inverse.a, inverse.b, inverse.d, inverse.e, inverse.c, inverse.f])
        left, top, right, bottom = pixels.bounds
        # Допуск убирает погрешность прямого и обратного affine у рамок ровно 512×512.
        x, y = floor(left + 1e-7), floor(top + 1e-7)
        width, height = ceil(right - 1e-7) - x, ceil(bottom - 1e-7) - y
        if width < 1 or height < 1:
            raise TrainingUIAPIError("Тестовая область меньше одного пикселя.")
        window = Window(x, y, width, height)
        transform = src.window_transform(window)
        profile = dict(driver="GTiff", width=width, height=height, count=count, dtype="uint8",
                       crs=src.crs, transform=transform, tiled=True, blockxsize=256, blockysize=256,
                       compress="deflate", BIGTIFF="IF_SAFER", ALPHA="NO", nodata=0)
        indexes = tuple(range(1, count + 1))
        valid_count = 0
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(target, "w", **profile) as dst:
            dst.colorinterp = tuple(src.colorinterp[:count])
            dst.descriptions = tuple(src.descriptions[:count])
            for _, block in dst.block_windows(1):
                bx, by, bw, bh = map(int, (block.col_off, block.row_off, block.width, block.height))
                data = np.zeros((count, bh, bw), dtype=np.uint8)
                valid = np.zeros((bh, bw), dtype=bool)
                sx, sy = max(0, x + bx), max(0, y + by)
                ex, ey = min(src.width, x + bx + bw), min(src.height, y + by + bh)
                if sx < ex and sy < ey:
                    read_window = Window(sx, sy, ex - sx, ey - sy)
                    dx, dy = sx - x - bx, sy - y - by
                    data[:, dy:dy + ey - sy, dx:dx + ex - sx] = src.read(indexes, window=read_window)
                    valid[dy:dy + ey - sy, dx:dx + ex - sx] = valid_data_mask(src, read_window, indexes)
                valid &= geometry_mask([region], out_shape=(bh, bw), transform=dst.window_transform(block), invert=True)
                valid_count += int(valid.sum())
                data[:, ~valid] = 0
                dst.write(data, window=block)
                dst.write_mask(valid.astype(np.uint8) * 255, window=block)
            if not valid_count:
                raise TrainingUIAPIError("В тестовой области нет полезных пикселей снимка.")
            factors = [factor for factor in (2, 4, 8, 16) if min(width, height) // factor >= 1]
            if factors:
                dst.build_overviews(factors, Resampling.average)
                dst.update_tags(ns="rio_overview", resampling="average")
    result = deepcopy(payload)
    originals, intersections = [], []
    for feature in payload.get("features", []):
        if (feature.get("properties") or {}).get("_mlsystem2_role", "positive") != "positive":
            continue
        geometry = shape(feature["geometry"])
        if not geometry.is_valid:
            raise TrainingUIAPIError("В выбранной версии есть некорректные полигоны. Исправьте их в редакторе датасета.")
        clipped = _polygonal_geometry(geometry.intersection(polygon))
        if not clipped.is_empty:
            originals.append(feature)
            intersections.append(clipped)
    with rasterio.open(target) as dst:
        clipped_geometries = clip_geometries_to_valid_data(dst, intersections)
    result["features"] = [{**deepcopy(feature), "geometry": dict(mapping(geometry))}
                          for feature, geometry in zip(originals, clipped_geometries) if not geometry.is_empty]
    return result
