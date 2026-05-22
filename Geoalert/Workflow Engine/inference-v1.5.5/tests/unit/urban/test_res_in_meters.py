from rasterio.crs import CRS
import math
from urban.functional.raster_ops.res_in_meters import get_resolution_in_meters


def test_get_resolution_latlon():
    crs, res = CRS.from_epsg(4326), 0.000001
    resolution = get_resolution_in_meters(crs, res)
    assert math.isclose(resolution, 0.1, abs_tol=0.02)


def test_get_resolution_meters():
    crs, res = CRS.from_epsg(3857), 0.5
    resolution = get_resolution_in_meters(crs, res)
    assert math.isclose(resolution, 0.5, abs_tol=0.01)


def test_get_resolution_foot():
    crs, res = CRS.from_epsg(6360), 1
    resolution = get_resolution_in_meters(crs, res)
    assert math.isclose(resolution, 0.3, abs_tol=0.01)
