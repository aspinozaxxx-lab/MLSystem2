import pytest

from data_validator_lib.functional import get_pixel_size, read_profile_from_s3
from data_validator_lib.functional.geometry import _get_mgs_poly, aoi_intersects_cell
from shapely.geometry import Polygon
GRID_FILE = '/opt/source-validator/data-validator-lib/data_validator_lib/static/mgrs_grid.csv'


def test_read_profile_from_s3():
    pass


def test_get_pixel_size():
    pass


def test_get_mgs_poly():
    cell = '38TKK'
    geometry = Polygon(
        [(41.4545362985, 40.5964014683),
         (42.75086522, 40.6289489934),
         (42.7832353435, 39.6404485626),
         (41.5054873524, 39.609013147),
         (41.4545362985, 40.5964014683)])
    assert _get_mgs_poly(cell, GRID_FILE).equals(geometry)


def test_aoi_intersects_cell():
    cell_ok = '38VNP'
    cell_not_ok = '38TKK'
    aoi = {"type": "Polygon",
           "coordinates": [[[45.722466910376255, 61.52350527049347],
                            [45.74091348225352, 61.52350527049347],
                            [45.74091348225352, 61.514322015710434],
                            [45.722466910376255, 61.514322015710434],
                            [45.722466910376255, 61.52350527049347]]]
           }
    assert aoi_intersects_cell(cell_ok, aoi, GRID_FILE)
    assert not aoi_intersects_cell(cell_not_ok, aoi, GRID_FILE)