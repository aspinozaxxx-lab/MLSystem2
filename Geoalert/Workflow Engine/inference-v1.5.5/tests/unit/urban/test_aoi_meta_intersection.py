import pytest
import shapely

from gpdadapter import FeatureCollection
from tests.testutils.utils import create_square_polygon
from urban.functional.metaangles import check_aoi_meta_intersection


def test_aoi_within_single_polygon():
    aoi = FeatureCollection(create_square_polygon((0, 0), 0, 5, 5))
    meta = FeatureCollection(create_square_polygon((0, 0), 0, 10, 10))
    check_aoi_meta_intersection(aoi, meta)


def test_aoi_within_multiple_polygons():
    aoi = FeatureCollection(create_square_polygon((0, 0), 0, 5, 5))
    meta = FeatureCollection([create_square_polygon((-2, -2), 0, 4, 4),
                              create_square_polygon((2, -2), 0, 4, 4),
                              create_square_polygon((-2, 2), 0, 4, 4),
                              create_square_polygon((2, 2), 0, 4, 4)])
    check_aoi_meta_intersection(aoi, meta)


def test_multipolygon_aoi_within_single_polygon():
    aoi = FeatureCollection(shapely.unary_union([create_square_polygon((0, 2), 0, 2, 2),
                                                 create_square_polygon((0, -2), 0, 2, 2)]))
    meta = FeatureCollection(create_square_polygon((0, 0), 0, 10, 10))
    check_aoi_meta_intersection(aoi, meta)


def test_multipolygon_aoi_within_multiple_polygons():
    aoi = FeatureCollection(shapely.unary_union([create_square_polygon((0, 2), 0, 2, 2),
                                                 create_square_polygon((0, -2), 0, 2, 2)]))
    meta = FeatureCollection([create_square_polygon((-2, -2), 0, 4, 4),
                              create_square_polygon((2, -2), 0, 4, 4),
                              create_square_polygon((-2, 2), 0, 4, 4),
                              create_square_polygon((2, 2), 0, 4, 4)])
    check_aoi_meta_intersection(aoi, meta)


def test_aoi_bigger_than_single_polygon():
    aoi = FeatureCollection(create_square_polygon((0, 0), 0, 15, 15))
    meta = FeatureCollection(create_square_polygon((0, 0), 0, 10, 10))
    with pytest.raises(ValueError):
        check_aoi_meta_intersection(aoi, meta)


def test_aoi_outside_single_polygon():
    aoi = FeatureCollection(create_square_polygon((10, 0), 0, 5, 5))
    meta = FeatureCollection(create_square_polygon((0, 0), 0, 10, 10))
    with pytest.raises(ValueError):
        check_aoi_meta_intersection(aoi, meta)


def test_aoi_not_covered_by_multiple_polygons():
    aoi = FeatureCollection(create_square_polygon((0, 0), 0, 5, 5))
    meta = FeatureCollection([create_square_polygon((-2, -2), 0, 2, 2),
                              create_square_polygon((2, -2), 0, 2, 2),
                              create_square_polygon((-2, 2), 0, 2, 2),
                              create_square_polygon((2, 2), 0, 2, 2)])
    with pytest.raises(ValueError):
        check_aoi_meta_intersection(aoi, meta)
