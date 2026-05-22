import pytest
from urban import Brick
from shapely.geometry import Polygon, Point
from gpdadapter import FeatureCollection


def get_data():
    triangle = Polygon([(0, 0), (1, 0), (0, 10)])
    l_shape = Polygon([(0, 0), (10, 0), (10, 1), (1, 1), (1, 10), (0, 10), (0, 0)])
    hole = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
                   [[(1, 1), (1, 9), (9, 9), (9, 1)]])
    fc = FeatureCollection({'geometry': [triangle, l_shape, hole]})
    return fc


def test_value_check():
    with pytest.raises(ValueError):
        p2p_centroid = Brick.from_config({"brick_class": "Polygons2Points",
                                          "input": "none",
                                          "method": "invalid"})


def assert_points_equal(p1, p2):
    assert pytest.approx(p1.x) == p2.x
    assert pytest.approx(p1.y) == p2.y


def test_points2polygons_centroid():
    p2p_centroid = Brick.from_config({"brick_class": "Polygons2Points",
                                      "input": "none",
                                      "method": "centroid"})
    result = p2p_centroid.process(get_data())
    assert_points_equal(result[0, 'geometry'], Point(0.3333333333333334, 3.3333333333333335))
    assert_points_equal(result[1, 'geometry'], Point(2.8684210526315788, 2.8684210526315788))
    assert_points_equal(result[2, 'geometry'], Point(5, 5))


def test_points2polygons_represent():

    p2p_represent = Brick.from_config({"brick_class": "Polygons2Points",
                                       "input": "none",
                                       "method": "representative_point"})
    result = p2p_represent.process(get_data())
    assert_points_equal(result[0, 'geometry'], Point(0.25, 5))
    assert_points_equal(result[1, 'geometry'], Point(0.5, 5.5))
    assert_points_equal(result[2, 'geometry'], Point(0.5, 5))


def test_points2polygons_opt():

    p2p_opt = Brick.from_config({"brick_class": "Polygons2Points",
                                 "input": "none",
                                 "method": "optimal"})
    result = p2p_opt.process(get_data())
    assert_points_equal(result[0, 'geometry'], Point(0.3333333333333334, 3.3333333333333335))
    assert_points_equal(result[1, 'geometry'], Point(0.5, 5.5))
    assert_points_equal(result[2, 'geometry'], Point(0.5, 5))
