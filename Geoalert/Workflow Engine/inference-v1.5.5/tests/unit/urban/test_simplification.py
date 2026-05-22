import pytest
from shapely.geometry import Polygon
from urban.functional.postprocessing.simplification import _make_rectangles


def test_simplify_geometry_with_rectangle():
    geom = Polygon([[0, 10], [10, 10], [10, 0], [0, 0]])
    simplified = _make_rectangles(geom, negative_buffer=-1)

    assert simplified.area == 64.0
    assert simplified == Polygon([[9, 1], [1, 1], [1, 9], [9, 9], [9, 1]])

    with pytest.raises(ValueError):
        _make_rectangles(geom, negative_buffer=0.5)

    with pytest.raises(ValueError):
        _make_rectangles(geom, negative_buffer=3)