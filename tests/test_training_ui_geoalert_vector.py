"""Совместимость имен бриков и защиты границы с установленным Geoalert."""

import sys
from types import ModuleType

import numpy as np
import pytest
from pydantic import BaseModel
from shapely.geometry import Polygon, box
from rasterio.transform import from_origin

from mlsystem2.training_ui_api._geoalert_vector import _touches_boundary, register_vector_bricks


class _Feature(dict):
    @property
    def geometry(self):
        return self["geometry"]


class _Collection(list):
    def filter(self, predicate):
        return _Collection(feature for feature in self if predicate(feature))


@pytest.fixture
def bricks(monkeypatch):
    registry = {}
    class PolygonProcessingBrick(BaseModel):
        input: str
        output: str | None = None
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            registry[cls.__name__] = cls
    class FilterSmallObjects(PolygonProcessingBrick):
        min_area: float
        area_tag: str | None = None
        def process(self, fc):
            return fc.filter(lambda feature: feature.geometry.area > self.min_area)
    class VectorizeMasks(PolygonProcessingBrick):
        pass
    urban = ModuleType("urban")
    urban.FilterSmallObjects = FilterSmallObjects
    urban.VectorizeMasks = VectorizeMasks
    base = ModuleType("urban.base.brick")
    base.PolygonProcessingBrick = PolygonProcessingBrick
    functional = ModuleType("urban.functional")
    functional.io = None
    monkeypatch.setitem(sys.modules, "urban", urban)
    monkeypatch.setitem(sys.modules, "urban.base.brick", base)
    monkeypatch.setitem(sys.modules, "urban.functional", functional)
    register_vector_bricks()
    return registry


def test_registered_compact_filters_preserve_boundary_and_invalid_shapes(bricks):
    invalid = Polygon([(0, 0), (10, 10), (0, 10), (10, 0), (0, 0)])
    features = _Collection([
        _Feature(geometry=box(0, 0, 10, 10), name="компактный"),
        _Feature(geometry=box(0, 0, 100, 10), name="вытянутый"),
        _Feature(geometry=box(0, 0, 1, 1), name="граничный", _touches_raster_boundary=True),
        _Feature(geometry=invalid, name="невалидный"),
        _Feature(geometry=box(0, 0, 10, 10), name="без отметки", _touches_raster_boundary=np.nan),
    ])
    remove = bricks["FilterCompactObjects"](input="output")
    keep = bricks["FilterNonCompactObjects"](input="output")
    assert [feature["name"] for feature in remove.process(features)] == ["вытянутый", "граничный", "невалидный"]
    assert [feature["name"] for feature in keep.process(features)] == ["компактный", "граничный", "невалидный", "без отметки"]


def test_small_object_filter_preserves_marked_boundary_only_when_requested(bricks):
    features = _Collection([_Feature(geometry=box(0, 0, 1, 1), _touches_raster_boundary=True),
                            _Feature(geometry=box(5, 5, 6, 6))])
    protected = bricks["FilterSmallObjects"](input="output", min_area=10, preserve_boundary_objects=True)
    ordinary = bricks["FilterSmallObjects"](input="output", min_area=10)
    assert len(protected.process(features)) == 1
    assert ordinary.process(features) == []


def test_boundary_protection_uses_pixel_tolerance_even_on_rotated_raster():
    from affine import Affine
    from shapely.ops import transform

    grid = from_origin(100, 200, 2, 2) * Affine.rotation(25)
    def from_pixels(x, y, z=None):
        return grid.a * x + grid.b * y + grid.c, grid.d * x + grid.e * y + grid.f
    edge = transform(from_pixels, box(0.5, 10, 8, 20))
    interior = transform(from_pixels, box(3, 10, 8, 20))
    assert _touches_boundary(edge, ~grid, 100, 100, 1)
    assert not _touches_boundary(interior, ~grid, 100, 100, 1)
