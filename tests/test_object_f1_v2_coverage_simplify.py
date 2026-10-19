"""Общие границы экземпляров упрощаются совместно, фон и ID сохраняются."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import shapely
from pydantic import BaseModel
from pyproj import Transformer
from rasterio.features import shapes
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box, shape
from shapely.ops import transform

from mlsystem2.training_ui_api import _geoalert_vector as subject


def _raster_parcels():
    labels = np.zeros((80, 100), np.int32)
    for y in range(5, 75):
        first = 35 + (y // 2) % 2
        second = 65 + (y // 3) % 2
        labels[y, 5:first], labels[y, first:second], labels[y, second:95] = 1, 2, 3
    labels[20:25, 12:20] = 0  # Настоящая пустота внутри участка.
    labels[48:54, 34:38] = 0  # Nodata прерывает общую границу.
    grid = from_origin(16_842_800, 8_307_500, .1, .1)
    parts = [(int(value), shape(geometry)) for geometry, value in
             shapes(labels, mask=labels > 0, transform=grid)]
    return parts


def _no_overlaps(geometries):
    for i, left in enumerate(geometries):
        for right in geometries[i+1:]:
            assert left.intersection(right).area == 0


def test_shared_staircase_keeps_ids_holes_and_nodata_without_independent_simplify():
    parts = _raster_parcels()
    ids = [value for value, _ in parts]
    original = [geometry for _, geometry in parts]
    matched, matching = subject._match_coverage_edges(original)
    assert matching["status"] in {"matched", "unchanged"}
    assert shapely.coverage_is_valid(matched)
    assert all(before.equals(after) for before, after in zip(original, matched))
    result, diagnostic = subject._simplify_coverage(matched, .5)
    assert diagnostic["status"] == "simplified"
    assert diagnostic["coordinates_after"] < diagnostic["coordinates_before"]
    assert len(result) == len(ids) == 3
    assert [len(g.interiors) for g in result] == [len(g.interiors) for g in original]
    assert shapely.coverage_union_all(result).equals(shapely.union_all(original))
    assert all(g.is_valid and not g.is_empty for g in result)
    _no_overlaps(result)


def test_t_junction_is_noded_before_reprojection_without_moving_any_boundary():
    original = [box(0, 0, 1, 2), box(1, 0, 2, 1), box(1, 1, 2, 2)]
    assert not shapely.coverage_is_valid(original)
    result, diagnostic = subject._match_coverage_edges(original)
    assert diagnostic["status"] == "matched"
    assert shapely.coverage_is_valid(result)
    assert all(left.equals(right) for left, right in zip(original, result))
    assert shapely.count_coordinates(result) == 16


def test_projected_native_coverage_stays_edge_matched_through_geojson_coordinates():
    original = [geometry for _, geometry in _raster_parcels()]
    matched, _ = subject._match_coverage_edges(original)
    to_wgs = Transformer.from_crs(3857, 4326, always_xy=True).transform
    to_utm = Transformer.from_crs(4326, 32656, always_xy=True).transform
    projected = [transform(to_utm, transform(to_wgs, geometry)) for geometry in matched]
    assert shapely.coverage_is_valid(projected)
    result, diagnostic = subject._simplify_coverage(projected, .5)
    assert diagnostic["status"] == "simplified"
    assert shapely.coverage_is_valid(result)
    _no_overlaps(result)


@pytest.mark.parametrize("geometries", [
    [box(0, 0, 2, 2), box(1, 1, 3, 3)],
    [Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])],
])
def test_invalid_coverage_falls_back_exactly_without_trimming_or_losing_objects(geometries):
    matched, match_info = subject._match_coverage_edges(geometries)
    result, simplify_info = subject._simplify_coverage(geometries, .5)
    assert match_info["status"] == simplify_info["status"] == "fallback"
    assert [g.wkb for g in matched] == [g.wkb for g in geometries]
    assert [g.wkb for g in result] == [g.wkb for g in geometries]


def test_older_geos_keeps_exact_geometry_and_returns_diagnostic(monkeypatch):
    geometries = [box(0, 0, 2, 2)]
    monkeypatch.setattr(shapely, "geos_version", (3, 11, 0))
    for operation in (subject._match_coverage_edges, lambda values: subject._simplify_coverage(values, .5)):
        result, diagnostic = operation(geometries)
        assert result is geometries
        assert diagnostic["reason"] == "coverage_unavailable"


def test_bad_simplifier_cannot_expand_nodata_or_change_object_count(monkeypatch):
    geometries = [Polygon(box(0, 0, 4, 4).exterior, [box(1, 1, 2, 2).exterior])]
    monkeypatch.setattr(shapely, "coverage_simplify", lambda *args, **kwargs: [box(0, 0, 5, 5)])
    result, diagnostic = subject._simplify_coverage(geometries, .5)
    assert result is geometries
    assert diagnostic["status"] == "fallback"


def test_final_projection_retry_changes_whole_coverage_from_original(monkeypatch):
    original, _ = subject._match_coverage_edges([geometry for _, geometry in _raster_parcels()])
    real_simplify = shapely.coverage_simplify
    calls, projections = [], []

    def observed(geometries, tolerance, **kwargs):
        assert geometries is original
        calls.append(tolerance)
        return real_simplify(geometries, tolerance, **kwargs)

    def final_projection(geometries):
        projections.append(geometries)
        if len(projections) < 3:
            # Валидность в рабочем CRS недостаточна: последующая проекция
            # способна превратить один контур в самопересекающуюся фигуру.
            return [Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)]), *geometries[1:]]
        return geometries

    monkeypatch.setattr(shapely, "coverage_simplify", observed)
    result, diagnostic = subject._simplify_coverage(original, .5, final_projection=final_projection)
    assert calls == [.5, .25, .125]
    assert diagnostic["tolerance_m"] == .5
    assert diagnostic["effective_tolerance_m"] == .125
    assert diagnostic["simplification_attempts"] == 3
    assert diagnostic["final_crs_validated"] is True
    assert diagnostic["status"] == "simplified"
    assert len(result) == len(original)
    assert [len(g.interiors) for g in result] == [len(g.interiors) for g in original]
    assert shapely.union_all(result).equals(shapely.union_all(original))
    _no_overlaps(result)


def test_valid_polygons_with_overlapping_final_coverage_exhaust_bounded_retries():
    original = [box(0, 0, 1, 2), box(1, 0, 2, 2)]
    calls = []

    def unsafe_projection(geometries):
        calls.append(geometries)
        return [box(0, 0, 2, 2), box(1, 0, 3, 2)]

    result, diagnostic = subject._simplify_coverage(original, .5, final_projection=unsafe_projection)
    assert result is original
    assert [g.wkb for g in result] == [g.wkb for g in original]
    assert len(calls) == diagnostic["simplification_attempts"] == 8
    assert diagnostic["effective_tolerance_m"] is None
    assert diagnostic["final_crs_validated"] is False
    assert diagnostic["reason"] == "unsafe_final_projection"
    assert diagnostic["status"] == "fallback"


@pytest.fixture
def registered_vector_types(monkeypatch):
    registry, calls = {}, []

    class Base(BaseModel):
        input: str = "output"
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            registry[cls.__name__] = cls

    class OriginalVectorizeMasks(Base):
        value_property_name: str | None = None
        def __call__(self, path):
            calls.append(path)

    urban, base, functional = ModuleType("urban"), ModuleType("urban.base.brick"), ModuleType("urban.functional")
    urban.VectorizeMasks, urban.FilterSmallObjects = OriginalVectorizeMasks, Base
    base.Brick, base.PolygonProcessingBrick, functional.io = Base, Base, None
    monkeypatch.setitem(sys.modules, "urban", urban)
    monkeypatch.setitem(sys.modules, "urban.base.brick", base)
    monkeypatch.setitem(sys.modules, "urban.functional", functional)
    subject.register_vector_bricks()
    return registry, calls


def test_registration_preserves_legacy_vectorization_and_feature_properties(registered_vector_types):
    registry, calls = registered_vector_types
    registry["VectorizeMasks"]()("неизменный путь legacy")
    assert calls == ["неизменный путь legacy"]
    parts = _raster_parcels()
    geometries, _ = subject._match_coverage_edges([g for _, g in parts])
    class Collection(SimpleNamespace):
        def copy(self):
            return Collection(geometry=list(self.geometry), properties=self.properties.copy())

        def to_crs(self, crs, *, inplace):
            assert crs == "EPSG:4326" and inplace
            converter = Transformer.from_crs(3857, crs, always_xy=True).transform
            self.geometry = list(shapely.transform(self.geometry, converter, interleaved=False))

    collection = Collection(geometry=geometries, properties=[{"instance_id": value, "name": "участок"} for value, _ in parts])
    original_properties = collection.properties.copy()
    result = registry["CoverageSimplify"](rate=.5).process(collection)
    assert result is collection
    assert result.properties == original_properties
    assert len(result.geometry) == len(original_properties)


@pytest.mark.parametrize("safe_original", [False, True])
def test_registered_fallback_checks_original_final_projection_before_sdk_save(
    monkeypatch, registered_vector_types, safe_original,
):
    registry, _ = registered_vector_types
    original = [box(0, 0, 1, 2), box(1, 0, 2, 2)]
    candidate = [box(0, 0, 1.1, 2), box(1.1, 0, 2, 2)]
    attempts, projections, diagnostics = [], [], []

    def simplify(geometries, tolerance, **kwargs):
        attempts.append(tolerance)
        return candidate

    class Collection(SimpleNamespace):
        def copy(self):
            return Collection(geometry=list(self.geometry), properties=self.properties.copy())

        def to_crs(self, crs, *, inplace):
            assert crs == "EPSG:4326" and inplace
            projections.append(self.geometry)
            if not safe_original or self.geometry[0].bounds[2] > 1.05:
                self.geometry = [box(0, 0, 2, 2), box(1, 0, 3, 2)]

    monkeypatch.setattr(shapely, "coverage_simplify", simplify)
    monkeypatch.setattr(subject, "_log_coverage", lambda operation, diagnostic: diagnostics.append(diagnostic))
    collection = Collection(geometry=original, properties=[{"instance_id": 10}, {"instance_id": 20}])
    before = [geometry.wkb for geometry in original]
    if safe_original:
        assert registry["CoverageSimplify"](rate=.5).process(collection) is collection
        assert diagnostics[-1]["status"] == "fallback"
        assert diagnostics[-1]["fallback_original_validated"] is True
        assert diagnostics[-1]["final_crs_validated"] is True
    else:
        with pytest.raises(RuntimeError, match="Исходные границы нельзя безопасно сохранить"):
            registry["CoverageSimplify"](rate=.5).process(collection)
        assert diagnostics[-1]["reason"] == "unsafe_original_final_projection"
    assert attempts == [.5 / (2 ** index) for index in range(8)]
    assert len(projections) == 9  # Восемь кандидатов и обязательная проверка исходника.
    assert [geometry.wkb for geometry in collection.geometry] == before
    assert collection.properties == [{"instance_id": 10}, {"instance_id": 20}]
