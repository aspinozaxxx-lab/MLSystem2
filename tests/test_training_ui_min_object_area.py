"""Фильтрация площади учитывает геопривязку, отверстия и составные объекты."""

import pytest
from pydantic import ValidationError
from pyproj import Geod, Transformer
from shapely.geometry import MultiPolygon, Polygon, box, mapping
from shapely.ops import transform

from mlsystem2.training_ui_api import _markup_export
from mlsystem2.training_ui_api.contracts import (
    TestSampleBatchItemCreate as BatchItemCreate,
    TestSampleCreate as SampleCreate,
    TestSampleCreationSettings as CreationSettings,
)


def _annotations(geometries, crs):
    return _markup_export._annotations_from_payload({
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": [{"type": "Feature", "id": index, "properties": {},
                      "geometry": mapping(geometry)} for index, geometry in enumerate(geometries)],
    })


@pytest.mark.parametrize("crs", ["EPSG:4326", "EPSG:3857", "EPSG:32636"])
def test_object_area_is_measured_on_earth_in_any_annotation_crs(crs):
    x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(33, 60)
    geometries = [box(x, y, x + 100, y + 100), box(x, y, x + 200, y + 200)]
    projected = [transform(Transformer.from_crs(3857, crs, always_xy=True).transform, item)
                 for item in geometries]
    filtered = _markup_export._filter_annotation_objects_by_area(_annotations(projected, crs), 5000)
    # В Web Mercator первый объект занимает 10 000 единиц², но на земле — около 2 500 м².
    assert [item.feature_id for item in filtered.features] == [1]
    assert "Исключено объектов" in filtered.warnings[-1]


def test_area_filter_subtracts_holes_and_adds_oppositely_oriented_parts():
    outer = box(33, 60, 33.002, 60.002)
    hole = box(33.0001, 60.0001, 33.0019, 60.0019)
    ring = Polygon(outer.exterior.coords, [hole.exterior.coords])
    part = box(33.003, 60, 33.004, 60.001)
    second = box(33.005, 60, 33.006, 60.001)
    combined = MultiPolygon([part, Polygon(list(second.exterior.coords)[::-1])])
    filtered = _markup_export._filter_annotation_objects_by_area(
        _annotations([ring, combined], "EPSG:4326"), 10_000
    )
    assert [item.feature_id for item in filtered.features] == [1]


def test_object_exactly_at_minimum_area_is_kept():
    geometry = box(33, 60, 33.001, 60.001)
    area = abs(Geod(ellps="WGS84").geometry_area_perimeter(geometry)[0])
    assert len(_markup_export._filter_annotation_objects_by_area(
        _annotations([geometry], "EPSG:4326"), area
    ).features) == 1


def test_large_area_filter_checks_queue_cancellation():
    geometry = box(33, 60, 33.001, 60.001)
    checks = []

    def cancel():
        checks.append(True)
        if len(checks) == 2:
            raise RuntimeError("Создание отменено")

    with pytest.raises(RuntimeError, match="Создание отменено"):
        _markup_export._filter_annotation_objects_by_area(
            _annotations([geometry] * 300, "EPSG:4326"), 1, check_cancelled=cancel
        )
    assert len(checks) == 2


@pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf")])
@pytest.mark.parametrize("model", [CreationSettings, BatchItemCreate, SampleCreate])
def test_area_threshold_requires_finite_nonnegative_number(model, threshold):
    arguments = {} if model is CreationSettings else {"dataset_key": "Демонстрационный\\main"}
    with pytest.raises(ValidationError):
        model(min_object_area_m2=threshold, **arguments)
