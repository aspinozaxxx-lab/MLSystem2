from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pyproj import Transformer
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import transform, unary_union

from mlsystem2.training_ui_api._test_samples import _single_test_sample_geojson
from mlsystem2.training_ui_api.contracts import TrainingUIAPIError


def _write(root: Path, index: int, features: list[dict], *, crs="EPSG:3857") -> None:
    (root / f"tile_{index:03d}.geojson").write_text(json.dumps({
        "type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": crs}},
        "_mlsystem2_schema_version": 1, "_mlsystem2_task": "multiclass",
        "_mlsystem2_classes": [{"id": 1, "slug": "вода"}, {"id": 2, "slug": "лес"}],
        "features": features,
    }, ensure_ascii=False), encoding="utf-8")


def _feature(geometry, slug="вода", **properties):
    return {"type": "Feature", "geometry": mapping(geometry), "properties": {
        "_mlsystem2_role": "positive", "_mlsystem2_class": slug, **properties,
    }}


def test_single_geojson_glues_edges_overlaps_and_duplicates_preserving_holes_types_and_area(tmp_path):
    lake = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], [[(1, 1), (2, 1), (2, 2), (1, 2)]])
    edge = box(4, 0, 6, 4)
    overlap = box(3, 0, 5, 3)
    point_contact = box(6, 4, 7, 5)
    isolated = box(9, 0, 10, 1)
    forest = box(0, 0, 3, 3)
    _write(tmp_path, 1, [_feature(lake, id=1, category="водоём"), _feature(forest, "лес")])
    _write(tmp_path, 2, [_feature(g, id=index, category="водоём") for index, g in
                         enumerate((edge, overlap, lake, point_contact, isolated), start=2)])
    # Невыбранный снимок не должен попасть в общий файл или даже читаться.
    (tmp_path / "tile_003.geojson").write_text("повреждённый файл", encoding="utf-8")
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    result = _single_test_sample_geojson(tmp_path, [SimpleNamespace(tile_index=i) for i in (1, 2)])
    assert result["_mlsystem2_task"] == "multiclass"
    assert len(result["_mlsystem2_classes"]) == 2
    assert len({feature["id"] for feature in result["features"]}) == 4
    water = [f for f in result["features"] if f["properties"]["_mlsystem2_class"] == "вода"]
    assert len(water) == 3  # Точечный контакт и отдельный участок не склеены.
    combined = unary_union([shape(f["geometry"]) for f in water])
    expected = unary_union([lake, edge, overlap, point_contact, isolated])
    assert combined.symmetric_difference(expected).area == 0
    largest = max(water, key=lambda f: shape(f["geometry"]).area)
    assert len(shape(largest["geometry"]).interiors) == 1
    assert largest["properties"]["category"] == "водоём"
    assert "id" not in largest["properties"]
    assert all(shape(f["geometry"]).is_valid for f in result["features"])
    assert shape(next(f for f in result["features"] if f["properties"]["_mlsystem2_class"] == "лес")["geometry"]).equals(forest)
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_single_geojson_keeps_native_coordinates_and_transforms_only_other_crs(tmp_path):
    native = box(1_234_567.123456789, 2_345_678.987654321, 1_234_568.123456789, 2_345_679.987654321)
    geographic = box(40, 50, 40.001, 50.001)
    _write(tmp_path, 1, [_feature(native)])
    _write(tmp_path, 2, [_feature(geographic)], crs="EPSG:4326")
    result = _single_test_sample_geojson(tmp_path, [SimpleNamespace(tile_index=i) for i in (1, 2)])
    geometries = [shape(f["geometry"]) for f in result["features"]]
    assert result["crs"]["properties"]["name"] == "EPSG:3857"
    assert any(g.bounds == native.bounds for g in geometries)
    projected = transform(Transformer.from_crs(4326, 3857, always_xy=True).transform, geographic)
    assert unary_union(geometries).symmetric_difference(unary_union([native, projected])).area == 0


def test_single_geojson_allows_empty_annotations(tmp_path):
    _write(tmp_path, 1, [])
    result = _single_test_sample_geojson(tmp_path, [SimpleNamespace(tile_index=1)])
    assert result["features"] == []
    assert result["crs"]["properties"]["name"] == "EPSG:3857"


@pytest.mark.parametrize("bad", [
    Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)]),
    Polygon(),
])
def test_single_geojson_rejects_invalid_or_empty_polygons(tmp_path, bad):
    _write(tmp_path, 1, [_feature(bad)])
    with pytest.raises(TrainingUIAPIError, match="некорректный полигон"):
        _single_test_sample_geojson(tmp_path, [SimpleNamespace(tile_index=1)])
