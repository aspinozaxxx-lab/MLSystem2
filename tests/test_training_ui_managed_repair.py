from __future__ import annotations

from copy import deepcopy

import pytest
from pyproj import Transformer
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union

from mlsystem2.training_ui_api._managed_repair import _repair_scene_payloads


def _payload(crs: str, features: list[dict] | None = None) -> dict:
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": features or [],
    }


def _feature(
    geometry,
    *,
    role: str = "positive",
    slug: str | None = None,
    feature_id: str,
    label: str | None = None,
) -> dict:
    properties = {"_mlsystem2_role": role}
    if slug is not None:
        properties["_mlsystem2_class"] = slug
    if label is not None:
        properties["label"] = label
    return {
        "type": "Feature",
        "id": feature_id,
        "properties": properties,
        "geometry": mapping(geometry),
    }


def _role_geometries(payload: dict, role: str) -> list:
    return [
        shape(feature["geometry"])
        for feature in payload["features"]
        if feature["properties"].get("_mlsystem2_role", "positive") == role
    ]


def test_repair_restores_missing_and_partial_positives_without_removing_new_markup() -> None:
    historical = _payload(
        "EPSG:3857",
        [
            _feature(box(0, 0, 10, 10), slug="first", feature_id="old-first"),
            _feature(box(20, 0, 30, 10), slug="second", feature_id="old-second"),
        ],
    )
    later = _feature(box(-10, 0, -5, 5), feature_id="later", label="keep")
    first = _payload(
        "EPSG:3857",
        [
            _feature(box(0, 0, 5, 10), feature_id="partial"),
            later,
        ],
    )

    repaired = _repair_scene_payloads(
        historical,
        {"first": first, "second": None},
        target_key="managed-key",
        annotation_name="scene.geojson",
    )

    assert repaired.positive_features_added == {"first": 1, "second": 1}
    assert repaired.positive_area_added["first"] == pytest.approx(50.0)
    assert repaired.positive_area_added["second"] == pytest.approx(100.0)
    first_union = unary_union(_role_geometries(repaired.payloads["first"], "positive"))
    second_union = unary_union(_role_geometries(repaired.payloads["second"], "positive"))
    assert first_union.covers(box(0, 0, 10, 10))
    assert first_union.covers(box(-10, 0, -5, 5))
    assert second_union.covers(box(20, 0, 30, 10))
    assert any(
        feature["properties"].get("label") == "keep"
        for feature in repaired.payloads["first"]["features"]
    )


def test_repair_copies_hard_negatives_to_every_source_and_keeps_later_additions() -> None:
    historical = _payload(
        "EPSG:3857",
        [
            _feature(
                box(0, 0, 10, 10),
                role="hard_negative",
                feature_id="old-negative",
                label="old",
            )
        ],
    )
    later_negative = _feature(
        box(20, 0, 30, 10),
        role="hard_negative",
        feature_id="later-negative",
        label="later",
    )

    repaired = _repair_scene_payloads(
        historical,
        {
            "first": _payload("EPSG:3857", [later_negative]),
            "second": _payload("EPSG:3857"),
        },
        target_key="managed-key",
        annotation_name="scene.geojson",
    )

    assert repaired.hard_negative_features == 2
    assert repaired.hard_negative_changed_slugs == {"first", "second"}
    first = _role_geometries(repaired.payloads["first"], "hard_negative")
    second = _role_geometries(repaired.payloads["second"], "hard_negative")
    assert len(first) == len(second) == 2
    assert unary_union(first).equals(unary_union(second))
    assert {
        feature["properties"].get("label") for feature in repaired.payloads["first"]["features"]
    } == {
        "old",
        "later",
    }
    first_origins = {
        feature["properties"]["_mlsystem2_origin_key"]
        for feature in repaired.payloads["first"]["features"]
    }
    second_origins = {
        feature["properties"]["_mlsystem2_origin_key"]
        for feature in repaired.payloads["second"]["features"]
    }
    assert first_origins == second_origins


def test_repair_is_idempotent() -> None:
    historical = _payload(
        "EPSG:3857",
        [
            _feature(box(0, 0, 10, 10), slug="first", feature_id="positive"),
            _feature(
                box(20, 0, 30, 10),
                role="hard_negative",
                feature_id="negative",
            ),
        ],
    )
    first = _repair_scene_payloads(
        historical,
        {"first": _payload("EPSG:3857"), "second": None},
        target_key="managed-key",
        annotation_name="scene.geojson",
    )
    second = _repair_scene_payloads(
        historical,
        deepcopy(first.payloads),
        target_key="managed-key",
        annotation_name="scene.geojson",
    )

    assert second.positive_features_added == {}
    assert second.hard_negative_changed_slugs == set()
    assert second.payloads == first.payloads


def test_repair_transforms_source_crs_before_comparing_geometry() -> None:
    historical_geometry = box(30.0, 60.0, 30.01, 60.01)
    historical = _payload(
        "EPSG:4326",
        [_feature(historical_geometry, slug="first", feature_id="positive")],
    )
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    projected = transform_geometry(transformer.transform, historical_geometry)
    current = _payload(
        "EPSG:3857",
        [_feature(projected, feature_id="projected")],
    )

    repaired = _repair_scene_payloads(
        historical,
        {"first": current, "second": _payload("EPSG:4326")},
        target_key="managed-key",
        annotation_name="scene.geojson",
    )

    assert repaired.positive_features_added == {}
    assert repaired.payloads["first"] == current
