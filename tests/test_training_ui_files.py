from __future__ import annotations

import json
import zipfile
from io import BytesIO

import pytest

from mlsystem2.training_ui_api._routes.files import _multiclass_geojson_archive


def test_multiclass_geojson_archive_splits_features_by_type() -> None:
    schema = [
        {
            "id": 1,
            "slug": "flooding",
            "name": "Переувлажнения",
            "color": "#3B82F6",
            "priority": 100,
        },
        {
            "id": 2,
            "slug": "waterlogging",
            "name": "Заболачивание",
            "color": "#22C55E",
            "priority": 0,
        },
    ]
    payload = {
        "type": "FeatureCollection",
        "features": [
            _pseudo_feature("flooding", 1),
            _pseudo_feature("waterlogging", 2),
            _pseudo_feature("flooding", 3),
        ],
        "metadata": {"task": "multiclass", "class_schema": schema},
    }

    with zipfile.ZipFile(BytesIO(_multiclass_geojson_archive(payload))) as archive:
        assert set(archive.namelist()) == {
            "flooding.geojson",
            "waterlogging.geojson",
        }
        flooding = json.loads(archive.read("flooding.geojson"))
        waterlogging = json.loads(archive.read("waterlogging.geojson"))

    assert [item["id"] for item in flooding["features"]] == [1, 3]
    assert [item["id"] for item in waterlogging["features"]] == [2]
    assert flooding["metadata"]["class_schema"] == [schema[0]]
    assert waterlogging["metadata"]["class_schema"] == [schema[1]]


def test_multiclass_geojson_archive_rejects_unsafe_slug() -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [],
        "metadata": {"class_schema": [{"id": 1, "slug": "../escape"}]},
    }

    with pytest.raises(ValueError, match="slug"):
        _multiclass_geojson_archive(payload)


def _pseudo_feature(slug: str, feature_id: int) -> dict[str, object]:
    return {
        "type": "Feature",
        "id": feature_id,
        "properties": {"object_type_slug": slug},
        "geometry": None,
    }
