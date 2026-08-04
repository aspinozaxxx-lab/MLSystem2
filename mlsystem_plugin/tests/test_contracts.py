from __future__ import annotations

import pytest

from mlsystem_plugin.contracts import (
    PluginContractError,
    build_job_payload,
    validate_feature_collection,
)


# Proveriaet otsutstvie tyazhelyh i proizvolnyh polei zaprosa.
def test_job_payload_contains_only_minimal_fields() -> None:
    payload = build_job_payload(
        "class-1",
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        "EPSG:4326",
    )

    assert set(payload) == {"class_id", "aoi", "aoi_crs"}


def test_job_payload_includes_selected_imagery_source() -> None:
    payload = build_job_payload(
        "class-1",
        {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        "EPSG:4326",
        "ortho",
    )

    assert payload["source_id"] == "ortho"


# Proveriaet unikalnost stabilnyh ID pri zagruzke.
def test_feature_collection_rejects_duplicate_candidate_ids() -> None:
    feature = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": []},
        "properties": {"candidate_id": "same"},
    }

    with pytest.raises(PluginContractError, match="повторяется"):
        validate_feature_collection(
            {"type": "FeatureCollection", "features": [feature, dict(feature)]}
        )


def test_feature_collection_rejects_confidence_outside_probability_range() -> None:
    feature = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": []},
        "properties": {"candidate_id": "candidate-1", "confidence": 1.1},
    }

    with pytest.raises(PluginContractError, match="от 0 до 1"):
        validate_feature_collection({"type": "FeatureCollection", "features": [feature]})
