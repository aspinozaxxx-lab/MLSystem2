from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from mlsystem2.dataset_preparing.contracts import (
    DatasetClassDefinition,
    DatasetManifest,
    DatasetSourceRevision,
)
from mlsystem2.training_ui_api import _combined_dataset
from mlsystem2.training_ui_api._dataset_editor import (
    _local_rebuild_changes,
    _merge_rebuild_payloads,
    _rebuild_conflicts,
    _replace_dataset_files_atomically,
)
from mlsystem2.training_ui_api._combined_dataset import build_combined_dataset
from mlsystem2.training_ui_api._markup_export import IntersectingImage, IntersectingImages


def test_combined_builder_applies_priority_background_and_stable_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "markup"
    first = repo / "first" / "main"
    second = repo / "second" / "main"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    _write_geojson(first / "first.geojson", [_feature("a", box(1, 1, 6, 6))])
    _write_geojson(
        first / "hard_negative.geojson",
        [_feature("hn", box(0, 0, 3, 3)), {"type": "Feature", "properties": {}, "geometry": None}],
    )
    _write_geojson(second / "second.geojson", [_feature("b", box(4, 4, 8, 8))])

    image = tmp_path / "images" / "scene.tif"
    image.parent.mkdir()
    with rasterio.open(
        image,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(0, 10, 1, 1),
        nodata=0,
    ) as target:
        target.write(np.ones((1, 10, 10), dtype=np.uint8))

    monkeypatch.setattr(
        _combined_dataset,
        "find_intersecting_images",
        lambda *_args, **_kwargs: IntersectingImages(
            images=(IntersectingImage(source_id="scene", path=image),),
            coverage_percent=100.0,
            warnings=(),
        ),
    )
    monkeypatch.setattr(_combined_dataset, "_git_head", lambda _root: "git-test")
    manifest = DatasetManifest(
        schema_version=1,
        task="multiclass",
        combined=True,
        classes=[
            DatasetClassDefinition(id=1, slug="first", name="Первый", color="#F59E0B", priority=100),
            DatasetClassDefinition(id=2, slug="second", name="Второй", color="#8B5CF6", priority=0),
        ],
        sources=[
            DatasetSourceRevision(
                path="first/main",
                class_slug="first",
                git_revision="seed",
                tree_revision="seed",
            ),
            DatasetSourceRevision(
                path="second/main",
                class_slug="second",
                git_revision="seed",
                tree_revision="seed",
            ),
        ],
    )

    first_build = build_combined_dataset(
        manifest=manifest,
        repo_root=repo,
        images_root=image.parent,
        build_id="test-build",
        built_at="2026-08-12T00:00:00Z",
        code_revision="test-code",
    )
    second_build = build_combined_dataset(
        manifest=manifest,
        repo_root=repo,
        images_root=image.parent,
        build_id="test-build-2",
        built_at="2026-08-12T00:00:01Z",
        code_revision="test-code",
    )

    assert len(first_build.files) == 1
    payload = next(iter(first_build.files.values()))
    assert payload["_mlsystem2_task"] == "multiclass"
    assert payload["_mlsystem2_classes"] == [item.model_dump(mode="json") for item in manifest.classes]
    features = payload["features"]
    by_class = {
        item["properties"].get("_mlsystem2_class") or "hard_negative": shape(item["geometry"])
        for item in features
    }
    assert by_class["first"].intersection(by_class["second"]).area == 0.0
    assert by_class["first"].intersection(by_class["hard_negative"]).area == 0.0
    assert by_class["second"].area == pytest.approx(12.0, abs=1e-9)
    assert by_class["hard_negative"].area == pytest.approx(5.0, abs=1e-9)
    assert any("пустая геометрия" in warning for warning in first_build.warnings)
    assert [item["id"] for item in features] == [
        item["id"] for item in next(iter(second_build.files.values()))["features"]
    ]
    assert all(item["properties"].get("_mlsystem2_origin_key") for item in features)
    assert first_build.manifest.scene_ids == ["scene"]
    assert first_build.manifest.baseline_hashes


def test_source_features_disambiguate_duplicate_source_ids(tmp_path: Path) -> None:
    repo = tmp_path / "markup"
    source = repo / "first" / "main" / "first.geojson"
    source.parent.mkdir(parents=True)
    _write_geojson(
        source,
        [
            _feature("duplicate", box(0, 0, 1, 1)),
            _feature("duplicate", box(2, 0, 3, 1)),
            _feature("duplicate", box(4, 0, 5, 1)),
        ],
    )

    first = _combined_dataset._load_source_features(
        source,
        repo,
        role="positive",
        class_slug="first",
        warnings_list=[],
    )
    second = _combined_dataset._load_source_features(
        source,
        repo,
        role="positive",
        class_slug="first",
        warnings_list=[],
    )

    assert len({item.origin_key for item in first}) == 3
    assert [item.origin_key for item in first] == [item.origin_key for item in second]


def test_target_priorities_remove_reprojection_overlap() -> None:
    high = _source_feature("high", "first", "positive")
    low = _source_feature("low", "second", "positive")
    hard_negative = _source_feature("hard", None, "hard_negative")
    classes = [
        DatasetClassDefinition(id=1, slug="first", name="Первый", color="#F59E0B", priority=100),
        DatasetClassDefinition(id=2, slug="second", name="Второй", color="#8B5CF6", priority=0),
    ]

    resolved = _combined_dataset._apply_target_priorities(
        [
            (high, box(0, 0, 2, 2)),
            (low, box(1, 1, 3, 3)),
            (hard_negative, box(0, 0, 4, 4)),
        ],
        classes,
    )
    by_key = {item.origin_key: geometry for item, geometry in resolved}

    assert by_key["high"].intersection(by_key["low"]).area == 0.0
    assert unary_union([by_key["high"], by_key["low"]]).intersection(by_key["hard"]).area == 0.0
    assert by_key["high"].distance(by_key["low"]) > 0.0
    assert by_key["low"].area == pytest.approx(3.0, abs=1e-9)
    assert by_key["hard"].area == pytest.approx(9.0, abs=1e-9)


def test_rebuild_merge_preserves_manual_edits_additions_and_deletions() -> None:
    original = _origin_feature("source:a", "original", box(0, 0, 2, 2))
    deleted = _origin_feature("source:deleted", "deleted", box(3, 0, 4, 1))
    manual_edit = _origin_feature("source:a", "manual-edit", box(0, 0, 3, 3))
    manual_add = _origin_feature("manual:new", "manual-add", box(5, 0, 6, 1))
    source_edit = _origin_feature("source:a", "source-edit", box(0, 0, 4, 4))
    source_add = _origin_feature("source:new", "source-add", box(7, 0, 8, 1))
    manifest = _rebuild_manifest(
        {
            "scene.geojson": {
                "source:a": _combined_dataset.feature_hash(original),
                "source:deleted": _combined_dataset.feature_hash(deleted),
            }
        }
    )
    current = {"scene.geojson": _multiclass_payload(manifest, [manual_edit, manual_add])}
    candidate = {
        "scene.geojson": _multiclass_payload(
            manifest,
            [source_edit, deleted, source_add],
        )
    }

    local_changes = _local_rebuild_changes(manifest, current)
    conflicts = _rebuild_conflicts(manifest, current, candidate, local_changes)
    merged = _merge_rebuild_payloads(manifest, current, candidate)["scene.geojson"]
    by_origin = {
        feature["properties"]["_mlsystem2_origin_key"]: feature
        for feature in merged["features"]
    }

    assert set(by_origin) == {"source:a", "manual:new", "source:new"}
    assert by_origin["source:a"]["id"] == "manual-edit"
    assert by_origin["manual:new"]["id"] == "manual-add"
    assert by_origin["source:new"]["id"] == "source-add"
    assert {item.origin_key for item in conflicts} == {"source:a"}
    assert merged["_mlsystem2_task"] == "multiclass"
    assert merged["_mlsystem2_classes"] == [
        item.model_dump(mode="json") for item in manifest.classes
    ]


def test_rebuild_replace_is_atomic_and_removes_obsolete_geojson(tmp_path: Path) -> None:
    dataset = tmp_path / "combined" / "main"
    dataset.mkdir(parents=True)
    (dataset / "obsolete.geojson").write_text("{}", encoding="utf-8")
    (dataset / "keep.md").write_text("keep", encoding="utf-8")
    manifest = _rebuild_manifest({})
    payload = _multiclass_payload(
        manifest,
        [_origin_feature("source:a", "replacement", box(0, 0, 1, 1))],
    )

    _replace_dataset_files_atomically(
        dataset,
        {"replacement.geojson": payload},
        manifest.model_dump(mode="json"),
    )

    assert not (dataset / "obsolete.geojson").exists()
    assert json.loads(
        (dataset / "replacement.geojson").read_text(encoding="utf-8")
    ) == json.loads(json.dumps(payload))
    assert (dataset / "keep.md").read_text(encoding="utf-8") == "keep"
    saved_manifest = json.loads(
        (dataset / ".mlsystem2-dataset.json").read_text(encoding="utf-8")
    )
    assert saved_manifest["task"] == "multiclass"


def _write_geojson(path: Path, features: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
                "features": features,
            }
        ),
        encoding="utf-8",
    )


def _feature(feature_id: str, geometry) -> dict[str, object]:
    return {
        "type": "Feature",
        "id": feature_id,
        "properties": {"name": feature_id},
        "geometry": mapping(geometry),
    }


def _source_feature(
    origin_key: str,
    class_slug: str | None,
    role: str,
) -> _combined_dataset._SourceFeature:
    return _combined_dataset._SourceFeature(
        geometry_wgs84=box(0, 0, 1, 1),
        properties={},
        feature_id=origin_key,
        origin_key=origin_key,
        origin_hash=f"hash:{origin_key}",
        source_path="source.geojson",
        role=role,
        class_slug=class_slug,
    )


def _origin_feature(origin_key: str, feature_id: str, geometry) -> dict[str, object]:
    return {
        "type": "Feature",
        "id": feature_id,
        "properties": {
            "_mlsystem2_role": "positive",
            "_mlsystem2_class": "first",
            "_mlsystem2_origin_key": origin_key,
        },
        "geometry": mapping(geometry),
    }


def _rebuild_manifest(
    baseline_hashes: dict[str, dict[str, str]],
) -> DatasetManifest:
    return DatasetManifest(
        schema_version=1,
        task="multiclass",
        combined=True,
        classes=[
            DatasetClassDefinition(
                id=1,
                slug="first",
                name="Первый",
                color="#F59E0B",
                priority=100,
            ),
            DatasetClassDefinition(
                id=2,
                slug="second",
                name="Второй",
                color="#8B5CF6",
                priority=0,
            ),
        ],
        sources=[
            DatasetSourceRevision(
                path="first/main",
                class_slug="first",
                git_revision="seed",
                tree_revision="seed",
            ),
            DatasetSourceRevision(
                path="second/main",
                class_slug="second",
                git_revision="seed",
                tree_revision="seed",
            ),
        ],
        baseline_hashes=baseline_hashes,
    )


def _multiclass_payload(
    manifest: DatasetManifest,
    features: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "type": "FeatureCollection",
        "_mlsystem2_schema_version": manifest.schema_version,
        "_mlsystem2_task": manifest.task,
        "_mlsystem2_classes": [
            item.model_dump(mode="json") for item in manifest.classes
        ],
        "features": features,
    }
