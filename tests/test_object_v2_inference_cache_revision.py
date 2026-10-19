from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from types import SimpleNamespace
import uuid

import pytest

from mlsystem2.training_ui_api import _dataset_editor, _test_samples


_V2_PROFILE = "object_f1_instances_v2_interior_geodesic_fp32_shared_edges_v1"


def _digest(payload: dict) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    ("variant", "previous_profile"),
    [
        (None, None),
        ("legacy", None),
        ("next_gen", None),
        ("next_gen2", "next_gen2_eval_v1"),
        ("next_gen3", "next_gen2_eval_v1"),
        ("next_gen4", None),
        ("object_f1", "object_f1_instances_v1"),
        ("object_f1_v2", "object_f1_instances_v2"),
        ("object_f1_v2", "object_f1_instances_v2_interior_geodesic"),
    ],
)
@pytest.mark.parametrize("managed", [False, True])
def test_f1_revision_changes_only_affected_profiles(variant, previous_profile, managed) -> None:
    source_job = SimpleNamespace(config={"train.pipeline_variant": variant})
    session = SimpleNamespace(scalar=lambda _query: None, get=lambda _kind, _key: source_job)
    result = SimpleNamespace(job_id="training-job")
    scope = [{"sample_id": "выборка", "revision": 3}] if managed else None
    previous_payload = {
        "evaluator_version": _test_samples.TEST_SAMPLE_F1_EVALUATOR_VERSION,
        "postprocess_profile": "none",
        "template_config": {},
    }
    if previous_profile is not None:
        previous_payload["inference_profile"] = previous_profile
    if scope is not None:
        previous_payload.update(evaluation_scope=scope, f1_aggregation="macro")

    _, values, current = _test_samples._effective_inference_template(
        session, "test", "none", training_result=result, evaluation_scope=scope,
    )

    assert values == {}
    if variant == "object_f1_v2":
        assert current != _digest(previous_payload)
        assert current == _digest({**previous_payload, "inference_profile": _V2_PROFILE})
    elif variant in {"next_gen2", "next_gen3", "next_gen4"}:
        assert current != _digest(previous_payload)
        assert current == _digest({**previous_payload, "inference_profile": "notebook_valid_data_v2"})
    else:
        assert current == _digest(previous_payload)


def _selected_model():
    return SimpleNamespace(
        inference_template_id=uuid.UUID("2e8424b4-55a3-4a1e-9b15-4c55ecf4a701"),
        inference_template_config={"postprocess.filter_compact_objects.mode": "keep_compact"},
        checkpoint=SimpleNamespace(
            threshold=0.5, artifact_uri="s3://artifacts/best.pt",
            artifact_path="checkpoints/best.pt", f1_score=0.7, epoch=10,
        ),
        input_channels=3, dataset_name="ЗУ500 / main", external_model=None,
    )


def _previous_editor_payload(selected, tile_size):
    return {
        "algorithm_version": _dataset_editor._EDITOR_PSEUDO_ALGORITHM_VERSION,
        "inference_template_id": str(selected.inference_template_id),
        "inference_template_config": selected.inference_template_config,
        "checkpoint_threshold": selected.checkpoint.threshold,
        "tile_size": tile_size,
    }


def test_editor_revision_preserves_previous_payload_and_invalidates_v2_dedup() -> None:
    selected = _selected_model()
    previous_payload = _previous_editor_payload(selected, 768)
    previous_hash = _digest(previous_payload)

    assert _dataset_editor._editor_pseudo_inference_config_hash(selected, 768) == previous_hash
    assert _dataset_editor._editor_pseudo_inference_config_hash(
        selected, 768, inference_profile=None,
    ) == previous_hash
    current_hash = _dataset_editor._editor_pseudo_inference_config_hash(
        selected, 768, inference_profile=_V2_PROFILE,
    )
    assert current_hash == _digest({**previous_payload, "inference_profile": _V2_PROFILE})
    assert current_hash != previous_hash
    previous_v2_hash = _dataset_editor._editor_pseudo_inference_config_hash(
        selected, 768, inference_profile="object_f1_instances_v2_interior_geodesic",
    )
    assert current_hash != previous_v2_hash
    result_id = uuid.UUID("79275782-d161-443f-866c-f2a259b62ad4")
    assert _dataset_editor._editor_pseudo_dedup_key(
        "test", "снимок.geojson", result_id, "raster-revision", current_hash,
    ) != _dataset_editor._editor_pseudo_dedup_key(
        "test", "снимок.geojson", result_id, "raster-revision", previous_hash,
    )


@pytest.mark.parametrize("variant", [None, "legacy", "next_gen2", "next_gen3", "next_gen4", "object_f1", "object_f1_v2"])
def test_editor_scene_job_uses_source_variant_and_records_only_v2_revision(
    tmp_path, monkeypatch, variant,
) -> None:
    image_path = tmp_path / "снимок.tif"
    image_path.write_bytes(b"raster")
    source_job = SimpleNamespace(config={"train.pipeline_variant": variant, "tile_preparation.tile_size": 768})
    selected = _selected_model()
    primary = SimpleNamespace(
        id=uuid.uuid4(), job_id=uuid.uuid4(), model_name="SegFormer B2",
        architecture="smp_segformer_b2", task="binary", class_schema=[], mlflow_run_id="run",
    )
    selected.result = primary
    dataset = SimpleNamespace(
        key="test", class_key="zu500", name="ЗУ500 / main", class_name="ЗУ500",
        input_channels=3, version="revision", imagery_type=SimpleNamespace(value="ortho"),
    )
    scene = SimpleNamespace(annotation_name="снимок.geojson", image_name=image_path.name)
    rows = []

    def flush():
        for row in rows:
            row.id = uuid.uuid4()

    session = SimpleNamespace(
        get=lambda _kind, _key: source_job, scalar=lambda _query: None,
        begin_nested=nullcontext, add=rows.append, flush=flush,
    )
    monkeypatch.setattr(_dataset_editor, "_editor_read_lock", lambda _config: nullcontext())
    monkeypatch.setattr(_dataset_editor, "_editor_dataset_context", lambda *_args: (dataset, tmp_path))
    monkeypatch.setattr(_dataset_editor, "_scene_by_annotation", lambda *_args: scene)
    monkeypatch.setattr(_dataset_editor, "_matched_image_path", lambda *_args: image_path)
    monkeypatch.setattr(_dataset_editor, "_dataset_images_root", lambda *_args: tmp_path)
    monkeypatch.setattr(_dataset_editor, "dataset_training_result", lambda *_args: primary)
    monkeypatch.setattr(_dataset_editor, "_editor_pseudo_compatibility_error", lambda *_args: None)
    monkeypatch.setattr(_dataset_editor, "_latest_covering_pseudo_result", lambda *_args: None)
    monkeypatch.setattr(_dataset_editor, "_select_model", lambda *_args, **_kwargs: selected)
    monkeypatch.setattr(_dataset_editor, "next_queue_position", lambda *_args: 1)

    result = _dataset_editor.editor_scene_pseudo_markup(
        session, SimpleNamespace(), dataset.key, scene.annotation_name, ensure=True,
    )

    assert result.status == "queued"
    assert len(rows) == 1
    snapshot = rows[0].config["editor_pseudo"]
    previous_payload = _previous_editor_payload(selected, 768)
    assert snapshot["algorithm_version"] == previous_payload["algorithm_version"]
    if variant == "object_f1_v2":
        assert snapshot["inference_profile"] == _V2_PROFILE
        expected_hash = _digest({**previous_payload, "inference_profile": _V2_PROFILE})
    elif variant in {"next_gen2", "next_gen3", "next_gen4"}:
        assert snapshot["inference_profile"] == "notebook_valid_data_v2"
        expected_hash = _digest({**previous_payload, "inference_profile": "notebook_valid_data_v2"})
    else:
        assert "inference_profile" not in snapshot
        expected_hash = _digest(previous_payload)
    assert snapshot["inference_config_hash"] == expected_hash
    assert rows[0].dedup_key == _dataset_editor._editor_pseudo_dedup_key(
        dataset.key, scene.annotation_name, primary.id,
        snapshot["raster_revision"], expected_hash,
    )
