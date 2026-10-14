"""Выбор весов ручной псевдоразметки не меняет контрольные и неявные прогнозы."""

from __future__ import annotations

import hashlib
import io
import json
import uuid
import zipfile
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from mlsystem2.mlflow_adapter.contracts import MLflowAdapterError, MLflowBestCheckpoint
from mlsystem2.training_ui_api import _dataset_editor, _geoalert_runner, _model_export, _pseudo_runner, _pseudo_viewer, _service, _test_samples, _worker
from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import Base, configure_schema, create_session_factory
from mlsystem2.training_ui_api._models import JobRow, PseudoMarkupResultRow, StoredFileRow, TrainingResultRow
from mlsystem2.training_ui_api.contracts import DatasetInfo, TrainingUIAPIError


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "files"))
    config = get_config()
    configure_schema(None)
    factory = create_session_factory(config)
    Base.metadata.create_all(factory.kw["bind"])
    with factory() as session:
        job = JobRow(type="training", source="manual", status="completed", queue_position=1,
                     dataset_key="ЗУ500/main", dataset_name="ЗУ500\\main", model_name="SegFormer B2",
                     architecture="smp_segformer_b2", config={"train.pipeline_variant": "object_f1_v2", "dataset.input_channels": 3})
        session.add(job)
        session.flush()
        result = TrainingResultRow(dataset_key=job.dataset_key, class_key=job.dataset_key,
                                   class_display_name=job.dataset_name, architecture=job.architecture,
                                   model_name=job.model_name, mlflow_run_id="run-fixed", epoch=7,
                                   status="ok", job_id=job.id)
        session.add(result)
        session.flush()
        monkeypatch.setattr(_service, "external_result_manifest", lambda *_args: None)
        monkeypatch.setattr(_service, "get_best_training_checkpoint", lambda *_args: MLflowBestCheckpoint(
            tracking_uri="http://mlflow", run_id="run-fixed", artifact_path="checkpoints/best.pt",
            artifact_uri="s3://run-fixed/best.pt", metric_name="val/object_f1", f1_score=0.42,
            epoch=7, threshold=0.5))
        yield SimpleNamespace(session=session, config=config, result=result, tmp=tmp_path)


def test_default_best_preserves_checkpoint_contract(environment):
    env = environment
    snapshot = _service._checkpoint_config(env.session, env.result, env.config)
    assert snapshot["checkpoint_kind"] == "best"
    assert snapshot["checkpoint_artifact_path"] == "checkpoints/best.pt"
    assert snapshot["checkpoint_epoch"] == 7
    assert snapshot["checkpoint_threshold"] == 0.5


def test_last_checks_exact_finished_artifact_without_best_metadata(environment, monkeypatch):
    env = environment
    calls = []
    def artifact(*args):
        calls.append(args)
        return SimpleNamespace(artifact_path="checkpoints/final.pt", artifact_uri="s3://run-fixed/final.pt")
    monkeypatch.setattr(_service, "get_finished_run_artifact", artifact)
    monkeypatch.setattr(_service, "get_best_training_checkpoint", lambda *_args: pytest.fail("Последние веса не читают лучшие"))
    snapshot = _service._checkpoint_config(env.session, env.result, env.config, checkpoint="last")
    assert calls == [(env.config.mlflow_tracking_uri, "run-fixed", "checkpoints/final.pt")]
    assert snapshot == {"mlflow_run_id": "run-fixed", "checkpoint_kind": "last",
                        "checkpoint_artifact_path": "checkpoints/final.pt", "checkpoint_uri": "s3://run-fixed/final.pt"}


@pytest.mark.parametrize("failure", ["missing", "unavailable", "external", "no_result", "running"])
def test_last_rejects_unavailable_weights_without_silent_fallback(environment, monkeypatch, failure):
    env = environment
    monkeypatch.setattr(_service, "get_finished_run_artifact", lambda *_args: None)
    if failure == "unavailable":
        def unavailable(*_args):
            raise MLflowAdapterError("Недоступно")
        monkeypatch.setattr(_service, "get_finished_run_artifact", unavailable)
    if failure == "external":
        monkeypatch.setattr(_service, "external_result_manifest", lambda *_args: SimpleNamespace())
    if failure == "running":
        env.result.status = "running"
    monkeypatch.setattr(_service, "get_best_training_checkpoint", lambda *_args: pytest.fail("Подмена на лучшие запрещена"))
    with pytest.raises(TrainingUIAPIError):
        _service._checkpoint_config(env.session, None if failure == "no_result" else env.result, env.config, checkpoint="last")


@pytest.mark.parametrize("variant,expected", [("legacy", 0.73), ("next_gen2", 0.9), ("object_f1_v2", 0.5)])
def test_worker_reads_final_epoch_threshold_and_parameters_from_selected_file(environment, monkeypatch, variant, expected):
    env = environment
    path = env.tmp / "final.pt"
    path.write_bytes(b"exact-final-not-best")
    calls = []
    monkeypatch.setattr(_worker, "get_finished_run_artifact", lambda *_args: SimpleNamespace())
    def download(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(local_path=str(path))
    monkeypatch.setattr(_worker, "download_run_artifact", download)
    metadata = {"epoch": 28, "confidence_threshold": 0.73, "pipeline_variant": variant,
                "preprocessing": {"profile": "min_max"}, "val_quality_f1": 0.25}
    parameters = {"pipeline_variant": variant, "object_separation": {"foreground_threshold": 0.5, "boundary_threshold": 0.65}}
    def load(request):
        assert request.map_location == "cpu"
        assert request.checkpoint_uri == str(path)
        return SimpleNamespace(artifact=SimpleNamespace(metadata=metadata), model=SimpleNamespace(spec=SimpleNamespace(parameters=parameters)))
    monkeypatch.setattr(_worker, "load_checkpoint", load)
    snapshot = _worker._last_pseudo_checkpoint_snapshot({"mlflow_run_id": "run-fixed", "checkpoint_kind": "last",
        "checkpoint_artifact_path": "checkpoints/final.pt", "checkpoint_epoch": 7, "checkpoint_threshold": 0.1}, env.config, env.tmp / "job")
    assert calls[0]["artifact_path"] == "checkpoints/final.pt"
    assert snapshot["checkpoint_epoch"] == 28
    assert snapshot["checkpoint_threshold"] == expected
    assert snapshot["checkpoint_metadata"] == metadata
    assert snapshot["checkpoint_model_parameters"] == parameters
    assert snapshot["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert _pseudo_runner._resolve_checkpoint(snapshot, env.tmp / "ignored") == path


def test_worker_fails_if_selected_final_disappeared(environment, monkeypatch):
    monkeypatch.setattr(_worker, "get_finished_run_artifact", lambda *_args: None)
    monkeypatch.setattr(_worker, "download_run_artifact", lambda **_kwargs: pytest.fail("Отсутствующие веса не скачиваются"))
    with pytest.raises(RuntimeError, match="больше недоступен"):
        _worker._last_pseudo_checkpoint_snapshot({"mlflow_run_id": "run-fixed", "checkpoint_kind": "last",
            "checkpoint_artifact_path": "checkpoints/final.pt"}, environment.config, environment.tmp)


@pytest.mark.parametrize("checkpoint", ["best", "last"])
def test_manual_creation_and_worker_keep_exact_selected_checkpoint(environment, monkeypatch, checkpoint):
    env = environment
    monkeypatch.setattr(_service, "ensure_seed_templates", lambda *_args: None)
    monkeypatch.setattr(_service, "find_managed_dataset", lambda *_args: DatasetInfo(key=env.result.dataset_key, name="ЗУ500\\main", input_channels=3))
    monkeypatch.setattr(_service, "effective_inference_template_row", lambda *_args: SimpleNamespace(id=uuid.uuid4(), default_config={}))
    monkeypatch.setattr(_service, "count_scenes_file_images", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(_service, "get_finished_run_artifact", lambda *_args: SimpleNamespace(artifact_path="checkpoints/final.pt", artifact_uri="s3://run-fixed/final.pt"))
    monkeypatch.setattr(_worker, "markup_read_lock", lambda *_args: nullcontext())
    monkeypatch.setattr(_worker, "external_result_manifest", lambda *_args: None)
    expected_epoch = 7
    if checkpoint == "last":
        path = env.tmp / "selected-final.pt"
        path.write_bytes(b"real-last-not-best")
        monkeypatch.setattr(_worker, "get_finished_run_artifact", lambda *_args: SimpleNamespace())
        monkeypatch.setattr(_worker, "download_run_artifact", lambda **_kwargs: SimpleNamespace(local_path=str(path)))
        monkeypatch.setattr(_worker, "load_checkpoint", lambda _request: SimpleNamespace(
            artifact=SimpleNamespace(metadata={"epoch": 28, "confidence_threshold": 0.75, "pipeline_variant": "object_f1_v2"}),
            model=SimpleNamespace(spec=SimpleNamespace(parameters={"object_separation": {"boundary_threshold": 0.65}}))))
        expected_epoch = 28
    detail = _service.create_pseudo_markup_job(env.session, class_key=env.result.class_key, dataset_key=None,
        image_folder_key=None, training_result_id=env.result.id, scenes_name="снимки.txt", scenes_content_type="text/plain",
        scenes_bytes=b"zu/scene\n", config=env.config, checkpoint=checkpoint)
    job = env.session.get(JobRow, detail.id)
    result = env.session.scalar(select(PseudoMarkupResultRow).where(PseudoMarkupResultRow.job_id == job.id))
    payload = _worker._build_pseudo_markup_config(env.session, job, env.config, env.tmp / "worker")
    assert payload["checkpoint_kind"] == checkpoint
    assert payload["checkpoint_artifact_path"] == ("checkpoints/final.pt" if checkpoint == "last" else "checkpoints/best.pt")
    assert payload["checkpoint_epoch"] == expected_epoch
    assert job.config["checkpoint_epoch"] == expected_epoch
    assert _service._pseudo_markup_info(env.session, result).checkpoint_epoch == expected_epoch
    assert _service._pseudo_markup_info(env.session, result).checkpoint_kind == checkpoint
    if checkpoint == "last":
        assert payload["local_checkpoint_path"] == str(path)
        assert job.config["checkpoint_model_parameters"]["object_separation"]["boundary_threshold"] == 0.65


def _pseudo(env, kind, *, epoch=None):
    file_path = env.tmp / f"{uuid.uuid4()}.geojson"
    file_path.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    file = StoredFileRow(kind="pseudo_markup_geojson", original_name=file_path.name, path=str(file_path), size_bytes=file_path.stat().st_size)
    job = JobRow(type="inference", source="manual", status="completed", queue_position=1,
                 dataset_key=env.result.dataset_key, dataset_name="ЗУ500\\main", model_name=env.result.model_name,
                 architecture="pseudo-markup", config=({"checkpoint_kind": kind, "checkpoint_epoch": epoch} if kind else {}))
    env.session.add_all([file, job])
    env.session.flush()
    result = PseudoMarkupResultRow(training_result_id=env.result.id, dataset_key=env.result.dataset_key,
                                  class_key=env.result.class_key, source_dataset_name="ЗУ500\\main",
                                  status="ok", geojson_file_id=file.id, job_id=job.id)
    env.session.add(result)
    env.session.flush()
    return result, job


def test_result_and_viewer_use_saved_checkpoint_not_best_epoch(environment):
    last, job = _pseudo(environment, "last", epoch=28)
    info = _service._pseudo_markup_info(environment.session, last)
    assert info.checkpoint_kind == "last"
    assert info.checkpoint_epoch == 28
    assert _pseudo_viewer.pseudo_checkpoint_details(last, job) == ("last", 28)
    job.config = {"checkpoint_kind": "last"}
    assert _pseudo_viewer.pseudo_checkpoint_details(last, job) == ("last", None)
    historic, historic_job = _pseudo(environment, None)
    assert _pseudo_viewer.pseudo_checkpoint_details(historic, historic_job) == ("best", 7)


def test_implicit_test_and_editor_sources_ignore_newer_last(environment, monkeypatch):
    env = environment
    best, _ = _pseudo(env, None)
    last, _ = _pseudo(env, "last", epoch=28)
    monkeypatch.setattr(_dataset_editor, "_pseudo_result_covers_image", lambda *_args: True)
    dataset = DatasetInfo(key=env.result.dataset_key, name="ЗУ500\\main")
    assert _test_samples.dataset_test_sample_pseudo_markup(env.session, dataset.key, env.result.id, config=None).id == best.id
    assert _dataset_editor._latest_covering_pseudo_result(env.session, dataset, env.result.id, env.tmp / "scene.tif").id == best.id
    assert last.id != best.id
    sample = SimpleNamespace(source_training_result_id=env.result.id, source_pseudo_result_id=last.id, dataset_key=dataset.key)
    assert _test_samples.test_sample_source_pseudo_markup(env.session, sample, config=None).id == last.id


def test_geoalert_cache_distinguishes_best_and_final_bytes_but_reuses_exact_weights(tmp_path, monkeypatch):
    exports = []
    def export(**kwargs):
        name = kwargs["model_name"]
        exports.append((name, kwargs["checkpoint_bytes"]))
        model_zip = io.BytesIO()
        with zipfile.ZipFile(model_zip, "w") as archive:
            archive.writestr(f"{name}/config.pbtxt", f'name: "{name}"')
        path = tmp_path / f"{name}.zip"
        metadata = {"model_archive": "model.zip", "pipeline": "pipeline.yaml", "input_channels": 3,
                    "sample_size": 512, "task": "binary", "output_kind": "probabilities", "threshold": 0.5}
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("export_metadata.json", json.dumps(metadata))
            archive.writestr("model.zip", model_zip.getvalue())
            archive.writestr("pipeline.yaml", _model_export._pipeline_yaml(name, 512, 3, probability_output=True, threshold=0.5))
        return SimpleNamespace(zip_path=path, cleanup=lambda: None)
    monkeypatch.setattr(_geoalert_runner, "build_triton_model_export_zip", export)
    config = {"geoalert_model_repository": str(tmp_path / "models"), "geoalert_pipeline_root": str(tmp_path / "pipelines"), "tile_size": 512, "threshold": 0.5}
    paths = [tmp_path / "best.pt", tmp_path / "final.pt"]
    paths[0].write_bytes(b"best-weights")
    paths[1].write_bytes(b"final-weights")
    best = _geoalert_runner._ensure_runtime_export(config, paths[0], external_manifest=None, postprocess_config={})
    final = _geoalert_runner._ensure_runtime_export(config, paths[1], external_manifest=None, postprocess_config={})
    repeated = _geoalert_runner._ensure_runtime_export(config, paths[1], external_manifest=None, postprocess_config={})
    assert best.model_name != final.model_name
    assert repeated.model_name == final.model_name
    assert len(exports) == 2
    assert exports[0][1] == b"best-weights" and exports[1][1] == b"final-weights"


def test_openapi_checkpoint_selection_default_and_enum(monkeypatch):
    from mlsystem2.training_ui_api.api import get_openapi_schema
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    schema = get_openapi_schema()
    body = schema["paths"]["/api/v1/results/datasets/{dataset_key}/pseudo-markup"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    request = schema["components"]["schemas"][body["$ref"].rsplit("/", 1)[-1]]
    assert request["properties"]["checkpoint"]["enum"] == ["best", "last"]
    assert request["properties"]["checkpoint"]["default"] == "best"
    assert "checkpoint" not in request.get("required", [])
