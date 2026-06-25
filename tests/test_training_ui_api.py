from __future__ import annotations

import json
import os
import re
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import BigInteger
from sqlalchemy import select

from mlsystem2.mlflow_adapter.contracts import MLflowBestCheckpoint, MLflowExperiment, MLflowTrainingProgress
from mlsystem2.models.contracts import ModelsError
from mlsystem2.training_ui_api import _automation, _model_export, _service, _worker
from mlsystem2.training_ui_api._routes import export as _export_routes
from mlsystem2.training_ui_api.api import create_app
from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import Base, configure_schema, create_session_factory
from mlsystem2.training_ui_api._models import JobRow, PseudoMarkupResultRow, StoredFileRow, TrainingResultRow
from mlsystem2.training_ui_api._service import create_training_job, ensure_seed_templates
from mlsystem2.training_ui_api._templates import sanitize_template_config
from mlsystem2.training_ui_api._worker import (
    dispatch_inference_queue_once,
    dispatch_queue_once,
    dispatch_training_queue_once,
)
from mlsystem2.training_ui_api.contracts import (
    AutomationEnabledUpdate,
    AutomationRuleUpdate,
    JobSource,
    JobStatus,
    JobType,
    ResultStatus,
    StoredFileKind,
    TrainingJobCreate,
    TrainingTemplateCreate,
    TrainingTemplateUpdate,
    TrainingUIAPIError,
)


def test_pseudo_report_success_requires_processed_scene() -> None:
    assert _worker._pseudo_report_allows_success({"status": "ok", "processed": 1}) is True
    assert _worker._pseudo_report_allows_success({"status": "partial", "processed": 1}) is True
    assert _worker._pseudo_report_allows_success({"status": "ok", "processed": 0}) is False
    assert _worker._pseudo_report_allows_success({"status": "error", "processed": 1}) is False


def test_pseudo_geojson_download_name_normalizes_legacy_slashes() -> None:
    row = StoredFileRow(
        kind=StoredFileKind.PSEUDO_MARKUP_GEOJSON.value,
        original_name="Засоления\\main_segformer b2_07_38_06_06.geojson",
        path="/tmp/file.geojson",
        size_bytes=1,
    )

    assert _service.stored_file_download_name(row) == "Засоления_main_segformer b2_07_38_06_06.geojson"


def test_training_ui_queue_snapshot_returns_unified_priority_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    created_at = datetime(2026, 6, 10, tzinfo=timezone.utc)

    with session_factory() as session:
        manual_training = _queue_test_job(JobType.TRAINING, JobSource.MANUAL, 1, created_at)
        session.add_all(
            [
                _queue_test_job(JobType.TRAINING, JobSource.AUTOMATION, 1, created_at),
                _queue_test_job(JobType.INFERENCE, JobSource.AUTOMATION, 1, created_at),
                manual_training,
                _queue_test_job(JobType.INFERENCE, JobSource.MANUAL, 1, created_at),
            ]
        )
        session.flush()

        snapshot = _service.queues(session)
        _service.move_job(session, manual_training.id, direction=-1)
        moved_snapshot = _service.queues(session)

    assert [(item.type, item.source) for item in snapshot.jobs] == [
        (JobType.INFERENCE, JobSource.MANUAL),
        (JobType.TRAINING, JobSource.MANUAL),
        (JobType.INFERENCE, JobSource.AUTOMATION),
        (JobType.TRAINING, JobSource.AUTOMATION),
    ]
    assert [item.type for item in snapshot.training_jobs] == [JobType.TRAINING, JobType.TRAINING]
    assert [item.type for item in snapshot.inference_jobs] == [JobType.INFERENCE, JobType.INFERENCE]
    assert [(item.type, item.source) for item in moved_snapshot.jobs[:2]] == [
        (JobType.TRAINING, JobSource.MANUAL),
        (JobType.INFERENCE, JobSource.MANUAL),
    ]


def test_training_ui_running_training_progress_is_returned_for_queue_and_class_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    created_at = datetime(2026, 6, 10, 17, 0, tzinfo=timezone.utc)
    started_at = datetime(2026, 6, 10, 17, 24, tzinfo=timezone.utc)
    now = datetime(2026, 6, 10, 17, 40, tzinfo=timezone.utc)

    def fake_epoch_progress(tracking_uri: str, run_id: str) -> MLflowTrainingProgress:
        assert tracking_uri == config.mlflow_tracking_uri
        assert run_id == "run-123"
        return MLflowTrainingProgress(completed_epochs=23)

    monkeypatch.setattr(_service, "_now", lambda: now)
    monkeypatch.setattr(_service, "get_training_epoch_progress", fake_epoch_progress)

    with session_factory() as session:
        job = _queue_test_job(JobType.TRAINING, JobSource.MANUAL, 1, created_at)
        job.status = JobStatus.RUNNING.value
        job.started_at = started_at
        job.dataset_key = "class-key"
        job.config = {"train.epochs": 64}
        job.tmp_path = str(tmp_path / "run")
        Path(job.tmp_path).mkdir(parents=True)
        (Path(job.tmp_path) / "mlflow_run_id").write_text("run-123\n", encoding="utf-8")
        session.add(job)
        session.flush()
        session.add(
            TrainingResultRow(
                source=JobSource.MANUAL.value,
                dataset_key="class-key",
                class_key="class-key",
                class_display_name="class",
                architecture="smp_segformer_b2",
                model_name="segformer b2",
                status=ResultStatus.RUNNING.value,
                job_id=job.id,
            )
        )
        session.flush()

        queue_progress = _service.queues(session).jobs[0].progress
        class_progress = _service.class_results(session, "class-key", config).results[0].progress

    assert queue_progress is not None
    assert queue_progress.current == 23
    assert queue_progress.total == 64
    assert queue_progress.elapsed_minutes == 16
    assert class_progress is not None
    assert class_progress.current == 23
    assert class_progress.elapsed_minutes == 16


def test_training_ui_running_training_progress_falls_back_without_mlflow_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    def failing_epoch_progress(tracking_uri: str, run_id: str) -> MLflowTrainingProgress:
        del tracking_uri, run_id
        raise RuntimeError("mlflow unavailable")

    monkeypatch.setattr(_service, "get_training_epoch_progress", failing_epoch_progress)

    with session_factory() as session:
        job = _queue_test_job(JobType.TRAINING, JobSource.MANUAL, 1, datetime(2026, 6, 10, tzinfo=timezone.utc))
        job.status = JobStatus.RUNNING.value
        job.started_at = datetime(2026, 6, 10, tzinfo=timezone.utc)
        job.config = {"train.epochs": 8}
        job.tmp_path = str(tmp_path / "run")
        Path(job.tmp_path).mkdir(parents=True)
        (Path(job.tmp_path) / "mlflow_run_id").write_text("run-err\n", encoding="utf-8")
        session.add(job)
        session.flush()

        progress = _service.queues(session).jobs[0].progress

    assert progress is not None
    assert progress.current == 0
    assert progress.total == 8


def test_training_ui_pseudo_progress_is_returned_for_queue_and_class_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    created_at = datetime(2026, 6, 10, tzinfo=timezone.utc)

    with session_factory() as session:
        training_result = TrainingResultRow(
            source=JobSource.MANUAL.value,
            dataset_key="class-key",
            class_key="class-key",
            class_display_name="class",
            architecture="smp_segformer_b2",
            model_name="segformer b2",
            status=ResultStatus.OK.value,
        )
        session.add(training_result)
        session.flush()

        job = _queue_test_job(JobType.INFERENCE, JobSource.MANUAL, 1, created_at)
        job.status = JobStatus.RUNNING.value
        job.started_at = created_at
        job.dataset_key = "class-key"
        job.tmp_path = str(tmp_path / "pseudo-run")
        progress_dir = Path(job.tmp_path) / "scratch"
        progress_dir.mkdir(parents=True)
        (progress_dir / "progress.json").write_text(
            json.dumps({"current": 13, "total": 64, "elapsed_sec": 17.5}),
            encoding="utf-8",
        )
        session.add(job)
        session.flush()
        session.add(
            PseudoMarkupResultRow(
                source=JobSource.MANUAL.value,
                dataset_key="class-key",
                training_result_id=training_result.id,
                class_key="class-key",
                source_dataset_name="source",
                status=ResultStatus.RUNNING.value,
                job_id=job.id,
            )
        )
        session.flush()

        queue_progress = _service.queues(session).jobs[0].progress
        pseudo_progress = (
            _service.class_results(session, "class-key", config)
            .results[0]
            .pseudo_markup_results[0]
            .progress
        )

    assert queue_progress is not None
    assert queue_progress.current == 13
    assert queue_progress.total == 64
    assert pseudo_progress is not None
    assert pseudo_progress.current == 13
    assert pseudo_progress.total == 64


def test_class_results_exposes_queued_job_statuses(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    created_at = datetime(2026, 6, 10, tzinfo=timezone.utc)

    with session_factory() as session:
        training_job = _queue_test_job(JobType.TRAINING, JobSource.MANUAL, 1, created_at)
        training_job.status = JobStatus.QUEUED.value
        training_job.dataset_key = "class-key"
        session.add(training_job)
        session.flush()
        queued_training = TrainingResultRow(
            source=JobSource.MANUAL.value,
            dataset_key="class-key",
            class_key="class-key",
            class_display_name="class",
            architecture="smp_segformer_b2",
            model_name="segformer b2",
            status=ResultStatus.RUNNING.value,
            job_id=training_job.id,
        )
        completed_training = TrainingResultRow(
            source=JobSource.MANUAL.value,
            dataset_key="class-key",
            class_key="class-key",
            class_display_name="class",
            architecture="smp_segformer_b2",
            model_name="segformer b2",
            status=ResultStatus.OK.value,
        )
        session.add_all([queued_training, completed_training])
        session.flush()

        pseudo_job = _queue_test_job(JobType.INFERENCE, JobSource.MANUAL, 2, created_at)
        pseudo_job.status = JobStatus.QUEUED.value
        pseudo_job.dataset_key = "class-key"
        pseudo_job.config = {"class_key": "class-key"}
        session.add(pseudo_job)
        session.flush()
        queued_pseudo = PseudoMarkupResultRow(
            source=JobSource.MANUAL.value,
            dataset_key="class-key",
            training_result_id=completed_training.id,
            class_key="class-key",
            source_dataset_name="source",
            status=ResultStatus.RUNNING.value,
            job_id=pseudo_job.id,
        )
        session.add(queued_pseudo)
        session.flush()

        response = _service.class_results(session, "class-key", config)

    training_result = next(item for item in response.results if item.id == queued_training.id)
    completed_result = next(item for item in response.results if item.id == completed_training.id)
    pseudo_result = completed_result.pseudo_markup_results[0]
    assert training_result.status == "queued"
    assert training_result.job_id == training_job.id
    assert pseudo_result.status == "queued"
    assert pseudo_result.job_id == pseudo_job.id


def test_training_ui_job_log_api_reads_failed_start_worker_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_USER", "mluser")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_PASSWORD", "secret")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        job = _queue_test_job(JobType.TRAINING, JobSource.MANUAL, 1, datetime(2026, 6, 10, tzinfo=timezone.utc))
        job.status = JobStatus.FAILED.value
        session.add(job)
        session.flush()
        run_dir = config.scratch_root / "jobs" / str(job.id)
        run_dir.mkdir(parents=True)
        (run_dir / "worker_error.txt").write_text(
            "Не удалось запустить обучение.\n\nRuntimeError: Датасет не найден или неполный",
            encoding="utf-8",
        )
        job_id = job.id
        session.commit()

    with TestClient(create_app()) as client:
        assert client.get(f"/api/v1/jobs/{job_id}/log").status_code == 401
        login = client.post("/api/v1/auth/login", json={"username": "mluser", "password": "secret"})
        assert login.status_code == 200
        response = client.get(f"/api/v1/jobs/{job_id}/log")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_name"] == "worker_error.txt"
    assert payload["truncated"] is False
    assert "Датасет не найден или неполный" in payload["content"]


def test_training_ui_worker_dispatches_inference_before_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    started: list[str] = []

    def fake_start_training(session, row, config, *, popen_factory) -> None:
        del session, config, popen_factory
        started.append(row.type)
        row.status = JobStatus.RUNNING.value

    def fake_start_inference(session, row, config, *, popen_factory) -> None:
        del session, config, popen_factory
        started.append(row.type)
        row.status = JobStatus.RUNNING.value

    monkeypatch.setattr(_worker, "_start_training_job", fake_start_training)
    monkeypatch.setattr(_worker, "_start_inference_job", fake_start_inference)
    created_at = datetime(2026, 6, 10, tzinfo=timezone.utc)

    with session_factory() as session:
        training = _queue_test_job(JobType.TRAINING, JobSource.MANUAL, 1, created_at)
        inference = _queue_test_job(JobType.INFERENCE, JobSource.MANUAL, 1, created_at)
        session.add_all([training, inference])
        session.flush()

        dispatch_queue_once(session, config)
        session.flush()

        assert started == [JobType.INFERENCE.value]
        assert session.get(JobRow, inference.id).status == JobStatus.RUNNING.value
        assert session.get(JobRow, training.id).status == JobStatus.QUEUED.value


def _queue_test_job(
    job_type: JobType,
    source: JobSource,
    position: int,
    created_at: datetime,
) -> JobRow:
    dataset_name = f"{job_type.value}-{source.value}"
    return JobRow(
        type=job_type.value,
        source=source.value,
        status=JobStatus.QUEUED.value,
        queue_position=position,
        dataset_name=dataset_name,
        training_dataset_name=dataset_name if job_type == JobType.TRAINING else "training-source",
        inference_dataset_name=dataset_name if job_type == JobType.INFERENCE else None,
        model_name="segformer b2" if job_type == JobType.TRAINING else "pseudo-markup",
        architecture="smp_segformer_b2" if job_type == JobType.TRAINING else "pseudo-markup",
        config={},
        created_at=created_at,
    )


def test_stored_file_size_uses_bigint() -> None:
    from mlsystem2.training_ui_api._models import StoredFileRow

    assert isinstance(StoredFileRow.__table__.columns["size_bytes"].type, BigInteger)


def test_model_export_zip_layout_config_and_pipeline(tmp_path: Path, monkeypatch) -> None:
    def fake_load_binary_checkpoint(path: Path) -> SimpleNamespace:
        assert path.name == "checkpoint.pt"
        return SimpleNamespace(
            model=SimpleNamespace(
                spec=SimpleNamespace(input_channels=4, output_channels=1),
                model=object(),
            ),
            artifact=SimpleNamespace(metadata={"val_best_threshold": 0.73, "sample_size": 768}),
        )

    def fake_export_onnx(**kwargs: object) -> None:
        assert kwargs["input_channels"] == 4
        assert kwargs["sample_size"] == 768
        assert kwargs["threshold"] == 0.73
        onnx_path = kwargs["onnx_path"]
        assert isinstance(onnx_path, Path)
        onnx_path.write_bytes(b"onnx")
        (onnx_path.parent / "model.onnx.data").write_bytes(b"weights")

    monkeypatch.setattr(_model_export, "_load_binary_checkpoint", fake_load_binary_checkpoint)
    monkeypatch.setattr(_model_export, "_export_binary_mask_onnx", fake_export_onnx)

    archive = _model_export.build_triton_model_export_zip(
        model_name="deforestation-b2",
        checkpoint_filename="best.pt",
        checkpoint_bytes=b"checkpoint",
        sample_size=None,
    )
    try:
        extract_dir = tmp_path / "extract"
        with zipfile.ZipFile(archive.zip_path) as zip_file:
            names = set(zip_file.namelist())
            zip_file.extractall(extract_dir)
        assert "export_metadata.json" in names
        assert "pipelines/deforestation-b2_triton.yaml" in names
        assert "models-serving-service/deforestation-b2.zip" in names
        assert "deforestation-b2/config.pbtxt" not in names
        assert "deforestation-b2/export_metadata.json" not in names
        service_zip_path = extract_dir / "models-serving-service" / "deforestation-b2.zip"
        service_extract_dir = tmp_path / "service"
        with zipfile.ZipFile(service_zip_path) as service_zip:
            service_names = set(service_zip.namelist())
            service_zip.extractall(service_extract_dir)
        assert service_names == {
            "deforestation-b2/config.pbtxt",
            "deforestation-b2/1/model.onnx",
            "deforestation-b2/1/model.onnx.data",
        }
        assert (service_extract_dir / "deforestation-b2").exists()
        config = (service_extract_dir / "deforestation-b2" / "config.pbtxt").read_text(encoding="utf-8")
        assert 'name: "deforestation-b2"' in config
        assert "dims: [ 1, 4, -1, -1 ]" in config
        assert "dims: [ -1, 1, -1, -1 ]" in config
        assert "dims: [ 1, 1, -1, -1 ]" not in config
        assert "KIND_CPU" in config
        assert "KIND_GPU" not in config
        pipeline = (extract_dir / "pipelines" / "deforestation-b2_triton.yaml").read_text(encoding="utf-8")
        assert 'name: "deforestation-b2"' in pipeline
        assert "sample_size:\n        - 768\n        - 768" in pipeline
        metadata = json.loads((extract_dir / "export_metadata.json").read_text(encoding="utf-8"))
        assert metadata["threshold"] == 0.73
        assert metadata["threshold_source"] == "checkpoint_metadata"
        assert metadata["sample_size"] == 768
        assert metadata["sample_size_source"] == "checkpoint_metadata"
        assert metadata["model_archive"] == "models-serving-service/deforestation-b2.zip"
        assert metadata["pipeline"] == "pipelines/deforestation-b2_triton.yaml"
        assert metadata["onnx_opset"] == 17
        assert metadata["onnx_ir_version"] == 8
    finally:
        archive.cleanup()


def test_model_export_normalizes_onnx_ir_for_old_triton(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    onnx_path = _minimal_onnx_path(tmp_path, opset=17, ir_version=10)

    _model_export._normalize_onnx_for_triton(onnx_path)

    model = onnx.load_model(onnx_path)
    assert model.ir_version == 8
    assert [item.version for item in model.opset_import if item.domain in ("", "ai.onnx")] == [17]
    onnx.checker.check_model(str(onnx_path))


def test_model_export_rejects_too_new_onnx_opset(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    onnx_path = _minimal_onnx_path(tmp_path, opset=18, ir_version=10)

    with pytest.raises(TrainingUIAPIError, match="opset"):
        _model_export._normalize_onnx_for_triton(onnx_path)


def _minimal_onnx_path(tmp_path: Path, *, opset: int, ir_version: int) -> Path:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "minimal",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", opset)],
    )
    model.ir_version = ir_version
    onnx_path = tmp_path / "model.onnx"
    onnx.save_model(model, onnx_path)
    return onnx_path


def test_model_export_manual_sample_size_is_used_for_old_checkpoint(monkeypatch) -> None:
    captured: list[tuple[float, int]] = []

    def fake_load_binary_checkpoint(path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            model=SimpleNamespace(
                spec=SimpleNamespace(input_channels=4, output_channels=1),
                model=object(),
            ),
            artifact=SimpleNamespace(metadata={"val_best_threshold": 0.73}),
        )

    def fake_export_onnx(**kwargs: object) -> None:
        captured.append((kwargs["threshold"], kwargs["sample_size"]))
        onnx_path = kwargs["onnx_path"]
        assert isinstance(onnx_path, Path)
        onnx_path.write_bytes(b"onnx")

    monkeypatch.setattr(_model_export, "_load_binary_checkpoint", fake_load_binary_checkpoint)
    monkeypatch.setattr(_model_export, "_export_binary_mask_onnx", fake_export_onnx)

    archive = _model_export.build_triton_model_export_zip(
        model_name="erosion-b2",
        checkpoint_filename="best.pt",
        checkpoint_bytes=b"checkpoint",
        sample_size=768,
    )
    try:
        assert captured == [(0.73, 768)]
        with zipfile.ZipFile(archive.zip_path) as zip_file:
            metadata = json.loads(zip_file.read("export_metadata.json").decode("utf-8"))
        assert metadata["threshold"] == 0.73
        assert metadata["threshold_source"] == "checkpoint_metadata"
        assert metadata["sample_size"] == 768
        assert metadata["sample_size_source"] == "request"
    finally:
        archive.cleanup()


def test_model_export_accepts_underscore_model_name(monkeypatch) -> None:
    def fake_load_binary_checkpoint(path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            model=SimpleNamespace(
                spec=SimpleNamespace(input_channels=4, output_channels=1),
                model=object(),
            ),
            artifact=SimpleNamespace(metadata={"val_best_threshold": 0.73, "sample_size": 768}),
        )

    def fake_export_onnx(**kwargs: object) -> None:
        onnx_path = kwargs["onnx_path"]
        assert isinstance(onnx_path, Path)
        onnx_path.write_bytes(b"onnx")

    monkeypatch.setattr(_model_export, "_load_binary_checkpoint", fake_load_binary_checkpoint)
    monkeypatch.setattr(_model_export, "_export_binary_mask_onnx", fake_export_onnx)

    archive = _model_export.build_triton_model_export_zip(
        model_name="rivers_kanopus",
        checkpoint_filename="best.pt",
        checkpoint_bytes=b"checkpoint",
        sample_size=None,
    )
    try:
        assert archive.filename == "rivers_kanopus_export.zip"
    finally:
        archive.cleanup()


def test_model_export_requires_threshold_metadata(monkeypatch) -> None:
    def fake_load_binary_checkpoint(path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            model=SimpleNamespace(
                spec=SimpleNamespace(input_channels=4, output_channels=1),
                model=object(),
            ),
            artifact=SimpleNamespace(metadata={"sample_size": 768}),
        )

    monkeypatch.setattr(_model_export, "_load_binary_checkpoint", fake_load_binary_checkpoint)

    with pytest.raises(TrainingUIAPIError, match="metadata.val_best_threshold"):
        _model_export.build_triton_model_export_zip(
            model_name="erosion-b2",
            checkpoint_filename="best.pt",
            checkpoint_bytes=b"checkpoint",
            sample_size=None,
        )


def test_model_export_requires_sample_size_metadata_or_request(monkeypatch) -> None:
    def fake_load_binary_checkpoint(path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            model=SimpleNamespace(
                spec=SimpleNamespace(input_channels=4, output_channels=1),
                model=object(),
            ),
            artifact=SimpleNamespace(metadata={"val_best_threshold": 0.73}),
        )

    monkeypatch.setattr(_model_export, "_load_binary_checkpoint", fake_load_binary_checkpoint)

    with pytest.raises(TrainingUIAPIError, match="metadata.sample_size"):
        _model_export.build_triton_model_export_zip(
            model_name="erosion-b2",
            checkpoint_filename="best.pt",
            checkpoint_bytes=b"checkpoint",
            sample_size=None,
        )


def test_model_export_rejects_invalid_model_name() -> None:
    with pytest.raises(TrainingUIAPIError, match="Имя модели"):
        _model_export.build_triton_model_export_zip(
            model_name="Вырубки_b2",
            checkpoint_filename="best.pt",
            checkpoint_bytes=b"checkpoint",
            sample_size=768,
        )


def test_model_export_checkpoint_without_model_spec_is_reported(monkeypatch) -> None:
    from mlsystem2.models import api as models_api

    def fake_load_checkpoint(request) -> None:
        raise ModelsError("Checkpoint не содержит model_spec, а request.model_spec не задан.")

    monkeypatch.setattr(models_api, "load_checkpoint", fake_load_checkpoint)

    with pytest.raises(TrainingUIAPIError, match="Checkpoint не содержит model_spec"):
        _model_export._load_binary_checkpoint(Path("checkpoint.pt"))


def test_model_export_api_returns_zip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_USER", "mluser")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_PASSWORD", "secret")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    def fake_build_zip(**kwargs: object) -> SimpleNamespace:
        assert kwargs["model_name"] == "deforestation-b2"
        assert kwargs["checkpoint_filename"] == "best.pt"
        assert kwargs["checkpoint_bytes"] == b"checkpoint"
        assert kwargs["sample_size"] is None
        zip_path = tmp_path / "deforestation-b2.zip"
        with zipfile.ZipFile(zip_path, "w") as zip_file:
            zip_file.writestr("deforestation-b2/config.pbtxt", "name")
        return SimpleNamespace(
            zip_path=zip_path,
            filename="deforestation-b2_export.zip",
            cleanup=lambda: None,
        )

    monkeypatch.setattr(_export_routes, "build_triton_model_export_zip", fake_build_zip)

    with TestClient(create_app()) as client:
        assert client.post("/api/v1/model-export/triton-zip").status_code == 401
        login = client.post("/api/v1/auth/login", json={"username": "mluser", "password": "secret"})
        assert login.status_code == 200
        response = client.post(
            "/api/v1/model-export/triton-zip",
            data={"model_name": "deforestation-b2"},
            files={"checkpoint": ("best.pt", b"checkpoint", "application/octet-stream")},
        )
    assert response.status_code == 200
    assert response.headers["content-disposition"].endswith('filename="deforestation-b2_export.zip"')
    assert response.content.startswith(b"PK")


def test_training_result_model_export_api_downloads_best_checkpoint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLFLOW_TRACKING_URI", "http://mlflow.local")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_USER", "mluser")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_PASSWORD", "secret")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    with session_factory() as session:
        result = TrainingResultRow(
            source=JobSource.MANUAL.value,
            dataset_key="Реки\\main",
            class_key="Реки\\main",
            class_display_name="Реки\\main",
            architecture="smp_segformer_b2",
            model_name="segformer b2",
            status=ResultStatus.OK.value,
            mlflow_run_id="run-zip",
        )
        session.add(result)
        session.commit()
        result_id = result.id

    def fake_download_run_artifact(**kwargs: object) -> SimpleNamespace:
        assert kwargs["tracking_uri"] == "http://mlflow.local"
        assert kwargs["run_id"] == "run-zip"
        assert kwargs["artifact_path"] == "checkpoints/best.pt"
        checkpoint_path = Path(kwargs["dst_dir"]) / "best.pt"
        checkpoint_path.write_bytes(b"checkpoint")
        return SimpleNamespace(local_path=str(checkpoint_path))

    def fake_build_zip(**kwargs: object) -> SimpleNamespace:
        assert kwargs["model_name"] == "rivers_kanopus"
        assert kwargs["checkpoint_filename"] == "best.pt"
        assert kwargs["checkpoint_bytes"] == b"checkpoint"
        assert kwargs["sample_size"] is None
        zip_path = tmp_path / "rivers_kanopus.zip"
        with zipfile.ZipFile(zip_path, "w") as zip_file:
            zip_file.writestr("export_metadata.json", "{}")
        return SimpleNamespace(
            zip_path=zip_path,
            filename="rivers_kanopus_export.zip",
            cleanup=lambda: None,
        )

    monkeypatch.setattr(_service, "download_run_artifact", fake_download_run_artifact)
    monkeypatch.setattr(_service, "build_triton_model_export_zip", fake_build_zip)

    with TestClient(create_app()) as client:
        assert client.post(f"/api/v1/results/training/{result_id}/triton-zip").status_code == 401
        login = client.post("/api/v1/auth/login", json={"username": "mluser", "password": "secret"})
        assert login.status_code == 200
        response = client.post(
            f"/api/v1/results/training/{result_id}/triton-zip",
            data={"model_name": "rivers_kanopus"},
        )
    assert response.status_code == 200
    assert response.headers["content-disposition"].endswith('filename="rivers_kanopus_export.zip"')
    assert response.content.startswith(b"PK")


def test_training_result_model_export_requires_ok_status_and_mlflow_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        running = TrainingResultRow(
            source=JobSource.MANUAL.value,
            dataset_key="Реки\\main",
            class_key="Реки\\main",
            class_display_name="Реки\\main",
            architecture="smp_segformer_b2",
            model_name="segformer b2",
            status=ResultStatus.RUNNING.value,
            mlflow_run_id="run-running",
        )
        without_run = TrainingResultRow(
            source=JobSource.MANUAL.value,
            dataset_key="Реки\\main",
            class_key="Реки\\main",
            class_display_name="Реки\\main",
            architecture="smp_segformer_b2",
            model_name="segformer b2",
            status=ResultStatus.OK.value,
        )
        session.add_all([running, without_run])
        session.flush()

        with pytest.raises(TrainingUIAPIError, match="только для успешного"):
            _service.export_training_result_triton_zip(
                session,
                result_id=running.id,
                model_name="rivers_kanopus",
                sample_size=None,
                config=config,
            )
        with pytest.raises(TrainingUIAPIError, match="нет MLflow run id"):
            _service.export_training_result_triton_zip(
                session,
                result_id=without_run.id,
                model_name="rivers_kanopus",
                sample_size=None,
                config=config,
            )


def test_training_ui_frontend_is_react_vite_app() -> None:
    package_json = json.loads(Path("frontend/package.json").read_text(encoding="utf-8"))
    app_tsx = Path("frontend/src/App.tsx").read_text(encoding="utf-8")
    api_client = Path("frontend/src/api/client.ts").read_text(encoding="utf-8")
    api_types = Path("frontend/src/api/types.ts").read_text(encoding="utf-8")
    assert "react" in package_json["dependencies"]
    assert "vite" in package_json["devDependencies"]
    assert "openapi-typescript" in package_json["devDependencies"]
    assert package_json["scripts"]["generate:api"].startswith("npm run generate:openapi")
    assert not Path("frontend/src/app.js").exists()
    assert not Path("frontend/src/assets/app.css").exists()
    assert 'head === "model-export"' in app_tsx
    assert "/bootstrap" in app_tsx
    assert "/model-export/triton-zip" in app_tsx
    assert "/results/training/" in app_tsx
    assert "/triton-zip" in app_tsx
    assert "metadata.sample_size" in app_tsx
    assert "showJobLog" in app_tsx
    assert "/log" in app_tsx
    assert "log-view" in app_tsx
    assert "hasActiveClassResults" in app_tsx
    assert "recommended_range" in app_tsx
    assert "downloadBlob(response.blob" in app_tsx
    assert 'pattern="[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?"' in app_tsx
    assert "components[\"schemas\"][\"BootstrapInfo\"]" in api_types
    assert "credentials: \"same-origin\"" in api_client


def test_frontend_build_runs_vite_wrapper(tmp_path: Path, monkeypatch) -> None:
    from frontend import build

    root = tmp_path / "frontend"
    dist = root / "dist"
    root.mkdir()
    (root / "package.json").write_text('{"scripts":{"build":"vite build"}}', encoding="utf-8")

    def fake_frontend_build() -> None:
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text(
            '<!doctype html><script type="module" src="/assets/index-test.js"></script>',
            encoding="utf-8",
        )

    monkeypatch.setattr(build, "ROOT", root)
    monkeypatch.setattr(build, "DIST", dist)
    monkeypatch.setattr(build, "run_frontend_build", fake_frontend_build)

    build.main()

    index = (dist / "index.html").read_text(encoding="utf-8")
    assert "/assets/index-test.js" in index
    assert "dashboard" not in index.lower()


def test_training_ui_api_contract_flow(tmp_path: Path, monkeypatch) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки" / "main"
    class_dir.mkdir(parents=True)
    (class_dir / "deforestation.txt").write_text("scene-1\n", encoding="utf-8")
    (class_dir / "deforestation.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    images_root = tmp_path / "prepared_images"
    image_folder = images_root / "kanopus" / "irkutsk"
    image_folder.mkdir(parents=True)
    (image_folder / "scene-1.tif").touch()
    (image_folder / "scene-2.tif").touch()
    frontend_dist = tmp_path / "frontend" / "dist"
    (frontend_dist / "assets").mkdir(parents=True)
    (frontend_dist / "index.html").write_text(
        '<!doctype html><title>MLSystem2</title><script type="module" src="/assets/index-test.js"></script>',
        encoding="utf-8",
    )
    (frontend_dist / "assets" / "index-test.js").write_text("console.log('MLSystem2')", encoding="utf-8")
    (frontend_dist / "assets" / "index-test.css").write_text("body{margin:0}", encoding="utf-8")

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "files"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_FRONTEND_DIST", str(frontend_dist))
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(mlmarkup_root))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(images_root))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_USER", "mluser")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_PASSWORD", "secret")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    with TestClient(create_app()) as client:
        assert client.get("/api/v1/health").json()["status"] == "ok"
        assert client.get("/").text.startswith("<!doctype html>")
        app_js = client.get("/assets/index-test.js")
        assert app_js.text == "console.log('MLSystem2')"
        assert app_js.headers["content-type"].split(";")[0] in {"text/javascript", "application/javascript"}
        assert client.get("/assets/index-test.css").text == "body{margin:0}"
        assert client.get("/not-a-real-frontend-route").text.startswith("<!doctype html>")
        unauthorized = client.get("/api/v1/datasets")
        assert unauthorized.status_code == 401
        assert client.get("/auth/proxy-check").status_code == 401

        login = client.post(
            "/api/v1/auth/login",
            json={"username": "mluser", "password": "secret"},
        )
        assert login.status_code == 200
        proxy_check = client.get("/auth/proxy-check")
        assert proxy_check.status_code == 204
        assert proxy_check.headers["x-remote-user"] == "mluser"

        bootstrap = client.get("/api/v1/bootstrap").json()
        assert set(bootstrap) == {
            "links",
            "datasets",
            "image_folders",
            "classes",
            "models",
            "training_templates",
            "inference_templates",
        }
        assert [item["name"] for item in bootstrap["datasets"]] == ["Вырубки\\main", "Custom"]
        assert bootstrap["image_folders"][0]["key"] == "kanopus/irkutsk"
        assert len(bootstrap["training_templates"]) == 7

        datasets = client.get("/api/v1/datasets").json()["datasets"]
        assert [item["name"] for item in datasets] == ["Вырубки\\main", "Custom"]
        assert datasets[0]["image_count"] == 1
        assert datasets[0]["hard_negative_annotation_file"] is None
        image_folders = client.get("/api/v1/image-folders").json()["folders"]
        assert image_folders == [
            {
                "key": "kanopus/irkutsk",
                "name": "kanopus/irkutsk",
                "path": str(image_folder),
                "image_count": 2,
            }
        ]
        second_image_folder = images_root / "kanopus" / "toguchinsk"
        second_image_folder.mkdir(parents=True)
        (second_image_folder / "scene-3.tif").touch()

        new_dir = mlmarkup_root / "Пожары" / "main"
        new_dir.mkdir(parents=True)
        (new_dir / "fires.txt").write_text("scene-2\n", encoding="utf-8")
        (new_dir / "fires.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
        refreshed = client.get("/api/v1/datasets").json()["datasets"]
        assert [item["name"] for item in refreshed] == ["Вырубки\\main", "Пожары\\main", "Custom"]

        models = client.get("/api/v1/models").json()["models"]
        assert [item["display_name"] for item in models] == [
            "deeplabV3+",
            "segformer b2",
            "segformer b3",
            "unet + resnet34",
            "unet + resnet50",
            "unet + resnet101",
            "unet + resnet152",
        ]
        automation = client.get("/api/v1/automation").json()
        assert automation["enabled"] is False
        assert [item["name"] for item in automation["datasets"]] == [
            "Вырубки\\main",
            "Пожары\\main",
        ]
        assert len(automation["rules"]) == len(automation["datasets"]) * len(models)
        enabled_automation = client.put("/api/v1/automation/enabled", json={"enabled": True}).json()
        assert enabled_automation["enabled"] is True
        rule = client.put(
            "/api/v1/automation/rules",
            json={
                "dataset_key": "Вырубки\\main",
                "architecture": "smp_segformer_b2",
                "training_enabled": True,
                "pseudo_markup_enabled": True,
            },
        ).json()
        assert rule["training_enabled"] is True
        assert rule["pseudo_markup_enabled"] is True

        templates = client.get("/api/v1/training-templates").json()["templates"]
        assert len(templates) == 7
        segformer_template = client.get("/api/v1/training-templates/smp_segformer_b2").json()
        assert segformer_template["source"] == "hpo_best"
        template_keys = {item["key"] for item in segformer_template["config_schema"]["fields"]}
        assert "dataset.split_granularity" not in template_keys
        assert "tile_preparation.num_workers" not in template_keys
        assert "train.device" not in template_keys
        assert "train.max_train_batches_per_epoch" in template_keys
        assert "train.max_val_batches_per_epoch" in template_keys
        assert "train.max_training_time_sec" in template_keys
        assert "train.hard_negative_weight" in template_keys
        assert "tile_preparation.positive_factor" in template_keys
        assert "tile_preparation.hard_negative_factor" in template_keys
        assert "tile_preparation.background_factor" in template_keys
        assert segformer_template["default_config"]["tile_preparation.positive_factor"] == 0.8
        assert segformer_template["default_config"]["tile_preparation.hard_negative_factor"] == 0.0
        assert segformer_template["default_config"]["tile_preparation.background_factor"] == 0.2
        assert segformer_template["default_config"]["train.hard_negative_weight"] == 1.0
        assert segformer_template["default_config"]["train.max_train_batches_per_epoch"] == 72
        assert segformer_template["default_config"]["train.max_val_batches_per_epoch"] == 1000
        assert segformer_template["default_config"]["train.max_training_time_sec"] == 1800
        loss_field = next(
            item for item in segformer_template["config_schema"]["fields"] if item["key"] == "train.loss"
        )
        assert loss_field["options"] == ["bce_dice", "focal_dice", "focal_tversky"]
        hard_weight_field = next(
            item for item in segformer_template["config_schema"]["fields"] if item["key"] == "train.hard_negative_weight"
        )
        assert "размеченных hard-negative зон" in hard_weight_field["tooltip"]
        assert "Остальной background" in hard_weight_field["tooltip"]
        assert "1..5" in hard_weight_field["recommended_range"]
        changed_config = dict(segformer_template["default_config"])
        changed_config["train.batch_size"] = 3
        updated = client.put(
            "/api/v1/training-templates/smp_segformer_b2",
            json={"default_config": changed_config},
        ).json()
        assert updated["source"] == "manual"
        assert updated["version"] == segformer_template["version"] + 1
        reset = client.put(
            "/api/v1/training-templates/smp_segformer_b2",
            json={"reset_to_baseline": True},
        ).json()
        assert reset["source"] == "hpo_best"
        dataset_template = client.post(
            "/api/v1/training-templates",
            json={"architecture": "smp_segformer_b2", "dataset_key": "Вырубки\\main"},
        ).json()
        assert dataset_template["architecture"] == "smp_segformer_b2"
        assert dataset_template["dataset_key"] == "Вырубки\\main"
        assert dataset_template["parent_template_id"] == reset["id"]
        assert dataset_template["default_config"] == reset["default_config"]
        updated_dataset_template = client.put(
            f"/api/v1/training-templates/by-id/{dataset_template['id']}",
            json={"default_config": {**dataset_template["default_config"], "train.batch_size": 5}},
        ).json()
        assert updated_dataset_template["source"] == "manual"
        assert updated_dataset_template["default_config"]["train.batch_size"] == 5
        applied = client.put(
            f"/api/v1/training-templates/by-id/{dataset_template['id']}/apply-field-to-all",
            json={"key": "train.batch_size", "value": 9},
        ).json()["templates"]
        assert {item["default_config"]["train.batch_size"] for item in applied} == {9}

        inference_templates = client.get("/api/v1/inference-templates").json()["templates"]
        assert len(inference_templates) == 8
        inference_template = client.get("/api/v1/inference-templates/smp_segformer_b2").json()
        inference_keys = {item["key"] for item in inference_template["config_schema"]["fields"]}
        assert "postprocess.min_area_m2" in inference_keys
        assert "postprocess.filter_compact_objects.enabled" in inference_keys
        assert "train.batch_size" not in inference_keys
        river_inference_template = next(
            item for item in inference_templates if item.get("dataset_key") == "Реки\\main"
        )
        assert river_inference_template["default_config"]["postprocess.min_area_m2"] == 10000.0
        assert river_inference_template["default_config"]["postprocess.min_hole_area_m2"] == 5000.0
        assert river_inference_template["default_config"]["postprocess.simplify_m"] == 15.0
        assert river_inference_template["default_config"]["postprocess.filter_compact_objects.enabled"] is True
        inference_dataset_template = client.post(
            "/api/v1/inference-templates",
            json={"architecture": "smp_segformer_b2", "dataset_key": "Вырубки\\main"},
        ).json()
        assert inference_dataset_template["parent_template_id"] == inference_template["id"]
        updated_inference_dataset_template = client.put(
            f"/api/v1/inference-templates/by-id/{inference_dataset_template['id']}",
            json={
                "default_config": {
                    **inference_dataset_template["default_config"],
                    "postprocess.min_area_m2": 2222.0,
                }
            },
        ).json()
        assert updated_inference_dataset_template["source"] == "manual"
        assert updated_inference_dataset_template["default_config"]["postprocess.min_area_m2"] == 2222.0

        custom = client.post(
            "/api/v1/custom-datasets",
            data={"name": "Custom"},
            files={
                "scenes_txt": ("custom.txt", b"scene-custom\n", "text/plain"),
                "annotation_geojson": (
                    "custom.geojson",
                    b'{"type":"FeatureCollection","features":[]}',
                    "application/geo+json",
                ),
            },
        ).json()
        assert custom["name"] == "Custom"
        scenes_download = client.get(custom["scenes_file"]["download_url"])
        assert scenes_download.status_code == 200
        assert scenes_download.text == "scene-custom\n"

        job = client.post(
            "/api/v1/training-jobs",
            json={
                "mlflow_experiment_id": "45",
                "mlflow_experiment_name": "Segformer-b2-HPO-deforest-2605",
                "dataset_key": "custom",
                "custom_dataset_id": custom["id"],
                "architecture": "smp_segformer_b2",
                "config": reset["default_config"],
            },
        ).json()
        assert job["status"] == "queued"
        assert job["dataset_name"] == "Custom"
        assert job["mlflow_run_name"] is None
        assert "train.device" not in job["config"]
        assert "dataset.split_granularity" not in job["config"]

        queues = client.get("/api/v1/queues").json()
        assert queues["training_enabled"] is True
        assert len(queues["training_jobs"]) == 1

        detail = client.get(f"/api/v1/jobs/{job['id']}").json()
        assert detail["readonly"] is True

        custom_results = client.get("/api/v1/results/classes/custom").json()
        training_result_id = custom_results["results"][0]["id"]
        pseudo = client.post(
            "/api/v1/results/classes/custom/pseudo-markup",
            data={"dataset_key": "Вырубки\\main", "training_result_id": training_result_id},
        ).json()
        assert pseudo["type"] == "inference"
        assert pseudo["config"]["inference_template_id"] == updated_inference_dataset_template["id"]
        assert pseudo["config"]["inference_template_config"]["postprocess.min_area_m2"] == 2222.0
        conflict = client.post(
            "/api/v1/results/classes/custom/pseudo-markup",
            data={
                "dataset_key": "Вырубки\\main",
                "image_folder_key": "kanopus/irkutsk",
                "training_result_id": training_result_id,
            },
        )
        assert conflict.status_code == 400
        assert conflict.json()["detail"] == "Выберите только один источник снимков"
        folder_pseudo = client.post(
            "/api/v1/results/classes/custom/pseudo-markup",
            data={"image_folder_key": "kanopus/irkutsk", "training_result_id": training_result_id},
        ).json()
        assert folder_pseudo["type"] == "inference"
        assert folder_pseudo["config"]["image_folder_key"] == "kanopus/irkutsk"
        second_folder_pseudo = client.post(
            "/api/v1/results/classes/custom/pseudo-markup",
            data={"image_folder_key": "kanopus/toguchinsk", "training_result_id": training_result_id},
        ).json()
        assert second_folder_pseudo["type"] == "inference"
        assert second_folder_pseudo["config"]["image_folder_key"] == "kanopus/toguchinsk"
        uploaded_txt_pseudo = client.post(
            "/api/v1/results/classes/custom/pseudo-markup",
            data={"training_result_id": training_result_id},
            files={"scenes_txt": ("manual.txt", b"kanopus/irkutsk\n", "text/plain")},
        ).json()
        assert uploaded_txt_pseudo["type"] == "inference"
        inference_queue = client.get("/api/v1/queues").json()["inference_jobs"]
        assert len(inference_queue) == 4
        pseudo_with_empty_upload = client.post(
            "/api/v1/results/classes/custom/pseudo-markup",
            data={"dataset_key": "Вырубки\\main", "training_result_id": training_result_id},
            files={"scenes_txt": ("", b"", "application/octet-stream")},
        ).json()
        assert pseudo_with_empty_upload["type"] == "inference"
        class_results = client.get("/api/v1/results/classes/custom").json()
        pseudo_results = class_results["results"][0]["pseudo_markup_results"]
        pseudo_scenes = next(
            item["scenes_file"]
            for item in pseudo_results
            if item["source_dataset_name"] == "Вырубки\\main"
        )
        assert client.get(pseudo_scenes["download_url"]).text.splitlines() == ["scene-1"]
        folder_scenes = next(
            item["scenes_file"]
            for item in pseudo_results
            if item["source_dataset_name"] == "kanopus/irkutsk"
        )
        assert client.get(folder_scenes["download_url"]).text.splitlines() == ["kanopus/irkutsk"]
        folder_result = next(item for item in pseudo_results if item["source_dataset_name"] == "kanopus/irkutsk")
        assert folder_result["image_count"] == 2
        second_folder_scenes = next(
            item["scenes_file"]
            for item in pseudo_results
            if item["source_dataset_name"] == "kanopus/toguchinsk"
        )
        assert client.get(second_folder_scenes["download_url"]).text.splitlines() == ["kanopus/toguchinsk"]
        uploaded_txt_result = next(
            item
            for item in pseudo_results
            if item["source_dataset_name"] == "Custom" and item["scenes_file"]["original_name"] == "manual.txt"
        )
        assert uploaded_txt_result["image_count"] == 2
        deleted_folder_pseudo = client.delete(f"/api/v1/jobs/{second_folder_pseudo['id']}")
        assert deleted_folder_pseudo.status_code == 200
        queue_after_delete = client.get("/api/v1/queues").json()["inference_jobs"]
        assert second_folder_pseudo["id"] not in {item["id"] for item in queue_after_delete}
        deleted_uploaded_pseudo = client.delete(f"/api/v1/results/pseudo-markup/{uploaded_txt_result['id']}")
        assert deleted_uploaded_pseudo.status_code == 200
        assert deleted_uploaded_pseudo.json()["id"] == uploaded_txt_result["id"]
        queue_after_pseudo_delete = client.get("/api/v1/queues").json()["inference_jobs"]
        assert uploaded_txt_pseudo["id"] not in {item["id"] for item in queue_after_pseudo_delete}
        result_id = class_results["results"][0]["id"]
        session_factory = create_session_factory(get_config())
        with session_factory() as session:
            training_result = session.get(TrainingResultRow, UUID(result_id))
            assert training_result is not None
            training_result.status = ResultStatus.OK.value
            training_result.updated_at = training_result.created_at
            session.flush()
            session.commit()
        changes = client.get("/api/v1/results/changes").json()["changes"]
        assert changes[0]["item_type"] == "job"
        assert changes[0]["status"] in {"queued", "running"}
        completed_change = next(item for item in changes if item["action"] == "обучена сеть")
        assert completed_change["item_type"] == "training_result"
        assert completed_change["class_key"] == "custom"

        deleted = client.delete(f"/api/v1/jobs/{job['id']}").json()
        assert deleted["status"] == "cancelled"
        assert client.get(f"/api/v1/jobs/{job['id']}").status_code == 400
        assert client.get("/api/v1/queues").json()["training_jobs"] == []
        assert client.get("/api/v1/results/classes/custom").json()["results"] == []
        deleted_template = client.delete(f"/api/v1/training-templates/by-id/{dataset_template['id']}").json()
        assert deleted_template["dataset_key"] == "Вырубки\\main"
        deleted_inference_template = client.delete(
            f"/api/v1/inference-templates/by-id/{inference_dataset_template['id']}"
        ).json()
        assert deleted_inference_template["dataset_key"] == "Вырубки\\main"


def test_class_results_removes_cancelled_results_from_database(tmp_path: Path, monkeypatch) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки" / "main"
    class_dir.mkdir(parents=True)
    (class_dir / "scenes.txt").write_text("scene-1\n", encoding="utf-8")
    (class_dir / "annotation.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(mlmarkup_root))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        job = JobRow(
            type=JobType.TRAINING.value,
            source=JobSource.MANUAL.value,
            status=JobStatus.CANCELLED.value,
            queue_position=1,
            dataset_key="Вырубки\\main",
            dataset_name="Вырубки\\main",
            model_name="segformer b2",
            architecture="smp_segformer_b2",
            config={},
        )
        session.add(job)
        session.flush()
        result = TrainingResultRow(
            source=JobSource.MANUAL.value,
            dataset_key="Вырубки\\main",
            class_key="Вырубки\\main",
            class_display_name="Вырубки\\main",
            architecture="smp_segformer_b2",
            model_name="segformer b2",
            status=ResultStatus.CANCELLED.value,
            job_id=job.id,
        )
        session.add(result)
        session.flush()
        pseudo = PseudoMarkupResultRow(
            source=JobSource.MANUAL.value,
            dataset_key="Вырубки\\main",
            training_result_id=result.id,
            class_key="Вырубки\\main",
            source_dataset_name="Вырубки\\main",
            status=ResultStatus.CANCELLED.value,
            job_id=job.id,
        )
        session.add(pseudo)
        session.flush()

        response = _service.class_results(session, "Вырубки\\main", config)

        assert response.results == []
        assert session.get(JobRow, job.id) is None
        assert session.get(TrainingResultRow, result.id) is None
        assert session.get(PseudoMarkupResultRow, pseudo.id) is None


def test_training_ui_worker_starts_first_training_job(tmp_path: Path, monkeypatch) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки" / "main"
    class_dir.mkdir(parents=True)
    (class_dir / "scenes.txt").write_text("scene-1\n", encoding="utf-8")
    (class_dir / "annotation.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )
    (class_dir / "hard_negative.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(mlmarkup_root))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "files"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("MLSYSTEM2_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    started: list[tuple[list[str], dict[str, object]]] = []

    def fake_popen(args: list[str], **kwargs: object) -> SimpleNamespace:
        started.append((args, kwargs))
        return SimpleNamespace(pid=4321)

    with session_factory() as session:
        ensure_seed_templates(session)
        template = session.query(JobRow).first()
        assert template is None
        job = create_training_job(
            session,
            TrainingJobCreate(
                mlflow_experiment_id="1",
                mlflow_experiment_name="ui-test",
                mlflow_run_name="worker-test",
                dataset_key="Вырубки\\main",
                architecture="smp_segformer_b2",
                config={
                    "dataset.val_fraction": 0.2,
                    "tile_preparation.tile_size": 32,
                    "tile_preparation.stride": 32,
                    "tile_preparation.augmentation_level": 0,
                    "tile_preparation.positive_factor": 0.5,
                    "tile_preparation.hard_negative_factor": 0.3,
                    "tile_preparation.background_factor": 0.2,
                    "train.epochs": 1,
                    "train.batch_size": 1,
                    "train.learning_rate": 0.0001,
                    "train.weight_decay": 0.0,
                    "train.loss": "bce_dice",
                    "train.focal_alpha": 0.6,
                    "train.pos_weight": 1.0,
                    "train.hard_negative_weight": 2.0,
                    "train.tversky_alpha": 0.4,
                    "train.tversky_beta": 0.6,
                    "train.threshold": 0.5,
                    "train.early_stopping_patience": 1,
                    "train.max_train_batches_per_epoch": 72,
                    "train.max_val_batches_per_epoch": 1000,
                    "train.max_training_time_sec": None,
                },
            ),
            config,
        )
        row = session.get(JobRow, job.id)
        assert row is not None
        legacy_config = dict(row.config)
        legacy_config["tile_preparation.positive_factor"] = 0.8
        legacy_config["tile_preparation.hard_negative_factor"] = 0.3
        legacy_config["tile_preparation.background_factor"] = 0.2
        row.config = legacy_config
        dispatch_training_queue_once(session, config, popen_factory=fake_popen)
        session.commit()

        row = session.get(JobRow, job.id)
        assert row is not None
        assert row.status == JobStatus.RUNNING.value
        assert row.process_pid == 4321
        assert row.tmp_path is not None
        run_dir = Path(row.tmp_path)
        assert (run_dir / "run.yml").is_file()
        assert (run_dir / "run_training.sh").is_file()
        config_yaml = (run_dir / "run.yml").read_text(encoding="utf-8")
        run_script = (run_dir / "run_training.sh").read_text(encoding="utf-8")
        assert "split_granularity" not in config_yaml
        assert "num_workers" not in config_yaml
        assert "input_channels" not in config_yaml
        assert "images_dir" not in config_yaml
        assert "inference:" not in config_yaml
        assert "hard_negative_annotation_file:" in config_yaml
        assert "positive_factor: 0.5" in config_yaml
        assert "hard_negative_factor: 0.3" in config_yaml
        assert "background_factor: 0.2" in config_yaml
        assert "hard_negative_weight: 2.0" in config_yaml
        assert "max_train_batches_per_epoch: 72" in config_yaml
        assert "max_val_batches_per_epoch: 1000" in config_yaml
        assert "max_training_time_sec: null" in config_yaml
        assert "--settings" in run_script
        assert "--run" in run_script
        assert "run.yml" in run_script
        assert "MLSYSTEM2_MLFLOW_RUN_ID_FILE" in run_script

    assert started
    assert started[0][0][0] == "bash"
    assert started[0][1]["cwd"] == str(tmp_path)
    assert started[0][1]["start_new_session"] is True


def test_training_ui_accepts_hard_negative_factor_without_dataset_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки" / "main"
    class_dir.mkdir(parents=True)
    (class_dir / "scenes.txt").write_text("scene-1\n", encoding="utf-8")
    (class_dir / "annotation.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(mlmarkup_root))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    job_config = _short_training_config()
    job_config["tile_preparation.positive_factor"] = 0.5
    job_config["tile_preparation.hard_negative_factor"] = 0.2
    job_config["tile_preparation.background_factor"] = 0.3

    with session_factory() as session:
        ensure_seed_templates(session)
        job = create_training_job(
            session,
            TrainingJobCreate(
                mlflow_experiment_id="1",
                mlflow_experiment_name="ui-test",
                mlflow_run_name="worker-test",
                dataset_key="Вырубки\\main",
                architecture="smp_segformer_b2",
                config=job_config,
            ),
            config,
        )

    assert job.status == JobStatus.QUEUED
    assert job.config["tile_preparation.hard_negative_factor"] == 0.2


def test_training_ui_sanitizes_invalid_marked_template_factors() -> None:
    config = sanitize_template_config(
        {
            "tile_preparation.positive_factor": 0.8,
            "tile_preparation.hard_negative_factor": 0.3,
            "tile_preparation.background_factor": 0.2,
        }
    )

    assert config["tile_preparation.positive_factor"] == pytest.approx(0.5)
    assert config["tile_preparation.hard_negative_factor"] == pytest.approx(0.3)
    assert config["tile_preparation.background_factor"] == pytest.approx(0.2)


def test_training_ui_rejects_invalid_job_factor_sum(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки" / "main"
    class_dir.mkdir(parents=True)
    (class_dir / "scenes.txt").write_text("scene-1\n", encoding="utf-8")
    (class_dir / "annotation.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(mlmarkup_root))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    job_config = _short_training_config()
    job_config["tile_preparation.positive_factor"] = 0.8
    job_config["tile_preparation.hard_negative_factor"] = 0.3
    job_config["tile_preparation.background_factor"] = 0.2

    with session_factory() as session:
        ensure_seed_templates(session)
        with pytest.raises(TrainingUIAPIError, match="Сумма"):
            create_training_job(
                session,
                TrainingJobCreate(
                    mlflow_experiment_id="1",
                    mlflow_experiment_name="ui-test",
                    mlflow_run_name="worker-test",
                    dataset_key="Вырубки\\main",
                    architecture="smp_segformer_b2",
                    config=job_config,
                ),
                config,
            )


def test_training_ui_automation_has_lower_priority_than_manual_jobs(tmp_path: Path, monkeypatch) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки" / "main"
    class_dir.mkdir(parents=True)
    (class_dir / "scenes.txt").write_text("scene-1\n", encoding="utf-8")
    (class_dir / "annotation.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(mlmarkup_root))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "files"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("MLSYSTEM2_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    started: list[tuple[list[str], dict[str, object]]] = []

    def fake_popen(args: list[str], **kwargs: object) -> SimpleNamespace:
        started.append((args, kwargs))
        return SimpleNamespace(pid=4321)

    def fake_experiment(request) -> MLflowExperiment:
        return MLflowExperiment(experiment_id="auto-exp", name=request.name)

    monkeypatch.setattr(_automation, "create_experiment", fake_experiment)

    with session_factory() as session:
        ensure_seed_templates(session)
        dataset_template = _service.create_training_template(
            session,
            TrainingTemplateCreate(
                architecture="smp_segformer_b2",
                dataset_key="Вырубки\\main",
            ),
            config,
        )
        dataset_config = dict(dataset_template.default_config)
        dataset_config["train.batch_size"] = 11
        _service.update_training_template_by_id(
            session,
            dataset_template.id,
            TrainingTemplateUpdate(default_config=dataset_config),
        )
        _service.set_automation(session, AutomationEnabledUpdate(enabled=True), config)
        _service.update_automation(
            session,
            AutomationRuleUpdate(
                dataset_key="Вырубки\\main",
                architecture="smp_segformer_b2",
                training_enabled=True,
                pseudo_markup_enabled=False,
            ),
            config,
        )
        _automation.sync_automation_once(session, config)
        auto_job = session.scalar(select(JobRow).where(JobRow.source == JobSource.AUTOMATION.value))
        assert auto_job is not None
        assert auto_job.status == JobStatus.QUEUED.value
        assert auto_job.dataset_key == "Вырубки\\main"
        assert auto_job.dataset_version is not None
        assert auto_job.config["train.batch_size"] == 11
        old_auto_job_id = auto_job.id
        old_version = auto_job.dataset_version
        annotation_path = class_dir / "annotation.geojson"
        annotation_path.write_text('{"type":"FeatureCollection","features":[{"type":"Feature"}]}', encoding="utf-8")
        stat = annotation_path.stat()
        os.utime(annotation_path, ns=(stat.st_atime_ns + 2_000_000_000, stat.st_mtime_ns + 2_000_000_000))
        _automation.sync_automation_once(session, config)
        assert session.get(JobRow, old_auto_job_id).status == JobStatus.CANCELLED.value
        auto_job = session.scalar(
            select(JobRow).where(
                JobRow.source == JobSource.AUTOMATION.value,
                JobRow.status == JobStatus.QUEUED.value,
            )
        )
        assert auto_job is not None
        assert auto_job.dataset_version != old_version
        stale_auto_job_id = auto_job.id
        _service.set_automation(session, AutomationEnabledUpdate(enabled=False), config)
        dispatch_training_queue_once(session, config, popen_factory=fake_popen)
        assert started == []
        assert session.get(JobRow, stale_auto_job_id).status == JobStatus.CANCELLED.value
        _service.set_automation(session, AutomationEnabledUpdate(enabled=True), config)
        auto_job = session.scalar(
            select(JobRow).where(
                JobRow.source == JobSource.AUTOMATION.value,
                JobRow.status == JobStatus.QUEUED.value,
            )
        )
        assert auto_job is not None
        assert auto_job.id != stale_auto_job_id

        manual_job = create_training_job(
            session,
            TrainingJobCreate(
                mlflow_experiment_id="1",
                mlflow_experiment_name="ui-test",
                mlflow_run_name="manual-test",
                dataset_key="Вырубки\\main",
                architecture="smp_segformer_b2",
                config=_short_training_config(),
            ),
            config,
        )
        dispatch_training_queue_once(session, config, popen_factory=fake_popen)
        session.commit()

        running = session.scalar(select(JobRow).where(JobRow.status == JobStatus.RUNNING.value))
        assert running is not None
        assert running.id == manual_job.id
        assert running.source == JobSource.MANUAL.value
        assert session.get(JobRow, auto_job.id).status == JobStatus.QUEUED.value
        with pytest.raises(TrainingUIAPIError):
            _service.delete_job(session, auto_job.id)
        with pytest.raises(TrainingUIAPIError):
            _service.move_job(session, auto_job.id, direction=-1)


def test_training_ui_disabling_automation_cancels_running_jobs_and_kills_mlflow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки" / "main"
    class_dir.mkdir(parents=True)
    (class_dir / "scenes.txt").write_text("scene-1\n", encoding="utf-8")
    (class_dir / "annotation.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(mlmarkup_root))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "files"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("MLSYSTEM2_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    killed_runs: list[tuple[str, str]] = []
    terminated_pids: list[int | None] = []

    def fake_experiment(request) -> MLflowExperiment:
        return MLflowExperiment(experiment_id="auto-exp", name=request.name)

    def fake_terminate(row: JobRow) -> None:
        terminated_pids.append(row.process_pid)
        row.process_pid = None

    def fake_mark_run_killed(tracking_uri: str, run_id: str) -> None:
        killed_runs.append((tracking_uri, run_id))

    monkeypatch.setattr(_automation, "create_experiment", fake_experiment)
    monkeypatch.setattr(_automation, "terminate_job_process", fake_terminate)
    monkeypatch.setattr(_automation, "mark_run_killed", fake_mark_run_killed)

    with session_factory() as session:
        ensure_seed_templates(session)
        _service.set_automation(session, AutomationEnabledUpdate(enabled=True), config)
        _service.update_automation(
            session,
            AutomationRuleUpdate(
                dataset_key="Вырубки\\main",
                architecture="smp_segformer_b2",
                training_enabled=True,
                pseudo_markup_enabled=False,
            ),
            config,
        )
        _automation.sync_automation_once(session, config)
        auto_job = session.scalar(select(JobRow).where(JobRow.source == JobSource.AUTOMATION.value))
        assert auto_job is not None
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "mlflow_run_id").write_text("run-auto-kill\n", encoding="utf-8")
        auto_job.status = JobStatus.RUNNING.value
        auto_job.process_pid = 9876
        auto_job.tmp_path = str(run_dir)
        result = session.scalar(select(TrainingResultRow).where(TrainingResultRow.job_id == auto_job.id))
        assert result is not None
        result.mlflow_run_id = "run-auto-kill"

        _service.set_automation(session, AutomationEnabledUpdate(enabled=False), config)

        assert auto_job.status == JobStatus.CANCELLED.value
        assert result.status == ResultStatus.CANCELLED.value
        assert auto_job.tmp_path is None
        assert not run_dir.exists()
        assert terminated_pids == [9876]
        assert killed_runs == [(config.mlflow_tracking_uri, "run-auto-kill")]


def test_training_ui_automation_creates_pseudo_after_training_and_does_not_retry_failed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки" / "main"
    class_dir.mkdir(parents=True)
    (class_dir / "scenes.txt").write_text("scene-1\n", encoding="utf-8")
    (class_dir / "annotation.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(mlmarkup_root))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "files"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("MLSYSTEM2_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    def fake_experiment(request) -> MLflowExperiment:
        return MLflowExperiment(experiment_id="auto-exp", name=request.name)

    def fake_best_checkpoint(tracking_uri: str, run_id: str) -> MLflowBestCheckpoint:
        return MLflowBestCheckpoint(
            tracking_uri=tracking_uri,
            run_id=run_id,
            metric_name="val/best_threshold_pixel_f1",
            f1_score=0.91,
            epoch=4,
            artifact_path="checkpoints/best.pt",
            artifact_uri="s3://mlflow-artifacts/auto/run/artifacts/checkpoints/best.pt",
            threshold=0.63,
        )

    monkeypatch.setattr(_automation, "create_experiment", fake_experiment)
    monkeypatch.setattr(_automation, "get_best_training_checkpoint", fake_best_checkpoint)

    with session_factory() as session:
        ensure_seed_templates(session)
        _service.set_automation(session, AutomationEnabledUpdate(enabled=True), config)
        _service.update_automation(
            session,
            AutomationRuleUpdate(
                dataset_key="Вырубки\\main",
                architecture="smp_segformer_b2",
                training_enabled=True,
                pseudo_markup_enabled=True,
            ),
            config,
        )
        _automation.sync_automation_once(session, config)
        training_job = session.scalar(
            select(JobRow).where(
                JobRow.source == JobSource.AUTOMATION.value,
                JobRow.type == "training",
            )
        )
        assert training_job is not None
        training_result = session.scalar(select(TrainingResultRow).where(TrainingResultRow.job_id == training_job.id))
        assert training_result is not None
        training_job.status = JobStatus.COMPLETED.value
        training_result.status = ResultStatus.OK.value
        training_result.mlflow_run_id = "run-auto"
        training_result.f1_score = 0.91
        training_result.epoch = 4

        _automation.sync_automation_once(session, config)
        pseudo_job = session.scalar(
            select(JobRow).where(
                JobRow.source == JobSource.AUTOMATION.value,
                JobRow.type == "inference",
            )
        )
        assert pseudo_job is not None
        assert pseudo_job.dataset_key == "Вырубки\\main"
        assert pseudo_job.dataset_version == training_job.dataset_version
        assert pseudo_job.config["training_result_id"] == str(training_result.id)
        assert pseudo_job.config["checkpoint_uri"] == "s3://mlflow-artifacts/auto/run/artifacts/checkpoints/best.pt"
        pseudo_result = session.scalar(select(PseudoMarkupResultRow).where(PseudoMarkupResultRow.job_id == pseudo_job.id))
        assert pseudo_result is not None
        assert pseudo_result.scenes_file is not None
        assert Path(pseudo_result.scenes_file.path).read_text(encoding="utf-8") == "scene-1\n"

        training_result.status = ResultStatus.ERROR.value
        pseudo_job.status = JobStatus.CANCELLED.value
        pseudo_result.status = ResultStatus.CANCELLED.value
        _automation.sync_automation_once(session, config)
        training_jobs = session.scalars(
            select(JobRow).where(
                JobRow.source == JobSource.AUTOMATION.value,
                JobRow.type == "training",
            )
        ).all()
        assert len(training_jobs) == 1


def test_training_ui_worker_records_best_mlflow_metric(tmp_path: Path, monkeypatch) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки" / "main"
    class_dir.mkdir(parents=True)
    (class_dir / "scenes.txt").write_text("scene-1\n", encoding="utf-8")
    (class_dir / "annotation.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(mlmarkup_root))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "files"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("MLSYSTEM2_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    def fake_popen(args: list[str], **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(pid=4321)

    def fake_best_checkpoint(tracking_uri: str, run_id: str) -> MLflowBestCheckpoint:
        assert tracking_uri == config.mlflow_tracking_uri
        assert run_id == "run-123"
        return MLflowBestCheckpoint(
            tracking_uri=tracking_uri,
            run_id=run_id,
            metric_name="val/best_threshold_pixel_f1",
            f1_score=0.8123,
            epoch=7,
            artifact_path="checkpoints/best.pt",
            artifact_uri="s3://mlflow-artifacts/1/run-123/artifacts/checkpoints/best.pt",
            threshold=0.7,
        )

    monkeypatch.setattr(_worker, "get_best_training_checkpoint", fake_best_checkpoint)

    with session_factory() as session:
        ensure_seed_templates(session)
        job = create_training_job(
            session,
            TrainingJobCreate(
                mlflow_experiment_id="1",
                mlflow_experiment_name="ui-test",
                mlflow_run_name="worker-test",
                dataset_key="Вырубки\\main",
                architecture="smp_segformer_b2",
                config={
                    "dataset.val_fraction": 0.2,
                    "tile_preparation.tile_size": 32,
                    "tile_preparation.stride": 32,
                    "tile_preparation.augmentation_level": 0,
                    "tile_preparation.positive_factor": 0.5,
                    "train.epochs": 1,
                    "train.batch_size": 1,
                    "train.learning_rate": 0.0001,
                    "train.weight_decay": 0.0,
                    "train.loss": "bce_dice",
                    "train.focal_alpha": 0.6,
                    "train.pos_weight": 1.0,
                    "train.tversky_alpha": 0.4,
                    "train.tversky_beta": 0.6,
                    "train.threshold": 0.5,
                    "train.early_stopping_patience": 1,
                    "train.max_training_time_sec": None,
                },
            ),
            config,
        )
        dispatch_training_queue_once(session, config, popen_factory=fake_popen)
        session.flush()

        row = session.get(JobRow, job.id)
        assert row is not None
        assert row.tmp_path is not None
        run_dir = Path(row.tmp_path)
        (run_dir / "mlflow_run_id").write_text("run-123\n", encoding="utf-8")
        _worker._sync_training_run_id(session, row, config)
        running_result = session.scalar(select(TrainingResultRow).where(TrainingResultRow.job_id == job.id))
        assert running_result is not None
        assert running_result.mlflow_run_id == "run-123"
        assert running_result.mlflow_run_url is not None

        (run_dir / "train.log").write_text("status=succeeded\nmlflow_run=run-123\n", encoding="utf-8")
        (run_dir / "exit_code").write_text("0\n", encoding="utf-8")

        dispatch_training_queue_once(session, config, popen_factory=fake_popen)
        session.commit()

        finished = session.get(JobRow, job.id)
        assert finished is not None
        assert finished.status == JobStatus.COMPLETED.value
        result = session.scalar(select(TrainingResultRow).where(TrainingResultRow.job_id == job.id))
        assert result is not None
        assert result.status == ResultStatus.OK.value
        assert result.mlflow_run_id == "run-123"
        assert result.f1_score == 0.8123
        assert result.epoch == 7

        monkeypatch.setattr(_service, "get_best_training_checkpoint", fake_best_checkpoint)
        pseudo_job = _service.create_pseudo_markup_job(
            session,
            class_key="Вырубки\\main",
            dataset_key="Вырубки\\main",
            image_folder_key=None,
            training_result_id=result.id,
            scenes_name=None,
            scenes_content_type=None,
            scenes_bytes=None,
            config=config,
        )
        assert pseudo_job.config["mlflow_run_id"] == "run-123"
        assert pseudo_job.config["checkpoint_artifact_path"] == "checkpoints/best.pt"
        assert (
            pseudo_job.config["checkpoint_uri"]
            == "s3://mlflow-artifacts/1/run-123/artifacts/checkpoints/best.pt"
        )
        assert pseudo_job.config["checkpoint_f1_score"] == 0.8123
        assert pseudo_job.config["checkpoint_epoch"] == 7
        assert pseudo_job.config["checkpoint_threshold"] == 0.7

        dispatch_inference_queue_once(session, config, popen_factory=fake_popen)
        session.flush()

        pseudo_row = session.get(JobRow, pseudo_job.id)
        assert pseudo_row is not None
        assert pseudo_row.status == JobStatus.RUNNING.value
        assert pseudo_row.tmp_path is not None
        pseudo_run_dir = Path(pseudo_row.tmp_path)
        pseudo_config = (pseudo_run_dir / "pseudo_config.yaml").read_text(encoding="utf-8")
        pseudo_config_payload = yaml.safe_load(pseudo_config)
        assert "threshold: 0.7" in pseudo_config
        assert "checkpoint_artifact_path: checkpoints/best.pt" in pseudo_config
        assert pseudo_config_payload["inference_backend"] == "pytorch_one_off"
        for forbidden_key in ("triton_model", "pipeline", "model_repository", "model_archive"):
            assert forbidden_key not in pseudo_config_payload

        output = pseudo_run_dir / "scratch" / "pseudo_markup.geojson"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
        (pseudo_run_dir / "scratch" / "report.json").write_text(
            '{"status":"ok","processed":1,"feature_count":0}',
            encoding="utf-8",
        )
        (pseudo_run_dir / "exit_code").write_text("0\n", encoding="utf-8")

        dispatch_inference_queue_once(session, config, popen_factory=fake_popen)
        session.commit()

        completed_pseudo = session.get(JobRow, pseudo_job.id)
        assert completed_pseudo is not None
        assert completed_pseudo.status == JobStatus.COMPLETED.value
        assert completed_pseudo.started_at is not None
        completed_pseudo.started_at = completed_pseudo.finished_at - timedelta(minutes=30)
        session.flush()
        assert not (pseudo_run_dir / "scratch").exists()
        refreshed_results = _service.class_results(session, "Вырубки\\main", config)
        pseudo_results = refreshed_results.results[0].pseudo_markup_results
        assert pseudo_results[0].status == ResultStatus.OK
        assert pseudo_results[0].runtime_minutes == 30
        assert pseudo_results[0].geojson_file is not None
        assert re.fullmatch(
            r"Вырубки_main_segformer_b2_\d{2}_\d{2}_\d{2}_\d{2}\.geojson",
            pseudo_results[0].geojson_file.original_name,
        )
        geojson_path = config.stored_files_root / "pseudo_markup_geojson"
        assert any(path.suffix == ".geojson" for path in geojson_path.iterdir())
        pseudo_db_row = session.get(PseudoMarkupResultRow, pseudo_results[0].id)
        assert pseudo_db_row is not None
        geojson_file = pseudo_db_row.geojson_file
        assert geojson_file is not None
        stored_geojson_path = Path(geojson_file.path)
        assert stored_geojson_path.is_file()
        geojson_file_id = geojson_file.id

        deleted = _service.delete_pseudo_markup_result(session, pseudo_results[0].id, config)
        session.commit()

        assert deleted.id == pseudo_results[0].id
        assert session.get(PseudoMarkupResultRow, pseudo_results[0].id) is None
        assert session.get(JobRow, pseudo_job.id) is None
        assert session.get(StoredFileRow, geojson_file_id) is None
        assert not stored_geojson_path.exists()


def _short_training_config() -> dict[str, object]:
    return {
        "dataset.val_fraction": 0.2,
        "tile_preparation.tile_size": 32,
        "tile_preparation.stride": 32,
        "tile_preparation.augmentation_level": 0,
        "tile_preparation.positive_factor": 0.5,
        "tile_preparation.hard_negative_factor": 0.0,
        "tile_preparation.background_factor": 0.5,
        "train.epochs": 1,
        "train.batch_size": 1,
        "train.learning_rate": 0.0001,
        "train.weight_decay": 0.0,
        "train.loss": "bce_dice",
        "train.focal_alpha": 0.6,
        "train.pos_weight": 1.0,
        "train.hard_negative_weight": 1.0,
        "train.tversky_alpha": 0.4,
        "train.tversky_beta": 0.6,
        "train.threshold": 0.5,
        "train.early_stopping_patience": 1,
        "train.max_train_batches_per_epoch": 72,
        "train.max_val_batches_per_epoch": 1000,
        "train.max_training_time_sec": None,
    }
