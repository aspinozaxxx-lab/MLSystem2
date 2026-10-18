"""Комментарии, независимые постоянные номера и обратимый скрытый архив сетей."""

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from fastapi.testclient import TestClient

from mlsystem2.training_ui_api.api import create_app
from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import create_session_factory
from mlsystem2.training_ui_api._models import DatasetClassRow, DatasetRow, JobRow, PseudoMarkupResultRow, TrainingResultRow
from mlsystem2.training_ui_api._result_metadata import next_result_number
from mlsystem2.training_ui_api._queueing import track_manual_job_author
from mlsystem2.training_ui_api import _service, _dataset_catalog
from mlsystem2.training_ui_api.contracts import TrainingUIAPIError


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_COOKIE_SECURE", "false")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SESSION_SECRET", "тестовый-секрет")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_USERS_JSON", json.dumps([
        {"username": "Alex", "password": "test", "role": "user"},
    ]))
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(tmp_path / "разметки"))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "снимки"))
    app = create_app()
    client = TestClient(app)
    config = get_config()
    factory = create_session_factory(config)
    with factory() as session:
        class_row = DatasetClassRow(key="класс", name="Реки", technical_name="rivers")
        session.add(class_row)
        session.flush()
        session.add(DatasetRow(key="датасет", name="main", class_id=class_row.id,
            source_type="mlmarkup", source_path="Реки/main"))
        job = JobRow(type="training", source="manual", status="completed", queue_position=1,
            dataset_key="датасет", dataset_name="Реки\\main", model_name="Сеть",
            architecture="smp_segformer_b1", config={"ui.started_by": "Alex", "train.epochs": 10})
        session.add(job)
        session.flush()
        result = TrainingResultRow(class_key="датасет", dataset_key="датасет", class_display_name="Реки\\main",
            model_name="Сеть", architecture="smp_segformer_b1", status="ok", job_id=job.id,
            dataset_number=next_result_number(session, "датасет"), f1_score=0.71,
            training_metrics={"pixel_f1": 0.71}, mlflow_run_id="сохранённые-веса")
        session.add(result)
        session.commit()
        result_id, job_id, class_id = result.id, job.id, class_row.id
    yield client, factory, config, result_id, job_id, class_id
    client.close()


def login(client):
    assert client.post("/api/v1/auth/login", json={"username": "Alex", "password": "test"}).status_code == 200


def test_comment_requires_authentication_and_is_shared_persistent_plain_text(context):
    client, factory, config, result_id, job_id, _ = context
    endpoint = f"/api/v1/results/training/{result_id}/comment"
    assert client.get(endpoint).status_code == 401
    assert client.put(endpoint, json={"comment": "текст", "expected_comment": ""}).status_code == 401
    login(client)
    assert client.get(endpoint).json() == {"id": str(result_id), "comment": ""}
    with factory() as session:
        session.get(TrainingResultRow, result_id).updated_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        session.commit()
    text = "  Проверить края\n<script>не HTML</script>  "
    response = client.put(endpoint, json={"comment": text, "expected_comment": ""})
    assert response.status_code == 200
    assert response.json()["comment"] == text.strip()
    assert client.get(endpoint).json()["comment"] == text.strip()
    with factory() as session:
        row = session.get(TrainingResultRow, result_id)
        info = _service._training_result_info(session, row, config=config)
        assert info.comment == text.strip() and info.dataset_number == 1
        assert info.started_by == "Alex" and info.f1_score == 0.71
        assert row.mlflow_run_id == "сохранённые-веса"
        assert row.training_metrics == {"pixel_f1": 0.71}
        assert row.updated_at.year == 2020
        assert session.get(JobRow, job_id).config == {"ui.started_by": "Alex", "train.epochs": 10}
    assert client.put(endpoint, json={"comment": " \n ", "expected_comment": text.strip()}).json()["comment"] == ""


def test_comment_conflict_does_not_overwrite_another_edit(context):
    client, _, _, result_id, _, _ = context
    login(client)
    endpoint = f"/api/v1/results/training/{result_id}/comment"
    assert client.put(endpoint, json={"comment": "Первая правка", "expected_comment": ""}).status_code == 200
    assert client.put(endpoint, json={"comment": "Устаревшая правка", "expected_comment": ""}).status_code == 409
    assert client.get(endpoint).json()["comment"] == "Первая правка"
    assert client.put(endpoint, json={"comment": "Уточнение", "expected_comment": "Первая правка"}).status_code == 200


@pytest.mark.parametrize("body", [
    {"comment": "Нет исходного значения"}, {"comment": None, "expected_comment": ""},
    {"comment": "я" * 2001, "expected_comment": ""}, {"comment": "текст\x00", "expected_comment": ""},
    {"comment": "текст", "expected_comment": "", "status": "ok"},
])
def test_comment_validates_request(context, body):
    client, _, _, result_id, _, _ = context
    login(client)
    assert client.put(f"/api/v1/results/training/{result_id}/comment", json=body).status_code == 422


def test_missing_result_is_not_created(context):
    client, _, _, _, _, _ = context
    login(client)
    endpoint = f"/api/v1/results/training/{uuid4()}"
    assert client.get(endpoint + "/comment").status_code == 404
    assert client.put(endpoint + "/comment", json={"comment": "текст", "expected_comment": ""}).status_code == 404
    assert client.post(endpoint + "/archive", json={"confirmed": True}).status_code == 404


def test_numbering_is_per_dataset_atomic_and_not_reused(context):
    _, factory, _, _, _, _ = context
    def allocate(key):
        with factory() as session:
            value = next_result_number(session, key)
            session.commit()
            return value
    assert allocate("другой") == 1
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(allocate, ["датасет"] * 10))
    assert sorted(values) == list(range(2, 12))
    assert allocate("другой") == 2
    assert allocate("датасет") == 12


def test_archive_requires_confirmation_preserves_records_and_hides_result(context):
    client, factory, config, result_id, job_id, class_id = context
    endpoint = f"/api/v1/results/training/{result_id}"
    assert client.post(endpoint + "/archive", json={"confirmed": True}).status_code == 401
    login(client)
    assert client.post(endpoint + "/archive", json={}).status_code == 422
    assert client.post(endpoint + "/archive", json={"confirmed": False}).status_code == 422
    with factory() as session:
        session.get(TrainingResultRow, result_id).comment = "Сохранить для сравнения"
        session.get(DatasetClassRow, class_id).primary_training_result_id = result_id
        session.commit()
    assert client.post(endpoint + "/archive", json={"confirmed": True}).status_code == 204
    assert client.post(endpoint + "/archive", json={"confirmed": True}).status_code == 204
    assert client.get(endpoint + "/comment").status_code == 404
    assert client.put(endpoint + "/comment", json={"comment": "правка", "expected_comment": "Сохранить для сравнения"}).status_code == 404
    assert client.post(endpoint + "/test-f1").status_code == 404
    assert client.get(endpoint + "/continue").status_code == 400
    assert client.get(endpoint + "/test-f1/view").status_code == 400
    with factory() as session:
        row = session.get(TrainingResultRow, result_id)
        assert row.archived_at is not None and row.archived_by == "Alex"
        assert row.status == "ok" and row.dataset_number == 1 and row.f1_score == 0.71
        assert row.comment == "Сохранить для сравнения" and row.mlflow_run_id == "сохранённые-веса"
        assert session.get(JobRow, job_id) is not None
        assert session.get(DatasetClassRow, class_id).primary_training_result_id is None
        assert _dataset_catalog.dataset_training_result(session, "датасет") is None
        assert _dataset_catalog.successful_training_results(session, "датасет") == []
        assert _service.dataset_results(session, "датасет", config).results == []
        assert _service.result_changes(session, config).changes == []
        with pytest.raises(TrainingUIAPIError, match="не найден"):
            _service._training_result_row_for_export(session, result_id)
        assert next_result_number(session, "датасет") == 2
        # Восстановление не требует пересоздания сети, изменения номера или метрик.
        row.archived_at = row.archived_by = None
        session.commit()
    assert client.get(endpoint + "/comment").json()["comment"] == "Сохранить для сравнения"


@pytest.mark.parametrize("dependency", [False, True])
def test_active_training_or_dependent_inference_blocks_archive(context, dependency):
    client, factory, _, result_id, job_id, _ = context
    login(client)
    with factory() as session:
        if dependency:
            session.add(JobRow(type="inference", source="manual", status="queued", queue_position=2,
                dataset_name="Реки\\main", model_name="Сеть", architecture="smp_segformer_b1",
                config={"editor_pseudo": {"training_result_id": str(result_id)}}))
        else:
            session.get(JobRow, job_id).status = "running"
        session.commit()
    assert client.post(f"/api/v1/results/training/{result_id}/archive", json={"confirmed": True}).status_code == 409
    with factory() as session:
        assert session.get(TrainingResultRow, result_id).archived_at is None


def test_archive_protects_linked_pseudo_and_cancelled_cleanup(context):
    client, factory, config, result_id, job_id, _ = context
    login(client)
    with factory() as session:
        job = JobRow(type="inference", source="manual", status="running", queue_position=2,
                     dataset_name="Реки\\main", model_name="Сеть", architecture="pseudo-markup", config={})
        session.add(job)
        session.flush()
        pseudo = PseudoMarkupResultRow(training_result_id=result_id, class_key="датасет",
            source_dataset_name="Реки\\main", status="running", job_id=job.id)
        session.add(pseudo)
        session.commit()
        pseudo_id, pseudo_job_id = pseudo.id, job.id
    endpoint = f"/api/v1/results/training/{result_id}/archive"
    assert client.post(endpoint, json={"confirmed": True}).status_code == 409
    with factory() as session:
        session.get(JobRow, pseudo_job_id).status = "cancelled"
        session.get(PseudoMarkupResultRow, pseudo_id).status = "cancelled"
        session.get(JobRow, job_id).status = "cancelled"
        session.get(TrainingResultRow, result_id).status = "cancelled"
        session.commit()
    assert client.post(endpoint, json={"confirmed": True}).status_code == 204
    assert client.get(f"/api/v1/jobs/{job_id}").status_code == 400
    assert client.delete(f"/api/v1/jobs/{job_id}").status_code == 400
    assert client.delete(f"/api/v1/results/pseudo-markup/{pseudo_id}").status_code == 400
    assert client.get(f"/api/v1/results/pseudo-markup/{pseudo_id}/view").status_code == 400
    with factory() as session:
        _service.dataset_results(session, "датасет", config)
        _service._delete_cancelled_manual_jobs(session)
        session.commit()
        assert session.get(TrainingResultRow, result_id).archived_at is not None
        assert session.get(JobRow, job_id) is not None
        assert session.get(JobRow, pseudo_job_id) is not None
        assert session.get(PseudoMarkupResultRow, pseudo_id) is not None


def test_job_author_tracks_only_new_manual_jobs(context):
    _, factory, _, _, job_id, _ = context
    with factory() as session:
        track_manual_job_author(session, "Aspinoza")
        previous = session.get(JobRow, job_id)
        previous.config = {**previous.config, "новое_поле": True}
        manual = JobRow(type="inference", source="manual", status="queued", queue_position=2,
            dataset_name="Учебный", model_name="Сеть", architecture="pseudo-markup", config={"ui.started_by": "Подмена"})
        automatic = JobRow(type="training", source="automation", status="queued", queue_position=3,
            dataset_name="Учебный", model_name="Сеть", architecture="smp_segformer_b1", config={})
        session.add_all([manual, automatic])
        session.flush()
        assert previous.config["ui.started_by"] == "Alex"
        assert manual.config["ui.started_by"] == "Aspinoza"
        assert "ui.started_by" not in automatic.config
        assert _service._job_summary(session, manual).started_by == "Aspinoza"
        assert _service._job_detail(session, manual).started_by == "Aspinoza"


def test_http_inference_queue_records_authenticated_author(context, monkeypatch):
    from mlsystem2.training_ui_api._routes import results as routes
    client, factory, _, _, _, _ = context
    login(client)
    def create_job(session, **kwargs):
        row = JobRow(type="inference", source="manual", status="queued", queue_position=2,
            dataset_name="Учебный", model_name="Сеть", architecture="pseudo-markup", config={})
        session.add(row)
        session.flush()
        return _service._job_detail(session, row)
    monkeypatch.setattr(routes, "create_pseudo_markup_job", create_job)
    response = client.post("/api/v1/results/datasets/датасет/pseudo-markup")
    assert response.status_code == 200, response.text
    assert response.json()["started_by"] == "Alex"
    jobs = client.get("/api/v1/queues").json()["jobs"]
    assert next(row for row in jobs if row["id"] == response.json()["id"])["started_by"] == "Alex"


def test_metadata_migration_numbers_oldest_first_and_preserves_existing_data(monkeypatch):
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    path = Path(__file__).parents[1] / "migrations/versions/20261018_0035_training_result_metadata.py"
    spec = importlib.util.spec_from_file_location("metadata_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(sa.text("CREATE TABLE training_results (id TEXT PRIMARY KEY, class_key TEXT, created_at TEXT, status TEXT, metrics TEXT)"))
        for row in [("позже", "А", "2026-10-18", "ok"), ("раньше", "А", "2026-10-17", "ok"),
                    ("отмена", "А", "2026-10-16", "cancelled"), ("другая", "Б", "2026-10-18", "error")]:
            connection.execute(sa.text("INSERT INTO training_results VALUES (:id, :key, :date, :status, 'исходные метрики')"),
                dict(zip(["id", "key", "date", "status"], row)))
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
            rows = {row.id: row for row in connection.execute(sa.text("SELECT * FROM training_results"))}
            assert rows["раньше"].dataset_number == 1 and rows["позже"].dataset_number == 2
            assert rows["другая"].dataset_number == 1 and rows["отмена"].dataset_number is None
            assert all(row.comment == "" and row.metrics == "исходные метрики" and row.archived_at is None for row in rows.values())
            assert dict(connection.execute(sa.text("SELECT dataset_key, last_number FROM training_result_counters")).all()) == {"А": 2, "Б": 1}
            connection.execute(sa.text("UPDATE training_results SET comment = 'важное' WHERE id = 'раньше'"))
            with pytest.raises(RuntimeError, match="Откат удалит комментарии"):
                migration.downgrade()
            connection.execute(sa.text("UPDATE training_results SET comment = ''"))
            migration.downgrade()
            assert connection.scalar(sa.text("SELECT count(*) FROM training_results")) == 4
