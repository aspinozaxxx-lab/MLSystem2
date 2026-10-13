"""Полная история выбранного класса не ограничена глобальными последними изменениями."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select

from mlsystem2.training_ui_api.api import create_app
from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import Base, configure_schema, create_session_factory
from mlsystem2.training_ui_api._dataset_catalog import synchronize_dataset_catalog
from mlsystem2.training_ui_api._models import DatasetClassRow, DatasetRow, JobRow, TrainingResultRow


def test_class_history_includes_old_failed_archived_and_active_training_without_duplicates(
    tmp_path: Path, monkeypatch,
):
    for variable, value in {
        "MLSYSTEM2_TRAINING_UI_DATABASE_URL": f"sqlite:///{tmp_path / 'ui.db'}",
        "MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA": "",
        "MLSYSTEM2_MLMARKUP_ROOT": str(tmp_path / "MLMarkup"),
        "MLSYSTEM2_IMAGES_ROOT": str(tmp_path / "images"),
        "MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT": str(tmp_path / "stored"),
        "MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT": str(tmp_path / "scratch"),
        "MLSYSTEM2_TRAINING_UI_WORKER_ENABLED": "false",
        "MLSYSTEM2_TRAINING_UI_USER": "mlsystem",
        "MLSYSTEM2_TRAINING_UI_PASSWORD": "secret",
        "MLSYSTEM2_TRAINING_UI_SESSION_SECRET": "тестовый-ключ",
    }.items():
        monkeypatch.setenv(variable, value)
    for source in ["Лес/main", "Лес/архивный", "Реки/main", "Пустой"]:
        (tmp_path / "MLMarkup" / source).mkdir(parents=True)
    config = get_config()
    configure_schema(None)
    factory = create_session_factory(config)
    Base.metadata.create_all(factory.kw["bind"])
    instant = datetime(2026, 10, 1, tzinfo=timezone.utc)
    with factory() as session:
        synchronize_dataset_catalog(session, config)
        forest = session.scalar(select(DatasetClassRow).where(DatasetClassRow.name == "Лес"))
        empty = session.scalar(select(DatasetClassRow).where(DatasetClassRow.name == "Пустой"))
        datasets = {row.source_path: row for row in session.scalars(select(DatasetRow))}
        for source, count, offset in [("Лес/main", 30, 0), ("Лес/архивный", 3, 30), ("Реки/main", 40, 100)]:
            dataset = datasets[source]
            for index in range(count):
                timestamp = instant + timedelta(minutes=offset + index)
                session.add(TrainingResultRow(
                    source="manual", dataset_key=dataset.key, class_key=dataset.key,
                    class_display_name=source.replace("/", "\\"), architecture="smp_segformer_b2",
                    model_name=f"Сеть {index}", status="ok", created_at=timestamp, updated_at=timestamp,
                ))
        archived = datasets["Лес/архивный"]
        archived.deleted_at = instant
        active_dataset = datasets["Лес/main"]
        failed = TrainingResultRow(
            source="manual", dataset_key=active_dataset.key, class_key=active_dataset.key,
            class_display_name="Лес\\main", architecture="smp_segformer_b2",
            model_name="Неуспешная сеть", status="error", created_at=instant, updated_at=instant,
        )
        job = JobRow(
            type="training", source="manual", status="running", queue_position=1,
            dataset_key=active_dataset.key, dataset_name="Лес\\main", architecture="smp_segformer_b2",
            model_name="Активная сеть", config={},
        )
        session.add_all([failed, job])
        session.flush()
        unfinished = TrainingResultRow(
            source="manual", dataset_key=active_dataset.key, class_key=active_dataset.key,
            class_display_name="Лес\\main", architecture="smp_segformer_b2",
            model_name="Активная сеть", status="running", job_id=job.id,
        )
        session.add(unfinished)
        session.commit()
        class_key, empty_key = forest.key, empty.key
        main_key, archived_key = active_dataset.key, archived.key
        job_id, failed_id, unfinished_id = str(job.id), str(failed.id), str(unfinished.id)

    with TestClient(create_app()) as client:
        login = client.post("/api/v1/auth/login", json={"username": "mlsystem", "password": "secret"})
        assert login.status_code == 200
        response = client.get("/api/v1/results/changes", params={"class_key": class_key})
        assert response.status_code == 200
        changes = response.json()["changes"]
        assert len(changes) == 35
        assert {item["class_key"] for item in changes} == {class_key}
        assert {item["dataset_key"] for item in changes} == {main_key, archived_key}
        assert sum(item["dataset_key"] == archived_key for item in changes) == 3
        assert sum(item["job_id"] == job_id for item in changes) == 1
        assert changes[0]["id"] == job_id
        assert next(item for item in changes if item["id"] == failed_id)["status"] == "error"
        assert unfinished_id not in {item["id"] for item in changes}
        assert len({item["id"] for item in changes}) == len(changes)
        completed = [item["changed_at"] for item in changes if item["item_type"] != "job"]
        assert completed == sorted(completed, reverse=True)
        assert client.get("/api/v1/results/changes", params={"class_key": empty_key}).json() == {"changes": []}

        # Прежний вызов сохраняет активные задания и ровно 20 успешных последних изменений.
        global_changes = client.get("/api/v1/results/changes").json()["changes"]
        assert len(global_changes) == 21
        assert all(item["status"] == "ok" for item in global_changes if item["item_type"] != "job")
        assert client.get("/api/v1/results/changes", params={"class_key": "несуществующий"}).status_code == 400
