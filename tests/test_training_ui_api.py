from __future__ import annotations

import hashlib
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
from sqlalchemy import BigInteger, Integer
from sqlalchemy import select

from mlsystem2.mlflow_adapter.contracts import (
    MLflowBestCheckpoint,
    MLflowExperiment,
    MLflowTrainingProgress,
)
from mlsystem2.models.contracts import ModelsError
from mlsystem2.training_ui_api import _auth, _automation, _model_export, _service, _worker
from mlsystem2.training_ui_api._routes import export as _export_routes
from mlsystem2.training_ui_api.api import create_app
from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import Base, configure_schema, create_session_factory
from mlsystem2.training_ui_api._models import (
    DatasetClassRow,
    DatasetRow,
    InferenceTemplateRow,
    JobRow,
    PseudoMarkupResultRow,
    StoredFileRow,
    TestSampleRow as _TestSampleRow,
    TestSampleTileRow as _TestSampleTileRow,
    TrainingResultTestMetricRow,
    TrainingResultRow,
    TrainingTemplateRow,
)
from mlsystem2.training_ui_api._service import create_training_job, ensure_seed_templates
from mlsystem2.training_ui_api._templates import initial_templates, sanitize_template_config
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
    TrainingResultBatchExportRequest,
    TrainingResultExportItem,
    TrainingTemplateCreate,
    TrainingTemplateUpdate,
    TrainingUIAPIError,
)


def test_frontend_credentials_accept_configured_username_and_legacy_alias(monkeypatch) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_USER", "mlsystem")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_USER_ALIASES", "mluser")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_PASSWORD", "secret")
    config = get_config()

    assert _auth.verify_credentials("mlsystem", "secret", config) is True
    assert _auth.verify_credentials("mluser", "secret", config) is True
    assert _auth.verify_credentials("other", "secret", config) is False
    assert _auth.verify_credentials("mluser", "wrong", config) is False


def test_app_links_use_prepared_images_browser(monkeypatch) -> None:
    monkeypatch.setenv("MLSYSTEM2_IMAGES_UI_URL", "/prepared-images/")

    config = get_config()
    links = {item.key: item for item in _service.app_links(config).links}

    assert links["images"].title == "Снимки"
    assert links["images"].url == "/prepared-images/"
    assert "minio" not in links


def test_next_gen_hf_template_and_legacy_defaults_are_explicit() -> None:
    templates = {item["architecture"]: item for item in initial_templates()}

    assert templates["smp_segformer_b0"]["default_config"]["train.pipeline_variant"] == (
        "legacy"
    )
    hf = templates["segformer_b0"]["default_config"]
    assert hf["train.pipeline_variant"] == "next_gen"
    assert hf["train.pretrained"] is True
    assert hf["train.max_val_batches_per_epoch"] is None
    assert hf["next_gen.normalization"] == "imagenet_rgb_red_nir"
    assert hf["tile_preparation.context"] == 128


def test_next_gen_job_validation_rejects_unsupported_combinations() -> None:
    base = {
        "train.pipeline_variant": "next_gen",
        "train.input_channels": 4,
        "train.pretrained": False,
        "train.max_val_batches_per_epoch": None,
        "dataset.task": "binary",
        "dataset.imagery_type": "kanopus",
    }

    _service._validate_training_pipeline_variant(dict(base), "smp_segformer_b0")
    with pytest.raises(TrainingUIAPIError, match="полную validation"):
        _service._validate_training_pipeline_variant(
            {**base, "train.max_val_batches_per_epoch": 1},
            "smp_segformer_b0",
        )
    with pytest.raises(TrainingUIAPIError, match="только binary"):
        _service._validate_training_pipeline_variant(
            {**base, "dataset.task": "multiclass"},
            "smp_segformer_b0",
        )
    with pytest.raises(TrainingUIAPIError, match="доступны только"):
        _service._validate_training_pipeline_variant(
            {**base, "train.pretrained": True},
            "smp_segformer_b0",
        )


def test_frontend_credentials_support_canonical_users_roles_and_aliases(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "MLSYSTEM2_TRAINING_UI_USERS_JSON",
        json.dumps(
            [
                {
                    "username": "Aspinoza",
                    "password": "admin-password",
                    "role": "admin",
                    "aliases": ["mlsystem"],
                },
                {
                    "username": "Alice",
                    "password": "user-password",
                    "role": "user",
                },
            ]
        ),
    )
    config = get_config()

    admin = _auth.authenticate_user("mlsystem", "admin-password", config)
    user = _auth.authenticate_user("Alice", "user-password", config)
    assert admin is not None and (admin.username, admin.role) == ("Aspinoza", "admin")
    assert user is not None and (user.username, user.role) == ("Alice", "user")
    assert _auth.authenticate_user("mluser", "admin-password", config) is None


def test_pseudo_report_success_requires_processed_scene() -> None:
    assert _worker._pseudo_report_allows_success({"status": "ok", "processed": 1}) is True
    assert _worker._pseudo_report_allows_success({"status": "partial", "processed": 1}) is False
    assert _worker._pseudo_report_allows_success({"status": "ok", "processed": 0}) is False
    assert _worker._pseudo_report_allows_success({"status": "error", "processed": 1}) is False
    assert (
        _worker._pseudo_report_allows_success(
            {"status": "ok", "processed": 1, "unique_image_count": 2}
        )
        is False
    )
    assert (
        _worker._pseudo_report_allows_success(
            {"status": "ok", "processed": 2, "unique_image_count": 2, "failed": 1}
        )
        is False
    )


def test_partial_pseudo_markup_is_not_published(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "stored"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        job = _queue_test_job(
            JobType.INFERENCE,
            JobSource.MANUAL,
            1,
            datetime(2026, 8, 17, tzinfo=timezone.utc),
        )
        job.status = JobStatus.RUNNING.value
        job.tmp_path = str(tmp_path / "job")
        session.add(job)
        session.flush()
        result = PseudoMarkupResultRow(
            source=JobSource.MANUAL.value,
            dataset_key="zu500-main",
            training_result_id=None,
            class_key="zu500-main",
            source_dataset_name="ЗУ500\\main",
            image_count=2,
            status=ResultStatus.RUNNING.value,
            job_id=job.id,
        )
        session.add(result)
        session.flush()

        scratch = Path(job.tmp_path) / "scratch"
        scratch.mkdir(parents=True)
        (scratch / "pseudo_markup.geojson").write_text(
            '{"type":"FeatureCollection","features":[]}',
            encoding="utf-8",
        )
        (scratch / "report.json").write_text(
            json.dumps(
                {
                    "status": "partial",
                    "processed": 1,
                    "unique_image_count": 2,
                    "failed": 1,
                    "failures": [
                        {
                            "scene_id": "failed-scene",
                            "error": "Triton shared memory exhausted",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        _worker._finish_inference_job(session, job, config, succeeded=True)
        session.flush()

        assert job.status == JobStatus.FAILED.value
        assert job.error is not None
        assert "1 из 2" in job.error
        assert "shared memory exhausted" in job.error
        assert result.status == ResultStatus.ERROR.value
        assert result.geojson_file_id is None
        assert session.scalar(select(StoredFileRow.id)) is None


def test_aoi_report_requires_every_selected_image_to_finish() -> None:
    complete = {"status": "ok", "processed": 2, "unique_image_count": 2, "failed": 0}
    partial = {"status": "partial", "processed": 1, "unique_image_count": 2, "failed": 1}

    assert _worker._pseudolabel_aoi_report_allows_success(complete) is True
    assert _worker._pseudolabel_aoi_report_allows_success(partial) is False


def test_pseudo_geojson_download_name_normalizes_legacy_slashes() -> None:
    row = StoredFileRow(
        kind=StoredFileKind.PSEUDO_MARKUP_GEOJSON.value,
        original_name="Засоления\\main_segformer b2_07_38_06_06.geojson",
        path="/tmp/file.geojson",
        size_bytes=1,
    )

    assert (
        _service.stored_file_download_name(row) == "Засоления_main_segformer b2_07_38_06_06.geojson"
    )


def test_pseudo_geojson_download_name_uses_display_name_for_uuid_key() -> None:
    training_result = TrainingResultRow(
        class_display_name="Разрушки\\main",
        model_name="segformer b2",
    )
    pseudo_result = PseudoMarkupResultRow(
        class_key="9788af3a-000d-4b0a-aa2f-c9b2b3b09cb1",
        training_result=training_result,
    )
    job = JobRow(
        dataset_name="Разрушки\\main",
        training_dataset_name="Разрушки\\main",
        model_name="pseudo-markup",
    )

    name = _worker._pseudo_geojson_download_name(
        job,
        [pseudo_result],
        datetime(2026, 7, 21, 11, 54, tzinfo=timezone.utc),
    )

    assert name == "Разрушки_main_segformer_b2_11_54_21_07.geojson"


def test_primary_training_result_switches_for_whole_class(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(tmp_path / "mlmarkup"))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    with session_factory() as session:
        class_row = DatasetClassRow(key="forest", name="Лес", technical_name="forest")
        session.add(class_row)
        session.flush()
        dataset = DatasetRow(
            key="forest-main",
            class_id=class_row.id,
            name="main",
            source_type="mlmarkup",
            source_path="Лес/main",
        )
        first = TrainingResultRow(
            source="manual",
            dataset_key=dataset.key,
            class_key=dataset.key,
            class_display_name="Лес\\main",
            architecture="smp_segformer_b2",
            model_name="первая",
            mlflow_run_id="first-run",
            trained_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            status="ok",
        )
        second = TrainingResultRow(
            source="manual",
            dataset_key=dataset.key,
            class_key=dataset.key,
            class_display_name="Лес\\main",
            architecture="smp_segformer_b2",
            model_name="вторая",
            mlflow_run_id="second-run",
            trained_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
            status="ok",
        )
        sample = _TestSampleRow(
            name="Основная разметка",
            dataset_key=dataset.key,
            dataset_name="Лес\\main",
            class_key=dataset.key,
            class_name="Лес",
            dataset_short_name="main",
            tile_width=768,
            tile_height=768,
            image_count=1,
            requested_object_count=1,
            actual_object_count=1,
            territory_count=1,
            is_primary=True,
            tiles=[
                _TestSampleTileRow(
                    tile_index=1,
                    source_name="tile001.tif",
                    territory="Лес",
                    object_count=1,
                    enabled=True,
                )
            ],
        )
        session.add_all([dataset, first, second, sample])
        session.flush()

        effective = _service.primary_training_result(session, class_row.key)
        unmarked = _service.dataset_results(session, dataset.key, config)
        assert effective is not None
        assert effective.id == second.id
        assert class_row.primary_training_result_id is None
        assert all(item.is_primary is False for item in unmarked.results)

        selected_effective = _service.set_primary_training_result(session, second.id, config)
        assert selected_effective.is_primary is True
        assert class_row.primary_training_result_id == second.id
        assert session.scalars(select(JobRow)).all() == []
        assert session.scalars(select(TrainingResultTestMetricRow)).all() == []

        class_row.primary_training_result_id = first.id
        session.flush()

        before = _service.recalculate_dataset_test_f1(session, dataset.key, config)
        before_by_id = {item.id: item for item in before.results}
        assert before.test_f1_status == "running"
        assert before_by_id[first.id].is_primary is True
        assert before_by_id[second.id].is_primary is False
        assert before_by_id[first.id].test_f1 is not None
        assert before_by_id[first.id].test_f1.status == "queued"
        assert before_by_id[second.id].test_f1 is not None
        assert before_by_id[second.id].test_f1.status == "queued"

        cleared = _service.clear_primary_training_result(session, first.id, config)
        fallback = _service.dataset_results(session, dataset.key, config)
        fallback_by_id = {item.id: item for item in fallback.results}
        assert cleared.is_primary is False
        assert class_row.primary_training_result_id is None
        assert all(item.is_primary is False for item in fallback.results)
        assert _service.primary_training_result(session, class_row.key).id == second.id
        session.refresh(sample)
        fallback_job = session.get(JobRow, sample.evaluation_job_id)
        assert fallback_job is not None
        assert fallback_job.config["training_result_id"] == str(second.id)
        assert fallback_by_id[first.id].is_primary is False
        assert fallback_by_id[second.id].is_primary is False
        with pytest.raises(TrainingUIAPIError, match="не отмечена основной"):
            _service.clear_primary_training_result(session, first.id, config)

        selected = _service.set_primary_training_result(session, second.id, config)
        after = _service.dataset_results(session, dataset.key, config)
        after_by_id = {item.id: item for item in after.results}

        assert selected.is_primary is True
        assert selected.test_f1 is not None
        assert after_by_id[first.id].is_primary is False
        assert after_by_id[second.id].is_primary is True
        assert after_by_id[first.id].test_f1 is not None
        session.refresh(sample)
        direct_job = session.get(JobRow, sample.evaluation_job_id)
        assert sample.metric_status == JobStatus.QUEUED.value
        assert direct_job is not None
        assert direct_job.config["metric_target"] == "test_sample"
        assert direct_job.config["training_result_id"] == str(second.id)

        inference_template = _service.create_inference_template(
            session,
            TrainingTemplateCreate(
                architecture="smp_segformer_b2",
                dataset_key=dataset.key,
            ),
            config,
        )
        session.refresh(sample)
        templated_job = session.get(JobRow, sample.evaluation_job_id)
        assert direct_job.status == JobStatus.CANCELLED.value
        assert templated_job is not None
        assert templated_job.id != direct_job.id
        assert templated_job.config["metric_target"] == "test_sample"
        assert templated_job.config["inference_template_id"] == str(inference_template.id)
        assert templated_job.config["inference_template_version"] == 1

        updated_template = _service.update_inference_template_by_id(
            session,
            inference_template.id,
            TrainingTemplateUpdate(
                default_config={
                    **inference_template.default_config,
                    "postprocess.min_area_m2": 321.0,
                }
            ),
            config,
        )
        session.refresh(sample)
        updated_template_job = session.get(JobRow, sample.evaluation_job_id)
        assert templated_job.status == JobStatus.CANCELLED.value
        assert updated_template_job is not None
        assert updated_template_job.id != templated_job.id
        assert updated_template_job.config["inference_template_id"] == str(updated_template.id)
        assert updated_template_job.config["inference_template_version"] == 2


def test_seed_inference_template_uses_active_dataset_key_after_migration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    with session_factory() as session:
        class_row = DatasetClassRow(key="rivers", name="Реки", technical_name="rivers")
        session.add(class_row)
        session.flush()
        active_key = "f9776773-5273-41f8-8d10-0b06c68b19e6"
        session.add_all(
            [
                DatasetRow(
                    key=active_key,
                    class_id=class_row.id,
                    name="main",
                    source_type="mlmarkup",
                    source_path="Реки/main",
                    legacy_version=False,
                ),
                DatasetRow(
                    key="Реки\\main",
                    class_id=class_row.id,
                    name="main [legacy]",
                    source_type="mlmarkup",
                    source_path="__archive__/Реки/main",
                    deleted_at=datetime(2026, 8, 15, tzinfo=timezone.utc),
                ),
            ]
        )
        session.flush()

        ensure_seed_templates(session)
        session.flush()
        river_templates = session.scalars(
            select(InferenceTemplateRow).where(
                InferenceTemplateRow.architecture == "smp_segformer_b2",
                InferenceTemplateRow.dataset_key.is_not(None),
                InferenceTemplateRow.dataset_name == "Реки\\main",
            )
        ).all()
        assert len(river_templates) == 1
        assert river_templates[0].dataset_key == active_key
        template_id = river_templates[0].id

        ensure_seed_templates(session)
        session.flush()
        river_templates = session.scalars(
            select(InferenceTemplateRow).where(
                InferenceTemplateRow.architecture == "smp_segformer_b2",
                InferenceTemplateRow.dataset_key.is_not(None),
                InferenceTemplateRow.dataset_name == "Реки\\main",
            )
        ).all()
        assert [(row.id, row.dataset_key) for row in river_templates] == [(template_id, active_key)]


def test_seed_inference_template_backfills_defaults_and_preserves_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        ensure_seed_templates(session)
        river = session.scalar(
            select(InferenceTemplateRow).where(
                InferenceTemplateRow.architecture == "smp_segformer_b2",
                InferenceTemplateRow.dataset_key == "Реки\\main",
            )
        )
        assert river is not None
        old_baseline = dict(river.baseline_default_config)
        old_baseline.pop("postprocess.smooth.enabled")
        old_baseline.pop("postprocess.smooth.iterations")
        old_baseline.pop("postprocess.smooth.offset")
        old_baseline.pop("postprocess.filter_compact_objects.mode")
        old_baseline["postprocess.simplify_m"] = 15.0
        current = {
            **old_baseline,
            "postprocess.filter_compact_objects.max_bbox_ratio": 4.25,
        }
        river.baseline_default_config = old_baseline
        river.default_config = current
        session.flush()

        ensure_seed_templates(session)
        session.flush()

        assert river.default_config["postprocess.smooth.enabled"] is True
        assert river.default_config["postprocess.smooth.iterations"] == 1
        assert river.default_config["postprocess.smooth.offset"] == 0.125
        assert river.default_config["postprocess.simplify_m"] == 1.0
        assert river.default_config["postprocess.filter_compact_objects.mode"] == "remove_compact"
        assert river.default_config["postprocess.filter_compact_objects.max_bbox_ratio"] == 4.25
        first_reconciled = dict(river.default_config)

        ensure_seed_templates(session)
        session.flush()

        assert river.default_config == first_reconciled
        assert river.baseline_default_config["postprocess.simplify_m"] == 1.0


def test_seed_training_template_backfills_background_weight_and_preserves_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        ensure_seed_templates(session)
        template = session.scalar(
            select(TrainingTemplateRow).where(
                TrainingTemplateRow.architecture == "smp_segformer_b2",
                TrainingTemplateRow.dataset_key.is_(None),
            )
        )
        assert template is not None
        current = dict(template.default_config)
        baseline = dict(template.baseline_default_config)
        current.pop("train.background_weight")
        baseline.pop("train.background_weight")
        current["train.batch_size"] = 3
        template.default_config = current
        template.baseline_default_config = baseline
        session.flush()

        ensure_seed_templates(session)
        session.flush()

        assert template.default_config["train.background_weight"] == 1.0
        assert template.baseline_default_config["train.background_weight"] == 1.0
        assert template.default_config["train.batch_size"] == 3


def test_result_classes_show_dataset_specific_network_f1(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    primary_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    latest_main_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    latest_test_id = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    failed_id = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
    datasets = [
        SimpleNamespace(
            key="forest-main",
            name="main",
            dataset_name="Лес\\main",
            class_key="forest",
            class_name="Лес",
            quality_metric="pixel",
            is_primary=True,
            image_count=12,
        ),
        SimpleNamespace(
            key="forest-test",
            name="test",
            dataset_name="Лес\\test",
            class_key="forest",
            class_name="Лес",
            quality_metric="pixel",
            is_primary=False,
            image_count=8,
        ),
        SimpleNamespace(
            key="forest-empty",
            name="empty",
            dataset_name="Лес\\empty",
            class_key="forest",
            class_name="Лес",
            quality_metric="pixel",
            is_primary=False,
            image_count=3,
        ),
    ]
    class_info = SimpleNamespace(
        key="forest",
        name="Лес",
        updated_at=None,
        datasets=datasets,
        is_custom=False,
        quality_metric="pixel",
    )
    metrics = {
        primary_id: (
            0.61,
            "current",
            {"pixel": {"per_class": {"forest": {"f1": 0.61}, "scrub": {"f1": 0.42}}}},
        ),
        latest_main_id: (
            0.72,
            "current",
            {"pixel": {"per_class": {"forest": {"f1": 0.72}, "scrub": {"f1": 0.55}}}},
        ),
        latest_test_id: (
            0.83,
            "stale",
            {"pixel": {"per_class": {"forest": {"f1": 0.83}, "scrub": {"f1": 0.74}}}},
        ),
    }
    metric_calls: list[UUID] = []

    monkeypatch.setattr(_service, "list_managed_classes", lambda *_args: [class_info])

    def metric_info(_session, result, _config):
        metric_calls.append(result.id)
        f1, status, per_class = metrics[result.id]
        return SimpleNamespace(f1=f1, status=status, metrics=per_class)

    monkeypatch.setattr(_service, "training_result_test_f1_info", metric_info)

    with session_factory() as session:
        class_row = DatasetClassRow(key="forest", name="Лес", technical_name="forest")
        session.add(class_row)
        session.flush()
        session.add_all(
            [
                DatasetRow(
                    key=dataset.key,
                    class_id=class_row.id,
                    name=dataset.name,
                    source_path=f"Лес/{dataset.name}",
                )
                for dataset in datasets
            ]
        )
        results = [
            TrainingResultRow(
                id=primary_id,
                source="manual",
                dataset_key="forest-main",
                class_key="forest-main",
                class_display_name="Лес\\main",
                architecture="smp_segformer_b2",
                model_name="основная сеть main",
                trained_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
                created_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
                status="ok",
            ),
            TrainingResultRow(
                id=latest_main_id,
                source="manual",
                dataset_key="forest-main",
                class_key="forest-main",
                class_display_name="Лес\\main",
                architecture="smp_segformer_b2",
                model_name="последняя сеть main",
                trained_at=datetime(2026, 8, 3, tzinfo=timezone.utc),
                created_at=datetime(2026, 8, 3, tzinfo=timezone.utc),
                status="ok",
            ),
            TrainingResultRow(
                id=latest_test_id,
                source="manual",
                dataset_key="forest-test-legacy",
                class_key="forest-test",
                class_display_name="Лес\\test",
                architecture="smp_segformer_b2",
                model_name="последняя сеть test",
                trained_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
                created_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
                status="ok",
            ),
            TrainingResultRow(
                id=failed_id,
                source="manual",
                dataset_key="forest-main",
                class_key="forest-main",
                class_display_name="Лес\\main",
                architecture="smp_segformer_b2",
                model_name="неуспешная сеть main",
                trained_at=datetime(2026, 8, 4, tzinfo=timezone.utc),
                created_at=datetime(2026, 8, 4, tzinfo=timezone.utc),
                status="error",
            ),
        ]
        session.add_all(results)
        session.flush()
        class_row.primary_training_result_id = primary_id
        session.flush()

        response = _service.result_classes(session, config)
        cards = response.classes[0].datasets

        assert metric_calls == [primary_id, latest_test_id]
        assert [item.is_primary for item in cards] == [True, False, False]
        assert [item.test_f1 for item in cards] == [0.61, 0.83, None]
        assert [item.test_f1_status for item in cards] == ["current", "stale", None]
        assert [item.test_f1_metrics for item in cards] == [
            {"pixel": {"per_class": {"forest": {"f1": 0.61}, "scrub": {"f1": 0.42}}}},
            {"pixel": {"per_class": {"forest": {"f1": 0.83}, "scrub": {"f1": 0.74}}}},
            {},
        ]
        assert [item.test_f1_training_result_id for item in cards] == [
            primary_id,
            latest_test_id,
            None,
        ]

        class_row.primary_training_result_id = None
        session.flush()
        metric_calls.clear()

        fallback = _service.result_classes(session, config).classes[0].datasets

        assert metric_calls == [latest_main_id, latest_test_id]
        assert [item.test_f1 for item in fallback] == [0.72, 0.83, None]
        assert [item.test_f1_training_result_id for item in fallback] == [
            latest_main_id,
            latest_test_id,
            None,
        ]


def test_successful_training_does_not_assign_primary_star(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    reconciled_class_keys: list[set[str]] = []
    post_training_inference_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        _worker,
        "queue_training_result_test_f1",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        _worker,
        "reconcile_test_sample_evaluations",
        lambda _session, _config, *, class_keys: reconciled_class_keys.append(class_keys) or 0,
    )
    monkeypatch.setattr(
        _service,
        "create_pseudo_markup_job",
        lambda _session, **kwargs: (
            post_training_inference_calls.append(kwargs)
            or SimpleNamespace(id=UUID("00000000-0000-0000-0000-000000000123"))
        ),
    )

    with session_factory() as session:
        class_row = DatasetClassRow(key="forest", name="Лес", technical_name="forest")
        session.add(class_row)
        session.flush()
        dataset = DatasetRow(
            key="forest-main",
            class_id=class_row.id,
            name="main",
            source_type="mlmarkup",
            source_path="Лес/main",
        )
        job = JobRow(
            type=JobType.TRAINING.value,
            source=JobSource.MANUAL.value,
            status=JobStatus.RUNNING.value,
            queue_position=1,
            dataset_key=dataset.key,
            dataset_name="Лес\\main",
            model_name="segformer b2",
            architecture="smp_segformer_b2",
            config={
                "ui.run_inference_after_training": True,
                "ui.secondary_priority": True,
            },
        )
        session.add_all([dataset, job])
        session.flush()
        result = TrainingResultRow(
            source=JobSource.MANUAL.value,
            dataset_key=dataset.key,
            class_key=dataset.key,
            class_display_name="Лес\\main",
            architecture="smp_segformer_b2",
            model_name="segformer b2",
            status=ResultStatus.RUNNING.value,
            job_id=job.id,
        )
        session.add(result)
        session.flush()

        _worker._finish_training_job(session, job, config, succeeded=True)

        assert result.status == ResultStatus.OK.value
        assert class_row.primary_training_result_id is None
        assert _service.primary_training_result(session, class_row.key).id == result.id
        assert reconciled_class_keys == [{class_row.key}]
        assert len(post_training_inference_calls) == 1
        assert post_training_inference_calls[0]["dataset_key"] == dataset.key
        assert post_training_inference_calls[0]["training_result_id"] == result.id
        assert post_training_inference_calls[0]["secondary_priority"] is True
        assert job.config["ui.post_training_inference_job_ids"] == [
            "00000000-0000-0000-0000-000000000123"
        ]


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
        class_progress = _service.dataset_results(session, "class-key", config).results[0].progress

    assert queue_progress is not None
    assert queue_progress.current == 23
    assert queue_progress.total == 64
    assert queue_progress.elapsed_minutes == 16
    assert class_progress is not None
    assert class_progress.current == 23
    assert class_progress.elapsed_minutes == 16


def test_class_results_returns_failed_training_dates(
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

    with session_factory() as session:
        job = _queue_test_job(JobType.TRAINING, JobSource.MANUAL, 1, created_at)
        job.status = JobStatus.FAILED.value
        job.started_at = started_at
        job.finished_at = started_at + timedelta(minutes=30)
        job.dataset_key = "class-key"
        job.error = "RuntimeError: не удалось подготовить датасет рек"
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
                status=ResultStatus.ERROR.value,
                job_id=job.id,
                created_at=created_at,
            )
        )
        session.flush()

        result = _service.dataset_results(session, "class-key", config).results[0]

    assert result.created_at.replace(tzinfo=timezone.utc) == created_at
    assert result.started_at == started_at
    assert result.trained_at is None
    assert result.error == "RuntimeError: не удалось подготовить датасет рек"


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
        job = _queue_test_job(
            JobType.TRAINING, JobSource.MANUAL, 1, datetime(2026, 6, 10, tzinfo=timezone.utc)
        )
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
            _service.dataset_results(session, "class-key", config)
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

        response = _service.dataset_results(session, "class-key", config)

    training_result = next(item for item in response.results if item.id == queued_training.id)
    completed_result = next(item for item in response.results if item.id == completed_training.id)
    pseudo_result = completed_result.pseudo_markup_results[0]
    assert training_result.status == "queued"
    assert training_result.job_id == training_job.id
    assert pseudo_result.status == "queued"
    assert pseudo_result.job_id == pseudo_job.id


def test_class_results_uses_stored_pseudo_image_count_for_legacy_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(tmp_path / "MLMarkup"))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    scenes_path = tmp_path / "legacy-scenes.txt"
    scenes_path.write_text("scene-1\n", encoding="utf-8")
    geojson_path = tmp_path / "legacy.geojson"
    geojson_path.write_text("not json", encoding="utf-8")

    def fail_runtime_image_count(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("legacy class_results must not recalculate image_count")

    monkeypatch.setattr(_service, "count_scenes_file_images", fail_runtime_image_count)

    with session_factory() as session:
        scenes_row = StoredFileRow(
            kind=StoredFileKind.SCENES_TXT.value,
            original_name="legacy-scenes.txt",
            path=str(scenes_path),
            size_bytes=scenes_path.stat().st_size,
        )
        geojson_row = StoredFileRow(
            kind=StoredFileKind.PSEUDO_MARKUP_GEOJSON.value,
            original_name="legacy.geojson",
            path=str(geojson_path),
            size_bytes=geojson_path.stat().st_size,
            object_count=None,
        )
        training_result = TrainingResultRow(
            source=JobSource.MANUAL.value,
            dataset_key="class-key",
            class_key="class-key",
            class_display_name="class",
            architecture="smp_segformer_b2",
            model_name="segformer b2",
            status=ResultStatus.OK.value,
        )
        session.add_all([scenes_row, geojson_row, training_result])
        session.flush()
        session.add(
            PseudoMarkupResultRow(
                source=JobSource.MANUAL.value,
                dataset_key="class-key",
                training_result_id=training_result.id,
                class_key="class-key",
                source_dataset_name="source",
                scenes_file_id=scenes_row.id,
                geojson_file_id=geojson_row.id,
                image_count=None,
                status=ResultStatus.OK.value,
            )
        )
        session.flush()

        response = _service.dataset_results(session, "class-key", config)

    assert response.results[0].pseudo_markup_results[0].image_count is None
    assert response.results[0].pseudo_markup_results[0].geojson_file is not None
    assert response.results[0].pseudo_markup_results[0].geojson_file.object_count is None


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
        job = _queue_test_job(
            JobType.TRAINING, JobSource.MANUAL, 1, datetime(2026, 6, 10, tzinfo=timezone.utc)
        )
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


def test_training_ui_job_log_falls_back_to_journalctl(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_JOURNAL_UNIT", "test-training-ui.service")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        job = _queue_test_job(
            JobType.TRAINING, JobSource.MANUAL, 1, datetime(2026, 6, 10, tzinfo=timezone.utc)
        )
        job.status = JobStatus.FAILED.value
        job.finished_at = job.created_at + timedelta(minutes=1)
        session.add(job)
        session.flush()
        run_dir = config.scratch_root / "jobs" / str(job.id)
        run_dir.mkdir(parents=True)
        (run_dir / "worker_error.txt").write_text(
            "Не удалось запустить обучение. Подробности в journalctl.",
            encoding="utf-8",
        )
        job.tmp_path = str(run_dir)
        job_id = job.id
        session.flush()

        def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
            assert "test-training-ui.service" in args[0]
            assert kwargs["timeout"] == 10
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    f"2026-06-10T00:00:00 service[1]: Failed to start training job {job_id}\n"
                    "2026-06-10T00:00:00 service[1]: Traceback (most recent call last):\n"
                    "2026-06-10T00:00:00 service[1]: RuntimeError: dataset is incomplete\n"
                    "2026-06-10T00:00:01 service[1]: INFO: unrelated request\n"
                ),
            )

        monkeypatch.setattr(_service.subprocess, "run", fake_run)
        log = _service.job_log(session, job_id, config)

    assert log.source_name == "journalctl:test-training-ui.service"
    assert "Failed to start training job" in log.content
    assert "dataset is incomplete" in log.content
    assert "unrelated request" not in log.content


def test_training_ui_job_log_reports_missing_when_journal_has_no_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        job = _queue_test_job(
            JobType.TRAINING, JobSource.MANUAL, 1, datetime(2026, 6, 10, tzinfo=timezone.utc)
        )
        job.status = JobStatus.FAILED.value
        session.add(job)
        session.flush()
        job_id = job.id

        monkeypatch.setattr(
            _service.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="no matching lines\n"),
        )
        with pytest.raises(TrainingUIAPIError, match="Лог задания не найден"):
            _service.job_log(session, job_id, config)


def test_training_ui_job_log_uses_persisted_error_when_runtime_is_gone(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_JOURNAL_UNIT", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        job = _queue_test_job(
            JobType.TRAINING, JobSource.MANUAL, 1, datetime(2026, 6, 10, tzinfo=timezone.utc)
        )
        job.status = JobStatus.FAILED.value
        job.error = "TrainPipelineError: процесс обучения был прерван"
        session.add(job)
        session.flush()

        log = _service.job_log(session, job.id, config)

    assert log.source_name == "сохранённая ошибка"
    assert log.content == "TrainPipelineError: процесс обучения был прерван"
    assert log.truncated is False


def test_training_worker_reads_failed_training_log_for_persistence(tmp_path: Path) -> None:
    job = _queue_test_job(
        JobType.TRAINING,
        JobSource.MANUAL,
        1,
        datetime(2026, 6, 10, tzinfo=timezone.utc),
    )
    run_dir = tmp_path / "job"
    run_dir.mkdir()
    (run_dir / "train.log").write_text(
        "Traceback (most recent call last):\nRuntimeError: недостаточно памяти GPU\n",
        encoding="utf-8",
    )
    job.tmp_path = str(run_dir)

    assert _worker._training_job_error(job).endswith("RuntimeError: недостаточно памяти GPU")


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


def test_training_worker_preempts_and_resumes_training_for_urgent_inference(
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
    started: list[UUID] = []

    def fake_start_inference(session, row, config, *, popen_factory) -> None:
        del session, config, popen_factory
        started.append(row.id)
        row.status = JobStatus.RUNNING.value
        row.process_pid = 5678

    monkeypatch.setattr(_worker, "_start_inference_job", fake_start_inference)
    monkeypatch.setattr(_worker, "_pid_is_alive", lambda _pid: True)
    created_at = datetime(2026, 8, 13, tzinfo=timezone.utc)

    with session_factory() as session:
        training = _queue_test_job(JobType.TRAINING, JobSource.MANUAL, 1, created_at)
        training.status = JobStatus.RUNNING.value
        training.process_pid = 4321
        training.tmp_path = str(tmp_path / "training")
        Path(training.tmp_path).mkdir(parents=True)
        urgent = _queue_test_job(JobType.INFERENCE, JobSource.MANUAL, 2, created_at)
        urgent.config = {"priority": "urgent"}
        session.add_all([training, urgent])
        session.flush()

        dispatch_queue_once(session, config)
        request_path = Path(training.tmp_path) / "control" / "pause.request"
        marker_path = Path(training.tmp_path) / "control" / "paused"
        assert request_path.is_file()
        assert started == []
        assert training.status == JobStatus.RUNNING.value

        marker_path.write_text(request_path.read_text(encoding="utf-8"), encoding="utf-8")
        dispatch_queue_once(session, config)
        assert training.status == JobStatus.PAUSED.value
        assert training.process_pid == 4321
        assert started == [urgent.id]

        urgent.status = JobStatus.COMPLETED.value
        urgent.process_pid = None
        session.flush()
        dispatch_queue_once(session, config)
        assert not request_path.exists()
        assert training.status == JobStatus.PAUSED.value

        marker_path.unlink()
        dispatch_queue_once(session, config)
        assert training.status == JobStatus.RUNNING.value
        assert training.process_pid == 4321


def test_legacy_f1_inference_starts_before_queued_manual_training(
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
    started: list[UUID] = []

    def fake_start(session, row, config, *, popen_factory) -> None:
        del session, config, popen_factory
        started.append(row.id)
        row.status = JobStatus.RUNNING.value

    monkeypatch.setattr(_worker, "_start_training_job", fake_start)
    monkeypatch.setattr(_worker, "_start_inference_job", fake_start)
    created_at = datetime(2026, 8, 25, tzinfo=timezone.utc)

    with session_factory() as session:
        training = _queue_test_job(JobType.TRAINING, JobSource.MANUAL, 20_001, created_at)
        f1 = _queue_test_job(JobType.INFERENCE, JobSource.AUTOMATION, 30_001, created_at)
        f1.config = {"operation": "test_sample_f1"}
        session.add_all([training, f1])
        session.flush()

        dispatch_queue_once(session, config)

        assert started == [f1.id]
        assert training.status == JobStatus.QUEUED.value


def test_secondary_job_waits_for_regular_job_regardless_of_queue_block(
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
    started: list[UUID] = []

    def fake_start(session, row, config, *, popen_factory) -> None:
        del session, config, popen_factory
        started.append(row.id)
        row.status = JobStatus.RUNNING.value

    monkeypatch.setattr(_worker, "_start_training_job", fake_start)
    monkeypatch.setattr(_worker, "_start_inference_job", fake_start)
    created_at = datetime(2026, 8, 23, tzinfo=timezone.utc)

    with session_factory() as session:
        secondary = _queue_test_job(JobType.INFERENCE, JobSource.MANUAL, 1, created_at)
        secondary.config = {"ui.secondary_priority": True}
        regular = _queue_test_job(JobType.TRAINING, JobSource.MANUAL, 99, created_at)
        session.add_all([secondary, regular])
        session.flush()

        dispatch_queue_once(session, config)

        assert started == [regular.id]
        assert secondary.status == JobStatus.QUEUED.value


def test_secondary_training_pauses_for_regular_job_and_resumes(
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
    started: list[UUID] = []

    def fake_start_inference(session, row, config, *, popen_factory) -> None:
        del session, config, popen_factory
        started.append(row.id)
        row.status = JobStatus.RUNNING.value
        row.process_pid = 5678

    monkeypatch.setattr(_worker, "_start_inference_job", fake_start_inference)
    monkeypatch.setattr(_worker, "_pid_is_alive", lambda _pid: True)
    created_at = datetime(2026, 8, 23, tzinfo=timezone.utc)

    with session_factory() as session:
        secondary = _queue_test_job(JobType.TRAINING, JobSource.MANUAL, 1, created_at)
        secondary.config = {"ui.secondary_priority": True}
        secondary.status = JobStatus.RUNNING.value
        secondary.process_pid = 4321
        secondary.tmp_path = str(tmp_path / "secondary-training")
        Path(secondary.tmp_path).mkdir(parents=True)
        regular = _queue_test_job(JobType.INFERENCE, JobSource.MANUAL, 2, created_at)
        session.add_all([secondary, regular])
        session.flush()

        dispatch_queue_once(session, config)
        request_path = Path(secondary.tmp_path) / "control" / "pause.request"
        marker_path = Path(secondary.tmp_path) / "control" / "paused"
        assert request_path.is_file()
        assert started == []

        regular.status = JobStatus.CANCELLED.value
        session.flush()
        dispatch_queue_once(session, config)
        assert not request_path.exists()
        assert secondary.status == JobStatus.RUNNING.value

        regular.status = JobStatus.QUEUED.value
        session.flush()
        dispatch_queue_once(session, config)
        assert request_path.is_file()

        marker_path.write_text(request_path.read_text(encoding="utf-8"), encoding="utf-8")
        dispatch_queue_once(session, config)
        assert secondary.status == JobStatus.PAUSED.value
        assert started == [regular.id]

        regular.status = JobStatus.COMPLETED.value
        regular.process_pid = None
        session.flush()
        dispatch_queue_once(session, config)
        assert not request_path.exists()
        assert secondary.status == JobStatus.PAUSED.value

        marker_path.unlink()
        dispatch_queue_once(session, config)
        assert secondary.status == JobStatus.RUNNING.value
        assert secondary.process_pid == 4321


def test_secondary_inference_pauses_for_regular_training_and_resumes(
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
    started: list[UUID] = []

    def fake_start_training(session, row, config, *, popen_factory) -> None:
        del session, config, popen_factory
        started.append(row.id)
        row.status = JobStatus.RUNNING.value
        row.process_pid = 8765

    monkeypatch.setattr(_worker, "_start_training_job", fake_start_training)
    monkeypatch.setattr(_worker, "_pid_is_alive", lambda _pid: True)
    created_at = datetime(2026, 8, 23, tzinfo=timezone.utc)

    with session_factory() as session:
        secondary = _queue_test_job(JobType.INFERENCE, JobSource.MANUAL, 1, created_at)
        secondary.config = {"ui.secondary_priority": True}
        secondary.status = JobStatus.RUNNING.value
        secondary.process_pid = 4321
        secondary.tmp_path = str(tmp_path / "secondary-inference")
        Path(secondary.tmp_path).mkdir(parents=True)
        regular = _queue_test_job(JobType.TRAINING, JobSource.MANUAL, 2, created_at)
        session.add_all([secondary, regular])
        session.flush()

        dispatch_queue_once(session, config)
        request_path = Path(secondary.tmp_path) / "control" / "pause.request"
        marker_path = Path(secondary.tmp_path) / "control" / "paused"
        assert request_path.is_file()
        assert started == []

        marker_path.write_text(request_path.read_text(encoding="utf-8"), encoding="utf-8")
        dispatch_queue_once(session, config)
        assert secondary.status == JobStatus.PAUSED.value
        assert started == [regular.id]

        regular.status = JobStatus.COMPLETED.value
        regular.process_pid = None
        session.flush()
        dispatch_queue_once(session, config)
        assert not request_path.exists()

        regular.status = JobStatus.QUEUED.value
        session.flush()
        dispatch_queue_once(session, config)
        assert request_path.is_file()
        assert started == [regular.id]

        marker_path.write_text(request_path.read_text(encoding="utf-8"), encoding="utf-8")
        dispatch_queue_once(session, config)
        assert started == [regular.id, regular.id]

        regular.status = JobStatus.COMPLETED.value
        regular.process_pid = None
        session.flush()
        dispatch_queue_once(session, config)
        assert not request_path.exists()

        marker_path.unlink()
        dispatch_queue_once(session, config)
        assert secondary.status == JobStatus.RUNNING.value
        assert secondary.process_pid == 4321


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


def test_stored_file_object_count_is_nullable_integer() -> None:
    from mlsystem2.training_ui_api._models import StoredFileRow

    column = StoredFileRow.__table__.columns["object_count"]
    assert isinstance(column.type, Integer)
    assert column.nullable is True


def test_pseudo_geojson_object_count_prefers_report_and_falls_back_to_geojson(
    tmp_path: Path,
) -> None:
    geojson_path = tmp_path / "pseudo.geojson"
    geojson_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "geometry": None, "properties": {}},
                    {"type": "Feature", "geometry": None, "properties": {}},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert _worker._pseudo_geojson_object_count(geojson_path, {"feature_count": 500}) == 500
    assert _worker._pseudo_geojson_object_count(geojson_path, {}) == 2


def test_model_export_zip_layout_config_and_pipeline(tmp_path: Path, monkeypatch) -> None:
    def fake_load_binary_checkpoint(path: Path) -> SimpleNamespace:
        assert path.name == "checkpoint.pt"
        return SimpleNamespace(
            model=SimpleNamespace(
                spec=SimpleNamespace(input_channels=4, output_channels=1),
                model=object(),
            ),
            artifact=SimpleNamespace(
                metadata={
                    "val_best_threshold": 0.73,
                    "sample_size": 768,
                    "inference_context": 128,
                    "inference_core_size": 512,
                }
            ),
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
        config = (service_extract_dir / "deforestation-b2" / "config.pbtxt").read_text(
            encoding="utf-8"
        )
        assert 'name: "deforestation-b2"' in config
        assert "dims: [ 1, 4, -1, -1 ]" in config
        assert "dims: [ -1, 1, -1, -1 ]" in config
        assert "dims: [ 1, 1, -1, -1 ]" not in config
        assert "KIND_CPU" in config
        assert "KIND_GPU" not in config
        pipeline = (extract_dir / "pipelines" / "deforestation-b2_triton.yaml").read_text(
            encoding="utf-8"
        )
        assert 'name: "deforestation-b2"' in pipeline
        assert "bounds: 128" in pipeline
        assert "sample_size:\n        - 512\n        - 512" in pipeline
        pipeline_config = yaml.safe_load(pipeline)["config"]["bricks"]
        assert pipeline_config[0]["output"] == ["RED", "GRN", "BLU", "NIR"]
        assert pipeline_config[1]["input_rasters"] == ["RED", "GRN", "BLU", "NIR"]
        metadata = json.loads((extract_dir / "export_metadata.json").read_text(encoding="utf-8"))
        assert metadata["threshold"] == 0.73
        assert metadata["threshold_source"] == "checkpoint_metadata"
        assert metadata["sample_size"] == 768
        assert metadata["sample_size_source"] == "checkpoint_metadata"
        assert metadata["inference_context"] == 128
        assert metadata["inference_context_source"] == "checkpoint_metadata"
        assert metadata["inference_core_size"] == 512
        assert metadata["model_archive"] == "models-serving-service/deforestation-b2.zip"
        assert metadata["pipeline"] == "pipelines/deforestation-b2_triton.yaml"
        assert metadata["onnx_opset"] == 17
        assert metadata["onnx_ir_version"] == 8
        assert metadata["postprocess_config"] == {}
        assert metadata["postprocess_config_sha256"] == hashlib.sha256(b"{}").hexdigest()
        assert "FilterCompactObjects" not in pipeline
        assert "Smooth" not in pipeline
    finally:
        archive.cleanup()


def test_model_export_pipeline_uses_rgb_for_three_channel_model() -> None:
    pipeline = yaml.safe_load(_model_export._pipeline_yaml("ortho-model", 512, 3))
    transforms = pipeline["config"]["bricks"]

    assert transforms[0]["output"] == ["RED", "GRN", "BLU"]
    assert transforms[1]["input_rasters"] == ["RED", "GRN", "BLU"]


def test_model_export_mask_postprocessing_never_overwrites_an_input_mask() -> None:
    pipeline = yaml.safe_load(
        _model_export._pipeline_yaml(
            "damaged-oks",
            768,
            3,
            context=128,
            postprocess_config={
                "postprocess.mask_min_object_pixels": 32,
                "postprocess.mask_min_hole_pixels": 32,
                "postprocess.binary_closing_radius": 2,
                "postprocess.min_area_m2": 20.0,
            },
        )
    )
    bricks = pipeline["config"]["bricks"]
    segmentation = bricks[1]
    morphology = [brick for brick in bricks if brick["_class"] == "MaskMorphology"]
    vectorize = next(brick for brick in bricks if brick["_class"] == "VectorizeMasks")

    assert segmentation["output_labels"] == ["mlsystem2_raw_1"]
    assert len(morphology) == 3
    assert morphology[0]["input_masks"] == ["mlsystem2_raw_1"]
    assert morphology[-1]["out_masks"] == ["mask"]
    assert vectorize["input_rasters"] == ["mask"]
    assert vectorize["output_fcs"] == ["output"]
    vector_postprocess = bricks[-1]
    assert vector_postprocess["_class"] == "UnifiedVectorProcessing"
    assert vector_postprocess["input"] == "output"
    assert vector_postprocess["output"] == "output"
    for brick in morphology:
        assert set(brick["input_masks"]).isdisjoint(brick["out_masks"])


def test_running_training_can_stop_and_publish_existing_best_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "MLSYSTEM2_TRAINING_UI_DATABASE_URL",
        f"sqlite:///{tmp_path / 'ui.db'}",
    )
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")
    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        job = _queue_test_job(
            JobType.TRAINING,
            JobSource.MANUAL,
            1,
            datetime(2026, 8, 24, tzinfo=timezone.utc),
        )
        job.status = JobStatus.RUNNING.value
        job.process_pid = 4321
        job.tmp_path = str(tmp_path / "job")
        best_path = Path(job.tmp_path) / "scratch" / "checkpoints" / "best.pt"
        best_path.parent.mkdir(parents=True)
        best_path.write_bytes(b"best-f1-checkpoint")
        session.add(job)
        session.flush()

        before = _service._job_summary(session, job)
        assert before.best_checkpoint_available is True
        assert "stop_and_save_best" in before.actions

        detail = _service.stop_training_job_and_save_best(session, job.id)
        request_path = (
            Path(job.tmp_path) / _service.JOB_CONTROL_DIR / _service.STOP_AND_SAVE_BEST_REQUEST_FILE
        )

        assert request_path.is_file()
        assert detail.status == JobStatus.RUNNING
        assert detail.stop_and_save_best_requested is True
        assert detail.best_checkpoint_available is True
        assert job.process_pid == 4321
        assert "stop_and_save_best" not in _service._job_summary(session, job).actions

        repeated = _service.stop_training_job_and_save_best(session, job.id)
        assert repeated.stop_and_save_best_requested is True


def test_training_cannot_stop_with_result_before_first_best_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "MLSYSTEM2_TRAINING_UI_DATABASE_URL",
        f"sqlite:///{tmp_path / 'ui.db'}",
    )
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")
    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        job = _queue_test_job(
            JobType.TRAINING,
            JobSource.MANUAL,
            1,
            datetime(2026, 8, 24, tzinfo=timezone.utc),
        )
        job.status = JobStatus.RUNNING.value
        job.tmp_path = str(tmp_path / "job")
        session.add(job)
        session.flush()

        with pytest.raises(TrainingUIAPIError, match="ещё не создан"):
            _service.stop_training_job_and_save_best(session, job.id)

        assert not (
            Path(job.tmp_path) / _service.JOB_CONTROL_DIR / _service.STOP_AND_SAVE_BEST_REQUEST_FILE
        ).exists()


def test_multiclass_export_can_replace_only_semantic_class_identifiers() -> None:
    checkpoint_schema = [
        {
            "id": 1,
            "slug": "type_legacy",
            "name": "Переувлажнение",
            "color": "#112233",
            "priority": 10,
        }
    ]
    canonical_schema = [
        {
            "id": 1,
            "slug": "floodings",
            "name": "Переувлажнение",
            "color": "#AABBCC",
            "priority": 20,
        }
    ]

    assert (
        _model_export._export_class_schema_override(
            "multiclass",
            checkpoint_schema,
            canonical_schema,
        )
        == canonical_schema
    )
    with pytest.raises(TrainingUIAPIError, match="назначение каналов"):
        _model_export._export_class_schema_override(
            "multiclass",
            checkpoint_schema,
            [{**canonical_schema[0], "name": "Другой класс"}],
        )


def test_model_export_pipeline_rejects_unsupported_channel_count() -> None:
    with pytest.raises(TrainingUIAPIError, match="только 3- и 4-канальные"):
        _model_export._pipeline_yaml("bad-model", 512, 2)


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
    postprocess_config = {
        "postprocess.smooth.offset": 0.125,
        "postprocess.smooth.enabled": True,
        "postprocess.smooth.iterations": 1,
    }

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
        context=128,
        postprocess_config=postprocess_config,
    )
    try:
        assert captured == [(0.73, 768)]
        with zipfile.ZipFile(archive.zip_path) as zip_file:
            metadata = json.loads(zip_file.read("export_metadata.json").decode("utf-8"))
            pipeline = zip_file.read("pipelines/erosion-b2_triton.yaml").decode("utf-8")
        assert metadata["threshold"] == 0.73
        assert metadata["threshold_source"] == "checkpoint_metadata"
        assert metadata["sample_size"] == 768
        assert metadata["sample_size_source"] == "request"
        assert metadata["inference_context"] == 128
        assert metadata["inference_context_source"] == "request"
        assert metadata["inference_core_size"] == 512
        normalized_postprocess = {
            key: postprocess_config[key] for key in sorted(postprocess_config)
        }
        normalized_json = json.dumps(
            normalized_postprocess,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        assert metadata["postprocess_config"] == normalized_postprocess
        assert metadata["postprocess_config_sha256"] == hashlib.sha256(normalized_json).hexdigest()
        assert "bounds: 128" in pipeline
        assert "_class: Smooth" in pipeline
        assert "sample_size: [" not in pipeline
        assert "sample_size:\n        - 512\n        - 512" in pipeline
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
    assert response.headers["content-disposition"].endswith(
        'filename="deforestation-b2_export.zip"'
    )
    assert response.content.startswith(b"PK")


def test_training_result_model_export_api_downloads_best_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
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
        ensure_seed_templates(session)
        river_template = session.scalar(
            select(InferenceTemplateRow).where(
                InferenceTemplateRow.architecture == "smp_segformer_b2",
                InferenceTemplateRow.dataset_key == "Реки\\main",
            )
        )
        assert river_template is not None
        river_config = dict(river_template.default_config)
        river_config["postprocess.smooth.offset"] = 0.2
        river_template.default_config = river_config
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
        postprocess_config = kwargs["postprocess_config"]
        assert isinstance(postprocess_config, dict)
        assert postprocess_config["postprocess.filter_compact_objects.enabled"] is True
        assert (
            postprocess_config["postprocess.filter_compact_objects.min_isoperimetric_quotient"]
            == 0.25
        )
        assert postprocess_config["postprocess.filter_compact_objects.max_bbox_ratio"] == 3.5
        assert postprocess_config["postprocess.smooth.enabled"] is True
        assert postprocess_config["postprocess.smooth.iterations"] == 1
        assert postprocess_config["postprocess.smooth.offset"] == 0.2
        assert postprocess_config["postprocess.simplify_m"] == 1.0
        assert "postprocess.filter_compact_objects.max_area_m2" not in postprocess_config
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


def test_training_results_batch_model_export_api_returns_flat_zip(
    tmp_path: Path, monkeypatch
) -> None:
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
        first = TrainingResultRow(
            source=JobSource.MANUAL.value,
            dataset_key="Реки\\main",
            class_key="Реки\\main",
            class_display_name="Реки\\main",
            architecture="smp_segformer_b2",
            model_name="segformer b2",
            status=ResultStatus.OK.value,
            mlflow_run_id="run-rivers",
        )
        second = TrainingResultRow(
            source=JobSource.MANUAL.value,
            dataset_key="Вырубки\\main",
            class_key="Вырубки\\main",
            class_display_name="Вырубки\\main",
            architecture="smp_segformer_b2",
            model_name="segformer b2",
            status=ResultStatus.OK.value,
            mlflow_run_id="run-deforest",
        )
        session.add_all([first, second])
        session.commit()
        first_id = first.id
        second_id = second.id

    download_calls: list[str] = []
    build_calls: list[dict[str, object]] = []

    def fake_download_run_artifact(**kwargs: object) -> SimpleNamespace:
        assert kwargs["tracking_uri"] == "http://mlflow.local"
        assert kwargs["artifact_path"] == "checkpoints/best.pt"
        download_calls.append(str(kwargs["run_id"]))
        checkpoint_path = Path(kwargs["dst_dir"]) / "best.pt"
        checkpoint_path.write_bytes(b"checkpoint")
        return SimpleNamespace(local_path=str(checkpoint_path))

    def fake_build_zip(**kwargs: object) -> SimpleNamespace:
        build_calls.append(dict(kwargs))
        model_name = str(kwargs["model_name"])
        zip_path = tmp_path / f"{model_name}_export.zip"
        with zipfile.ZipFile(zip_path, "w") as zip_file:
            zip_file.writestr(f"models-serving-service/{model_name}.zip", b"service-zip")
            zip_file.writestr(f"pipelines/{model_name}_triton.yaml", "pipeline")
            zip_file.writestr("export_metadata.json", json.dumps({"model_name": model_name}))
        return SimpleNamespace(
            zip_path=zip_path,
            filename=f"{model_name}_export.zip",
            cleanup=lambda: None,
        )

    monkeypatch.setattr(_service, "download_run_artifact", fake_download_run_artifact)
    monkeypatch.setattr(_service, "build_triton_model_export_zip", fake_build_zip)

    with TestClient(create_app()) as client:
        assert (
            client.post("/api/v1/results/training/triton-zip", json={"items": []}).status_code
            == 401
        )
        login = client.post("/api/v1/auth/login", json={"username": "mluser", "password": "secret"})
        assert login.status_code == 200
        response = client.post(
            "/api/v1/results/training/triton-zip",
            json={
                "items": [
                    {
                        "result_id": str(first_id),
                        "model_name": "rivers_kanopus",
                        "sample_size": 512,
                    },
                    {
                        "result_id": str(second_id),
                        "model_name": "deforest_kanopus",
                        "sample_size": None,
                    },
                ]
            },
        )
    assert response.status_code == 200
    assert response.headers["content-disposition"].endswith('filename="models_export.zip"')
    assert download_calls == ["run-rivers", "run-deforest"]
    assert [item["model_name"] for item in build_calls] == ["rivers_kanopus", "deforest_kanopus"]
    assert build_calls[0]["sample_size"] == 512
    assert build_calls[1]["sample_size"] is None
    river_postprocess = build_calls[0]["postprocess_config"]
    default_postprocess = build_calls[1]["postprocess_config"]
    assert isinstance(river_postprocess, dict)
    assert isinstance(default_postprocess, dict)
    assert river_postprocess["postprocess.filter_compact_objects.enabled"] is True
    assert river_postprocess["postprocess.smooth.enabled"] is True
    assert river_postprocess["postprocess.smooth.offset"] == 0.125
    assert river_postprocess["postprocess.simplify_m"] == 1.0
    assert default_postprocess["postprocess.filter_compact_objects.enabled"] is False
    assert default_postprocess["postprocess.smooth.enabled"] is False

    batch_zip_path = tmp_path / "models_export.zip"
    batch_zip_path.write_bytes(response.content)
    with zipfile.ZipFile(batch_zip_path) as zip_file:
        names = set(zip_file.namelist())
        assert "models-serving-service/rivers_kanopus.zip" in names
        assert "models-serving-service/deforest_kanopus.zip" in names
        assert "pipelines/rivers_kanopus_triton.yaml" in names
        assert "metadata/rivers_kanopus_export_metadata.json" in names
        metadata = json.loads(zip_file.read("export_metadata.json").decode("utf-8"))
    assert [item["model_name"] for item in metadata["models"]] == [
        "rivers_kanopus",
        "deforest_kanopus",
    ]


def test_training_results_batch_export_supports_native_and_external_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLFLOW_TRACKING_URI", "http://mlflow.local")
    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    manifest = {
        "version": 1,
        "adapter": "detectron2_instances",
        "artifact_path": "models/model.zip",
        "archive_sha256": "2" * 64,
        "model_member": "zu/1/model.pt",
        "model_root": "zu",
        "input_channels": 3,
        "target_resolution_m": 0.15,
        "tile_size": 1884,
        "stride": 628,
        "context": 628,
        "score_threshold": 0.0,
        "min_area_m2": 50.0,
        "min_hole_area_m2": 10.0,
        "nms_iou_threshold": 0.75,
        "nms_relative_intersection": 0.75,
        "max_shift_m": None,
        "shift_iterations": None,
        "shift_confidence": None,
        "correction_confidence": None,
    }
    with session_factory() as session:
        source_job = JobRow(
            type=JobType.TRAINING.value,
            source=JobSource.MANUAL.value,
            status=JobStatus.COMPLETED.value,
            queue_position=0,
            dataset_name="ЗУ500\\main",
            model_name="zu500_orto",
            architecture="external_torchscript",
            config={"external_model": manifest},
        )
        session.add(source_job)
        session.flush()
        native = TrainingResultRow(
            source=JobSource.MANUAL.value,
            class_key="Реки\\main",
            class_display_name="Реки\\main",
            architecture="smp_segformer_b2",
            model_name="segformer b2",
            status=ResultStatus.OK.value,
            mlflow_run_id="run-native",
        )
        external = TrainingResultRow(
            source=JobSource.MANUAL.value,
            class_key="ЗУ500\\main",
            class_display_name="ЗУ500\\main",
            architecture="external_torchscript",
            model_name="zu500_orto",
            status=ResultStatus.OK.value,
            mlflow_run_id="run-external",
            job_id=source_job.id,
        )
        session.add_all([native, external])
        session.commit()
        native_id = native.id
        external_id = external.id

    downloads: list[tuple[str, str]] = []
    builders: list[tuple[str, str]] = []

    def fake_download(**kwargs: object) -> SimpleNamespace:
        artifact_path = str(kwargs["artifact_path"])
        downloads.append((str(kwargs["run_id"]), artifact_path))
        path = Path(str(kwargs["dst_dir"])) / Path(artifact_path).name
        path.write_bytes(b"artifact")
        return SimpleNamespace(local_path=str(path))

    def archive_for(model_name: str, kind: str) -> SimpleNamespace:
        builders.append((model_name, kind))
        path = tmp_path / f"{model_name}-{kind}.zip"
        with zipfile.ZipFile(path, "w") as zip_file:
            zip_file.writestr(f"models-serving-service/{model_name}.zip", b"model")
            zip_file.writestr(f"pipelines/{model_name}_triton.yaml", "pipeline")
            zip_file.writestr("export_metadata.json", json.dumps({"kind": kind}))
        return SimpleNamespace(
            zip_path=path,
            filename=path.name,
            cleanup=lambda: None,
        )

    monkeypatch.setattr(_service, "download_run_artifact", fake_download)
    monkeypatch.setattr(
        _service,
        "build_triton_model_export_zip",
        lambda **kwargs: archive_for(str(kwargs["model_name"]), "native"),
    )

    def fake_external_builder(**kwargs: object) -> SimpleNamespace:
        assert kwargs["manifest"].archive_sha256 == "2" * 64
        return archive_for(str(kwargs["model_name"]), "external")

    monkeypatch.setattr(
        _service,
        "build_external_triton_model_export_zip",
        fake_external_builder,
    )
    with session_factory() as session:
        archive = _service.export_training_results_triton_zip(
            session,
            request=TrainingResultBatchExportRequest(
                items=[
                    TrainingResultExportItem(result_id=native_id, model_name="native_model"),
                    TrainingResultExportItem(
                        result_id=external_id,
                        model_name="external_model",
                        sample_size=512,
                    ),
                ]
            ),
            config=config,
        )
    try:
        assert downloads == [
            ("run-native", "checkpoints/best.pt"),
            ("run-external", "models/model.zip"),
        ]
        assert builders == [("native_model", "native"), ("external_model", "external")]
        with zipfile.ZipFile(archive.zip_path) as zip_file:
            assert "models-serving-service/native_model.zip" in zip_file.namelist()
            assert "models-serving-service/external_model.zip" in zip_file.namelist()
    finally:
        archive.cleanup()


def test_model_export_old_checkpoint_defaults_to_zero_context() -> None:
    assert _model_export._context_from_metadata_or_request({}, None) == (
        0,
        "legacy_default",
    )


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
        with pytest.raises(TrainingUIAPIError, match="Имена моделей"):
            _service.export_training_results_triton_zip(
                session,
                request=TrainingResultBatchExportRequest(
                    items=[
                        TrainingResultExportItem(result_id=running.id, model_name="same_name"),
                        TrainingResultExportItem(result_id=without_run.id, model_name="same_name"),
                    ]
                ),
                config=config,
            )
        with pytest.raises(TrainingUIAPIError, match="только для успешного"):
            _service.export_training_results_triton_zip(
                session,
                request=TrainingResultBatchExportRequest(
                    items=[
                        TrainingResultExportItem(result_id=running.id, model_name="rivers_kanopus")
                    ]
                ),
                config=config,
            )
        with pytest.raises(TrainingUIAPIError, match="нет MLflow run id"):
            _service.export_training_results_triton_zip(
                session,
                request=TrainingResultBatchExportRequest(
                    items=[
                        TrainingResultExportItem(
                            result_id=without_run.id, model_name="rivers_kanopus"
                        )
                    ]
                ),
                config=config,
            )


def test_class_results_includes_sample_size_hint_from_training_job(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    with session_factory() as session:
        job = JobRow(
            type=JobType.TRAINING.value,
            source=JobSource.MANUAL.value,
            status=JobStatus.RUNNING.value,
            queue_position=1,
            dataset_key="Реки\\main",
            dataset_name="Реки\\main",
            model_name="segformer b2",
            architecture="smp_segformer_b2",
            tile_size=768,
            config={},
        )
        session.add(job)
        session.flush()
        session.add(
            TrainingResultRow(
                source=JobSource.MANUAL.value,
                dataset_key="Реки\\main",
                class_key="Реки\\main",
                class_display_name="Реки\\main",
                architecture="smp_segformer_b2",
                model_name="segformer b2",
                status=ResultStatus.OK.value,
                job_id=job.id,
                mlflow_run_id="run-rivers",
            )
        )
        session.flush()

        response = _service.dataset_results(session, "Реки\\main", config)

    assert response.results[0].sample_size_hint == 768


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
    assert 'head === "scene-list-export"' in app_tsx
    assert 'head === "test-markups" && second === "create"' in app_tsx
    assert 'head === "markup-export"' not in app_tsx
    assert "/bootstrap" in app_tsx
    assert "Экспорт моделей" in app_tsx
    assert "Создать список сцен" in app_tsx
    assert "Каталог тестовых разметок" in app_tsx
    assert "Создание тестовых разметок" in app_tsx
    assert "Пересчитать основной сетью" in app_tsx
    assert "Оценить состав по псевдоразметке" in app_tsx
    assert "Оптимизация состава" in app_tsx
    assert "/optimize-preview" in app_tsx
    assert "/evaluate-preview" in app_tsx
    assert "Не сохранено" in app_tsx
    assert "[512, 768, 1024, 1536, 2048, 2560, 3072, 3584]" in app_tsx
    assert "useState(1536)" in app_tsx
    assert '"/test-sample-batches/latest"' in app_tsx
    assert "setTileSize(latest.tile_size)" in app_tsx
    assert "setMinImageCount(latest.min_image_count)" in app_tsx
    assert "setMaxImageCount(latest.image_count)" in app_tsx
    assert "minObjectCount: previous.min_object_count" in app_tsx
    assert "excludeBoundaryObjects" in app_tsx
    assert "Не учитывать объекты, выходящие за тайл" in app_tsx
    assert "defaultTrainingZipModelName" in app_tsx
    assert "metric: previous.metric" not in app_tsx
    assert "qualityMetricLabel(dataset.quality_metric)" in app_tsx
    assert 'head === "classes"' in app_tsx
    assert "/dataset-catalog/sync" in app_tsx
    assert 'className="test-sample-batch-class-list"' in app_tsx
    assert '"/test-sample-batches/options"' in app_tsx
    assert "Сеть: {dataset.training_model_name" in app_tsx
    assert "/test-samples" in app_tsx
    assert "apiDownloadJson" in app_tsx
    assert "/results/training/triton-zip" in app_tsx
    assert '"/scene-list-export"' in app_tsx
    assert "/results/training/" in app_tsx
    assert "/triton-zip" in app_tsx
    assert "metadata.sample_size" in app_tsx
    assert "sample_size_hint" in app_tsx
    assert 'min="32"' in app_tsx
    assert 'step="32"' in app_tsx
    assert "showJobLog" in app_tsx
    assert "/log" in app_tsx
    assert "log-view" in app_tsx
    assert "hasActiveDatasetResults" in app_tsx
    assert 'const sourceDatasetKey = String(source.get("dataset_key")' in app_tsx
    assert "encodeURIComponent(datasetKey)}/pseudo-markup" in app_tsx
    assert "recommended_range" in app_tsx
    assert "downloadBlob(response.blob" in app_tsx
    assert 'pattern="[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?"' in app_tsx
    assert 'components["schemas"]["BootstrapInfo"]' in api_types
    assert 'components["schemas"]["TestSampleDetail"]' in api_types
    assert 'components["schemas"]["TestSampleOptimizeRequest"]' in api_types
    assert 'credentials: "same-origin"' in api_client


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
    (class_dir / "deforestation.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}', encoding="utf-8"
    )
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
    (frontend_dist / "assets" / "index-test.js").write_text(
        "console.log('MLSystem2')", encoding="utf-8"
    )
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
        health = client.get("/api/v1/health")
        assert health.json()["status"] == "ok"
        assert health.headers["server-timing"].startswith("app;dur=")
        assert float(health.headers["x-process-time-ms"]) >= 0
        assert client.get("/").text.startswith("<!doctype html>")
        app_js = client.get("/assets/index-test.js")
        assert app_js.text == "console.log('MLSystem2')"
        assert app_js.headers["content-type"].split(";")[0] in {
            "text/javascript",
            "application/javascript",
        }
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
        assert [item["name"] for item in bootstrap["datasets"]] == [
            "Вырубки\\main",
            "Озера\\main",
            "Реки\\main",
            "Custom",
        ]
        assert bootstrap["image_folders"][0]["key"] == "kanopus/irkutsk"
        assert len(bootstrap["training_templates"]) == 10

        datasets = client.get("/api/v1/datasets").json()["datasets"]
        assert [item["name"] for item in datasets] == [
            "Вырубки\\main",
            "Озера\\main",
            "Реки\\main",
            "Custom",
        ]
        assert datasets[0]["image_count"] == 1
        assert datasets[0]["hard_negative_annotation_file"] is None
        image_folders = client.get("/api/v1/image-folders").json()["folders"]
        assert image_folders == [
            {
                "key": "kanopus/irkutsk",
                "name": "kanopus/irkutsk",
                "path": str(image_folder),
                "image_count": 2,
                "imagery_type": "kanopus",
            }
        ]
        second_image_folder = images_root / "kanopus" / "toguchinsk"
        second_image_folder.mkdir(parents=True)
        (second_image_folder / "scene-3.tif").touch()

        new_dir = mlmarkup_root / "Пожары" / "main"
        new_dir.mkdir(parents=True)
        (new_dir / "fires.txt").write_text("scene-2\n", encoding="utf-8")
        (new_dir / "fires.geojson").write_text(
            '{"type":"FeatureCollection","features":[]}', encoding="utf-8"
        )
        refreshed = client.get("/api/v1/datasets").json()["datasets"]
        assert [item["name"] for item in refreshed] == [
            "Вырубки\\main",
            "Озера\\main",
            "Пожары\\main",
            "Реки\\main",
            "Custom",
        ]

        models = client.get("/api/v1/models").json()["models"]
        assert [item["display_name"] for item in models] == [
                "deeplabV3+",
                "segformer b0",
                "SegFormer B0 HF (next-gen)",
                "segformer b1",
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
        assert len(templates) == 10
        segformer_b0_template = client.get("/api/v1/training-templates/smp_segformer_b0").json()
        assert segformer_b0_template["display_name"] == "segformer b0"
        assert segformer_b0_template["source"] == "analogy"
        assert segformer_b0_template["default_config"]["train.batch_size"] == 4
        segformer_b1_template = client.get("/api/v1/training-templates/smp_segformer_b1").json()
        assert segformer_b1_template["display_name"] == "segformer b1"
        assert segformer_b1_template["source"] == "analogy"
        assert segformer_b1_template["default_config"]["train.batch_size"] == 4
        segformer_template = client.get("/api/v1/training-templates/smp_segformer_b2").json()
        assert segformer_template["source"] == "hpo_best"
        template_keys = {item["key"] for item in segformer_template["config_schema"]["fields"]}
        assert "dataset.split_granularity" not in template_keys
        assert "tile_preparation.num_workers" not in template_keys
        assert "train.device" not in template_keys
        assert "train.max_train_batches_per_epoch" in template_keys
        assert "train.max_val_batches_per_epoch" in template_keys
        assert "train.max_training_time_sec" in template_keys
        assert "train.background_weight" in template_keys
        assert "train.hard_negative_weight" in template_keys
        assert "tile_preparation.positive_factor" in template_keys
        assert "tile_preparation.hard_negative_factor" in template_keys
        assert "tile_preparation.background_factor" in template_keys
        assert segformer_template["default_config"]["tile_preparation.positive_factor"] == 0.8
        assert segformer_template["default_config"]["tile_preparation.hard_negative_factor"] == 0.0
        assert segformer_template["default_config"]["tile_preparation.background_factor"] == 0.2
        assert segformer_template["default_config"]["train.background_weight"] == 1.0
        assert segformer_template["default_config"]["train.hard_negative_weight"] == 1.0
        assert segformer_template["default_config"]["train.max_train_batches_per_epoch"] == 72
        assert segformer_template["default_config"]["train.max_val_batches_per_epoch"] == 1000
        assert segformer_template["default_config"]["train.max_training_time_sec"] == 1800
        loss_field = next(
            item
            for item in segformer_template["config_schema"]["fields"]
            if item["key"] == "train.loss"
        )
        assert loss_field["options"] == [
            "bce_dice",
            "focal_dice",
            "focal_tversky",
            "cross_entropy",
            "cross_entropy_dice",
        ]
        hard_weight_field = next(
            item
            for item in segformer_template["config_schema"]["fields"]
            if item["key"] == "train.hard_negative_weight"
        )
        assert "размеченных hard-negative зон" in hard_weight_field["tooltip"]
        assert "умножается на background_weight" in hard_weight_field["tooltip"]
        assert "1..5" in hard_weight_field["recommended_range"]
        background_weight_field = next(
            item
            for item in segformer_template["config_schema"]["fields"]
            if item["key"] == "train.background_weight"
        )
        assert background_weight_field["label"] == "Вес фона"
        assert "nodata" in background_weight_field["tooltip"]
        assert "0.1..2" in background_weight_field["recommended_range"]
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
        assert len(inference_templates) == 12
        segformer_b0_inference = client.get("/api/v1/inference-templates/smp_segformer_b0").json()
        assert segformer_b0_inference["display_name"] == "segformer b0"
        assert segformer_b0_inference["source"] == "analogy"
        segformer_b1_inference = client.get("/api/v1/inference-templates/smp_segformer_b1").json()
        assert segformer_b1_inference["display_name"] == "segformer b1"
        assert segformer_b1_inference["source"] == "analogy"
        inference_template = client.get("/api/v1/inference-templates/smp_segformer_b2").json()
        inference_keys = {item["key"] for item in inference_template["config_schema"]["fields"]}
        assert "postprocess.min_area_m2" in inference_keys
        assert "postprocess.smooth.enabled" in inference_keys
        assert "postprocess.smooth.iterations" in inference_keys
        assert "postprocess.smooth.offset" in inference_keys
        assert "postprocess.filter_compact_objects.enabled" in inference_keys
        assert "postprocess.filter_compact_objects.mode" in inference_keys
        assert "train.batch_size" not in inference_keys
        river_inference_template = next(
            item for item in inference_templates if item.get("dataset_key") == "Реки\\main"
        )
        assert river_inference_template["default_config"]["postprocess.min_area_m2"] == 10000.0
        assert river_inference_template["default_config"]["postprocess.min_hole_area_m2"] == 5000.0
        assert river_inference_template["default_config"]["postprocess.smooth.enabled"] is True
        assert river_inference_template["default_config"]["postprocess.smooth.iterations"] == 1
        assert river_inference_template["default_config"]["postprocess.smooth.offset"] == 0.125
        assert river_inference_template["default_config"]["postprocess.simplify_m"] == 1.0
        assert (
            river_inference_template["default_config"]["postprocess.filter_compact_objects.enabled"]
            is True
        )
        assert (
            river_inference_template["default_config"]["postprocess.filter_compact_objects.mode"]
            == "remove_compact"
        )
        lake_inference_template = next(
            item for item in inference_templates if item.get("dataset_key") == "Озера\\main"
        )
        assert (
            lake_inference_template["default_config"]["postprocess.filter_compact_objects.enabled"]
            is True
        )
        assert (
            lake_inference_template["default_config"]["postprocess.filter_compact_objects.mode"]
            == "keep_compact"
        )
        assert (
            lake_inference_template["default_config"][
                "postprocess.filter_compact_objects.min_isoperimetric_quotient"
            ]
            == 0.25
        )
        assert (
            lake_inference_template["default_config"][
                "postprocess.filter_compact_objects.max_bbox_ratio"
            ]
            == 3.5
        )
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
        assert (
            updated_inference_dataset_template["default_config"]["postprocess.min_area_m2"]
            == 2222.0
        )

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
                "run_inference_after_training": True,
                "secondary_priority": True,
            },
        ).json()
        assert job["status"] == "queued"
        assert job["dataset_name"] == "Custom"
        assert job["mlflow_run_name"] is None
        assert "train.device" not in job["config"]
        assert "dataset.split_granularity" not in job["config"]
        assert job["run_inference_after_training"] is True
        assert job["secondary_priority"] is True
        assert "ui.run_inference_after_training" not in job["config"]
        assert "ui.secondary_priority" not in job["config"]

        queues = client.get("/api/v1/queues").json()
        assert queues["training_enabled"] is True
        assert len(queues["training_jobs"]) == 1
        assert queues["training_jobs"][0]["secondary_priority"] is True
        assert client.get("/api/v1/queues/count").json() == {"active_jobs": 1}

        detail = client.get(f"/api/v1/jobs/{job['id']}").json()
        assert detail["readonly"] is True
        assert detail["run_inference_after_training"] is True
        assert detail["secondary_priority"] is True

        custom_results = client.get("/api/v1/results/datasets/custom").json()
        training_result_id = custom_results["results"][0]["id"]
        pseudo = client.post(
            "/api/v1/results/datasets/custom/pseudo-markup",
            data={"dataset_key": "Вырубки\\main", "training_result_id": training_result_id},
        ).json()
        assert pseudo["type"] == "inference"
        assert pseudo["config"]["inference_template_id"] == inference_template["id"]
        assert (
            pseudo["config"]["inference_template_config"]["postprocess.min_area_m2"]
            == inference_template["default_config"]["postprocess.min_area_m2"]
        )
        conflict = client.post(
            "/api/v1/results/datasets/custom/pseudo-markup",
            data={
                "dataset_key": "Вырубки\\main",
                "image_folder_key": "kanopus/irkutsk",
                "training_result_id": training_result_id,
            },
        )
        assert conflict.status_code == 400
        assert conflict.json()["detail"] == "Выберите только один источник снимков"
        folder_pseudo = client.post(
            "/api/v1/results/datasets/custom/pseudo-markup",
            data={"image_folder_key": "kanopus/irkutsk", "training_result_id": training_result_id},
        ).json()
        assert folder_pseudo["type"] == "inference"
        assert folder_pseudo["config"]["image_folder_key"] == "kanopus/irkutsk"
        assert folder_pseudo["config"]["images_root"] == str(images_root / "kanopus")
        assert folder_pseudo["config"]["input_channels"] == 4
        assert folder_pseudo["config"]["inference_template_id"] == inference_template["id"]
        second_folder_pseudo = client.post(
            "/api/v1/results/datasets/custom/pseudo-markup",
            data={
                "image_folder_key": "kanopus/toguchinsk",
                "training_result_id": training_result_id,
            },
        ).json()
        assert second_folder_pseudo["type"] == "inference"
        assert second_folder_pseudo["config"]["image_folder_key"] == "kanopus/toguchinsk"
        uploaded_txt_pseudo = client.post(
            "/api/v1/results/datasets/custom/pseudo-markup",
            data={"training_result_id": training_result_id},
            files={"scenes_txt": ("manual.txt", b"kanopus/irkutsk\n", "text/plain")},
        ).json()
        assert uploaded_txt_pseudo["type"] == "inference"
        inference_queue = client.get("/api/v1/queues").json()["inference_jobs"]
        assert len(inference_queue) == 4
        pseudo_with_empty_upload = client.post(
            "/api/v1/results/datasets/custom/pseudo-markup",
            data={"dataset_key": "Вырубки\\main", "training_result_id": training_result_id},
            files={"scenes_txt": ("", b"", "application/octet-stream")},
        ).json()
        assert pseudo_with_empty_upload["type"] == "inference"

        def fail_runtime_image_count(*_args: object, **_kwargs: object) -> int:
            raise AssertionError("class_results must use stored pseudo image_count")

        monkeypatch.setattr(_service, "count_scenes_file_images", fail_runtime_image_count)
        class_results = client.get("/api/v1/results/datasets/custom").json()
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
        assert client.get(folder_scenes["download_url"]).text.splitlines() == [
            "irkutsk/scene-1.tif",
            "irkutsk/scene-2.tif",
        ]
        folder_result = next(
            item for item in pseudo_results if item["source_dataset_name"] == "kanopus/irkutsk"
        )
        assert folder_result["image_count"] == 2
        second_folder_scenes = next(
            item["scenes_file"]
            for item in pseudo_results
            if item["source_dataset_name"] == "kanopus/toguchinsk"
        )
        assert client.get(second_folder_scenes["download_url"]).text.splitlines() == [
            "toguchinsk/scene-3.tif"
        ]
        uploaded_txt_result = next(
            item
            for item in pseudo_results
            if item["source_dataset_name"] == "Custom"
            and item["scenes_file"]["original_name"] == "manual.txt"
        )
        assert uploaded_txt_result["image_count"] == 2
        session_factory = create_session_factory(get_config())
        with session_factory() as session:
            folder_db_row = session.get(PseudoMarkupResultRow, UUID(folder_result["id"]))
            uploaded_db_row = session.get(PseudoMarkupResultRow, UUID(uploaded_txt_result["id"]))
            assert folder_db_row is not None
            assert uploaded_db_row is not None
            assert folder_db_row.image_count == 2
            assert uploaded_db_row.image_count == 2
        deleted_folder_pseudo = client.delete(f"/api/v1/jobs/{second_folder_pseudo['id']}")
        assert deleted_folder_pseudo.status_code == 200
        queue_after_delete = client.get("/api/v1/queues").json()["inference_jobs"]
        assert second_folder_pseudo["id"] not in {item["id"] for item in queue_after_delete}
        deleted_uploaded_pseudo = client.delete(
            f"/api/v1/results/pseudo-markup/{uploaded_txt_result['id']}"
        )
        assert deleted_uploaded_pseudo.status_code == 200
        assert deleted_uploaded_pseudo.json()["id"] == uploaded_txt_result["id"]
        queue_after_pseudo_delete = client.get("/api/v1/queues").json()["inference_jobs"]
        assert uploaded_txt_pseudo["id"] not in {item["id"] for item in queue_after_pseudo_delete}
        result_id = class_results["results"][0]["id"]
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
        assert client.get("/api/v1/results/datasets/custom").json()["results"] == []
        deleted_template = client.delete(
            f"/api/v1/training-templates/by-id/{dataset_template['id']}"
        ).json()
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
    (class_dir / "annotation.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}', encoding="utf-8"
    )

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

        response = _service.dataset_results(session, "Вырубки\\main", config)

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
                    "train.background_weight": 0.4,
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
        assert "input_channels: 4" in config_yaml
        assert f"images_dir: {config.images_root / 'kanopus'}" in config_yaml
        assert "inference:" not in config_yaml
        assert "hard_negative_annotation_file:" in config_yaml
        assert "positive_factor: 0.5" in config_yaml
        assert "hard_negative_factor: 0.3" in config_yaml
        assert "background_factor: 0.2" in config_yaml
        assert "background_weight: 0.4" in config_yaml
        assert "hard_negative_weight: 2.0" in config_yaml
        assert "max_train_batches_per_epoch: 72" in config_yaml
        assert "max_val_batches_per_epoch: 1000" in config_yaml
        assert "max_training_time_sec: null" in config_yaml
        assert "--settings" in run_script
        assert "--run" in run_script
        assert "run.yml" in run_script
        assert "MLSYSTEM2_MLFLOW_RUN_ID_FILE" in run_script
        assert "MLSYSTEM2_TORCH_NUM_THREADS=4" in run_script
        assert "MLSYSTEM2_TORCH_NUM_INTEROP_THREADS=2" in run_script
        assert "OMP_NUM_THREADS=4" in run_script
        assert "MKL_NUM_THREADS=4" in run_script
        assert "OPENBLAS_NUM_THREADS=1" in run_script
        assert "renice -n 10" in run_script
        assert "ionice -c 2 -n 7" in run_script

    assert started
    assert started[0][0][0] == "bash"
    assert started[0][1]["cwd"] == str(tmp_path)
    assert started[0][1]["start_new_session"] is True


def test_training_ui_worker_snapshots_per_image_annotations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "MLMarkup" / "Реки" / "test"
    dataset_root.mkdir(parents=True)
    annotation = dataset_root / "Olskij_SCN06.geojson"
    original = '{"type":"FeatureCollection","features":[]}'
    annotation.write_text(original, encoding="utf-8")
    image = tmp_path / "images" / "kanopus" / "Olskij" / "SCN06.tif"
    image.parent.mkdir(parents=True)
    image.touch()

    monkeypatch.setenv(
        "MLSYSTEM2_TRAINING_UI_DATABASE_URL",
        f"sqlite:///{tmp_path / 'ui.db'}",
    )
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(tmp_path / "MLMarkup"))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv(
        "MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT",
        str(tmp_path / "files"),
    )
    monkeypatch.setenv(
        "MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT",
        str(tmp_path / "scratch"),
    )
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    run_dir = tmp_path / "run"
    with session_factory() as session:
        ensure_seed_templates(session)
        job = create_training_job(
            session,
            TrainingJobCreate(
                mlflow_experiment_id="1",
                mlflow_experiment_name="per-image-test",
                dataset_key="Реки\\test",
                architecture="smp_segformer_b2",
                config=_short_training_config(),
            ),
            config,
        )
        row = session.get(JobRow, job.id)
        assert row is not None

        payload = _worker._build_training_config(session, row, config, run_dir)
        pseudo = _service.create_pseudo_markup_job(
            session,
            class_key="Реки\\test",
            dataset_key="Реки\\test",
            image_folder_key=None,
            training_result_id=None,
            scenes_name=None,
            scenes_content_type=None,
            scenes_bytes=None,
            config=config,
        )
        pseudo_row = session.get(JobRow, pseudo.id)
        pseudo_result = session.scalar(
            select(PseudoMarkupResultRow).where(PseudoMarkupResultRow.job_id == pseudo.id)
        )
        assert pseudo_row is not None
        assert pseudo_result is not None
        assert pseudo_result.scenes_file is not None
        generated_scenes = Path(pseudo_result.scenes_file.path).read_text(encoding="utf-8")
        annotation_files = list(pseudo_row.config["annotation_files"])

    snapshot_dir = Path(payload["dataset"]["annotations_dir"])
    snapshot = snapshot_dir / annotation.name
    assert payload["dataset"]["images_dir"] == str(config.images_root / "kanopus")
    assert "scenes_file" not in payload["dataset"]
    assert snapshot.read_text(encoding="utf-8") == original
    assert generated_scenes == "Olskij/SCN06\n"
    assert annotation_files == [str(annotation)]
    annotation.write_text('{"changed":true}', encoding="utf-8")
    assert snapshot.read_text(encoding="utf-8") == original


def test_training_ui_builds_ortho_training_config_with_three_channels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "MLMarkup" / "Крыши" / "main"
    dataset_root.mkdir(parents=True)
    (dataset_root / "scenes.txt").write_text("ryazan\n", encoding="utf-8")
    (dataset_root / "annotation.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )
    ortho_root = tmp_path / "images" / "orto" / "ryazan"
    ortho_root.mkdir(parents=True)
    (ortho_root / "ortho.tif").touch()

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(tmp_path / "MLMarkup"))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv(
        "MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT",
        str(tmp_path / "stored-files"),
    )
    monkeypatch.setenv(
        "MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT",
        str(tmp_path / "scratch"),
    )
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    with session_factory() as session:
        ensure_seed_templates(session)
        job = create_training_job(
            session,
            TrainingJobCreate(
                mlflow_experiment_id="1",
                mlflow_experiment_name="ortho-test",
                dataset_key="Крыши\\main",
                architecture="smp_segformer_b2",
                config=_short_training_config(),
            ),
            config,
        )
        row = session.get(JobRow, job.id)
        assert row is not None
        payload = _worker._build_training_config(session, row, config, tmp_path / "run")
        training_result = session.scalar(
            select(TrainingResultRow).where(TrainingResultRow.job_id == row.id)
        )
        assert training_result is not None
        training_result.status = ResultStatus.OK.value
        inference_template = _service.create_inference_template(
            session,
            TrainingTemplateCreate(
                architecture="smp_segformer_b2",
                dataset_key="Крыши\\main",
            ),
            config,
        )
        inference_template = _service.update_inference_template_by_id(
            session,
            inference_template.id,
            TrainingTemplateUpdate(
                default_config={
                    **inference_template.default_config,
                    "postprocess.min_area_m2": 20.0,
                    "postprocess.simplify_m": 0.5,
                }
            ),
            config,
        )
        pseudo = _service.create_pseudo_markup_job(
            session,
            class_key="Крыши\\main",
            dataset_key=None,
            image_folder_key="orto/ryazan",
            training_result_id=training_result.id,
            scenes_name=None,
            scenes_content_type=None,
            scenes_bytes=None,
            config=config,
        )
        pseudo_row = session.get(JobRow, pseudo.id)
        pseudo_result = session.scalar(
            select(PseudoMarkupResultRow).where(PseudoMarkupResultRow.job_id == pseudo.id)
        )
        assert pseudo_row is not None
        assert pseudo_result is not None
        assert pseudo_result.scenes_file is not None
        pseudo_scenes = Path(pseudo_result.scenes_file.path).read_text(encoding="utf-8")
        pseudo_row.config = {**pseudo_row.config, "checkpoint_threshold": 0.5}
        pseudo_payload = _worker._build_pseudo_markup_config(
            session,
            pseudo_row,
            config,
            tmp_path / "pseudo-run",
        )

    assert row.config["train.input_channels"] == 3
    assert row.config["dataset.imagery_type"] == "ortho"
    assert payload["dataset"]["images_dir"] == str(tmp_path / "images" / "orto")
    assert payload["train"]["input_channels"] == 3
    assert pseudo.config["images_root"] == str(tmp_path / "images" / "orto")
    assert pseudo.config["imagery_type"] == "ortho"
    assert pseudo.config["input_channels"] == 3
    assert pseudo.config["inference_template_id"] == str(inference_template.id)
    assert pseudo.config["inference_template_config"]["postprocess.min_area_m2"] == 20.0
    assert pseudo.config["inference_template_config"]["postprocess.simplify_m"] == 0.5
    assert pseudo_scenes == "ryazan/ortho.tif\n"
    assert pseudo_payload["images_root"] == str(tmp_path / "images" / "orto")
    assert pseudo_payload["annotation_files"] == []
    assert pseudo_payload["imagery_type"] == "ortho"
    assert pseudo_payload["input_channels"] == 3
    assert pseudo_payload["inference_backend"] == "geoalert_workflow_engine"
    assert pseudo_payload["model_imagery_type"] == "ortho"


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


def test_training_ui_adds_context_to_legacy_768_template_and_preserves_explicit_zero() -> None:
    legacy = sanitize_template_config(
        {
            "tile_preparation.tile_size": 768,
            "tile_preparation.stride": 384,
        }
    )
    explicit = sanitize_template_config(
        {
            "tile_preparation.tile_size": 768,
            "tile_preparation.stride": 384,
            "tile_preparation.context": 0,
        }
    )

    assert legacy["tile_preparation.context"] == 128
    assert explicit["tile_preparation.context"] == 0


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


def test_training_ui_automation_has_lower_priority_than_manual_jobs(
    tmp_path: Path, monkeypatch
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
        stored_dataset_template = session.get(TrainingTemplateRow, dataset_template.id)
        assert stored_dataset_template is not None
        active_dataset = session.scalar(select(DatasetRow).where(DatasetRow.key == "Вырубки\\main"))
        assert active_dataset is not None
        legacy_key = "00000000-0000-0000-0000-000000000001"
        session.add(
            DatasetRow(
                key=legacy_key,
                class_id=active_dataset.class_id,
                name="main [legacy]",
                source_type="mlmarkup",
                source_path=".mlsystem2-archive/main",
                legacy_version=True,
                deleted_at=datetime.now(timezone.utc),
            )
        )
        stored_dataset_template.dataset_key = legacy_key
        session.flush()
        selected_template = _service.training_template_row_for_dataset(
            session,
            "smp_segformer_b2",
            "Вырубки\\main",
        )
        assert selected_template is stored_dataset_template
        _service.sync_dataset_catalog(session, config)
        assert stored_dataset_template.dataset_key == "Вырубки\\main"
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
        annotation_path.write_text(
            '{"type":"FeatureCollection","features":[{"type":"Feature"}]}', encoding="utf-8"
        )
        stat = annotation_path.stat()
        os.utime(
            annotation_path, ns=(stat.st_atime_ns + 2_000_000_000, stat.st_mtime_ns + 2_000_000_000)
        )
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
        result = session.scalar(
            select(TrainingResultRow).where(TrainingResultRow.job_id == auto_job.id)
        )
        assert result is not None
        result.mlflow_run_id = "run-auto-kill"

        _service.set_automation(session, AutomationEnabledUpdate(enabled=False), config)

        assert auto_job.status == JobStatus.CANCELLED.value
        assert result.status == ResultStatus.CANCELLED.value
        assert auto_job.tmp_path is None
        assert not run_dir.exists()
        assert terminated_pids == [9876]
        assert killed_runs == [(config.mlflow_tracking_uri, "run-auto-kill")]


def test_automation_current_results_ignore_newer_cancelled_attempts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    older = datetime(2026, 8, 15, 10, tzinfo=timezone.utc)
    newer = older + timedelta(minutes=1)

    with session_factory() as session:
        inherited = TrainingResultRow(
            source=JobSource.AUTOMATION.value,
            dataset_key="dataset-key",
            dataset_version="managed:2:git:current",
            class_key="dataset-key",
            class_display_name="Вырубки\\main",
            architecture="smp_segformer_b2",
            model_name="SegFormer B2",
            status=ResultStatus.OK.value,
            created_at=older,
        )
        cancelled = TrainingResultRow(
            source=JobSource.AUTOMATION.value,
            dataset_key="dataset-key",
            dataset_version="managed:2:git:current",
            class_key="dataset-key",
            class_display_name="Вырубки\\main",
            architecture="smp_segformer_b2",
            model_name="SegFormer B2",
            status=ResultStatus.CANCELLED.value,
            created_at=newer,
        )
        session.add_all([inherited, cancelled])
        session.flush()
        inherited_pseudo = PseudoMarkupResultRow(
            source=JobSource.AUTOMATION.value,
            dataset_key="dataset-key",
            dataset_version="managed:2:git:current",
            training_result_id=inherited.id,
            class_key="dataset-key",
            source_dataset_name="Вырубки\\main",
            status=ResultStatus.OK.value,
            created_at=older,
        )
        cancelled_pseudo = PseudoMarkupResultRow(
            source=JobSource.AUTOMATION.value,
            dataset_key="dataset-key",
            dataset_version="managed:2:git:current",
            training_result_id=inherited.id,
            class_key="dataset-key",
            source_dataset_name="Вырубки\\main",
            status=ResultStatus.CANCELLED.value,
            created_at=newer,
        )
        session.add_all([inherited_pseudo, cancelled_pseudo])
        session.flush()

        assert (
            _automation._current_training_result(
                session,
                dataset_key="dataset-key",
                architecture="smp_segformer_b2",
                dataset_version="managed:2:git:current",
            )
            is inherited
        )
        assert (
            _automation._current_pseudo_result(
                session,
                inherited,
                "managed:2:git:current",
            )
            is inherited_pseudo
        )


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
    images_root = tmp_path / "images"
    kanopus_root = images_root / "kanopus"
    kanopus_root.mkdir(parents=True)
    (kanopus_root / "scene-1.tif").touch()

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(mlmarkup_root))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(images_root))
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
        training_result = session.scalar(
            select(TrainingResultRow).where(TrainingResultRow.job_id == training_job.id)
        )
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
        assert (
            pseudo_job.config["checkpoint_uri"]
            == "s3://mlflow-artifacts/auto/run/artifacts/checkpoints/best.pt"
        )
        pseudo_result = session.scalar(
            select(PseudoMarkupResultRow).where(PseudoMarkupResultRow.job_id == pseudo_job.id)
        )
        assert pseudo_result is not None
        assert pseudo_result.scenes_file is not None
        assert pseudo_result.image_count == 1
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


def test_training_ui_automation_supports_per_image_dataset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "MLMarkup" / "Реки" / "test"
    dataset_root.mkdir(parents=True)
    annotation = dataset_root / "Olskij_SCN06.geojson"
    annotation.write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )
    image = tmp_path / "images" / "kanopus" / "Olskij" / "SCN06.tif"
    image.parent.mkdir(parents=True)
    image.touch()
    monkeypatch.setenv(
        "MLSYSTEM2_TRAINING_UI_DATABASE_URL",
        f"sqlite:///{tmp_path / 'ui.db'}",
    )
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(tmp_path / "MLMarkup"))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv(
        "MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT",
        str(tmp_path / "files"),
    )
    monkeypatch.setenv(
        "MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT",
        str(tmp_path / "scratch"),
    )
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")
    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    monkeypatch.setattr(
        _automation,
        "create_experiment",
        lambda request: MLflowExperiment(experiment_id="auto-exp", name=request.name),
    )
    monkeypatch.setattr(
        _automation,
        "get_best_training_checkpoint",
        lambda tracking_uri, run_id: MLflowBestCheckpoint(
            tracking_uri=tracking_uri,
            run_id=run_id,
            metric_name="val/best_threshold_pixel_f1",
            f1_score=0.9,
            epoch=3,
            artifact_path="checkpoints/best.pt",
            artifact_uri="s3://mlflow/run/best.pt",
            threshold=0.6,
        ),
    )

    with session_factory() as session:
        ensure_seed_templates(session)
        _service.set_automation(
            session,
            AutomationEnabledUpdate(enabled=True),
            config,
        )
        _service.update_automation(
            session,
            AutomationRuleUpdate(
                dataset_key="Реки\\test",
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
                JobRow.type == JobType.TRAINING.value,
            )
        )
        assert training_job is not None
        training_result = session.scalar(
            select(TrainingResultRow).where(TrainingResultRow.job_id == training_job.id)
        )
        assert training_result is not None
        training_job.status = JobStatus.COMPLETED.value
        training_result.status = ResultStatus.OK.value
        training_result.mlflow_run_id = "run-per-image"

        _automation.sync_automation_once(session, config)

        pseudo_job = session.scalar(
            select(JobRow).where(
                JobRow.source == JobSource.AUTOMATION.value,
                JobRow.type == JobType.INFERENCE.value,
            )
        )
        assert pseudo_job is not None
        pseudo_result = session.scalar(
            select(PseudoMarkupResultRow).where(PseudoMarkupResultRow.job_id == pseudo_job.id)
        )
        assert pseudo_result is not None
        assert pseudo_result.image_count == 1
        assert pseudo_result.scenes_file is not None
        assert Path(pseudo_result.scenes_file.path).read_text(encoding="utf-8") == (
            "Olskij/SCN06\n"
        )
        assert pseudo_job.config["annotation_files"] == [str(annotation.resolve())]


def test_training_ui_worker_records_best_mlflow_metric(tmp_path: Path, monkeypatch) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки" / "main"
    class_dir.mkdir(parents=True)
    (class_dir / "scenes.txt").write_text("scene-1\n", encoding="utf-8")
    (class_dir / "annotation.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )
    images_root = tmp_path / "images"
    kanopus_root = images_root / "kanopus"
    kanopus_root.mkdir(parents=True)
    (kanopus_root / "scene-1.tif").touch()

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(mlmarkup_root))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(images_root))
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

    def fake_list_experiments(tracking_uri: str) -> list[MLflowExperiment]:
        assert tracking_uri == config.mlflow_tracking_uri
        return [MLflowExperiment(experiment_id="1", name="ui-test", lifecycle_stage="active")]

    monkeypatch.setattr(_worker, "get_best_training_checkpoint", fake_best_checkpoint)
    monkeypatch.setattr(_worker, "list_experiments", fake_list_experiments)

    with session_factory() as session:
        ensure_seed_templates(session)
        job = create_training_job(
            session,
            TrainingJobCreate(
                mlflow_experiment_id=None,
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
        running_result = session.scalar(
            select(TrainingResultRow).where(TrainingResultRow.job_id == job.id)
        )
        assert running_result is not None
        assert running_result.mlflow_run_id == "run-123"
        assert row.mlflow_experiment_id == "1"
        assert (
            running_result.mlflow_run_url
            == f"{config.mlflow_ui_url.rstrip('/')}/#/experiments/1/runs/run-123"
        )

        (run_dir / "train.log").write_text(
            "status=succeeded\nmlflow_run=run-123\n", encoding="utf-8"
        )
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
        assert pseudo_job.config["annotation_files"] == [str(class_dir / "annotation.geojson")]
        legacy_pseudo_result = session.scalar(
            select(PseudoMarkupResultRow).where(PseudoMarkupResultRow.job_id == pseudo_job.id)
        )
        assert legacy_pseudo_result is not None
        assert legacy_pseudo_result.image_count == 1
        legacy_pseudo_result.image_count = None
        session.flush()

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
        assert pseudo_config_payload["input_channels"] == 4
        assert pseudo_config_payload["annotation_files"] == [str(class_dir / "annotation.geojson")]
        for forbidden_key in ("triton_model", "pipeline", "model_repository", "model_archive"):
            assert forbidden_key not in pseudo_config_payload

        output = pseudo_run_dir / "scratch" / "pseudo_markup.geojson"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
        (pseudo_run_dir / "scratch" / "report.json").write_text(
            '{"status":"ok","processed":1,"feature_count":500}',
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
        refreshed_results = _service.dataset_results(session, "Вырубки\\main", config)
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
        assert pseudo_db_row.image_count == 1
        geojson_file = pseudo_db_row.geojson_file
        assert geojson_file is not None
        assert geojson_file.object_count == 500
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
        "train.background_weight": 1.0,
        "train.hard_negative_weight": 1.0,
        "train.tversky_alpha": 0.4,
        "train.tversky_beta": 0.6,
        "train.threshold": 0.5,
        "train.early_stopping_patience": 1,
        "train.max_train_batches_per_epoch": 72,
        "train.max_val_batches_per_epoch": 1000,
        "train.max_training_time_sec": None,
    }
