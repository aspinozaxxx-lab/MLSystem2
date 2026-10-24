"""Контракты выбора и постановки отдельного объектного профиля второй версии."""

from types import SimpleNamespace

import pytest

from mlsystem2.training_ui_api import _service
from mlsystem2.training_ui_api._templates import (
    BASE_DEFAULT_CONFIG,
    OBJECT_F1_DEFAULT_CONFIG,
    OBJECT_F1_V2_DEFAULT_CONFIG,
    initial_templates,
    next_gen2_train_batch_size,
    sanitize_template_config,
)
from mlsystem2.training_ui_api.api import get_openapi_schema
from mlsystem2.training_ui_api.contracts import TrainingUIAPIError


@pytest.mark.parametrize("architecture", [f"smp_segformer_b{i}" for i in range(4)])
@pytest.mark.parametrize("imagery_type,channels", [("kanopus", 4), ("ortho", 3)])
def test_v2_catalog_profile_is_independent_and_validated(architecture, imagery_type, channels):
    templates = {item["architecture"]: item for item in initial_templates()}
    template = templates[architecture]
    schema = template["config_schema"]
    assert schema["fields"][0]["options"] == ["legacy", "next_gen2", "next_gen3", "next_gen4", "next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "object_f1", "object_f1_v2"]
    assert template["default_config"]["train.pipeline_variant"] == "legacy"
    assert "три взаимоисключающих класса" in schema["pipeline_descriptions"]["object_f1_v2"]
    assert "минимум три группы" in schema["pipeline_descriptions"]["object_f1_v2"]
    for size in (512, 768, 1024, 1536):
        value = sanitize_template_config(
            {"train.pipeline_variant": "object_f1_v2", "tile_preparation.tile_size": size,
             "train.pretrained": False, "train.epochs": 31, "train.learning_rate": 0.25,
             "train.max_val_batches_per_epoch": 3},
            architecture=architecture,
        )
        assert value["train.pipeline_variant"] == "object_f1_v2"
        assert value["train.pretrained"] is False and value["train.epochs"] == 31
        assert value["train.learning_rate"] == 1e-4
        assert value["train.max_val_batches_per_epoch"] is None
        assert value["train.batch_size"] == next_gen2_train_batch_size(size, architecture)
        assert value["tile_preparation.stride"] == size // 2
        job = {**value, "dataset.imagery_type": imagery_type, "dataset.task": "binary", "train.input_channels": channels}
        _service._validate_training_pipeline_variant(job, architecture)
        with pytest.raises(TrainingUIAPIError, match="только бинарную"):
            _service._validate_training_pipeline_variant({**job, "dataset.task": "multiclass"}, architecture)
        with pytest.raises(TrainingUIAPIError, match="train.input_channels"):
            _service._validate_training_pipeline_variant({**job, "train.input_channels": 7}, architecture)


def test_v2_switch_preserves_old_variants_and_does_not_alias_presets():
    old = dict(OBJECT_F1_DEFAULT_CONFIG)
    new = dict(OBJECT_F1_V2_DEFAULT_CONFIG)
    architecture = "smp_segformer_b0"
    current = sanitize_template_config({"train.pipeline_variant": "object_f1_v2"}, architecture=architecture)
    for variant in ("object_f1", "next_gen2", "legacy"):
        config = sanitize_template_config({"train.pipeline_variant": variant}, fallback=current, architecture=architecture)
        assert config["train.pipeline_variant"] == variant
        assert config["train.loss"] == (BASE_DEFAULT_CONFIG["train.loss"] if variant == "legacy" else "cross_entropy_tversky")
    assert OBJECT_F1_DEFAULT_CONFIG == old
    assert OBJECT_F1_V2_DEFAULT_CONFIG == new
    assert _service._job_pipeline_variant(SimpleNamespace(config=current)) == "object_f1_v2"
    assert _service._job_pipeline_variant(None) == "legacy"


@pytest.mark.parametrize("architecture", ["segformer_b0", "smp_unet_resnet34", "smp_deeplabv3plus_resnet50"])
def test_v2_rejects_unsupported_architectures(architecture):
    with pytest.raises(TrainingUIAPIError, match="object f1"):
        sanitize_template_config({"train.pipeline_variant": "object_f1_v2"}, architecture=architecture)
    with pytest.raises(TrainingUIAPIError, match="object f1 v2"):
        _service._validate_training_pipeline_variant({"train.pipeline_variant": "object_f1_v2"}, architecture)


def test_v2_openapi_carries_variant_on_jobs_and_results(monkeypatch):
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")
    schemas = get_openapi_schema()["components"]["schemas"]
    for name in ("JobSummary", "JobDetail", "TrainingResultInfo"):
        field = schemas[name]["properties"]["pipeline_variant"]
        assert "object_f1_v2" in field["enum"]
        assert field["default"] == "legacy"


@pytest.mark.parametrize("imagery_type,channels", [("kanopus", 4), ("ortho", 3)])
def test_v2_job_uses_object_metric_and_keeps_base_templates(tmp_path, monkeypatch, imagery_type, channels):
    from sqlalchemy import select
    from mlsystem2.training_ui_api._config import get_config
    from mlsystem2.training_ui_api._database import Base, configure_schema, create_session_factory
    from mlsystem2.training_ui_api._models import TrainingResultRow, TrainingTemplateRow
    from mlsystem2.training_ui_api.contracts import ImageryType, TrainingJobCreate

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    config = get_config()
    dataset = SimpleNamespace(
        quality_metric="pixel", task="binary", input_channels=channels,
        imagery_type=ImageryType(imagery_type), object_types=[], images_dir=str(tmp_path / imagery_type),
        version="зафиксированная-версия", name="ЗУ500/main",
    )
    monkeypatch.setattr(_service, "_resolve_dataset_name", lambda *args: dataset)
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    with session_factory() as session:
        job = _service.create_training_job(session, TrainingJobCreate(
            dataset_key="ЗУ500/main", architecture="smp_segformer_b2",
            mlflow_experiment_id="1", mlflow_experiment_name="Проверка выбора профиля",
            config={"train.pipeline_variant": "object_f1_v2", "train.pretrained": False},
        ), config)
        assert job.pipeline_variant == "object_f1_v2"
        assert job.config["train.quality_metric"] == "objects"
        assert job.config["train.input_channels"] == channels
        assert job.config["train.pretrained"] is False
        result = session.scalar(select(TrainingResultRow).where(TrainingResultRow.job_id == job.id))
        assert result is not None and result.quality_metric == "objects"
        assert result.dataset_version == "зафиксированная-версия"
        for template in session.scalars(select(TrainingTemplateRow)):
            assert template.default_config["train.pipeline_variant"] == "legacy"
