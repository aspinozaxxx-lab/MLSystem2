"""Справка не должна менять числа, шаги, статус запуска или пользовательский текст."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlsystem2.mlflow_adapter import _client
from mlsystem2.mlflow_adapter._metric_descriptions import (
    HTML_PATH, JSON_PATH, NOTE_START, SAMPLING, build_catalog, describe_metric,
    render_html, update_run_note,
)
from mlsystem2.mlflow_adapter.api import log_metric_descriptions
from mlsystem2.mlflow_adapter.contracts import MLflowAdapterError, MLflowRunRef
from mlsystem2.train.contracts import EpochMetrics, TrainResult


@pytest.mark.parametrize("variant", ["legacy", "next_gen", "next_gen2", "next_gen3", "next_gen4", "next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "object_f1", "object_f1_v2"])
@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_every_emitted_metric_has_a_description(monkeypatch, variant, task):
    logged = {}
    mlflow = SimpleNamespace(log_metric=lambda key, value, **kwargs: logged.setdefault(key, value))
    monkeypatch.setattr(_client, "_ensure_run_active", lambda run: mlflow)
    monkeypatch.setattr(_client, "_log_dict", lambda *args: None)
    monkeypatch.setattr(_client, "_try_log_metric_descriptions", lambda run: None)
    run = MLflowRunRef(run_id="проверка", experiment_name="проверка", tracking_uri="http://localhost", active=True)
    values = {field: 0.4 for field in EpochMetrics.model_fields if field.startswith("val_") and
              not any(word in field for word in ("per_class", "per_scene", "warnings", "sweep", "separation", "dense_window"))}
    rates = {"empty_pixel_fpr": 0.001, "empty_tile_fpr": 0.1, "foreground_recall": 0.8}
    metrics = EpochMetrics(
        epoch=1, train_loss=0.5, train_region_loss=0.4, train_boundary_loss=0.1,
        learning_rate=0.001, epoch_time_sec=2,
        adaptive_sampling={"current": {**{key: 1 for key in SAMPLING}, "rates": rates, "ema": rates}},
        val_per_class_metrics=[{"slug": "новый_тип", "precision": 0.4, "recall": 0.4, "f1": 0.4, "iou": 0.25}],
        val_per_scene_metrics=[{"scene_id": "Новый снимок", **{k: 0.4 for k in
            ("pixel_f1", "pixel_precision", "pixel_recall", "fixed_0_5_f1", "fixed_0_5_precision", "fixed_0_5_recall")}}],
        **values,
    )
    _client.log_training_epoch(run, metrics)
    test = {key: 0.4 for key in (
        "loss", "pixel_f1", "pixel_precision", "pixel_recall", "threshold", "checkpoint_epoch",
        "macro_pixel_f1", "micro_pixel_f1", "foreground_pixel_f1", "object_f1", "object_precision", "object_recall",
    )}
    _client.log_training_metrics(run, TrainResult(
        epochs_total=1, training_time_sec=2, best_threshold=0.5, history=[metrics],
        diagnostics={"test_metrics": test, "peak_vram_bytes": 1024},
    ))
    _client.log_training_metrics(run, TrainResult(
        epochs_total=1, training_time_sec=2, history=[metrics],
        diagnostics={"checkpoint_selection": {"metric": "val/object_f1", "object_f1": 0.6}},
    ))
    context = {"pipeline_variant": variant, "task": task}
    catalog = build_catalog(logged, context)
    assert set(catalog["metrics"]) == set(logged)
    assert len(logged) >= 80
    for item in catalog["metrics"].values():
        assert set(item) == {"meaning", "purpose", "unit", "direction"}
        assert all(item.values())


def test_context_changes_meaning_not_metric_names():
    binary = describe_metric("val/pixel_f1", {"task": "binary"})
    multiclass = describe_metric("val/pixel_f1", {"task": "multiclass"})
    assert "суммарным" in binary["meaning"]
    assert "macro" in multiclass["meaning"]
    assert "сцены" in describe_metric("val/macro_pixel_f1", {})["meaning"]
    assert "фонового класса" in describe_metric("val/macro_pixel_f1", {"task": "multiclass"})["meaning"]
    assert "Dice-loss" in describe_metric("val/boundary_loss", {"pipeline_variant": "object_f1_v2"})["meaning"]
    assert "BCE" in describe_metric("val/boundary_loss", {"pipeline_variant": "object_f1"})["meaning"]
    assert "best.pt" in describe_metric("train/best_quality_f1", {"pipeline_variant": "next_gen4", "checkpoint_selection_metric": "val_loss"})["meaning"]
    assert "всех проверенных эпох" in describe_metric("train/best_macro_pixel_f1", {})["meaning"]
    assert "не индикатор" in describe_metric("train/stopped_early", {})["purpose"]
    for name in ("val/quality_f1", "val/quality_precision", "val/quality_recall"):
        assert describe_metric(name, {})["meaning"]
    historical = build_catalog(["val/pixel_f1"], {"pipeline_variant": "next_gen2", "checkpoint_selection_metric": "quality_f1"})
    assert any("исторический тег" in note for note in historical["notes"])
    assert not any("выбираются по минимуму" in note for note in historical["notes"])


def test_v6_description_explains_two_ce_and_does_not_add_context_to_validation():
    context = {"pipeline_variant": "next_gen4_consistency_v6", "checkpoint_selection_metric": "val_loss"}
    train = describe_metric("train/loss", context)["meaning"]
    assert "среднее CE" in train and "собственной разметке" in train
    assert "вес0,25 применяется один раз" in train
    assert "Включает дополнительный KL" not in train
    assert "среднее CE" not in describe_metric("val/loss", context)["meaning"]
    assert "среднее CE" not in describe_metric("test/loss", context)["meaning"]


def test_html_is_offline_safe_and_has_tooltips():
    name = 'val/scene/снимок<img src=x onerror="alert(1)">/pixel_f1'
    html = render_html(build_catalog([name], {"task": '<script>ошибка</script>'}))
    assert '<html lang="ru">' in html and 'title="F1' in html
    assert "&lt;img" in html and "&lt;script&gt;" in html
    assert "<script" not in html and "<img" not in html
    assert "http://" not in html and "https://" not in html


def test_note_preserves_user_content_and_is_idempotent():
    original = "Комментарий пользователя.\n\nВажные условия эксперимента."
    updated = update_run_note(original, "56", "запуск")
    assert updated.startswith(original + "\n\n")
    assert "metric_descriptions.html" in updated
    assert update_run_note(updated, "56", "запуск") == updated
    assert update_run_note(updated + "\nПосле справки", "56", "запуск").endswith("\nПосле справки")
    assert updated.count(NOTE_START) == 1


def _fake_client(monkeypatch, *, note="Комментарий", names=None):
    state = SimpleNamespace(
        info=SimpleNamespace(experiment_id="56", status="FINISHED"),
        data=SimpleNamespace(tags={"mlflow.note.content": note, "pipeline_variant": "next_gen4"},
                             metrics=dict.fromkeys(names or ["train/loss", "val/pixel_f1"], 0.7)),
    )
    writes = []
    class Client:
        def get_run(self, run_id):
            return state

        def log_dict(self, run_id, payload, path):
            writes.append((path, payload))

        def log_text(self, run_id, payload, path):
            writes.append((path, payload))

        def set_tag(self, run_id, key, value):
            state.data.tags[key] = value

    monkeypatch.setattr(_client, "_mlflow", lambda: SimpleNamespace(
        tracking=SimpleNamespace(MlflowClient=lambda **kwargs: Client()),
    ))
    return state, writes, MLflowRunRef(run_id="запуск", experiment_name="проверка", tracking_uri="http://localhost", active=True)


def test_metadata_only_and_regenerated_only_for_new_keys_or_context(monkeypatch):
    state, writes, run = _fake_client(monkeypatch)
    log_metric_descriptions(run)
    assert {path for path, _ in writes} == {HTML_PATH, JSON_PATH}
    assert state.info.status == "FINISHED"
    assert state.data.metrics == {"train/loss": 0.7, "val/pixel_f1": 0.7}
    assert state.data.tags["mlflow.note.content"].startswith("Комментарий\n\n")
    state.data.metrics["train/loss"] = 0.5
    log_metric_descriptions(run)
    assert len(writes) == 2
    state.data.metrics["val/scene/Новый/pixel_f1"] = 0.8
    log_metric_descriptions(run)
    assert len(writes) == 4
    state.data.tags["task"] = "multiclass"
    log_metric_descriptions(run)
    assert len(writes) == 6
    assert "macro" in writes[-2][1]["metrics"]["val/pixel_f1"]["meaning"]


def test_long_user_description_is_never_truncated(monkeypatch):
    state, writes, run = _fake_client(monkeypatch, note="А" * 7990)
    log_metric_descriptions(run)
    assert state.data.tags["mlflow.note.content"] == "А" * 7990
    assert len(writes) == 2


def test_termination_covers_flushed_async_metrics(monkeypatch):
    state, writes, run = _fake_client(monkeypatch)
    mlflow = _client._mlflow()
    mlflow.active_run = lambda: SimpleNamespace(info=SimpleNamespace(run_id=run.run_id))
    def finish(*, status):
        state.info.status = status
        state.data.metrics["test/loss"] = 0.1
    mlflow.end_run = finish
    monkeypatch.setattr(_client, "_mlflow", lambda: mlflow)
    from mlsystem2.mlflow_adapter.contracts import MLflowRunStatus
    _client.end_run(run, MLflowRunStatus.FINISHED)
    assert state.info.status == "FINISHED"
    assert "test/loss" in writes[0][1]["metrics"]


def test_no_metrics_means_no_empty_glossary(monkeypatch):
    state, writes, run = _fake_client(monkeypatch)
    state.data.metrics.clear()
    log_metric_descriptions(run)
    assert not writes
    assert state.data.tags["mlflow.note.content"] == "Комментарий"


def test_unknown_metric_is_explicit_and_does_not_break_training(monkeypatch, caplog):
    state, writes, run = _fake_client(monkeypatch, names=["неизвестная/метрика"])
    with pytest.raises(MLflowAdapterError) as exc:
        log_metric_descriptions(run)
    assert isinstance(exc.value.__cause__, ValueError)
    assert not writes
    _client._try_log_metric_descriptions(run)
    assert "обучение продолжается" in caplog.text
    assert state.info.status == "FINISHED"


def test_disabled_run_does_not_contact_mlflow(monkeypatch):
    monkeypatch.setattr(_client, "_mlflow", lambda: pytest.fail("Отключённый запуск не должен обращаться к MLflow"))
    log_metric_descriptions(MLflowRunRef(run_id="disabled", experiment_name="проверка", tracking_uri="http://localhost", active=False))
