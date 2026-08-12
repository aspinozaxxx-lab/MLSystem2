from __future__ import annotations

import inspect
from pathlib import Path
from typing import get_type_hints

import pytest

from mlsystem2.dataset_preparing.contracts import (
    DatasetClassAnnotation,
    DatasetPreparationReport,
    DatasetPreparationResult,
    PreparedDataset,
    PreparedScene,
)
from mlsystem2.mlflow_adapter.contracts import MLflowRunRef
from mlsystem2.models.contracts import CheckpointArtifact, LoadedCheckpoint, ModelHandle, ModelSpec
from mlsystem2.settings.contracts import (
    DatasetClassSettings,
    DatasetSettings,
    InferenceSettings,
    MLflowSettings,
    RuntimeSettings,
    SystemSettings,
    TilePreparationSettings,
    TrainSettings,
)
from mlsystem2.train.contracts import EpochMetrics, TrainResult
from mlsystem2.train_pipeline.api import run_train_pipeline
from mlsystem2.train_pipeline import _runner
from mlsystem2.train_pipeline.contracts import TrainPipelineError, TrainPipelineRequest, TrainPipelineResult


def test_run_train_pipeline_signature_uses_request_contract() -> None:
    signature = inspect.signature(run_train_pipeline)
    parameters = list(signature.parameters.values())
    hints = get_type_hints(run_train_pipeline)
    assert len(parameters) == 1
    assert parameters[0].name == "request"
    assert hints["request"] is TrainPipelineRequest


def test_train_pipeline_result_fields() -> None:
    assert {"status", "mlflow_run", "timings", "report"} <= set(TrainPipelineResult.model_fields)


def test_train_pipeline_request_has_only_run_name() -> None:
    assert set(TrainPipelineRequest.model_fields) == {"run_name"}


def test_train_pipeline_sets_mlflow_dataset_from_binary_annotation_stem() -> None:
    request = _runner._mlflow_start_request(
        _settings(initial_checkpoint_uri=None),
        TrainPipelineRequest(),
    )

    assert request.dataset == "annotations"
    assert request.tags["class"] == "annotations"


def test_train_pipeline_sets_mlflow_class_tag_from_mlmarkup_class_and_dataset() -> None:
    settings = _settings(initial_checkpoint_uri=None)
    settings.dataset.scenes_file = "/data/MLMarkup/Абразия/main/scenes.txt"
    settings.dataset.annotation_file = "/data/MLMarkup/Абразия/main/annotation.geojson"

    request = _runner._mlflow_start_request(settings, TrainPipelineRequest())

    assert request.tags["class"] == "абразия_main"


def test_train_pipeline_sets_mlflow_dataset_from_multiclass_annotation_stems() -> None:
    request = _runner._mlflow_start_request(
        _multiclass_settings(),
        TrainPipelineRequest(),
    )

    assert request.dataset == "class_a+class_b"


def test_train_request_uses_tile_size_as_sample_size() -> None:
    settings = _settings(initial_checkpoint_uri=None)
    settings.tile_preparation = TilePreparationSettings(
        tile_size=768,
        stride=384,
        context=128,
        seed=42,
    )
    model = ModelHandle(
        spec=ModelSpec(name="segformer_b2", input_channels=4, output_channels=1),
        model=object(),
    )
    train_request = _runner._train_request(settings, model, object(), object())

    assert train_request.sample_size == settings.tile_preparation.tile_size
    assert train_request.config.inference_context == 128
    assert train_request.config.seed == 42


def test_train_pipeline_uses_load_checkpoint_branch() -> None:
    calls: list[str] = []
    dataset_artifacts: list[dict[str, str]] = []
    model = ModelHandle(
        spec=ModelSpec(name="segformer_b2", input_channels=4, output_channels=1),
        model=object(),
    )
    deps = _runner._PipelineDependencies(
        get_settings=lambda: _settings(initial_checkpoint_uri="/tmp/initial.pt"),
        get_settings_path=lambda: Path("config.yaml"),
        start_run=lambda request: MLflowRunRef(
            run_id="disabled",
            experiment_name=request.experiment_name,
            tracking_uri=request.tracking_uri,
            active=False,
        ),
        prepare_dataset=lambda request: _dataset_result(),
        create_tile_dataloader=lambda request: object(),
        create_model=lambda spec: calls.append("create_model") or model,
        load_checkpoint=lambda request: calls.append("load_checkpoint")
        or LoadedCheckpoint(
            model=model,
            artifact=CheckpointArtifact(uri=request.checkpoint_uri, format="torch_pt"),
        ),
        train_model=lambda request, progress_sink=None: _train_result(),
        log_dataset_preparation=lambda run, report: None,
        log_dataset_artifacts=lambda run, files: dataset_artifacts.append(files),
        log_tile_preparation=lambda run, report: None,
        log_run_config=lambda run, config_path: None,
        log_training_epoch=lambda run, metrics: None,
        log_training_metrics=lambda run, result: None,
        log_training_artifacts=lambda run, result: None,
        log_timing_report=lambda run, report: None,
        log_pipeline_report=lambda run, report: None,
        end_run=lambda run, status: None,
    )

    result = _runner.run_train_pipeline(TrainPipelineRequest(), dependencies=deps)

    assert result.status.value == "succeeded"
    assert calls == ["load_checkpoint"]
    assert dataset_artifacts == [{"scenes.txt": "./scenes.txt", "annotations.geojson": "./annotations.geojson"}]


def test_dataset_artifact_files_prefix_multiclass_sources() -> None:
    assert _runner._dataset_artifact_files(_multiclass_settings()) == {
        "class_a_scenes.txt": "./class_a.txt",
        "class_a_annotation.geojson": "./class_a.geojson",
        "class_a_hard_negative.geojson": "./class_a_hard_negative.geojson",
        "class_b_scenes.txt": "./class_b.txt",
        "class_b_annotation.geojson": "./class_b.geojson",
    }


def test_dataset_artifact_files_include_binary_hard_negative_source() -> None:
    settings = _settings(initial_checkpoint_uri=None)
    settings.dataset.hard_negative_annotation_file = "./hard_negative.geojson"

    assert _runner._dataset_artifact_files(settings) == {
        "scenes.txt": "./scenes.txt",
        "annotations.geojson": "./annotations.geojson",
        "hard_negative.geojson": "./hard_negative.geojson",
    }


def test_train_pipeline_builds_per_image_requests_and_logs_all_geojson(
    tmp_path: Path,
) -> None:
    annotations_dir = tmp_path / "MLMarkup" / "Реки" / "test"
    annotations_dir.mkdir(parents=True)
    first_annotation = annotations_dir / "Olskij_SCN06.geojson"
    second_annotation = annotations_dir / "Olskij_SCN07.geojson"
    first_annotation.write_text("{}", encoding="utf-8")
    second_annotation.write_text("{}", encoding="utf-8")
    manifest = annotations_dir / ".mlsystem2-dataset.json"
    manifest.write_text("{}", encoding="utf-8")
    (annotations_dir / "README.md").write_text("описание", encoding="utf-8")
    settings = _settings(initial_checkpoint_uri=None)
    settings.dataset = DatasetSettings(
        images_dir="./images",
        annotations_dir=str(annotations_dir),
        val_fraction=0.2,
    )
    prepared = PreparedDataset(
        format="per_image_binary",
        scenes=[
            PreparedScene(
                scene_id="Olskij/SCN06",
                image_path="./SCN06.tif",
                annotation_file=str(first_annotation),
            ),
            PreparedScene(
                scene_id="Olskij/SCN07",
                image_path="./SCN07.tif",
                annotation_file=str(second_annotation),
            ),
        ],
    )

    dataset_request = _runner._dataset_request(settings)
    tile_request = _runner._tile_request(prepared, 2, "train")
    mlflow_request = _runner._mlflow_start_request(settings, TrainPipelineRequest())

    assert dataset_request.annotations_dir == str(annotations_dir)
    assert dataset_request.scenes_file is None
    assert tile_request.annotation_file is None
    assert [str(scene.annotation_file) for scene in tile_request.scenes] == [
        str(first_annotation),
        str(second_annotation),
    ]
    assert _runner._dataset_artifact_files(settings) == {
        "per_image/Olskij_SCN06.geojson": first_annotation.as_posix(),
        "per_image/Olskij_SCN07.geojson": second_annotation.as_posix(),
        "per_image/.mlsystem2-dataset.json": manifest.as_posix(),
    }
    assert mlflow_request.dataset == "test"
    assert mlflow_request.tags["class"] == "реки_test"


def test_train_pipeline_logs_epoch_metrics_from_progress_sink() -> None:
    logged_epochs: list[int] = []
    metrics = _train_result().history[0]
    model = ModelHandle(
        spec=ModelSpec(name="segformer_b0", input_channels=4, output_channels=1),
        model=object(),
    )

    def train_model(request, progress_sink=None):
        if progress_sink is not None:
            from mlsystem2.train.contracts import TrainProgressEvent

            progress_sink(TrainProgressEvent(epoch=1, message="epoch_finished", metrics=metrics))
        return _train_result()

    deps = _runner._PipelineDependencies(
        get_settings=lambda: _settings(initial_checkpoint_uri=None),
        get_settings_path=lambda: Path("config.yaml"),
        start_run=lambda request: MLflowRunRef(
            run_id="run",
            experiment_name=request.experiment_name,
            tracking_uri=request.tracking_uri,
            active=True,
        ),
        prepare_dataset=lambda request: _dataset_result(),
        create_tile_dataloader=lambda request: object(),
        create_model=lambda spec: model,
        load_checkpoint=lambda request: None,
        train_model=train_model,
        log_dataset_preparation=lambda run, report: None,
        log_tile_preparation=lambda run, report: None,
        log_run_config=lambda run, config_path: None,
        log_training_epoch=lambda run, item: logged_epochs.append(item.epoch),
        log_training_metrics=lambda run, result: None,
        log_training_artifacts=lambda run, result: None,
        log_timing_report=lambda run, report: None,
        log_pipeline_report=lambda run, report: None,
        end_run=lambda run, status: None,
    )

    result = _runner.run_train_pipeline(TrainPipelineRequest(), dependencies=deps)

    assert result.status.value == "succeeded"
    assert logged_epochs == [1]


def test_train_pipeline_marks_mlflow_run_killed_on_interrupt() -> None:
    ended_statuses: list[str] = []
    model = ModelHandle(
        spec=ModelSpec(name="segformer_b0", input_channels=4, output_channels=1),
        model=object(),
    )

    deps = _runner._PipelineDependencies(
        get_settings=lambda: _settings(initial_checkpoint_uri=None),
        get_settings_path=lambda: Path("config.yaml"),
        start_run=lambda request: MLflowRunRef(
            run_id="run",
            experiment_name=request.experiment_name,
            tracking_uri=request.tracking_uri,
            active=True,
        ),
        prepare_dataset=lambda request: _dataset_result(),
        create_tile_dataloader=lambda request: object(),
        create_model=lambda spec: model,
        load_checkpoint=lambda request: None,
        train_model=lambda request, progress_sink=None: (_ for _ in ()).throw(InterruptedError("stop")),
        log_dataset_preparation=lambda run, report: None,
        log_tile_preparation=lambda run, report: None,
        log_run_config=lambda run, config_path: None,
        log_training_epoch=lambda run, metrics: None,
        log_training_metrics=lambda run, result: None,
        log_training_artifacts=lambda run, result: None,
        log_timing_report=lambda run, report: None,
        log_pipeline_report=lambda run, report: None,
        end_run=lambda run, status: ended_statuses.append(status.value),
    )

    with pytest.raises(TrainPipelineError):
        _runner.run_train_pipeline(TrainPipelineRequest(), dependencies=deps)

    assert ended_statuses == ["KILLED"]


def test_train_pipeline_builds_multiclass_requests() -> None:
    settings = _multiclass_settings()
    model = ModelHandle(
        spec=ModelSpec(name="segformer_b2", input_channels=4, output_channels=3),
        model=object(),
    )
    prepared = PreparedDataset(
        format="legacy_multiclass",
        scenes=[PreparedScene(scene_id="scene", image_path="./scene.tif")],
        class_annotations=[
            DatasetClassAnnotation(
                class_id=1,
                slug="class_a",
                name="Класс А",
                annotation_file="./class_a.geojson",
                hard_negative_annotation_file="./class_a_hard_negative.geojson",
                priority=7,
            )
        ],
    )

    dataset_request = _runner._dataset_request(settings)
    tile_request = _runner._tile_request(prepared, 2, "train")
    train_request = _runner._train_request(settings, model, object(), object())

    assert dataset_request.classes is not None
    assert [item.slug for item in dataset_request.classes] == ["class_a", "class_b"]
    assert [item.priority for item in dataset_request.classes] == [5, 0]
    assert dataset_request.classes[0].hard_negative_annotation_file == (
        "./class_a_hard_negative.geojson"
    )
    assert tile_request.annotation_file is None
    assert [item.slug for item in tile_request.class_annotations] == ["class_a"]
    assert [item.priority for item in tile_request.class_annotations] == [7]
    assert tile_request.class_annotations[0].hard_negative_annotation_file == (
        "./class_a_hard_negative.geojson"
    )
    assert train_request.config.task == "multiclass"
    assert train_request.config.loss == "cross_entropy"
    assert train_request.config.hard_negative_weight == 1.0
    assert train_request.config.class_slugs == ["class_a", "class_b"]


def test_train_pipeline_builds_tile_split_requests() -> None:
    settings = _settings(initial_checkpoint_uri=None)
    settings.dataset.hard_negative_annotation_file = "./hard_negative.geojson"
    prepared = PreparedDataset(
        format="legacy_binary",
        scenes=[PreparedScene(scene_id="scene", image_path="./scene.tif")],
        annotation_file="./annotations.geojson",
        hard_negative_annotation_file="./hard_negative.geojson",
    )

    dataset_request = _runner._dataset_request(settings)
    tile_split = _runner._tile_split_request(settings)
    train_request = _runner._tile_request(
        prepared,
        2,
        "train",
        tile_split,
    )
    val_request = _runner._tile_request(
        prepared,
        2,
        "val",
        tile_split,
    )

    assert not hasattr(dataset_request, "negative_scene_limit")
    assert not hasattr(dataset_request, "split_granularity")
    assert dataset_request.hard_negative_annotation_file == "./hard_negative.geojson"
    assert tile_split is not None
    assert tile_split.val_fraction == settings.dataset.val_fraction
    assert tile_split.seed == settings.tile_preparation.seed
    assert train_request.scenes[0].image_path == "./scene.tif"
    assert val_request.scenes[0].scene_id == "scene"
    assert train_request.hard_negative_annotation_file == "./hard_negative.geojson"
    assert val_request.hard_negative_annotation_file == "./hard_negative.geojson"
    assert train_request.tile_split == tile_split
    assert val_request.tile_split == tile_split


def test_train_pipeline_passes_batch_limits_to_tile_requests() -> None:
    settings = _settings(initial_checkpoint_uri=None)
    settings.train.max_train_batches_per_epoch = 128
    settings.train.max_val_batches_per_epoch = 256
    prepared = PreparedDataset(
        format="legacy_binary",
        scenes=[PreparedScene(scene_id="scene", image_path="./scene.tif")],
        annotation_file="./annotations.geojson",
    )
    requests = []
    model = ModelHandle(
        spec=ModelSpec(name="segformer_b2", input_channels=4, output_channels=1),
        model=object(),
    )

    deps = _runner._PipelineDependencies(
        get_settings=lambda: settings,
        get_settings_path=lambda: Path("config.yaml"),
        start_run=lambda request: MLflowRunRef(
            run_id="disabled",
            experiment_name=request.experiment_name,
            tracking_uri=request.tracking_uri,
            active=False,
        ),
        prepare_dataset=lambda request: _dataset_result(dataset=prepared),
        create_tile_dataloader=lambda request: requests.append(request) or object(),
        create_model=lambda spec: model,
        load_checkpoint=lambda request: None,
        train_model=lambda request, progress_sink=None: _train_result(),
        log_dataset_preparation=lambda run, report: None,
        log_tile_preparation=lambda run, report: None,
        log_run_config=lambda run, config_path: None,
        log_training_epoch=lambda run, metrics: None,
        log_training_metrics=lambda run, result: None,
        log_training_artifacts=lambda run, result: None,
        log_timing_report=lambda run, report: None,
        log_pipeline_report=lambda run, report: None,
        end_run=lambda run, status: None,
    )

    result = _runner.run_train_pipeline(TrainPipelineRequest(), dependencies=deps)

    assert result.status.value == "succeeded"
    assert [request.mode for request in requests] == ["train", "val"]
    assert requests[0].max_batches_per_epoch == 128
    assert requests[1].max_batches_per_epoch == 256


def test_counting_loader_counts_observed_tiles_and_augmentations() -> None:
    class Dataset:
        scene_count = 1
        candidate_window_count = 5
        candidate_window_count_before_valid_filter = 7
        black_filtered_window_count = 2
        valid_footprint_stride = 64
        valid_footprint_valid_cells = 10
        valid_footprint_total_cells = 12
        estimated_positive_tiles = 2
        estimated_hard_negative_tiles = 1
        estimated_background_tiles = 2
        class_balance_enabled = False
        class_balance_warnings: list[str] = []

        def __len__(self) -> int:
            return 5

    class Images:
        def __init__(self, batch_size: int) -> None:
            self.shape = (batch_size, 1, 4, 4)

    class Loader:
        dataset = Dataset()

        def __iter__(self):
            yield (
                Images(2),
                object(),
                {
                    "augmented_tile_count": 1,
                    "positive_tile_count": 2,
                    "hard_negative_tile_count": 0,
                    "background_tile_count": 0,
                    "augmented_positive_tile_count": 1,
                    "augmented_hard_negative_tile_count": 0,
                },
            )
            yield (
                Images(1),
                object(),
                {
                    "positive_tile_count": 0,
                    "hard_negative_tile_count": 0,
                    "background_tile_count": 1,
                },
            )

        def __len__(self) -> int:
            return 2

    loader = _runner._CountingLoader(Loader(), "train")

    assert len(list(loader)) == 2
    snapshot = loader.snapshot()
    assert not {
        "uses_vrt_source_rects",
        "tile_split_enabled",
        "estimated_class_positive_tiles",
        "target_positive_factor",
        "is_diagnostic_sampling",
        "observed_class_pixel_counts",
        "observed_class_positive_tile_counts",
        "estimated_negative_tiles",
    } & set(snapshot)
    assert snapshot == {
        "tile_count": 5,
        "batch_count": 2,
        "scene_count": 1,
        "candidate_window_count": 5,
        "candidate_window_count_before_valid_filter": 7,
        "black_filtered_window_count": 2,
        "valid_footprint_stride": 64,
        "valid_footprint_valid_cells": 10,
        "valid_footprint_total_cells": 12,
        "pool_window_count": None,
        "split_window_count": None,
        "estimated_positive_tiles": 2,
        "estimated_hard_negative_tiles": 1,
        "estimated_background_tiles": 2,
        "sampling_mode": "sequential",
        "positive_factor_used": None,
        "hard_negative_factor_used": None,
        "background_factor_used": None,
        "cache_mode": None,
        "cached_batches": None,
        "cached_tiles": None,
        "selected_batches": None,
        "selected_tiles": None,
        "cache_estimated_bytes": None,
        "cache_limit_bytes": None,
        "cache_fallback_reason": None,
        "class_balance_enabled": False,
        "observed_batches": 2,
        "observed_tiles": 3,
        "observed_positive_tiles": 2,
        "observed_hard_negative_tiles": 0,
        "observed_background_tiles": 1,
        "observed_positive_ratio": 2 / 3,
        "observed_hard_negative_ratio": 0.0,
        "observed_background_ratio": 1 / 3,
        "positive_ratio_abs_error": None,
        "hard_negative_ratio_abs_error": None,
        "background_ratio_abs_error": None,
        "observed_augmented_tiles": 1,
        "observed_augmented_positive_tiles": 1,
        "observed_augmented_hard_negative_tiles": 0,
        "observed_real_tiles": 2,
        "warnings": [],
    }


def test_tile_preparation_report_exposes_three_train_factors() -> None:
    class SnapshotLoader:
        def __init__(self, split: str) -> None:
            self.split = split

        def snapshot(self) -> dict[str, object]:
            return {"split": self.split}

    settings = _settings(initial_checkpoint_uri=None)
    report = _runner._tile_preparation_report(
        settings,
        SnapshotLoader("train"),
        SnapshotLoader("val"),
    )

    assert report["positive_factor"] == 0.5
    assert report["hard_negative_factor"] == 0.0
    assert report["background_factor"] == 0.5
    assert report["context"] == 0
    assert report["core_size"] == 512
    assert report["seed"] == 42
    assert report["splits"]["train"] == {"split": "train"}
    assert report["splits"]["val"] == {"split": "val"}


def test_counting_loader_reports_target_positive_ratio() -> None:
    class Dataset:
        scene_count = 1
        candidate_window_count = 4
        candidate_window_count_before_valid_filter = 4
        black_filtered_window_count = 0
        valid_footprint_stride = 64
        valid_footprint_valid_cells = 4
        valid_footprint_total_cells = 4
        estimated_positive_tiles = 2
        estimated_hard_negative_tiles = 1
        estimated_background_tiles = 1
        class_balance_enabled = True
        class_balance_warnings: list[str] = []
        sampling_warnings = [
            "hard_negative_factor_used уменьшен из-за отсутствия или малого числа hard_negative tiles."
        ]

        def __len__(self) -> int:
            return 4

    class Images:
        shape = (4, 1, 4, 4)

    class Loader:
        dataset = Dataset()

        def __iter__(self):
            yield (
                Images(),
                object(),
                {
                    "positive_tile_count": 2,
                    "hard_negative_tile_count": 1,
                    "background_tile_count": 1,
                },
            )

        def __len__(self) -> int:
            return 1

    loader = _runner._CountingLoader(
        Loader(),
        "train",
        sampling_mode="weighted_class_balance",
        positive_factor_used=0.5,
        hard_negative_factor_used=0.25,
        background_factor_used=0.25,
    )

    list(loader)
    snapshot = loader.snapshot()

    assert snapshot["positive_factor_used"] == 0.5
    assert snapshot["hard_negative_factor_used"] == 0.25
    assert snapshot["background_factor_used"] == 0.25
    assert snapshot["observed_positive_ratio"] == 0.5
    assert snapshot["observed_hard_negative_ratio"] == 0.25
    assert snapshot["observed_background_ratio"] == 0.25
    assert snapshot["positive_ratio_abs_error"] == 0.0
    assert snapshot["hard_negative_ratio_abs_error"] == 0.0
    assert snapshot["background_ratio_abs_error"] == 0.0
    assert snapshot["warnings"] == Dataset.sampling_warnings
    assert snapshot["pool_window_count"] is None
    assert snapshot["split_window_count"] is None
    assert "target_positive_factor" not in snapshot
    assert "estimated_class_positive_tiles" not in snapshot
    assert "tile_split_enabled" not in snapshot


def test_counting_loader_reports_cached_balanced_val_metadata() -> None:
    class Dataset:
        scene_count = 1
        candidate_window_count = 4
        candidate_window_count_before_valid_filter = 4
        black_filtered_window_count = 0
        valid_footprint_stride = 64
        valid_footprint_valid_cells = 4
        valid_footprint_total_cells = 4
        estimated_positive_tiles = 2
        estimated_hard_negative_tiles = 1
        estimated_background_tiles = 1
        class_balance_enabled = False
        class_balance_warnings: list[str] = []

        def __len__(self) -> int:
            return 4

    class Images:
        shape = (4, 1, 4, 4)

    class Loader:
        dataset = Dataset()
        cache_mode = "memory"
        cached_batches = 1
        cached_tiles = 4
        selected_batches = 1
        selected_tiles = 4
        cache_estimated_bytes = 1024
        cache_limit_bytes = 2048
        cache_fallback_reason = None
        warnings: list[str] = []
        sampler = None

        def __iter__(self):
            yield (
                Images(),
                object(),
                {
                    "positive_tile_count": 2,
                    "hard_negative_tile_count": 1,
                    "background_tile_count": 1,
                },
            )

        def __len__(self) -> int:
            return 1

    raw_loader = Loader()
    loader = _runner._CountingLoader(
        raw_loader,
        "val",
        sampling_mode=_runner._sampling_mode(_settings(initial_checkpoint_uri=None), raw_loader),
    )

    list(loader)
    snapshot = loader.snapshot()

    assert snapshot["sampling_mode"] == "cached_balanced"
    assert snapshot["cache_mode"] == "memory"
    assert snapshot["cached_batches"] == 1
    assert snapshot["cached_tiles"] == 4
    assert snapshot["selected_batches"] == 1
    assert snapshot["selected_tiles"] == 4
    assert snapshot["cache_estimated_bytes"] == 1024
    assert snapshot["cache_limit_bytes"] == 2048
    assert snapshot["cache_fallback_reason"] is None
    assert snapshot["observed_positive_ratio"] == 0.5
    assert snapshot["observed_hard_negative_ratio"] == 0.25
    assert snapshot["observed_background_ratio"] == 0.25
    assert snapshot["positive_ratio_abs_error"] is None
    assert snapshot["hard_negative_ratio_abs_error"] is None
    assert snapshot["background_ratio_abs_error"] is None


def test_counting_loader_reports_lazy_balanced_val_metadata() -> None:
    warning = "Val tile cache не помещается; используется ленивое чтение."

    class Dataset:
        scene_count = 1
        candidate_window_count = 4
        candidate_window_count_before_valid_filter = 4
        black_filtered_window_count = 0
        valid_footprint_stride = 64
        valid_footprint_valid_cells = 4
        valid_footprint_total_cells = 4
        estimated_positive_tiles = 2
        estimated_hard_negative_tiles = 0
        estimated_background_tiles = 2
        class_balance_enabled = False
        class_balance_warnings: list[str] = []

        def __len__(self) -> int:
            return 4

    class Loader:
        dataset = Dataset()
        cache_mode = "lazy"
        cached_batches = 0
        cached_tiles = 0
        selected_batches = 1
        selected_tiles = 4
        cache_estimated_bytes = 4096
        cache_limit_bytes = 2048
        cache_fallback_reason = warning
        warnings = [warning]
        sampler = object()

        def __iter__(self):
            return iter(())

        def __len__(self) -> int:
            return 1

    raw_loader = Loader()
    loader = _runner._CountingLoader(
        raw_loader,
        "val",
        sampling_mode=_runner._sampling_mode(_settings(initial_checkpoint_uri=None), raw_loader),
    )

    snapshot = loader.snapshot()

    assert snapshot["sampling_mode"] == "lazy_balanced"
    assert snapshot["cache_mode"] == "lazy"
    assert snapshot["cached_batches"] == 0
    assert snapshot["cached_tiles"] == 0
    assert snapshot["selected_batches"] == 1
    assert snapshot["selected_tiles"] == 4
    assert snapshot["cache_estimated_bytes"] == 4096
    assert snapshot["cache_limit_bytes"] == 2048
    assert snapshot["cache_fallback_reason"] == warning
    assert warning in snapshot["warnings"]


def _settings(*, initial_checkpoint_uri: str | None) -> SystemSettings:
    return SystemSettings(
        runtime=RuntimeSettings(
            project_root=".",
            scratch_root="./scratch",
            logs_root="./logs",
            cleanup_scratch_after_mlflow_log=False,
        ),
        dataset=DatasetSettings(
            images_dir="./images",
            scenes_file="./scenes.txt",
            annotation_file="./annotations.geojson",
            val_fraction=0.2,
        ),
        tile_preparation=TilePreparationSettings(tile_size=512, stride=512),
        train=TrainSettings(
            model_name="segformer_b2",
            input_channels=4,
            output_channels=1,
            pretrained=False,
            initial_checkpoint_uri=initial_checkpoint_uri,
            epochs=1,
            batch_size=1,
            device="cpu",
            learning_rate=0.00001,
            weight_decay=0.0001,
            loss="bce_dice",
            early_stopping_patience=1,
        ),
        inference=InferenceSettings(
            checkpoint_uri="./checkpoint.pt",
            threshold=0.5,
            batch_size=1,
            device="cpu",
        ),
        mlflow=MLflowSettings(
            enabled=False,
            tracking_uri="./mlruns",
            experiment_name="test",
        ),
    )


def _multiclass_settings() -> SystemSettings:
    return SystemSettings(
        runtime=RuntimeSettings(
            project_root=".",
            scratch_root="./scratch",
            logs_root="./logs",
            cleanup_scratch_after_mlflow_log=False,
        ),
        dataset=DatasetSettings(
            images_dir="./images",
            classes=[
                DatasetClassSettings(
                    slug="class_a",
                    name="Класс А",
                    scenes_file="./class_a.txt",
                    annotation_file="./class_a.geojson",
                    hard_negative_annotation_file="./class_a_hard_negative.geojson",
                    priority=5,
                ),
                DatasetClassSettings(
                    slug="class_b",
                    name="Класс Б",
                    scenes_file="./class_b.txt",
                    annotation_file="./class_b.geojson",
                ),
            ],
            val_fraction=0.2,
        ),
        tile_preparation=TilePreparationSettings(tile_size=512, stride=512),
        train=TrainSettings(
            task="multiclass",
            model_name="segformer_b2",
            input_channels=4,
            output_channels=3,
            pretrained=False,
            initial_checkpoint_uri=None,
            epochs=1,
            batch_size=1,
            device="cpu",
            learning_rate=0.00001,
            weight_decay=0.0001,
            loss="cross_entropy",
            early_stopping_patience=1,
        ),
        inference=InferenceSettings(
            checkpoint_uri="./checkpoint.pt",
            threshold=0.5,
            batch_size=1,
            device="cpu",
        ),
        mlflow=MLflowSettings(
            enabled=False,
            tracking_uri="./mlruns",
            experiment_name="test",
        ),
    )


def _dataset_result(dataset: PreparedDataset | None = None) -> DatasetPreparationResult:
    return DatasetPreparationResult(
        dataset=dataset
        or PreparedDataset(
            format="legacy_binary",
            scenes=[PreparedScene(scene_id="scene", image_path="./scene.tif")],
            annotation_file="./annotations.geojson",
        ),
        report=DatasetPreparationReport(
            status="ok",
            scenes_total=0,
            scenes_found=0,
            positive_objects=0,
            hard_negative_objects=0,
            objects_total=0,
            scenes=[],
            missing_files=[],
            errors=[],
        ),
    )


def _train_result() -> TrainResult:
    return TrainResult(
        history=[
            EpochMetrics(
                epoch=1,
                train_loss=1.0,
                val_loss=1.0,
                epoch_time_sec=0.1,
            )
        ],
        epochs_total=1,
        training_time_sec=0.1,
        best_checkpoint_path="/tmp/best.pt",
        final_checkpoint_path="/tmp/final.pt",
    )
