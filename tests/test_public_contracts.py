from __future__ import annotations

import importlib


EXPECTED_API = {
    "settings.api": ["load_settings", "get_settings", "get_settings_path"],
    "dataset_preparing.api": [
        "prepare_dataset",
        "resolve_scene_images",
        "footprint_name_for_annotation",
        "is_per_image_footprint_name",
        "per_image_annotation_name",
        "per_image_annotation_files",
        "per_image_footprint_name",
        "resolve_per_image_annotations",
        "load_dataset_manifest",
        "annotation_regions",
    ],
    "tile_preparation.api": ["create_tile_dataloader"],
    "models.api": ["list_supported_models", "create_model", "load_checkpoint", "save_checkpoint"],
    "metrics.api": ["compute_object_f1", "compute_pixel_f1", "summarize_epoch_metrics"],
    "train.api": ["train_model"],
    "mlflow_adapter.api": [
        "list_experiments",
        "create_experiment",
        "get_best_training_checkpoint",
        "get_usable_training_checkpoint",
        "get_finished_run_artifact",
        "get_training_epoch_progress",
        "download_run_artifact",
        "start_run",
        "log_dataset_preparation",
        "log_dataset_artifacts",
        "log_tile_preparation",
        "log_run_config",
        "log_training_epoch",
        "log_training_metrics",
        "log_training_artifacts",
        "log_timing_report",
        "log_pipeline_report",
        "end_run",
        "mark_run_killed",
    ],
    "train_pipeline.api": ["run_train_pipeline"],
    "inference.api": ["run_inference", "create_object_scene", "separate_objects", "object_window_origins"],
    "inference_pipeline.api": ["run_inference_pipeline"],
    "training_ui_api.api": ["create_app", "get_openapi_schema", "main", "worker_main", "feedback_main"],
}


def test_public_api_all_is_exact() -> None:
    for module_name, expected in EXPECTED_API.items():
        module = importlib.import_module(f"mlsystem2.{module_name}")
        assert list(module.__all__) == expected


def test_usage_config_contract(monkeypatch) -> None:
    from mlsystem2.training_ui_api.api import get_openapi_schema
    from mlsystem2.training_ui_api.contracts import UsageConfig

    assert set(UsageConfig.model_fields) == {"metrica_counter_id", "metrica_user_id"}
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    schema = get_openapi_schema()
    assert "get" in schema["paths"]["/api/v1/usage/config"]
    assert "/api/v1/usage/report" not in schema["paths"]


def test_inference_template_imagery_and_class_contract(monkeypatch) -> None:
    from mlsystem2.training_ui_api.api import get_openapi_schema
    from mlsystem2.training_ui_api.contracts import InferenceTemplate, InferenceTemplateCreate

    assert set(InferenceTemplateCreate.model_fields) == {"display_name", "description"}
    assert {"display_name", "description", "class_keys"} <= InferenceTemplate.model_fields.keys()
    assert not {"architecture", "dataset_key", "dataset_name"} & InferenceTemplate.model_fields.keys()
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    schema = get_openapi_schema()
    assert "/api/v1/inference-templates/{architecture}" not in schema["paths"]
    assert "put" in schema["paths"]["/api/v1/dataset-classes/{class_key}/inference-template"]
    assert "/api/v1/inference-templates/roots/{imagery_type}" not in schema["paths"]
    assert schema["components"]["schemas"]["ImageryType"]["enum"] == ["kanopus", "ortho"]


def test_completed_training_continuation_contract(monkeypatch) -> None:
    from mlsystem2.training_ui_api.api import get_openapi_schema
    from mlsystem2.training_ui_api.contracts import TrainingContinuationCreate, TrainingResultInfo

    assert set(TrainingContinuationCreate.model_fields) == {
        "additional_epochs", "additional_time_sec", "early_stopping_patience", "checkpoint", "request_id",
        "run_inference_after_training", "secondary_priority",
    }
    assert {"can_continue_training", "continued_from_result_id", "continued_from_checkpoint"} <= TrainingResultInfo.model_fields.keys()
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    schema = get_openapi_schema()
    path = schema["paths"]["/api/v1/results/training/{result_id}/continue"]
    assert {"get", "post"} <= path.keys()
    properties = schema["components"]["schemas"]["TrainingContinuationCreate"]["properties"]
    assert properties["checkpoint"]["enum"] == ["best", "last"]
    assert properties["additional_epochs"]["exclusiveMinimum"] == 0
    assert properties["request_id"]["format"] == "uuid"


def test_test_markup_creation_queue_contract(monkeypatch) -> None:
    from mlsystem2.training_ui_api.api import get_openapi_schema
    from mlsystem2.training_ui_api.contracts import TestSampleBatchCreate, TestSampleCreationSettings

    assert TestSampleBatchCreate.model_json_schema()["properties"]["items"]["maxItems"] == 1
    assert set(TestSampleCreationSettings.model_fields) == {
        "tile_size", "min_image_count", "image_count", "min_object_count",
        "use_optimization", "exclude_boundary_objects", "min_object_area_m2",
    }
    sizes = [512, 768, 1024, 1536, 2048, 2560, 3072, 3584, *range(4096, 8193, 512)]
    for contract in (TestSampleBatchCreate, TestSampleCreationSettings):
        assert contract.model_json_schema()["properties"]["tile_size"]["enum"] == sizes
        assert contract.model_json_schema()["properties"]["tile_size"]["default"] == 1536
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    schema = get_openapi_schema()
    paths = schema["paths"]
    imagery_type = schema["components"]["schemas"]["TestSampleBatchDatasetOption"]["properties"]["imagery_type"]
    assert imagery_type["anyOf"] == [{"$ref": "#/components/schemas/ImageryType"}, {"type": "null"}]
    for name in ("TestSampleCreationSettings", "TestSampleBatchItemCreate", "TestSampleCreate"):
        area = schema["components"]["schemas"][name]["properties"]["min_object_area_m2"]
        assert area["minimum"] == 0 and area["default"] == 0
    assert {"get", "post"} <= paths["/api/v1/test-sample-batches"].keys()
    settings = paths["/api/v1/test-sample-batches/options/{dataset_key}/settings"]["put"]
    assert settings["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith("/TestSampleCreationSettings")
    assert "delete" in paths["/api/v1/test-sample-batches/{batch_id}"]
    move = paths["/api/v1/test-sample-batches/{batch_id}/move"]["post"]
    assert move["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith("/TestSampleBatchMove")


def test_test_markup_fast_catalog_contract(monkeypatch) -> None:
    from mlsystem2.training_ui_api.api import get_openapi_schema
    from mlsystem2.training_ui_api.contracts import TestSampleCard, TestSampleSummary, TestSampleClassIndexItem

    assert set(TestSampleCard.model_fields) == set(TestSampleSummary.model_fields) - {"pseudo_markup"}
    assert set(TestSampleClassIndexItem.model_fields) == {"key", "name", "sample_count", "has_primary"}
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    paths = get_openapi_schema()["paths"]
    assert {p["name"] for p in paths["/api/v1/test-samples/classes"]["get"]["parameters"]} == {"include_empty"}
    assert {p["name"] for p in paths["/api/v1/test-samples/cards"]["get"]["parameters"]} == {"class_key"}
    assert "204" in paths["/api/v1/test-samples/classes/{class_key}/reconcile"]["post"]["responses"]
    assert {p["name"] for p in paths["/api/v1/test-sample-batches/options"]["get"]["parameters"]} == {"class_key"}
    assert "post" in paths["/api/v1/test-sample-batches/{batch_id}/cancel"]


def test_dataset_editor_import_contract(monkeypatch) -> None:
    from mlsystem2.training_ui_api.api import get_openapi_schema
    from mlsystem2.training_ui_api.contracts import DatasetEditorImportRequest, DatasetEditorImportSceneRequest

    assert set(DatasetEditorImportRequest.model_fields) == {"scenes", "clip_to_footprint"}
    assert DatasetEditorImportRequest(scenes=[{"annotation_name": "image.geojson", "geojson": {}}]).clip_to_footprint is False
    assert set(DatasetEditorImportSceneRequest.model_fields) == {"annotation_name", "geojson"}
    assert DatasetEditorImportRequest.model_json_schema()["properties"]["scenes"]["maxItems"] == 100
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    schema = get_openapi_schema()
    endpoint = schema["paths"]["/api/v1/dataset-editor/datasets/{dataset_key}/drafts/import"]["post"]
    assert endpoint["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith("/DatasetEditorImportRequest")
    assert endpoint["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/DatasetEditorSceneListResponse")
    assert endpoint["responses"]["400"]["content"]["application/json"]["schema"]["properties"]["code"]["enum"] == ["annotation_outside_footprint"]


def test_test_sample_annotation_merge_contract(monkeypatch) -> None:
    from mlsystem2.training_ui_api.api import get_openapi_schema
    from mlsystem2.training_ui_api.contracts import TestSampleAnnotationsMerge, TestSampleTileMerge

    assert set(TestSampleAnnotationsMerge.model_fields) == {"expected_revision", "name", "tiles"}
    assert set(TestSampleTileMerge.model_fields) == {"tile_index", "groups"}
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    schema = get_openapi_schema()
    assert "post" in schema["paths"]["/api/v1/test-samples/{sample_id}/merge-annotations"]
    assert "content_revision" in schema["components"]["schemas"]["TestSampleDetail"]["properties"]


def test_next_gen2_uses_existing_public_contracts() -> None:
    from mlsystem2.settings.contracts import TrainSettings
    from mlsystem2.tile_preparation.contracts import TileDataloaderRequest, TileSplitRequest
    from mlsystem2.train.contracts import TrainConfig
    from mlsystem2.training_ui_api.contracts import ConfigSchema, JobDetail, JobSummary, TrainingResultInfo

    for dto in (TrainSettings, TrainConfig, TileDataloaderRequest, JobDetail, JobSummary, TrainingResultInfo):
        assert dto.model_json_schema()["properties"]["pipeline_variant"]["enum"] == [
            "legacy", "next_gen", "next_gen2", "object_f1",
        ]
    assert TileSplitRequest.model_json_schema()["properties"]["strategy"]["enum"] == [
        "window_random", "scene_fold", "scene_groups",
    ]
    assert ConfigSchema(fields=[]).pipeline_defaults == {}
    assert ConfigSchema(fields=[]).pipeline_descriptions == {}
    assert TileSplitRequest.model_json_schema()["properties"]["test_fraction"]["default"] == 0.0
    assert TileDataloaderRequest.model_json_schema()["properties"]["mode"]["enum"] == ["train", "val", "test"]
    assert TrainConfig.model_fields["class_weights"].default_factory() == []


def test_pseudo_markup_view_contract(monkeypatch) -> None:
    from mlsystem2.training_ui_api.api import get_openapi_schema
    from mlsystem2.training_ui_api.contracts import PseudoMarkupSceneInfo, PseudoMarkupViewInfo

    assert set(PseudoMarkupSceneInfo.model_fields) == {"id", "name", "raster_url", "footprint_url", "bounds", "has_alpha", "has_nir", "nodata"}
    assert set(PseudoMarkupViewInfo.model_fields) == {
        "id", "training_result_id", "checkpoint_epoch", "model_name", "source_dataset_name", "training_dataset_name", "created_at",
        "geojson_url", "object_count", "expected_image_count", "scenes", "warnings", "class_name",
    }
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    paths = get_openapi_schema()["paths"]
    assert set(paths["/api/v1/results/pseudo-markup/{result_id}/view"]) == {"get"}
    assert set(paths["/api/v1/results/pseudo-markup/{result_id}/raster/{scene_id}"]) == {"get"}
    assert set(paths["/api/v1/results/pseudo-markup/{result_id}/footprint/{scene_id}"]) == {"get"}
    raster_parameters = paths["/api/v1/results/pseudo-markup/{result_id}/raster/{scene_id}"]["get"]["parameters"]
    assert any(item["name"] == "v" and item["in"] == "query" and item["required"] is False for item in raster_parameters)


def test_pseudo_markup_comparison_contract(monkeypatch) -> None:
    from mlsystem2.training_ui_api.api import get_openapi_schema
    from mlsystem2.training_ui_api.contracts import (
        PseudoMarkupComparisonRequest, PseudoMarkupPixelComparison,
        PseudoMarkupComparisonCounts, PseudoMarkupComparisonLayers,
        PseudoMarkupComparisonViewport,
    )

    assert set(PseudoMarkupComparisonRequest.model_fields) == {"result_ids", "scene_revisions", "scene_id", "viewport"}
    assert set(PseudoMarkupComparisonViewport.model_fields) == {"bounds", "width", "height"}
    assert set(PseudoMarkupPixelComparison.model_fields) == {"intersection", "only_first", "only_second"}
    assert set(PseudoMarkupComparisonCounts.model_fields) == {"result_ids", "scenes", "total", "warnings"}
    assert set(PseudoMarkupComparisonLayers.model_fields) == {"scene_id", "available_result_ids", "geojson", "counts"}
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    paths = get_openapi_schema()["paths"]
    assert set(paths["/api/v1/results/pseudo-markup/compare/counts"]) == {"post"}
    assert set(paths["/api/v1/results/pseudo-markup/compare/{scene_id}/layers"]) == {"post"}


def test_training_test_f1_view_contract(monkeypatch) -> None:
    from mlsystem2.training_ui_api.api import get_openapi_schema
    from mlsystem2.training_ui_api.contracts import TestF1ViewInfo, TestF1SceneInfo, TestF1ScoreInfo, PseudoMarkupSceneInfo

    assert set(TestF1ScoreInfo.model_fields) == {
        "precision", "recall", "f1", "true_positive", "false_positive", "false_negative",
    }
    assert set(TestF1SceneInfo.model_fields) == set(PseudoMarkupSceneInfo.model_fields) | {
        "layers_url", "raster_available", "sample_name", "sample_revision", "target_class_id",
        "class_schema", "pixel", "objects", "metrics",
    }
    assert set(TestF1ViewInfo.model_fields) == {
        "training_result_id", "model_name", "training_dataset_name", "status", "metric", "scenes", "warnings",
    }
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    paths = get_openapi_schema()["paths"]
    prefix = "/api/v1/results/training/{result_id}/test-f1"
    assert set(paths[f"{prefix}/view"]) == {"get", "post"}
    for suffix in ("raster", "footprint", "layers"):
        assert set(paths[f"{prefix}/{suffix}/{{scene_id}}"] ) == {"get"}
    parameters = paths[f"{prefix}/layers/{{scene_id}}"]["get"]["parameters"]
    assert any(item["name"] == "class_id" and not item["required"] for item in parameters)
