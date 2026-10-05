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
        "id", "training_result_id", "model_name", "source_dataset_name", "training_dataset_name", "created_at",
        "geojson_url", "object_count", "expected_image_count", "scenes", "warnings",
    }
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    paths = get_openapi_schema()["paths"]
    assert set(paths["/api/v1/results/pseudo-markup/{result_id}/view"]) == {"get"}
    assert set(paths["/api/v1/results/pseudo-markup/{result_id}/raster/{scene_id}"]) == {"get"}
    assert set(paths["/api/v1/results/pseudo-markup/{result_id}/footprint/{scene_id}"]) == {"get"}
    raster_parameters = paths["/api/v1/results/pseudo-markup/{result_id}/raster/{scene_id}"]["get"]["parameters"]
    assert any(item["name"] == "v" and item["in"] == "query" and item["required"] is False for item in raster_parameters)
