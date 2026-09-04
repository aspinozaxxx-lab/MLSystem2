from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from mlsystem2.dataset_preparing._raster_validation import validate_rasters
from mlsystem2.mlflow_adapter import _client as mlflow_client
from mlsystem2.mlflow_adapter.contracts import MLflowRunRef
from mlsystem2.models import _factory
from mlsystem2.models.api import load_checkpoint, save_checkpoint
from mlsystem2.models.contracts import (
    LoadCheckpointRequest,
    ModelHandle,
    ModelSpec,
    SaveCheckpointRequest,
)
from mlsystem2.tile_preparation._augmentations import apply_next_gen_augmentations
from mlsystem2.tile_preparation._dataset import TileDataset
from mlsystem2.tile_preparation.contracts import TileSceneSource, TileSplitRequest
from mlsystem2.train import _trainer
from mlsystem2.train.api import train_model
from mlsystem2.train.contracts import TrainConfig, TrainRequest, TrainResult
from mlsystem2.train_pipeline._next_gen import preprocessing_parameters
from mlsystem2.training_ui_api import _model_export


def test_scene_fold_assigns_each_of_six_scenes_to_validation_once(tmp_path: Path) -> None:
    annotation = _write_empty_annotation(tmp_path / "annotation.geojson")
    scenes = _scenes(tmp_path, {f"SCN{index:02d}": (20 + index, index * 10) for index in range(6)})
    held_out: list[str] = []

    for fold in range(6):
        dataset = TileDataset(
            scenes=scenes,
            annotation_file=annotation,
            tile_size=4,
            stride=4,
            mode="val",
            seed=42,
            augmentation_level=0,
            pipeline_variant="next_gen",
            tile_split=TileSplitRequest(
                strategy="scene_fold",
                val_fraction=0.2,
                seed=42,
                validation_fold=fold,
            ),
        )
        manifest = dataset.tile_split_manifest
        assert manifest["fold_count"] == 6
        assert len(dataset) == 1
        assert len(manifest["validation_scene_ids"]) == 1
        held_out.extend(manifest["validation_scene_ids"])
        dataset.close()

    assert sorted(held_out) == sorted(scene.scene_id for scene in scenes)


def test_scene_fold_spatial_purge_removes_only_intersecting_train_window(
    tmp_path: Path,
) -> None:
    annotation = _write_empty_annotation(tmp_path / "annotation.geojson")
    scene_ids = ["SCN03", "SCN04", "SCN09"]
    ordered = sorted(
        scene_ids,
        key=lambda scene_id: hashlib.sha256(f"7{scene_id}".encode()).digest(),
    )
    held_out = ordered[0]
    overlap = ordered[1]
    safe = ordered[2]
    scenes = _scenes(
        tmp_path,
        {
            held_out: (50, 0),
            overlap: (60, 0),
            safe: (70, 100),
        },
    )

    dataset = TileDataset(
        scenes=scenes,
        annotation_file=annotation,
        tile_size=4,
        stride=4,
        mode="train",
        seed=7,
        augmentation_level=0,
        pipeline_variant="next_gen",
        tile_split=TileSplitRequest(
            strategy="scene_fold",
            val_fraction=0.2,
            seed=7,
            validation_fold=0,
            spatial_purge=True,
        ),
    )

    manifest = dataset.tile_split_manifest
    assert manifest["validation_scene_ids"] == [held_out]
    assert manifest["purged_windows_by_scene"] == {overlap: 1}
    assert manifest["purged_window_count"] == 1
    assert manifest["geographic_overlap_after_purge"] == 0
    assert [item.scene_id for item in dataset._windows] == [safe]
    dataset.close()


def test_robust_histogram_reads_train_scenes_without_validation_leakage(
    tmp_path: Path,
) -> None:
    annotation = _write_empty_annotation(tmp_path / "annotation.geojson")
    scene_ids = ["A", "B", "C"]
    held_out = sorted(
        scene_ids,
        key=lambda scene_id: hashlib.sha256(f"11{scene_id}".encode()).digest(),
    )[0]
    values = {scene_id: (250 if scene_id == held_out else 10, index * 10) for index, scene_id in enumerate(scene_ids)}
    dataset = TileDataset(
        scenes=_scenes(tmp_path, values),
        annotation_file=annotation,
        tile_size=4,
        stride=4,
        mode="train",
        seed=11,
        augmentation_level=0,
        pipeline_variant="next_gen",
        collect_band_histogram=True,
        tile_split=TileSplitRequest(
            strategy="scene_fold",
            val_fraction=0.2,
            seed=11,
            validation_fold=0,
        ),
    )

    histogram = dataset.band_histogram
    assert histogram is not None
    assert held_out not in histogram["scene_ids"]
    assert all(channel[250] == 0 for channel in histogram["counts"])
    profile = preprocessing_parameters("robust_percentile", histogram)
    assert profile["low"] == [10.0] * 4
    assert profile["high"] == [11.0] * 4
    dataset.close()


def test_next_gen_augmentation_is_draw_aware_and_reproducible() -> None:
    image = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
    mask = np.zeros((1, 8, 8), dtype=np.float32)
    mask[:, 2:6, 2:6] = 1
    invalid = np.zeros((8, 8), dtype=bool)

    first = apply_next_gen_augmentations(
        image,
        mask,
        nodata_pixels=invalid,
        nodata=0,
        level=3,
        seed=17,
        sample_key="epoch=1/draw=2/scene=A/window=0,0",
    )
    repeated = apply_next_gen_augmentations(
        image,
        mask,
        nodata_pixels=invalid,
        nodata=0,
        level=3,
        seed=17,
        sample_key="epoch=1/draw=2/scene=A/window=0,0",
    )
    next_draw = apply_next_gen_augmentations(
        image,
        mask,
        nodata_pixels=invalid,
        nodata=0,
        level=3,
        seed=17,
        sample_key="epoch=1/draw=3/scene=A/window=0,0",
    )

    assert all(np.array_equal(left, right) for left, right in zip(first[:3], repeated[:3]))
    assert not np.array_equal(first[0], next_draw[0])


def test_preprocessing_wrapper_zeroes_nodata_after_imagenet_normalization() -> None:
    torch = pytest.importorskip("torch")

    class Capture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, value):
            self.seen = value
            return value[:, :1]

    spec = ModelSpec(
        name="smp_segformer_b0",
        input_channels=4,
        output_channels=1,
        parameters={
            "pipeline_variant": "next_gen",
            "preprocessing": preprocessing_parameters("imagenet_rgb_red_nir", None),
        },
    )
    capture = Capture()
    model = _factory._wrap_next_gen(spec, capture).model
    raw = torch.zeros((1, 4, 1, 2), dtype=torch.float32)
    raw[:, :, :, 1] = 255

    logits = model(raw)

    assert torch.equal(capture.seen[:, :, :, 0], torch.zeros((1, 4, 1)))
    expected = torch.tensor(
        [(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225, (1 - 0.485) / 0.229]
    )
    assert torch.allclose(capture.seen[0, :, 0, 1], expected)
    assert logits[0, 0, 0, 0] == -1000


def test_hf_four_channel_adapter_copies_red_weights_to_nir() -> None:
    torch = pytest.importorskip("torch")
    projection = torch.nn.Conv2d(3, 2, kernel_size=3, padding=1)
    with torch.no_grad():
        projection.weight.copy_(torch.arange(projection.weight.numel()).reshape_as(projection.weight))
    fake = SimpleNamespace(
        segformer=SimpleNamespace(
            encoder=SimpleNamespace(
                patch_embeddings=[SimpleNamespace(proj=projection)]
            )
        ),
        decode_head=SimpleNamespace(classifier=torch.nn.Conv2d(2, 150, kernel_size=1)),
        config=SimpleNamespace(),
    )

    _factory._adapt_pretrained_segformer(fake, 4, 1)

    adapted = fake.segformer.encoder.patch_embeddings[0].proj
    assert adapted.in_channels == 4
    assert torch.equal(adapted.weight[:, :3], projection.weight)
    assert torch.equal(adapted.weight[:, 3], projection.weight[:, 0])
    assert fake.decode_head.classifier.out_channels == 1


def test_hf_checkpoint_load_is_offline_and_preserves_logits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    calls = {"from_pretrained": 0}

    class FakeConfig:
        def __init__(self, **values):
            self.__dict__.update(values)

        def to_dict(self):
            return dict(self.__dict__)

    class FakeSegformer(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.conv = torch.nn.Conv2d(config.num_channels, config.num_labels, 1)

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            calls["from_pretrained"] += 1
            raise AssertionError("Сетевая загрузка checkpoint не разрешена")

        def forward(self, value):
            return SimpleNamespace(logits=self.conv(value))

    transformers = ModuleType("transformers")
    transformers.SegformerConfig = FakeConfig
    transformers.SegformerForSemanticSegmentation = FakeSegformer
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    spec = ModelSpec(
        name="segformer_b0",
        input_channels=4,
        output_channels=1,
        pretrained=True,
        parameters={
            "pipeline_variant": "next_gen",
            "preprocessing": {"mode": "scale_255", "nodata": 0.0},
            "hf_config": {"num_channels": 4, "num_labels": 1},
        },
    )
    original = _factory.create_model_for_checkpoint(spec)
    sample = torch.full((1, 4, 4, 4), 127.0)
    expected = original.model(sample).detach()
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(
        SaveCheckpointRequest(
            model=original,
            checkpoint_uri=str(checkpoint),
            metadata={"pipeline_variant": "next_gen"},
        )
    )

    loaded = load_checkpoint(
        LoadCheckpointRequest(checkpoint_uri=str(checkpoint), map_location="cpu")
    )
    actual = loaded.model.model(sample).detach()

    assert calls["from_pretrained"] == 0
    assert torch.equal(expected, actual)


def test_real_hf_b0_checkpoint_round_trip_is_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    spec = ModelSpec(
        name="segformer_b0",
        input_channels=4,
        output_channels=1,
        pretrained=False,
        parameters={
            "pipeline_variant": "next_gen",
            "preprocessing": {"mode": "scale_255", "nodata": 0.0},
        },
    )
    original = _factory.create_model(spec)
    sample = torch.full((1, 4, 32, 32), 127.0)
    sample[:, :, :2, :] = 0
    original.model.eval()
    with torch.no_grad():
        expected = original.model(sample)
    checkpoint = tmp_path / "real-hf-b0.pt"
    save_checkpoint(
        SaveCheckpointRequest(
            model=original,
            checkpoint_uri=str(checkpoint),
            metadata={"pipeline_variant": "next_gen"},
        )
    )

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    loaded = load_checkpoint(
        LoadCheckpointRequest(checkpoint_uri=str(checkpoint), map_location="cpu")
    )
    loaded.model.model.eval()
    with torch.no_grad():
        actual = loaded.model.model(sample)

    assert isinstance(original.spec.parameters.get("hf_config"), dict)
    assert expected.shape == (1, 1, 32, 32)
    assert torch.all(expected[:, :, :2, :] == -1000)
    assert torch.equal(expected, actual)

    onnx = pytest.importorskip("onnx")
    onnx_path = tmp_path / "real-hf-b0.onnx"
    _model_export._export_binary_mask_onnx(
        model=loaded.model.model,
        input_channels=4,
        sample_size=32,
        threshold=0.5,
        onnx_path=onnx_path,
    )
    exported = onnx.load_model(onnx_path)
    channel_dimension = exported.graph.output[0].type.tensor_type.shape.dim[1]
    assert channel_dimension.dim_value == 1
    assert channel_dimension.dim_param == ""


def test_next_gen_loss_ignores_invalid_pixels() -> None:
    torch = pytest.importorskip("torch")
    config = _next_gen_train_config(epochs=1)
    masks = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    masks[:, :, 0, 0] = 1
    valid = torch.zeros_like(masks, dtype=torch.bool)
    valid[:, :, 0, 0] = True
    first_logits = torch.zeros_like(masks)
    second_logits = first_logits.clone()
    second_logits[:, :, 1, 1] = 100
    second_masks = masks.clone()
    second_masks[:, :, 1, 1] = 1

    first = _trainer._loss(torch, first_logits, masks, config, valid_pixels=valid)
    second = _trainer._loss(torch, second_logits, masks, config, valid_pixels=valid)
    third = _trainer._loss(torch, first_logits, second_masks, config, valid_pixels=valid)

    assert torch.equal(first, second)
    assert torch.equal(first, third)


def test_next_gen_fixed_threshold_is_applied_exactly() -> None:
    torch = pytest.importorskip("torch")
    configured_threshold = 0.5001
    probabilities = torch.tensor([[[[0.50011, 0.50009]]]], dtype=torch.float32)

    class FixedProbabilities(torch.nn.Module):
        def forward(self, _images):
            return torch.logit(probabilities)

    images = torch.ones((1, 4, 1, 2), dtype=torch.float32)
    masks = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    loader = [
        (
            images,
            masks,
            {
                "valid_pixels": torch.ones_like(masks, dtype=torch.bool),
                "scene_ids": ["scene-a"],
            },
        )
    ]
    config = _next_gen_train_config(epochs=1).model_copy(
        update={"threshold_mode": "fixed", "threshold": configured_threshold}
    )

    metrics = _trainer._validate_epoch(
        torch,
        FixedProbabilities(),
        loader,
        torch.device("cpu"),
        config,
        1,
    )

    assert metrics["best_threshold"] == configured_threshold
    assert metrics["best_threshold_pixel_precision"] == 1.0
    assert metrics["best_threshold_pixel_recall"] == 1.0
    assert metrics["best_threshold_pixel_f1"] == 1.0
    assert metrics["fixed_0_5_pixel_f1"] == pytest.approx(2 / 3)


def test_legacy_checkpoint_config_omits_next_gen_defaults() -> None:
    config = TrainConfig(
        epochs=1,
        batch_size=1,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss="bce_dice",
        early_stopping_patience=1,
    )

    serialized = _trainer._checkpoint_train_config(config)

    assert "pipeline_variant" not in serialized
    assert "validation_interval_epochs" not in serialized
    assert "threshold_mode" not in serialized
    assert "evaluate_gaussian_blend" not in serialized


def test_legacy_cosine_scheduler_golden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(_trainer, "_validate_epoch", _constant_binary_validation)
    config = TrainConfig(
        epochs=4,
        batch_size=2,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss="bce_dice",
        early_stopping_patience=10,
    )

    result = train_model(
        TrainRequest(
            model=_tiny_model_handle(torch),
            train_loader=_next_gen_loader(torch),
            val_loader=_next_gen_loader(torch),
            config=config,
            checkpoint_dir=str(tmp_path / "legacy-checkpoints"),
            sample_size=4,
        )
    )

    assert [item.learning_rate for item in result.history] == pytest.approx(
        [0.0008535533905932737, 0.0005, 0.00014644660940672628, 0.0]
    )
    checkpoint = torch.load(result.best_checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    assert "pipeline_variant" not in metadata
    assert "run_metadata" not in metadata
    assert "validation_performed" not in metadata
    assert "val_per_scene_metrics" not in metadata
    assert "model_parameters" not in metadata
    assert "pipeline_variant" not in metadata["train_config"]


def test_next_gen_plateau_scheduler_counts_validation_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(_trainer, "_validate_epoch", _constant_binary_validation)
    config = _next_gen_train_config(epochs=5).model_copy(
        update={"validation_interval_epochs": 1}
    )

    result = train_model(
        TrainRequest(
            model=_tiny_model_handle(torch),
            train_loader=_next_gen_loader(torch),
            val_loader=_next_gen_loader(torch),
            config=config,
            checkpoint_dir=str(tmp_path / "next-gen-checkpoints"),
            sample_size=4,
        )
    )

    assert [item.learning_rate for item in result.history] == pytest.approx(
        [0.001, 0.001, 0.001, 0.001, 0.0005]
    )


def test_next_gen_validation_cadence_is_first_interval_and_final(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 1, 1)

        def forward(self, value):
            return self.conv(value)

    loader = _next_gen_loader(torch)
    result = train_model(
        TrainRequest(
            model=ModelHandle(
                spec=ModelSpec(name="smp_segformer_b0", input_channels=4, output_channels=1),
                model=Tiny(),
            ),
            train_loader=loader,
            val_loader=loader,
            config=_next_gen_train_config(epochs=6),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            sample_size=4,
        )
    )

    assert [item.validation_performed for item in result.history] == [
        True,
        False,
        False,
        False,
        True,
        True,
    ]
    assert result.history[1].val_loss is None
    assert result.history[1].val_quality_f1 is None
    assert result.history[1].val_best_threshold is None
    assert result.history[0].val_macro_pixel_f1 is not None


def test_raster_band_contract_warns_when_absent_and_rejects_conflict(tmp_path: Path) -> None:
    absent = tmp_path / "absent.tif"
    conflict = tmp_path / "conflict.tif"
    _write_raster(absent, value=50, x=0)
    _write_raster(conflict, value=50, x=0, descriptions=("BLU", "GRN", "RED", "NIR"))
    expected = ["RED", "GRN", "BLU", "NIR"]

    warning = validate_rasters(
        {"absent": absent},
        expected_band_count=4,
        expected_dtype="uint8",
        expected_band_names=expected,
    )
    error = validate_rasters(
        {"conflict": conflict},
        expected_band_count=4,
        expected_dtype="uint8",
        expected_band_names=expected,
    )

    assert warning.errors == []
    assert len(warning.warnings) == 1
    assert "принят контракт RED, GRN, BLU, NIR" in warning.warnings[0]
    assert error.errors and "Порядок описанных каналов" in error.errors[0]


def test_next_gen_mlflow_artifacts_include_reproducibility_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dictionaries: list[str] = []
    artifacts: list[tuple[str, str]] = []
    params: list[dict[str, object]] = []
    tags: dict[str, str] = {}

    class MLflow:
        @staticmethod
        def log_params(values):
            params.append(dict(values))

        @staticmethod
        def set_tag(key, value):
            tags[str(key)] = str(value)

    monkeypatch.setattr(mlflow_client, "_ensure_run_active", lambda _run: MLflow)
    monkeypatch.setattr(
        mlflow_client,
        "_log_dict",
        lambda _value, artifact_path: dictionaries.append(artifact_path),
    )
    monkeypatch.setattr(
        mlflow_client,
        "_log_artifact",
        lambda path, artifact_path: artifacts.append((str(path), artifact_path)),
    )
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"next-gen-checkpoint")
    result = TrainResult(
        history=[],
        epochs_total=0,
        training_time_sec=0,
        best_checkpoint_path=str(checkpoint),
        diagnostics={
            "pipeline_variant": "next_gen",
            "validation_fold": 2,
            "resolved_train_config": {"train": {"pipeline_variant": "next_gen"}},
            "split_manifest": {"validation_scene_ids": ["SCN03"]},
            "preprocessing": {"mode": "scale_255"},
            "runtime_environment": {"commit": "abc"},
            "validation_by_scene": {"events": []},
            "inference_merge_comparison": {"production_merge_unchanged": "core_crop"},
            "flattened_params": {"train.pipeline_variant": "next_gen"},
        },
    )

    mlflow_client.log_training_artifacts(
        MLflowRunRef(
            run_id="run",
            experiment_name="test",
            tracking_uri="file://mlruns",
            active=True,
        ),
        result,
    )

    assert {
        "config/resolved_train_config.json",
        "reports/split_manifest.json",
        "reports/preprocessing.json",
        "reports/runtime_environment.json",
        "reports/validation_by_scene.json",
        "reports/inference_merge_comparison.json",
        "reports/checkpoint_hashes.json",
    }.issubset(dictionaries)
    assert artifacts == [(str(checkpoint), "checkpoints")]
    assert params == [{"train.pipeline_variant": "next_gen"}]
    assert tags == {
        "pipeline_variant": "next_gen",
        "validation_fold": "2",
        "code_commit": "abc",
        "preprocessing": "scale_255",
    }


def _next_gen_train_config(*, epochs: int) -> TrainConfig:
    return TrainConfig(
        pipeline_variant="next_gen",
        validation_interval_epochs=5,
        threshold_mode="optimize",
        epochs=epochs,
        batch_size=2,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        loss="bce_dice",
        threshold=0.5,
        early_stopping_patience=10,
    )


def _next_gen_loader(torch):
    images = torch.full((2, 4, 4, 4), 100.0)
    masks = torch.zeros((2, 1, 4, 4))
    masks[0, :, 1:3, 1:3] = 1
    return [
        (
            images,
            masks,
            {
                "valid_pixels": torch.ones((2, 1, 4, 4), dtype=torch.bool),
                "scene_ids": ["scene-a", "scene-b"],
            },
        )
    ]


def _tiny_model_handle(torch) -> ModelHandle:
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 1, 1)

        def forward(self, value):
            return self.conv(value)

    return ModelHandle(
        spec=ModelSpec(name="smp_segformer_b0", input_channels=4, output_channels=1),
        model=Tiny(),
    )


def _constant_binary_validation(*_args, **_kwargs) -> dict[str, object]:
    return {
        "loss": 1.0,
        "best_threshold": 0.5,
        "best_pixel_threshold": 0.5,
        "best_threshold_pixel_f1": 0.5,
        "best_threshold_pixel_precision": 0.5,
        "best_threshold_pixel_recall": 0.5,
        "best_threshold_precision": 0.5,
        "best_threshold_recall": 0.5,
        "best_threshold_object_f1": None,
        "best_threshold_object_precision": None,
        "best_threshold_object_recall": None,
        "quality_f1": 0.5,
        "quality_precision": 0.5,
        "quality_recall": 0.5,
        "macro_pixel_f1": 0.5,
        "macro_pixel_precision": 0.5,
        "macro_pixel_recall": 0.5,
        "micro_pixel_f1": 0.5,
        "micro_pixel_precision": 0.5,
        "micro_pixel_recall": 0.5,
        "per_scene_metrics": [],
        "metric_warnings": [],
    }


def _scenes(tmp_path: Path, values_and_x: dict[str, tuple[int, int]]) -> list[TileSceneSource]:
    scenes: list[TileSceneSource] = []
    for scene_id, (value, x) in values_and_x.items():
        path = tmp_path / f"{scene_id}.tif"
        _write_raster(path, value=value, x=x)
        scenes.append(TileSceneSource(scene_id=scene_id, image_path=str(path)))
    return scenes


def _write_raster(
    path: Path,
    *,
    value: int,
    x: int,
    descriptions: tuple[str, str, str, str] | None = None,
) -> None:
    data = np.full((4, 4, 4), value, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=4,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(x, 4, 1, 1),
        nodata=0,
    ) as dataset:
        dataset.write(data)
        if descriptions is not None:
            for index, description in enumerate(descriptions, start=1):
                dataset.set_band_description(index, description)


def _write_empty_annotation(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
                "features": [],
            }
        ),
        encoding="utf-8",
    )
    return path
