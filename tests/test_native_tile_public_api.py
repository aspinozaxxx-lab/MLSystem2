"""Публичный однотайловый вход использует штатный runner, не очередь."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Для вычислительных проверок необходим необязательный PyTorch")

from mlsystem2.models.contracts import CheckpointArtifact, LoadedCheckpoint, ModelHandle, ModelSpec
from mlsystem2.training_ui_api.api import infer_native_tile
from mlsystem2.training_ui_api.contracts import (
    NativeTileInferenceRequest, NativeTileInferenceResult, TrainingUIAPIError,
)
from mlsystem2.training_ui_api import _pseudo_runner as runner


class WaterModel(torch.nn.Module):
    def forward(self, inputs, *, return_two_class_logits=False):
        assert return_two_class_logits
        output = torch.zeros((inputs.shape[0], 2, *inputs.shape[-2:]), dtype=inputs.dtype, device=inputs.device)
        output[:, 1] = 8
        return output


def checkpoint(*, variant="next_gen4_consistency_v5", sample_size=1024, output_channels=1):
    return LoadedCheckpoint(
        model=ModelHandle(spec=ModelSpec(name="smp_segformer_b2", input_channels=4,
            output_channels=output_channels, parameters={"pipeline_variant": variant}), model=WaterModel().eval()),
        artifact=CheckpointArtifact(uri="проверочные_веса.pt", format="torch",
            metadata={"sample_size": sample_size, "pipeline_variant": variant}),
    )


def test_public_native_tile_delegates_exact_profile_without_queue(monkeypatch):
    loaded = checkpoint()
    expected = np.ones((23, 31), dtype=np.uint8)
    captured = {}

    def infer(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(runner, "_infer_test_tile_mask", infer)
    monkeypatch.setattr(runner, "load_checkpoint", lambda *_a, **_k: pytest.fail("Повторная загрузка весов запрещена"))
    postprocess = {"postprocess.min_area_m2": 100, "postprocess.simplify_m": 1,
        "postprocess.filter_compact_objects.enabled": True,
        "postprocess.filter_compact_objects.mode": "remove_compact",
        "postprocess.filter_compact_objects.min_isoperimetric_quotient": .1,
        "postprocess.filter_compact_objects.max_bbox_ratio": 3.5}
    result = infer_native_tile(NativeTileInferenceRequest(loaded=loaded, image_path="тайл.tif", postprocess_config=postprocess))
    assert isinstance(result, NativeTileInferenceResult) and result.mask is expected
    assert captured["model"] is loaded.model.model and captured["image_path"] == Path("тайл.tif")
    assert captured["tile_size"] == 1024 and captured["stride"] == 512 and captured["context"] == 0
    assert captured["threshold"] == .9 and captured["batch_size"] == 2
    assert captured["notebook"] is True and captured["pad_edges"] is True and captured["device"] == "cpu"
    assert captured["input_channels"] == 4
    assert captured["postprocess_profile"].min_area_m2 == 100
    assert captured["postprocess_profile"].filter_compact_min_isoperimetric_quotient == .1
    assert postprocess["postprocess.min_area_m2"] == 100
    assert result.effective_config["inference_merge"] == "gaussian_probabilities"
    assert not loaded.model.model.training


@pytest.mark.parametrize("variant", ["legacy", "object_f1", "object_f1_v2"])
def test_public_native_tile_rejects_other_output_algorithms(monkeypatch, variant):
    monkeypatch.setattr(runner, "_infer_test_tile_mask", lambda **_kw: pytest.fail("Обработка не должна начинаться"))
    with pytest.raises(TrainingUIAPIError, match="бинарный Gaussian"):
        infer_native_tile(NativeTileInferenceRequest(loaded=checkpoint(variant=variant), image_path="тайл.tif"))


def test_public_native_tile_rejects_multiclass(monkeypatch):
    monkeypatch.setattr(runner, "_infer_test_tile_mask", lambda **_kw: pytest.fail("Обработка не должна начинаться"))
    with pytest.raises(TrainingUIAPIError, match="бинарный Gaussian"):
        infer_native_tile(NativeTileInferenceRequest(loaded=checkpoint(output_channels=3), image_path="тайл.tif"))


@pytest.mark.parametrize("sample_size", [None, 0, 513, 1024.0, True])
def test_public_native_tile_requires_saved_window(monkeypatch, sample_size):
    monkeypatch.setattr(runner, "_infer_test_tile_mask", lambda **_kw: pytest.fail("Обработка не должна начинаться"))
    with pytest.raises(TrainingUIAPIError, match="размер окна"):
        infer_native_tile(NativeTileInferenceRequest(loaded=checkpoint(sample_size=sample_size), image_path="тайл.tif"))


def test_public_native_tile_rejects_training_mode(monkeypatch):
    loaded = checkpoint()
    loaded.model.model.train()
    monkeypatch.setattr(runner, "_infer_test_tile_mask", lambda **_kw: pytest.fail("Обработка не должна начинаться"))
    with pytest.raises(TrainingUIAPIError, match="eval"):
        infer_native_tile(NativeTileInferenceRequest(loaded=loaded, image_path="тайл.tif"))
    assert loaded.model.model.training


def test_public_native_tile_real_tiff_preserves_valid_mask(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    image = np.full((4, 29, 31), 100, dtype=np.uint8)
    image[:, 8:13, 12:17] = 0
    path = tmp_path / "тайл.tif"
    with rasterio.open(path, "w", driver="GTiff", height=29, width=31, count=4,
        dtype="uint8", nodata=0, crs="EPSG:3857", transform=from_origin(100, 200, 1, 1)) as target:
        target.write(image)
    result = infer_native_tile(NativeTileInferenceRequest(loaded=checkpoint(sample_size=512), image_path=str(path)))
    assert result.mask.dtype == np.uint8 and result.mask.shape == (29, 31)
    np.testing.assert_array_equal(result.mask > 0, np.any(image != 0, axis=0))
    assert result.effective_config["tile_size"] == 512 and result.effective_config["stride"] == 256


def test_public_native_tile_contract_has_no_queue_or_checkpoint_reload_fields():
    assert set(NativeTileInferenceRequest.model_fields) == {"loaded", "image_path", "device", "postprocess_config"}
    assert set(NativeTileInferenceResult.model_fields) == {"mask", "effective_config"}
    with pytest.raises(ValueError):
        NativeTileInferenceRequest(loaded=checkpoint(), image_path="тайл.tif", device="auto")
