"""Общая train-нормализация, отсутствие утечки и сохранение маски исходных данных."""
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box, mapping

from mlsystem2.models import _factory
from mlsystem2.models.api import create_model, load_checkpoint, save_checkpoint
from mlsystem2.models.contracts import ModelSpec, ModelsError, LoadCheckpointRequest, SaveCheckpointRequest
from mlsystem2.tile_preparation._dataset import TileDataset
from mlsystem2.tile_preparation.contracts import TileSceneSource, TileSplitRequest, TileDataloaderRequest
from mlsystem2.train_pipeline._next_gen import global_minmax_parameters
from mlsystem2.train_pipeline import _runner
from mlsystem2.training_ui_api import _model_export, _service
from mlsystem2.training_ui_api._templates import initial_templates, sanitize_template_config

VARIANT = "next_gen4_global_norm"


@pytest.fixture(autouse=True, params=["next_gen4_global_norm", "next_gen4_augmentation_v3"])
def normalization_profile(request, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "VARIANT", request.param)


def dataset(tmp_path, mode="train", *, channels=4, workers=0, collect=True):
    path, annotation = tmp_path / "image.tif", tmp_path / "objects.geojson"
    if not path.exists():
        image = np.zeros((channels, 128, 160), dtype=np.uint8)
        for y in range(4):
            for x in range(5):
                image[:, y * 32:(y + 1) * 32, x * 32:(x + 1) * 32] = np.arange(channels)[:, None, None] + 40 + y * 5 + x
        with rasterio.open(path, "w", driver="GTiff", width=160, height=128, count=channels,
            dtype="uint8", transform=from_origin(0, 128, 1, 1), crs="EPSG:3857", nodata=0) as output:
            output.write(image)
            output.colorinterp = (rasterio.enums.ColorInterp.gray,) + (rasterio.enums.ColorInterp.undefined,) * (channels - 1)
            mask = np.full((128, 160), 255, dtype=np.uint8)
            mask[8:12, :] = 0
            output.write_mask(mask)
        annotation.write_text(json.dumps({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
            "features": [{"type": "Feature", "properties": {}, "geometry": mapping(box(0, 0, 160, 128))}]}), encoding="utf-8")
    return TileDataset(scenes=[TileSceneSource(scene_id="снимок", image_path=path, annotation_file=annotation)],
        tile_size=32, stride=32, mode=mode, seed=42, augmentation_level=0, pipeline_variant=VARIANT,
        input_channels=channels, tile_split=TileSplitRequest(val_fraction=0.2, test_fraction=0.2),
        collect_band_histogram=collect and mode == "train", statistics_workers=workers)


@pytest.mark.parametrize("channels", [3, 4])
def test_statistics_use_only_train_pixels_and_ignore_invalid(tmp_path, channels):
    held_out = [dataset(tmp_path, mode, channels=channels, collect=False) for mode in ("val", "test")]
    with rasterio.open(tmp_path / "image.tif", "r+") as source:
        for part in held_out:
            for tile in part._windows:
                source.write(np.full((channels, 32, 32), 250, dtype=np.uint8), window=Window(tile.window.x, tile.window.y, 32, 32))
        # Экстремум под raster mask тоже не должен попасть в границы.
        source.write(np.full((channels, 4, 160), 255, dtype=np.uint8), window=Window(0, 8, 160, 4))
    train = dataset(tmp_path, channels=channels)
    histogram = train.band_histogram
    parameters = global_minmax_parameters(histogram, channels)
    assert histogram["source"] == "retained_train_windows"
    assert histogram["window_count"] == len(train)
    assert max(parameters["high"]) < 100
    expected = np.zeros((channels, 256), dtype=np.uint64)
    valid_pixels = 0
    with rasterio.open(tmp_path / "image.tif") as source:
        for tile in train._windows:
            window = Window(tile.window.x, tile.window.y, 32, 32)
            image = source.read(window=window)
            valid = (source.dataset_mask(window=window) != 0) & np.any(image != 0, axis=0)
            valid_pixels += int(valid.sum())
            for index in range(channels):
                expected[index] += np.bincount(image[index][valid], minlength=256).astype(np.uint64)
    np.testing.assert_array_equal(histogram["counts"], expected)
    assert histogram["valid_pixel_count"] == valid_pixels
    with pytest.raises(ValueError, match="train-тайлов"):
        global_minmax_parameters({**histogram, "source": "retained_train_scenes"}, channels)
    for value in [train, *held_out]:
        value.close()


def test_parallel_statistics_exactly_match_single_process(tmp_path):
    first = dataset(tmp_path, workers=0)
    expected = first.band_histogram
    weights, tiles = first.notebook_class_weights, first.adaptive_sampling_tiles()
    first.close()
    second = dataset(tmp_path, workers=2)
    assert second.band_histogram == expected
    assert second.notebook_class_weights == weights and second.adaptive_sampling_tiles() == tiles
    second.close()


@pytest.mark.parametrize("nodata", [None, 0, 255])
@pytest.mark.parametrize("zoned", [False, True])
def test_alpha_raw_nodata_and_padding_are_excluded_from_global_statistics(tmp_path, nodata, zoned):
    path, annotation = tmp_path / "rgba.tif", tmp_path / "objects.geojson"
    height, width = 70, 90
    values = np.full((4, height, width), 255, dtype=np.uint8)
    if nodata == 255:
        values[3] = 254  # Валидная альфа не должна одновременно быть объявлена nodata.
    values[:3] = np.arange(3, dtype=np.uint8)[:, None, None] + 40
    values[:3, 4:8] = 240  # Яркие пиксели под прозрачной альфой не участвуют в статистике.
    values[3, 4:8] = 0
    values[:3, 12:16] = 0  # Исходный ноль не равен валидному нормализованному нулю.
    if nodata is not None:
        values[:3, 20:24] = nodata
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=4,
        dtype="uint8", transform=from_origin(0, height, 1, 1), crs="EPSG:3857", nodata=nodata) as output:
        output.write(values)
        output.colorinterp = (rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.alpha)
    geometry = mapping(box(0, 0, width, height))
    annotation.write_text(json.dumps({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [{"type": "Feature", "properties": {}, "geometry": geometry}]}), encoding="utf-8")
    scene = TileSceneSource(scene_id="снимок", image_path=path, annotation_file=annotation,
        **({"region_geometry": geometry, "region_window": (0, 0, width, height)} if zoned else {}))
    train = TileDataset(scenes=[scene], tile_size=32, stride=32, mode="train", seed=42,
        augmentation_level=0, pipeline_variant=VARIANT, input_channels=3,
        tile_split=TileSplitRequest(val_fraction=0.2, test_fraction=0.2),
        collect_band_histogram=True, statistics_workers=0)
    bounds = global_minmax_parameters(train.band_histogram, 3)
    assert bounds["low"] == bounds["high"] == [40.0, 41.0, 42.0]
    expected_count = 0
    saw_padding = False
    with rasterio.open(path) as source:
        for index, tile in enumerate(train._windows):
            window = Window(tile.window.x, tile.window.y, 32, 32)
            raw = source.read((1, 2, 3), window=window, boundless=True, fill_value=0)
            valid = ((source.dataset_mask(window=window, boundless=True) != 0)
                & (source.read(4, window=window, boundless=True, fill_value=0) != 0)
                & np.any(raw != 0, axis=0)
                & np.any(raw != (nodata if nodata is not None else 0), axis=0))
            image, _, meta = train[index]
            np.testing.assert_array_equal(meta["valid_pixels"], valid)
            assert not image[:, ~valid].any()
            expected_count += int(valid.sum())
            saw_padding |= tile.window.x + 32 > width or tile.window.y + 32 > height
    assert saw_padding and train.band_histogram["valid_pixel_count"] == expected_count
    train.close()


def parameters(channels=4):
    return {"mode": "global_minmax", "low": [40.0] * channels, "high": [200.0] * channels,
        "epsilon": 1e-6, "nodata": 0.0, "mask_output_nodata": True,
        "source": "retained_train_windows", "valid_pixel_count": 100, "window_count": 10}


def test_global_scale_does_not_depend_on_window_and_valid_zero_is_not_nodata():
    torch = pytest.importorskip("torch")
    class Observe(torch.nn.Module):
        def forward(self, image):
            return torch.cat((image[:, :1] * 0, image[:, :1]), dim=1)
    model = _factory._InputPreprocessingWrapper(Observe(), parameters())
    uniform = torch.full((1, 4, 32, 32), 80.0)
    mixed = uniform.clone()
    mixed[:, :, :16] = 160
    assert model(uniform)[0, 0, 20, 20] == pytest.approx(0.25)
    torch.testing.assert_close(model(uniform)[:, :, 16:], model(mixed)[:, :, 16:])
    class Foreground(torch.nn.Module):
        def forward(self, image):
            return torch.cat((image[:, :1] * 0, image[:, :1] * 0 + 10), dim=1)
    model = _factory._InputPreprocessingWrapper(Foreground(), parameters())
    uniform.fill_(40)  # Все нормализованные каналы нулевые, но исходные пиксели валидны.
    uniform[:, :, :4] = 0
    output = model(uniform)
    assert output[..., :4, :].max() < -900
    assert output[..., 4:, :].min() == 10
    constant = parameters()
    constant["high"] = constant["low"]
    assert torch.isfinite(_factory._InputPreprocessingWrapper(Observe(), constant)(uniform)).all()


@pytest.mark.parametrize("channels", [3, 4])
def test_checkpoint_restores_global_bounds_and_onnx_matches_pytorch(tmp_path, monkeypatch, channels):
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    reference = pytest.importorskip("onnx.reference")
    torch.set_num_threads(2)
    def local_model(spec, **kwargs):
        return torch.nn.Conv2d(spec.input_channels, 2, 1)
    monkeypatch.setattr(_factory, "_create_smp_segformer", local_model)
    preprocessing = parameters(channels)
    spec = ModelSpec(name="smp_segformer_b0", input_channels=channels, output_channels=1,
        parameters={"pipeline_variant": VARIANT, "preprocessing": preprocessing,
            **({"augmentation_policy": "validity_guard_v3"} if VARIANT == "next_gen4_augmentation_v3" else {})})
    model = create_model(spec)
    path = tmp_path / "best.pt"
    save_checkpoint(SaveCheckpointRequest(model=model, checkpoint_uri=str(path)))
    loaded = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=str(path), map_location="cpu"))
    assert loaded.model.spec.parameters["preprocessing"] == preprocessing
    if VARIANT == "next_gen4_augmentation_v3":
        assert loaded.model.spec.parameters["augmentation_policy"] == "validity_guard_v3"
    image = torch.randint(1, 255, (1, channels, 32, 32)).float()
    image[..., :4, :] = 0
    expected = loaded.model.model(image).sigmoid().detach().numpy()
    torch.testing.assert_close(model.model(image), loaded.model.model(image), rtol=0, atol=0)
    target = tmp_path / "model.onnx"
    _model_export._export_segmentation_mask_onnx(model=loaded.model.model, input_channels=channels,
        output_channels=1, sample_size=32, threshold=0.9, onnx_path=target, probability_output=True)
    graph = onnx.load(target)
    onnx.checker.check_model(graph)
    assert not any(node.op_type in {"ReduceMin", "ReduceMax"} for node in graph.graph.node)
    actual = reference.ReferenceEvaluator(graph).run(None, {"input": image.numpy()})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert not actual[..., :4, :].any()
    with pytest.raises(ModelsError, match="train-границы"):
        create_model(spec.model_copy(update={"parameters": {**spec.parameters, "preprocessing": {"mode": "window_minmax"}}}))


@pytest.mark.parametrize("size", [512, 768, 1024, 1536])
def test_profile_available_without_switching_old_templates(size):
    config = sanitize_template_config({"train.pipeline_variant": VARIANT, "tile_preparation.tile_size": size}, architecture="smp_segformer_b1")
    assert config["train.pipeline_variant"] == VARIANT and config["tile_preparation.stride"] == size
    _service._validate_training_pipeline_variant({**config, "dataset.task": "binary", "dataset.imagery_type": "kanopus", "train.input_channels": 4}, "smp_segformer_b1")
    templates = initial_templates()
    selected = next(item for item in templates if item["architecture"] == "smp_segformer_b1")
    assert VARIANT in selected["config_schema"]["fields"][0]["options"]
    assert selected["config_schema"]["pipeline_defaults"][VARIANT]["train.pipeline_variant"] == VARIANT
    assert all(item["default_config"]["train.pipeline_variant"] == "legacy" for item in templates)


def test_model_spec_reads_train_histogram_and_val_cannot_collect_it(tmp_path):
    train = dataset(tmp_path)
    settings = SimpleNamespace(train=SimpleNamespace(pipeline_variant=VARIANT, input_channels=4,
        output_channels=1, pretrained=False, model_name="smp_segformer_b0"))
    result = _runner._model_spec(settings, SimpleNamespace(dataset=train))
    assert result.parameters["preprocessing"] == global_minmax_parameters(train.band_histogram, 4)
    if VARIANT == "next_gen4_augmentation_v3":
        assert result.parameters["augmentation_policy"] == "validity_guard_v3"
    with pytest.raises(ValueError, match="только на train"):
        TileDataloaderRequest(scenes=[TileSceneSource(scene_id="снимок", image_path="image.tif", annotation_file="truth.geojson")],
            batch_size=1, mode="val", pipeline_variant=VARIANT, collect_band_histogram=True)
    train.close()
