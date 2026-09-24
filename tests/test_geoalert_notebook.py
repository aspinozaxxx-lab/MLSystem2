"""Оконное объединение Geoalert должно совпадать с полным массивом ноутбука."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from mlsystem2.training_ui_api._geoalert_notebook import (
    _sliding_window_probabilities,
    _split_unmasked,
)


def _image(tmp_path, channels, height, width):
    image = np.random.default_rng(42).integers(1, 256, (channels, height, width), dtype=np.uint8)
    path = tmp_path / "input.tif"
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height,
                       count=channels, dtype="uint8", crs="EPSG:3857",
                       transform=from_origin(100, 500, 0.5, 0.5), nodata=0) as target:
        target.write(image)
        mask = np.full((height, width), 255, dtype=np.uint8)
        mask[3:12, 3:12] = 0
        target.write_mask(mask)
    return image, path


@pytest.mark.parametrize("channels", [3, 4])
def test_split_preserves_pixels_under_tiff_mask(tmp_path, channels):
    image, path = _image(tmp_path, channels, 55, 69)
    labels = ["RED", "GRN", "BLU", "NIR"][:channels]
    _split_unmasked(path, tmp_path, labels)
    for index, label in enumerate(labels):
        with rasterio.open(tmp_path / f"{label}.tif") as band:
            np.testing.assert_array_equal(band.read(1), image[index])
            assert band.nodata is None
            assert band.transform == from_origin(100, 500, 0.5, 0.5)


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("height,width,stride", [(55, 69, 16), (65, 67, 32), (32, 32, 16), (20, 40, 16), (40, 20, 16)])
def test_stripe_merge_matches_full_notebook_arrays(tmp_path, channels, height, width, stride):
    image, path = _image(tmp_path, channels, height, width)
    inputs = ["RED", "GRN", "BLU", "NIR"][:channels]
    _split_unmasked(path, tmp_path, inputs)
    size = 32
    calls = []

    def predict(tile):
        # Пространственно неоднородный выход выявляет ошибки сдвига полос и взвешивания.
        values = tile.mean(axis=0, keepdims=True) / 255
        return (values * np.linspace(0.2, 1.0, size, dtype=np.float32)[None, None, :]).astype(np.float32)

    def adapter(tile):
        assert tile.dtype == np.float32
        calls.append(1)
        return predict(tile)

    _sliding_window_probabilities(tmp_path, inputs, ["probability"], adapter, size, stride, 0.25)
    expected = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros_like(expected)
    axis = np.arange(size)
    gaussian = np.exp(-((axis - size / 2) ** 2) / (2 * (size / 4) ** 2))
    weight = np.outer(gaussian, gaussian).astype(np.float32)
    windows = 0
    for y in range(0, height - size + 1, stride):
        for x in range(0, width - size + 1, stride):
            tile = image[:, y:y + size, x:x + size].astype(np.float32)
            expected[y:y + size, x:x + size] += predict(tile)[0] * weight
            counts[y:y + size, x:x + size] += weight
            windows += 1
    expected /= np.maximum(counts, 1e-6)
    with rasterio.open(tmp_path / "probability.tif") as result:
        np.testing.assert_array_equal(result.read(1), expected)
        assert result.dtypes == ("float32",)
        assert result.crs.to_epsg() == 3857
    assert len(calls) == windows


@pytest.mark.parametrize("bad_output", [np.zeros((1, 16, 16)), np.full((1, 32, 32), np.nan)])
def test_invalid_adapter_output_is_rejected(tmp_path, bad_output):
    _, path = _image(tmp_path, 3, 32, 32)
    _split_unmasked(path, tmp_path, ["RED", "GRN", "BLU"])
    with pytest.raises(ValueError, match="некорректную карту вероятностей"):
        _sliding_window_probabilities(tmp_path, ["RED", "GRN", "BLU"], ["probability"],
                                      lambda _: bad_output, 32, 16, 0.25)
