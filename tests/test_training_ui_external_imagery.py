from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box

from mlsystem2.training_ui_api import _external_imagery, _pseudo_runner


def _png_bytes() -> bytes:
    output = BytesIO()
    image = np.zeros((256, 256, 4), dtype=np.uint8)
    image[:, :, :3] = (10, 20, 30)
    image[:, :, 3] = 255
    Image.fromarray(image, mode="RGBA").save(output, format="PNG")
    return output.getvalue()


def test_xyz_and_tms_use_opposite_y_axis() -> None:
    aoi = box(30.0, 59.0, 30.01, 59.01)
    settings = {
        "url_template": "https://tiles.example/{z}/{x}/{y}.png",
        "min_zoom": 8,
        "max_zoom": 8,
        "context_pixels": 1,
    }
    xyz = _external_imagery._slippy_grid("xyz", settings, aoi, 500.0, "xyz")
    tms = _external_imagery._slippy_grid("tms", settings, aoi, 500.0, "tms")
    xyz_y = int(xyz.requests[0].url.rsplit("/", 1)[-1].split(".", 1)[0])
    tms_y = int(tms.requests[0].url.rsplit("/", 1)[-1].split(".", 1)[0])
    assert tms_y == 2**8 - 1 - xyz_y


def test_external_xyz_is_written_only_to_job_scratch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(_external_imagery, "_request_bytes", lambda *args, **kwargs: _png_bytes())
    result = _external_imagery.prepare_external_imagery(
        {
            "source_id": "local_xyz",
            "source_kind": "xyz",
            "source_protocol": "xyz",
            "source_attribution": "Тестовый XYZ",
            "source_license_url": "https://example.test/license",
            "source_settings": {
                "protocol": "xyz",
                "url_template": "https://tiles.example/{z}/{x}/{y}.png",
                "min_zoom": 8,
                "max_zoom": 8,
                "context_pixels": 1,
                "auth": {"type": "none"},
            },
            "target_resolution_m": 500.0,
            "external_http_workers": 2,
        },
        box(30.0, 59.0, 30.01, 59.01),
        tmp_path / "job-scratch",
    )
    paths = list(result.images_root.glob("*.tif"))
    assert len(paths) == 1
    assert result.coverage_percent == 100.0
    assert paths[0].is_relative_to(tmp_path / "job-scratch")
    with rasterio.open(paths[0]) as dataset:
        assert dataset.count == 4
        assert dataset.colorinterp[3] == ColorInterp.alpha


def test_ortho_alpha_is_mask_and_never_nir(tmp_path: Path) -> None:
    image_path = tmp_path / "ortho.tif"
    rgb_alpha = np.zeros((4, 4, 4), dtype=np.uint8)
    rgb_alpha[0] = 10
    rgb_alpha[1] = 20
    rgb_alpha[2] = 30
    rgb_alpha[3] = 255
    rgb_alpha[3, 0, 0] = 0
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=4,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 4, 1, 1),
    ) as dataset:
        dataset.write(rgb_alpha)
        dataset.colorinterp = (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        )
    with rasterio.open(image_path) as dataset:
        tile = _pseudo_runner._read_inference_window(
            dataset,
            image_path,
            Window(0, 0, 4, 4),
            4,
            input_channels=4,
            channel_mapping="rgb_zero_nir",
            source_imagery_type="ortho",
        )
    assert tile is not None
    assert np.all(tile[3] == 0)
    assert np.all(tile[:3, 0, 0] == 0)
    assert tuple(tile[:3, 1, 1]) == (10.0, 20.0, 30.0)


def test_inference_uses_configured_batch(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "scene.tif"
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=3,
        dtype="uint8",
        nodata=0,
        crs="EPSG:3857",
        transform=from_origin(0, 8, 1, 1),
    ) as output:
        output.write(np.ones((3, 8, 8), dtype=np.uint8))
    batch_sizes: list[int] = []

    def predict(_torch, _model, images, **_kwargs):
        batch_sizes.append(images.shape[0])
        confidence = np.ones((images.shape[0], 4, 4), dtype=np.float32)
        return confidence.astype(np.uint8), confidence

    monkeypatch.setattr(_pseudo_runner, "_predict_tiles", predict)
    with rasterio.open(image_path) as dataset:
        mask = np.zeros((8, 8), dtype=np.uint8)
        confidence = np.zeros((8, 8), dtype=np.float32)
        _pseudo_runner._infer_windows_into_mask(
            dataset=dataset,
            input_indexes=(1, 2, 3),
            nodata=0,
            windows=[
                Window(0, 0, 4, 4),
                Window(4, 0, 4, 4),
                Window(0, 4, 4, 4),
                Window(4, 4, 4, 4),
            ],
            mask_window=Window(0, 0, 8, 8),
            mask=mask,
            confidence_map=confidence,
            tile_size=4,
            torch=object(),
            model=object(),
            threshold=0.5,
            device="cpu",
            batch_size=2,
        )
    assert batch_sizes == [2, 2]
    assert np.all(mask == 1)
