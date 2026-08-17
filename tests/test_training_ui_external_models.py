from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from shapely.geometry import box, shape

from mlsystem2.training_ui_api._external_models import (
    ExternalModelError,
    ExternalModelManifest,
    LoadedExternalModel,
    _open_resampled_dataset,
    load_external_model,
    merge_external_instance_features,
    predict_external_test_tile,
    validate_external_archive,
)
from mlsystem2.training_ui_api._model_export import build_external_triton_model_export_zip


def _archive(
    path: Path,
    *,
    unsafe_name: str | None = None,
    model_bytes: bytes = b"torchscript",
) -> str:
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr(
            "sample/config.pbtxt",
            'name: "sample"\nplatform: "pytorch_libtorch"\n',
        )
        archive.writestr("sample/1/model.pt", model_bytes)
        if unsafe_name is not None:
            archive.writestr(unsafe_name, b"bad")
    return sha256(path.read_bytes()).hexdigest()


def _oks_manifest(archive_hash: str, *, model_root: str = "sample") -> ExternalModelManifest:
    return ExternalModelManifest(
        adapter="oks_multiclass_footprints",
        artifact_path="models/model.zip",
        archive_sha256=archive_hash,
        model_member=f"{model_root}/1/model.pt",
        model_root=model_root,
        target_resolution_m=1.0,
        tile_size=4,
        stride=2,
        context=1,
        min_area_m2=0.0,
        min_hole_area_m2=0.0,
        max_shift_m=50.0,
        shift_iterations=2,
        shift_confidence=0.2,
        correction_confidence=0.05,
    )


def _zu_manifest(archive_hash: str) -> ExternalModelManifest:
    return ExternalModelManifest(
        adapter="detectron2_instances",
        artifact_path="models/model.zip",
        archive_sha256=archive_hash,
        model_member="sample/1/model.pt",
        model_root="sample",
        target_resolution_m=1.0,
        tile_size=4,
        stride=2,
        context=1,
        score_threshold=0.0,
        min_area_m2=0.0,
        min_hole_area_m2=0.0,
        nms_iou_threshold=0.75,
        nms_relative_intersection=0.75,
    )


def test_external_archive_checks_hash_and_unsafe_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "model.zip"
    archive_hash = _archive(archive_path)
    validate_external_archive(archive_path, _oks_manifest(archive_hash))

    with pytest.raises(ExternalModelError, match="SHA-256"):
        validate_external_archive(archive_path, _oks_manifest("0" * 64))

    unsafe_path = tmp_path / "unsafe.zip"
    unsafe_hash = _archive(unsafe_path, unsafe_name="../outside.pt")
    with pytest.raises((ExternalModelError, ValueError), match="небезопасный путь"):
        validate_external_archive(unsafe_path, _oks_manifest(unsafe_hash))


def test_external_model_loads_on_cpu_before_moving_to_device(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive_path = tmp_path / "model.zip"
    manifest = _zu_manifest(_archive(archive_path))
    calls: dict[str, str] = {}

    class _Model:
        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = "yes"
            return self

    fake_torch = ModuleType("torch")
    fake_torch.device = lambda value: value
    fake_torch.ops = SimpleNamespace(
        torchvision=SimpleNamespace(nms=object()),
    )

    def _load(_path: str, *, map_location: str):
        calls["map_location"] = map_location
        return _Model()

    fake_torch.jit = SimpleNamespace(load=_load)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torchvision", ModuleType("torchvision"))

    loaded = load_external_model(
        archive_path,
        manifest,
        device="cuda",
        scratch_root=tmp_path / "scratch",
    )

    assert calls == {"map_location": "cpu", "device": "cuda", "eval": "yes"}
    assert loaded.device == "cuda"


def test_oks_model_patches_only_scratch_copy_for_cuda(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inner = BytesIO()
    code_entry = (
        "model/code/__torch__/segmentation_models_pytorch/encoders/mix_transformer.py"
    )
    with zipfile.ZipFile(inner, mode="w") as archive:
        archive.writestr(
            code_entry,
            b'skip = torch.empty([1, 0, 1, 1], device=torch.device("cpu"))\n',
        )
    archive_path = tmp_path / "model.zip"
    manifest = _oks_manifest(_archive(archive_path, model_bytes=inner.getvalue()))
    calls: dict[str, str] = {}

    class _Device:
        type = "cuda"
        index = None

    class _Model:
        def eval(self):
            return self

    fake_torch = ModuleType("torch")
    fake_torch.device = lambda _value: _Device()
    fake_torch.cuda = SimpleNamespace(current_device=lambda: 0)

    def _load(path: str, *, map_location: _Device):
        calls["map_location"] = map_location.type
        with zipfile.ZipFile(path) as archive:
            code = archive.read(code_entry)
        assert b'torch.device("cuda:0")' in code
        assert b'torch.device("cpu")' not in code
        return _Model()

    fake_torch.jit = SimpleNamespace(load=_load)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    loaded = load_external_model(
        archive_path,
        manifest,
        device="cuda",
        scratch_root=tmp_path / "scratch",
    )

    assert calls == {"map_location": "cuda"}
    assert loaded.device == "cuda"
    assert sha256(archive_path.read_bytes()).hexdigest() == manifest.archive_sha256


def test_external_export_renames_root_and_config(tmp_path: Path) -> None:
    archive_path = tmp_path / "model.zip"
    manifest = _oks_manifest(_archive(archive_path))

    result = build_external_triton_model_export_zip(
        model_name="imported_oks",
        source_archive=archive_path,
        manifest=manifest,
    )
    try:
        with zipfile.ZipFile(result.zip_path) as outer:
            metadata = json.loads(outer.read("export_metadata.json"))
            assert metadata["external_adapter"] == "oks_multiclass_footprints"
            assert "SimplifyAsShapes" not in outer.read(
                "pipelines/imported_oks_triton.yaml"
            ).decode("utf-8")
            service_archive = tmp_path / "service.zip"
            service_archive.write_bytes(
                outer.read("models-serving-service/imported_oks.zip")
            )
        with zipfile.ZipFile(service_archive) as model_zip:
            assert "imported_oks/1/model.pt" in model_zip.namelist()
            config = model_zip.read("imported_oks/config.pbtxt").decode("utf-8")
            assert 'name: "imported_oks"' in config
            assert 'name: "sample"' not in config
    finally:
        result.cleanup()


def test_zu_external_export_connects_python_backend_to_shared_torch(tmp_path: Path) -> None:
    archive_path = tmp_path / "model.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr(
            "sample/config.pbtxt",
            'name: "sample"\nbackend: "python"\n',
        )
        archive.writestr("sample/1/model.pt", b"torchscript")
        archive.writestr(
            "sample/1/model.py",
            "import numpy as np\nimport cv2\nimport torch\nimport torchvision\n",
        )
    manifest = _zu_manifest(sha256(archive_path.read_bytes()).hexdigest())

    result = build_external_triton_model_export_zip(
        model_name="imported_zu",
        source_archive=archive_path,
        manifest=manifest,
        python_site_packages="/mlsystem2-venv/lib/python3.12/site-packages",
    )
    try:
        with zipfile.ZipFile(result.zip_path) as outer:
            metadata = json.loads(outer.read("export_metadata.json"))
            assert metadata["python_site_packages"] == (
                "/mlsystem2-venv/lib/python3.12/site-packages"
            )
            service_archive = tmp_path / "zu-service.zip"
            service_archive.write_bytes(
                outer.read("models-serving-service/imported_zu.zip")
            )
        with zipfile.ZipFile(service_archive) as model_zip:
            model_code = model_zip.read("imported_zu/1/model.py").decode("utf-8")
            assert model_code.startswith(
                "import sys\n"
                "sys.path.insert(0, '/mlsystem2-venv/lib/python3.12/site-packages')\n"
            )
            assert "import torch" in model_code
            assert "import torchvision" in model_code
            assert "import cv2" not in model_code
    finally:
        result.cleanup()


def test_instance_merge_keeps_touching_objects_separate_and_resolves_overlap() -> None:
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
            },
            "properties": {"confidence": 0.9, "name": "first"},
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[1, 0], [3, 0], [3, 2], [1, 2], [1, 0]]],
            },
            "properties": {"confidence": 0.8, "name": "second"},
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[3, 0], [4, 0], [4, 2], [3, 2], [3, 0]]],
            },
            "properties": {"confidence": 0.7, "name": "touching"},
        },
    ]

    merged = merge_external_instance_features(features)

    assert [item["properties"]["name"] for item in merged] == [
        "first",
        "second",
        "touching",
    ]
    geometries = [shape(item["geometry"]) for item in merged]
    assert geometries[0].intersection(geometries[1]).area == 0.0
    assert geometries[1].touches(geometries[2])


class _ArrayTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def detach(self) -> "_ArrayTensor":
        return self

    def cpu(self) -> "_ArrayTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class _FakeTorch:
    uint8 = np.uint8

    @staticmethod
    def device(value: str) -> str:
        return value

    @staticmethod
    def as_tensor(value, **_kwargs):
        return np.asarray(value)

    @staticmethod
    def no_grad():
        return nullcontext()


class _RoofModel:
    def __call__(self, value: np.ndarray) -> _ArrayTensor:
        return _ArrayTensor(np.ones((value.shape[0], value.shape[-2], value.shape[-1]), dtype=np.uint8))


def test_oks_prediction_uses_alpha_and_returns_instances_in_original_grid(tmp_path: Path) -> None:
    image_path = tmp_path / "tile.tif"
    data = np.full((4, 4, 4), 100, dtype=np.uint8)
    data[3, :, 2:] = 0
    with rasterio.open(
        image_path,
        mode="w",
        driver="GTiff",
        width=4,
        height=4,
        count=4,
        dtype="uint8",
        crs="EPSG:32637",
        transform=from_origin(500_000, 1_000, 1, 1),
    ) as dataset:
        dataset.write(data)
        dataset.colorinterp = (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        )
    loaded = LoadedExternalModel(
        manifest=_oks_manifest("0" * 64),
        torch=_FakeTorch(),
        model=_RoofModel(),
        device="cpu",
    )

    prediction = predict_external_test_tile(loaded, image_path)

    assert prediction.mask.shape == (4, 4)
    assert np.all(prediction.mask[:, :2] == 1)
    assert np.all(prediction.mask[:, 2:] == 0)
    assert set(np.unique(prediction.instances)) == {0, 1}

    cropped = predict_external_test_tile(
        loaded,
        image_path,
        geometry_postprocessor=lambda geometry, _crs: geometry.intersection(
            box(500_000, 996, 500_001, 1_000)
        ),
    )
    assert np.all(cropped.mask[:, :1] == 1)
    assert np.all(cropped.mask[:, 1:] == 0)

    with pytest.raises(ExternalModelError, match="Постобработчик внешней модели"):
        predict_external_test_tile(
            loaded,
            image_path,
            geometry_postprocessor=lambda _geometry, _crs: None,
        )


def test_external_resampling_uses_local_utm_for_web_mercator(tmp_path: Path) -> None:
    image_path = tmp_path / "web-mercator.tif"
    with rasterio.open(
        image_path,
        mode="w",
        driver="GTiff",
        width=64,
        height=64,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(4_361_172, 6_741_324, 0.25, 0.25),
    ) as dataset:
        dataset.write(np.ones((3, 64, 64), dtype=np.uint8))

    with _open_resampled_dataset(image_path, 0.6) as dataset:
        assert dataset.crs == rasterio.crs.CRS.from_epsg(32637)
        assert dataset.res == pytest.approx((0.6, 0.6))
