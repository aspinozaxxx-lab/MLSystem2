"""Готовые файлы экспорта для вкладки пайплайна и API сценариев ПМО."""
import base64
import zipfile
from types import SimpleNamespace

import pytest
import yaml

from mlsystem2.training_ui_api import _model_export
from mlsystem2.training_ui_api import _service
from mlsystem2.training_ui_api.contracts import TrainingUIAPIError


@pytest.mark.parametrize("profile", ["legacy", "next_gen2", "object_f1"])
@pytest.mark.parametrize("channels", [3, 4])
def test_pmo_files_preserve_model_semantics_and_configure_gateway(tmp_path, profile, channels):
    original = _model_export.build_geoalert_pipeline_yaml(
        model_name="oks500_orto", sample_size=1024, input_channels=channels,
        probability_output=profile != "legacy", object_output=profile == "object_f1",
        threshold=0.5, postprocess_config={"postprocess.min_area_m2": 20, "postprocess.simplify_m": 0.5},
    )
    files = _model_export._write_export_pipelines(tmp_path, "oks500_orto", original)
    assert (tmp_path / "pipelines/oks500_orto_triton.yaml").read_text(encoding="utf-8") == original
    pmo_text = (tmp_path / files["pipeline_pmo"]).read_text(encoding="utf-8")
    assert base64.b64decode((tmp_path / files["pipeline_pmo_base64"]).read_text().strip(), validate=True).decode("utf-8") == pmo_text
    pmo = yaml.safe_load(pmo_text)
    initial = yaml.safe_load(original)
    block, = pmo["config"].pop("blocks")
    assert block["name"] == "Recognition" and block["optional"] is False
    assert block["inputs"] == pmo["config"]["inputs"] and block["outputs"] == pmo["config"]["outputs"]
    pmo["config"]["bricks"] = block["bricks"]
    adapters = []
    def restore(value):
        if isinstance(value, dict):
            if value.get("_class") == "TritonAdapter":
                assert value["host"] == "models-serving.service.consul" and value["port"] == 18070
                assert value["protocol"] == "http" and value["name"] == "oks500_orto"
                value.update(host="127.0.0.1", port=8000)
                adapters.append(value)
            for item in value.values():
                restore(item)
        elif isinstance(value, list):
            for item in value:
                restore(item)
    restore(pmo)
    assert len(adapters) == 1 and pmo == initial
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert files["pipeline_pmo"] in readme and files["pipeline_pmo_base64"] in readme
    assert "не устанавливают расширения" in readme


def test_blocked_external_pipeline_and_unicode_are_preserved(tmp_path):
    document = {"version": "0.1.4", "shared": {"label": "ОКС"}, "config": {
        "_class": "Compose", "inputs": ["input.tif"], "outputs": ["output.geojson"],
        "blocks": [{"name": "Segmentation", "optional": False, "inputs": ["input.tif"], "outputs": ["output.geojson"],
                    "bricks": [{"_class": "Segmentation", "adapter": {"_class": "TritonAdapter", "name": "oks500_orto", "host": "127.0.0.1", "port": 8000}}]}]}}
    files = _model_export._write_export_pipelines(tmp_path, "oks500_orto", yaml.safe_dump(document, allow_unicode=True))
    actual = yaml.safe_load((tmp_path / files["pipeline_pmo"]).read_text(encoding="utf-8"))
    assert actual["shared"] == document["shared"]
    assert len(actual["config"]["blocks"]) == 1
    actual["config"]["blocks"][0]["bricks"][0]["adapter"].update(host="127.0.0.1", port=8000)
    actual["config"]["blocks"][0]["bricks"][0]["adapter"].pop("protocol")
    assert actual == document
    assert base64.b64decode((tmp_path / files["pipeline_pmo_base64"]).read_bytes()).decode("utf-8") == (tmp_path / files["pipeline_pmo"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("extra_member", ["runtime/../escape.py", "runtime/sub\\escape.py", "runtime/C:/escape.py"])
def test_batch_export_rejects_unsafe_runtime_paths(tmp_path, extra_member):
    archive = _batch_fixture(tmp_path, {extra_member: b"invalid"})
    with pytest.raises(TrainingUIAPIError, match="недопустимый путь"):
        _copy_batch_fixture(tmp_path, archive)
    assert not (tmp_path / "export/escape.py").exists()


def test_batch_export_rejects_conflicting_runtime_versions(tmp_path):
    archive = _batch_fixture(tmp_path, {"runtime/module.py": b"first"})
    _copy_batch_fixture(tmp_path, archive)
    archive = _batch_fixture(tmp_path, {"runtime/module.py": b"different"})
    with pytest.raises(TrainingUIAPIError, match="несовместимые версии"):
        _copy_batch_fixture(tmp_path, archive)
    assert (tmp_path / "export/runtime/module.py").read_bytes() == b"first"


def test_batch_export_rejects_incomplete_pmo_files(tmp_path):
    archive = _batch_fixture(tmp_path, {"pipelines/model_pmo.yaml": b"pmo"})
    with pytest.raises(TrainingUIAPIError, match="один из файлов"):
        _copy_batch_fixture(tmp_path, archive)


def _batch_fixture(tmp_path, extra_files):
    path = tmp_path / "model_export.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("models-serving-service/model.zip", b"model")
        archive.writestr("pipelines/model_triton.yaml", "pipeline")
        archive.writestr("export_metadata.json", "{}")
        for name, content in extra_files.items():
            member = zipfile.ZipInfo()
            member.filename = name
            archive.writestr(member, content)
    return SimpleNamespace(zip_path=path)


def _copy_batch_fixture(tmp_path, archive):
    root = tmp_path / "export"
    for name in ("models-serving-service", "pipelines", "metadata"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return _service._copy_model_export_files(
        archive, model_name="model", service_zip_dir=root / "models-serving-service",
        pipeline_dir=root / "pipelines", metadata_dir=root / "metadata",
    )
