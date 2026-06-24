"""Экспорт checkpoint в архив Triton для models-serving-service."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Any

from mlsystem2.models.contracts import LoadCheckpointRequest, ModelsError

from .contracts import TrainingUIAPIError

MODEL_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,126}[a-z0-9])?$")
ONNX_OPSET = 17
ONNX_IR_VERSION = 8


@dataclass(frozen=True)
class ModelExportArchive:
    """Готовый временный zip-архив."""

    zip_path: Path
    filename: str
    cleanup_root: Path

    def cleanup(self) -> None:
        shutil.rmtree(self.cleanup_root, ignore_errors=True)


def build_triton_model_export_zip(
    *,
    model_name: str,
    checkpoint_filename: str,
    checkpoint_bytes: bytes,
    sample_size: int | None,
) -> ModelExportArchive:
    """Собрать zip модели для загрузчика models-serving-service."""

    parsed_model_name = _validate_model_name(model_name)
    _validate_checkpoint_filename(checkpoint_filename)
    request_sample_size = _validate_optional_sample_size(sample_size)

    temp_root = Path(tempfile.mkdtemp(prefix="mlsystem2-model-export-"))
    try:
        checkpoint_path = temp_root / "checkpoint.pt"
        checkpoint_path.write_bytes(checkpoint_bytes)
        loaded = _load_binary_checkpoint(checkpoint_path)
        effective_threshold = _threshold_from_metadata(loaded.artifact.metadata)
        parsed_sample_size, sample_size_source = _sample_size_from_metadata_or_request(
            loaded.artifact.metadata,
            request_sample_size,
        )
        input_channels = int(loaded.model.spec.input_channels)

        export_root = temp_root / "export"
        service_root = temp_root / "models-serving-service"
        model_dir = service_root / parsed_model_name
        version_dir = model_dir / "1"
        pipeline_dir = export_root / "pipelines"
        service_zip_dir = export_root / "models-serving-service"
        version_dir.mkdir(parents=True)
        pipeline_dir.mkdir(parents=True)
        service_zip_dir.mkdir(parents=True)

        onnx_path = version_dir / "model.onnx"
        _export_binary_mask_onnx(
            model=loaded.model.model,
            input_channels=input_channels,
            sample_size=parsed_sample_size,
            threshold=effective_threshold,
            onnx_path=onnx_path,
        )
        _write_text(model_dir / "config.pbtxt", _triton_config(parsed_model_name, input_channels))
        _write_text(
            pipeline_dir / f"{parsed_model_name}_triton.yaml",
            _pipeline_yaml(parsed_model_name, parsed_sample_size),
        )
        service_zip_path = service_zip_dir / f"{parsed_model_name}.zip"
        _zip_directory(service_root, service_zip_path)
        _write_json(
            export_root / "export_metadata.json",
            {
                "model_name": parsed_model_name,
                "model_archive": f"models-serving-service/{parsed_model_name}.zip",
                "pipeline": f"pipelines/{parsed_model_name}_triton.yaml",
                "format": "onnx",
                "triton_platform": "onnxruntime_onnx",
                "triton_instance_kind": "KIND_CPU",
                "input_channels": input_channels,
                "sample_size": parsed_sample_size,
                "sample_size_source": sample_size_source,
                "threshold": effective_threshold,
                "threshold_source": "checkpoint_metadata",
                "onnx_opset": ONNX_OPSET,
                "onnx_ir_version": ONNX_IR_VERSION,
                "checkpoint_filename": checkpoint_filename,
                "checkpoint_metadata": loaded.artifact.metadata,
            },
        )

        zip_path = temp_root / f"{parsed_model_name}_export.zip"
        _zip_directory(export_root, zip_path)
        return ModelExportArchive(
            zip_path=zip_path,
            filename=f"{parsed_model_name}_export.zip",
            cleanup_root=temp_root,
        )
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def _validate_model_name(value: str) -> str:
    model_name = value.strip()
    if not MODEL_NAME_RE.fullmatch(model_name):
        raise TrainingUIAPIError(
            "Имя модели должно содержать только латинские строчные буквы, цифры, дефис и подчеркивание, "
            "начинаться и заканчиваться буквой или цифрой."
        )
    return model_name


def _validate_checkpoint_filename(value: str) -> None:
    if not value.lower().endswith(".pt"):
        raise TrainingUIAPIError("Нужен checkpoint MLSystem2 в формате .pt.")


def _validate_sample_size(value: int, *, field_name: str = "sample_size") -> int:
    try:
        sample_size = int(value)
    except (TypeError, ValueError) as exc:
        raise TrainingUIAPIError(f"{field_name} должен быть целым числом.") from exc
    if sample_size <= 0 or sample_size % 32 != 0:
        raise TrainingUIAPIError(f"{field_name} должен быть положительным числом, кратным 32.")
    return sample_size


def _validate_optional_sample_size(value: int | None) -> int | None:
    if value is None:
        return None
    return _validate_sample_size(value)


def _load_binary_checkpoint(checkpoint_path: Path) -> Any:
    try:
        from mlsystem2.models.api import load_checkpoint

        loaded = load_checkpoint(
            LoadCheckpointRequest(
                checkpoint_uri=str(checkpoint_path),
                map_location="cpu",
            )
        )
    except ModelsError as exc:
        raise TrainingUIAPIError(str(exc)) from exc
    if loaded.model.spec.output_channels != 1:
        raise TrainingUIAPIError("Экспорт поддерживает только binary segmentation checkpoint с output_channels=1.")
    return loaded


def _threshold_from_metadata(metadata: dict[str, object]) -> float:
    if "val_best_threshold" not in metadata:
        raise TrainingUIAPIError(
            "Checkpoint не содержит metadata.val_best_threshold. "
            "Автоматический экспорт невозможен: выберите checkpoint, сохраненный после validation с best threshold."
        )
    raw_threshold = metadata["val_best_threshold"]
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError) as exc:
        raise TrainingUIAPIError("metadata.val_best_threshold должен быть числом от 0 до 1.") from exc
    if threshold < 0.0 or threshold > 1.0:
        raise TrainingUIAPIError("metadata.val_best_threshold должен быть числом от 0 до 1.")
    return threshold


def _sample_size_from_metadata_or_request(metadata: dict[str, object], request_sample_size: int | None) -> tuple[int, str]:
    raw_sample_size = metadata.get("sample_size")
    if raw_sample_size is not None:
        return _validate_sample_size(raw_sample_size, field_name="metadata.sample_size"), "checkpoint_metadata"
    if request_sample_size is not None:
        return request_sample_size, "request"
    raise TrainingUIAPIError(
        "Checkpoint не содержит metadata.sample_size. Для старого checkpoint задайте sample_size вручную."
    )


def _export_binary_mask_onnx(
    *,
    model: object,
    input_channels: int,
    sample_size: int,
    threshold: float,
    onnx_path: Path,
) -> None:
    try:
        import torch
    except ImportError as exc:
        raise TrainingUIAPIError(
            "Для экспорта ONNX требуется optional dependency torch. Установите пакет через `pip install -e .[torch]`."
        ) from exc

    class BinaryMaskWrapper(torch.nn.Module):
        def __init__(self, wrapped_model: object, threshold_value: float) -> None:
            super().__init__()
            self.wrapped_model = wrapped_model
            self.threshold_value = threshold_value

        def forward(self, x: Any) -> Any:
            logits = self.wrapped_model(x.float())
            if hasattr(logits, "logits"):
                logits = logits.logits
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.shape[-2:] != x.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            return (torch.sigmoid(logits) > self.threshold_value).to(torch.uint8)

    if not hasattr(model, "eval"):
        raise TrainingUIAPIError("Checkpoint содержит модель без метода eval().")
    model.eval()
    wrapper = BinaryMaskWrapper(model, threshold).eval()
    dummy = torch.zeros((1, input_channels, sample_size, sample_size), dtype=torch.float32)
    try:
        with torch.no_grad():
            export_kwargs = {
                "input_names": ["input"],
                "output_names": ["mask"],
                "opset_version": ONNX_OPSET,
                "do_constant_folding": True,
                "dynamic_axes": {
                    "input": {2: "height", 3: "width"},
                    "mask": {2: "height", 3: "width"},
                },
            }
            export_params = signature(torch.onnx.export).parameters
            if "external_data" in export_params:
                export_kwargs["external_data"] = True
            elif "use_external_data_format" in export_params:
                export_kwargs["use_external_data_format"] = True
            if "dynamo" in export_params:
                export_kwargs["dynamo"] = False
            torch.onnx.export(wrapper, dummy, str(onnx_path), **export_kwargs)
    except Exception as exc:
        raise TrainingUIAPIError("Не удалось экспортировать checkpoint в ONNX.") from exc
    if not onnx_path.is_file():
        raise TrainingUIAPIError("ONNX exporter не создал model.onnx.")
    _normalize_onnx_for_triton(onnx_path)


def _normalize_onnx_for_triton(onnx_path: Path) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise TrainingUIAPIError(
            "Для экспорта ONNX под старый Triton требуется optional dependency onnx. "
            "Установите пакет через `pip install -e .`."
        ) from exc

    try:
        model = onnx.load_model(onnx_path, load_external_data=False)
    except Exception as exc:
        raise TrainingUIAPIError("Не удалось прочитать экспортированный ONNX.") from exc

    default_opsets = [item.version for item in model.opset_import if item.domain in ("", "ai.onnx")]
    if default_opsets and max(default_opsets) > ONNX_OPSET:
        raise TrainingUIAPIError(
            f"ONNX exporter создал модель с opset {max(default_opsets)}, "
            f"а старый Triton export поддерживает не выше {ONNX_OPSET}."
        )

    model.ir_version = ONNX_IR_VERSION
    try:
        onnx.save_model(model, onnx_path, save_as_external_data=False)
        onnx.checker.check_model(str(onnx_path))
    except Exception as exc:
        raise TrainingUIAPIError(
            f"Не удалось привести ONNX к IR version {ONNX_IR_VERSION} для старого Triton."
        ) from exc


def _triton_config(model_name: str, input_channels: int) -> str:
    return f"""name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 0
input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ 1, {input_channels}, -1, -1 ]
  }}
]
output [
  {{
    name: "mask"
    data_type: TYPE_UINT8
    dims: [ 1, 1, -1, -1 ]
  }}
]
instance_group [
  {{
    kind: KIND_CPU
    count: 1
  }}
]
"""


def _pipeline_yaml(model_name: str, sample_size: int) -> str:
    return f"""version: 0.1.4
config:
  _class: Compose
  inputs:
    - input.tif
  outputs:
    - output.geojson
  bricks:
    - _class: SplitRaster
      input: input
      input_ext: tif
      output:
        - RED
        - GRN
        - BLU
        - NIR
    - _class: Segmentation
      bounds: 0
      sample_size:
        - {sample_size}
        - {sample_size}
      input_rasters:
        - RED
        - GRN
        - BLU
        - NIR
      output_labels:
        - mask
      nodata: 0
      adapter:
        _class: TritonAdapter
        name: "{model_name}"
        host: 127.0.0.1
        port: 8000
        protocol: http
        input_dtype: float32
        input_ndim: 4
        output_dtype: uint8
        output_ndim: 3
    - _class: VectorizeMasks
      input_rasters:
        - mask
      output_fcs:
        - output
"""


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def _write_json(path: Path, content: dict[str, object]) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _zip_directory(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir).as_posix())
