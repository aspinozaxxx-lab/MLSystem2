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

from ._external_models import ExternalModelManifest, validate_external_archive
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
    context: int | None = None,
    threshold: float | None = None,
    instance_kind: str = "KIND_CPU",
    postprocess_config: dict[str, object] | None = None,
    resolution_m: float | None = None,
    class_schema_override: list[dict[str, object]] | None = None,
) -> ModelExportArchive:
    """Собрать zip модели для загрузчика models-serving-service."""

    parsed_model_name = _validate_model_name(model_name)
    _validate_checkpoint_filename(checkpoint_filename)
    request_sample_size = _validate_optional_sample_size(sample_size)
    request_context = _validate_optional_context(context)

    temp_root = Path(tempfile.mkdtemp(prefix="mlsystem2-model-export-"))
    try:
        checkpoint_path = temp_root / "checkpoint.pt"
        checkpoint_path.write_bytes(checkpoint_bytes)
        loaded = _load_binary_checkpoint(checkpoint_path)
        task, class_schema = _checkpoint_task_schema(loaded)
        class_schema = _export_class_schema_override(
            task,
            class_schema,
            class_schema_override,
        )
        metadata_threshold = _threshold_from_metadata(loaded.artifact.metadata)
        effective_threshold = (
            _validate_threshold(threshold) if threshold is not None else metadata_threshold
        )
        parsed_instance_kind = _validate_instance_kind(instance_kind)
        parsed_sample_size, sample_size_source = _sample_size_from_metadata_or_request(
            loaded.artifact.metadata,
            request_sample_size,
        )
        parsed_context, context_source = _context_from_metadata_or_request(
            loaded.artifact.metadata,
            request_context,
        )
        inference_core_size = _validate_inference_window(
            parsed_sample_size,
            parsed_context,
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
        if task == "binary":
            _export_binary_mask_onnx(
                model=loaded.model.model,
                input_channels=input_channels,
                sample_size=parsed_sample_size,
                threshold=effective_threshold,
                onnx_path=onnx_path,
            )
        else:
            _export_segmentation_mask_onnx(
                model=loaded.model.model,
                input_channels=input_channels,
                output_channels=int(loaded.model.spec.output_channels),
                sample_size=parsed_sample_size,
                threshold=effective_threshold,
                onnx_path=onnx_path,
            )
        _write_text(
            model_dir / "config.pbtxt",
            _triton_config(
                parsed_model_name,
                input_channels,
                foreground_channels=(len(class_schema) if task == "multiclass" else 1),
                instance_kind=parsed_instance_kind,
            ),
        )
        _write_text(
            pipeline_dir / f"{parsed_model_name}_triton.yaml",
            _pipeline_yaml(
                parsed_model_name,
                parsed_sample_size,
                input_channels,
                class_schema=class_schema,
                context=parsed_context,
                postprocess_config=postprocess_config,
                resolution_m=resolution_m,
            ),
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
                "triton_instance_kind": parsed_instance_kind,
                "input_channels": input_channels,
                "sample_size": parsed_sample_size,
                "sample_size_source": sample_size_source,
                "inference_context": parsed_context,
                "inference_context_source": context_source,
                "inference_core_size": inference_core_size,
                "threshold": effective_threshold,
                "threshold_source": (
                    "request" if threshold is not None else "checkpoint_metadata"
                ),
                "task": task,
                "class_schema": class_schema,
                "onnx_opset": ONNX_OPSET,
                "onnx_ir_version": ONNX_IR_VERSION,
                "checkpoint_filename": checkpoint_filename,
                "resolution_m": resolution_m,
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


def build_external_triton_model_export_zip(
    *,
    model_name: str,
    source_archive: Path,
    manifest: ExternalModelManifest,
    python_site_packages: str | None = None,
) -> ModelExportArchive:
    """Переупаковать проверенную внешнюю TorchScript-модель под выбранным именем."""

    parsed_model_name = _validate_model_name(model_name)
    effective_python_site_packages = (
        python_site_packages if manifest.adapter == "detectron2_instances" else None
    )
    try:
        validate_external_archive(source_archive, manifest)
    except Exception as exc:
        raise TrainingUIAPIError(str(exc)) from exc
    temp_root = Path(tempfile.mkdtemp(prefix="mlsystem2-external-model-export-"))
    try:
        export_root = temp_root / "export"
        service_zip_dir = export_root / "models-serving-service"
        pipeline_dir = export_root / "pipelines"
        service_zip_dir.mkdir(parents=True)
        pipeline_dir.mkdir(parents=True)
        service_zip_path = service_zip_dir / f"{parsed_model_name}.zip"
        _rewrite_external_model_archive(
            source_archive=source_archive,
            target_archive=service_zip_path,
            source_root=manifest.model_root,
            target_root=parsed_model_name,
            python_site_packages=effective_python_site_packages,
        )
        _write_text(
            pipeline_dir / f"{parsed_model_name}_triton.yaml",
            _external_pipeline_yaml(parsed_model_name, manifest),
        )
        _write_json(
            export_root / "export_metadata.json",
            {
                "model_name": parsed_model_name,
                "model_archive": f"models-serving-service/{parsed_model_name}.zip",
                "pipeline": f"pipelines/{parsed_model_name}_triton.yaml",
                "format": "torchscript",
                "triton_platform": (
                    "python" if manifest.adapter == "detectron2_instances" else "pytorch_libtorch"
                ),
                "triton_instance_kind": "SOURCE_CONFIG",
                "source_triton_config_preserved": True,
                "input_channels": manifest.input_channels,
                "sample_size": manifest.stride,
                "sample_size_source": "external_model_manifest",
                "tile_size_with_context": manifest.tile_size,
                "context": manifest.context,
                "target_resolution_m": manifest.target_resolution_m,
                "threshold": manifest.score_threshold,
                "threshold_source": (
                    "external_model_manifest"
                    if manifest.score_threshold is not None
                    else "not_applicable"
                ),
                "source_archive_sha256": manifest.archive_sha256,
                "external_adapter": manifest.adapter,
                "python_site_packages": effective_python_site_packages,
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


def build_geoalert_pipeline_yaml(
    *,
    model_name: str,
    sample_size: int,
    input_channels: int,
    class_schema: list[dict[str, object]] | None = None,
    context: int = 0,
    postprocess_config: dict[str, object] | None = None,
    resolution_m: float | None = None,
    external_manifest: ExternalModelManifest | None = None,
) -> str:
    """Собрать pipeline повторно, не дублируя тяжёлый Triton model export."""

    parsed_model_name = _validate_model_name(model_name)
    if external_manifest is not None:
        return _external_pipeline_yaml(parsed_model_name, external_manifest)
    return _pipeline_yaml(
        parsed_model_name,
        sample_size,
        input_channels,
        class_schema=class_schema,
        context=context,
        postprocess_config=postprocess_config,
        resolution_m=resolution_m,
    )


def _rewrite_external_model_archive(
    *,
    source_archive: Path,
    target_archive: Path,
    source_root: str,
    target_root: str,
    python_site_packages: str | None = None,
) -> None:
    config_name = f"{source_root}/config.pbtxt"
    with zipfile.ZipFile(source_archive) as source, zipfile.ZipFile(
        target_archive,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as target:
        for item in sorted(source.infolist(), key=lambda value: value.filename):
            normalized = item.filename.replace("\\", "/").rstrip("/")
            if item.is_dir() or not normalized:
                continue
            relative = normalized.removeprefix(f"{source_root}/")
            if relative == normalized:
                raise TrainingUIAPIError("ZIP внешней модели содержит файл вне корневой папки.")
            content = source.read(item)
            if normalized == config_name:
                content = _renamed_triton_config(content, target_root)
            elif python_site_packages is not None and relative == "1/model.py":
                content = _python_backend_model_with_site_packages(
                    content,
                    python_site_packages,
                )
            target.writestr(f"{target_root}/{relative}", content)


def _python_backend_model_with_site_packages(content: bytes, site_packages: str) -> bytes:
    if not site_packages.startswith("/") or any(
        value in site_packages for value in ("\n", "\r", "\x00")
    ):
        raise TrainingUIAPIError("Путь site-packages для Triton Python backend должен быть абсолютным.")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrainingUIAPIError("model.py внешней модели должен быть UTF-8.") from exc
    text = re.sub(r"(?m)^\s*import\s+cv2\s*$", "", text)
    prefix = f"import sys\nsys.path.insert(0, {site_packages!r})\n"
    return (prefix + text.lstrip("\ufeff")).encode("utf-8")


def _renamed_triton_config(content: bytes, model_name: str) -> bytes:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrainingUIAPIError("config.pbtxt внешней модели должен быть UTF-8.") from exc
    updated, count = re.subn(
        r'(?m)^\s*name\s*:\s*"[^"]+"\s*$',
        f'name: "{model_name}"',
        text,
        count=1,
    )
    if count != 1:
        raise TrainingUIAPIError("В config.pbtxt внешней модели не найдено однозначное имя модели.")
    return updated.encode("utf-8")


def _external_pipeline_yaml(model_name: str, manifest: ExternalModelManifest) -> str:
    if manifest.adapter == "detectron2_instances":
        return _external_zu_pipeline_yaml(model_name, manifest)
    return _external_oks_pipeline_yaml(model_name, manifest)


def _external_zu_pipeline_yaml(model_name: str, manifest: ExternalModelManifest) -> str:
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
      output: [RED, GRN, BLU]
    - _class: NSPDParcels
      bounds: {manifest.context}
      input_rasters: [RED, GRN, BLU]
      output_fc: output
      sample_size: [{manifest.stride}, {manifest.stride}]
      crs: utm
      res: [{manifest.target_resolution_m}, {manifest.target_resolution_m}]
      adapter:
        _class: TritonAdapter
        name: "{model_name}"
        host: 127.0.0.1
        port: 8000
        protocol: http
        input_ndim: 3
        output_ndim: null
        output_dtype: null
        output_transpose: null
    - _class: UnifiedVectorProcessing
      input: output
      output: output
      crs: utm
      bricks:
        - _class: NMSVector
          algorithm: original
          corr_coef: {manifest.nms_relative_intersection}
          iou_threshold: {manifest.nms_iou_threshold}
        - _class: CorrectTopology
          correct_by_subtraction: -1
          distance_step: 0.0
        - _class: FilterOutput
          flatten_multipolygons: true
          min_hole_area: {manifest.min_hole_area_m2}
        - _class: FilterSmallObjects
          min_area: {manifest.min_area_m2}
          area_tag: area
"""


def _external_oks_pipeline_yaml(model_name: str, manifest: ExternalModelManifest) -> str:
    return f"""version: 0.1.4
config:
  _class: Compose
  inputs:
    - input.tif
  outputs:
    - output.geojson
  blocks:
    - name: Segmentation
      optional: false
      inputs: [input.tif]
      outputs: [roofs.geojson, walls.geojson]
      bricks:
        - _class: SplitRaster
          input: input
          input_ext: tif
          output: [RED, GRN, BLU]
        - _class: Segmentation
          bounds: {manifest.context}
          sample_size: [{manifest.stride}, {manifest.stride}]
          vectorize: false
          input_rasters: [RED, GRN, BLU]
          output_labels: [shadow, wall, markers, contour]
          crs: utm
          res: [{manifest.target_resolution_m}, {manifest.target_resolution_m}]
          adapter:
            _class: TritonAdapter
            name: "{model_name}"
            host: 127.0.0.1
            port: 8000
            protocol: http
            input_ndim: 4
            output_ndim: 3
          postprocessors:
            - _class: LabelsToOnehot
              class_map: 1, 2, 3, 4
        - _class: ApplyMask
          child_masks: [contour]
          parent_mask: markers
          out_masks: [roof]
          mask_operation: or
          reverse_parent: false
          sample_size: [2000, 2000]
        - _class: VectorizeMasks
          input_rasters: [roof, wall]
          output_fcs: [roofs, walls]
    - name: Подготовка геометрии
      optional: false
      inputs: [roofs.geojson]
      outputs: [roofs.geojson]
      bricks:
        - _class: UnifiedVectorProcessing
          input: roofs
          output: roofs
          crs: utm
          bricks:
            - _class: FilterBigObjects
              max_area_sq_m: 100000
            - _class: FilterSmallObjects
              min_area: {manifest.min_area_m2}
            - _class: RemoveSmallHoles
              min_hole_area: {manifest.min_hole_area_m2}
    - name: Смещение footprint
      optional: false
      inputs: [roofs.geojson, walls.geojson]
      outputs: [footprints.geojson]
      bricks:
        - _class: MeasureShift
          roofs: roofs
          walls: walls
          output: roof_shift
          max_shift: {manifest.max_shift_m}
          max_iterations: {manifest.shift_iterations}
        - _class: GenerateFootprints
          roofs: roof_shift
          output: footprint_shift
          x_shift_tag: _x_shift
          y_shift_tag: _y_shift
          confidence_shift_tag: _confidence_shift
          confidence_shift_thr: {manifest.shift_confidence}
        - _class: CorrectShift
          footprints: footprint_shift
          walls: walls
          output: footprint_corrected
          x_shift_tag: _x_shift
          y_shift_tag: _y_shift
          confidence_shift_tag: _confidence_shift
          corr_x_shift_tag: _x_shift_corr
          corr_y_shift_tag: _y_shift_corr
          corr_confidence_shift_tag: _confidence_shift_corr
          corr_threshold: {manifest.correction_confidence}
        - _class: GenerateFootprints
          roofs: footprint_corrected
          output: footprints
          x_shift_tag: _x_shift_corr
          y_shift_tag: _y_shift_corr
          confidence_shift_tag: _confidence_shift_corr
          confidence_shift_thr: {manifest.correction_confidence}
    - name: Финальная топология
      optional: false
      inputs: [footprints.geojson]
      outputs: [output.geojson]
      bricks:
        - _class: UnifiedVectorProcessing
          input: footprints
          output: output
          crs: utm
          bricks:
            - _class: CorrectTopology
              distance_step: 1.0
              correct_by_subtraction: 1
              buffer: 0.0
            - _class: FilterSmallObjects
              min_area: 5.0
              area_tag: area
            - _class: RemoveTags
              remove_underscore_tags: true
"""


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


def _validate_optional_context(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        context = int(value)
    except (TypeError, ValueError) as exc:
        raise TrainingUIAPIError("context должен быть целым неотрицательным числом.") from exc
    if context < 0:
        raise TrainingUIAPIError("context должен быть целым неотрицательным числом.")
    return context


def _load_binary_checkpoint(checkpoint_path: Path) -> Any:
    return _load_native_checkpoint(checkpoint_path)


def _load_native_checkpoint(checkpoint_path: Path) -> Any:
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
    return loaded


def _checkpoint_task_schema(loaded: Any) -> tuple[str, list[dict[str, object]]]:
    metadata = dict(loaded.artifact.metadata or {})
    train_config = metadata.get("train_config")
    train_config = dict(train_config) if isinstance(train_config, dict) else {}
    output_channels = int(loaded.model.spec.output_channels)
    task = str(metadata.get("task") or train_config.get("task") or "")
    if not task:
        task = "binary" if output_channels == 1 else "multiclass"
    if task == "binary":
        if output_channels != 1:
            raise TrainingUIAPIError("Binary checkpoint должен иметь output_channels=1.")
        return task, []
    if task != "multiclass":
        raise TrainingUIAPIError(f"Checkpoint содержит неизвестный task={task!r}.")
    raw_schema = metadata.get("class_schema") or train_config.get("class_schema")
    if not isinstance(raw_schema, list) or not raw_schema:
        slugs = train_config.get("class_slugs")
        if not isinstance(slugs, list) or not slugs:
            raise TrainingUIAPIError("Multiclass checkpoint не содержит class schema.")
        raw_schema = [
            {
                "id": index,
                "slug": str(slug),
                "name": str(slug),
                "color": "#808080",
                "priority": 0,
            }
            for index, slug in enumerate(slugs, start=1)
        ]
    schema: list[dict[str, object]] = []
    for index, raw in enumerate(raw_schema, start=1):
        if not isinstance(raw, dict):
            raise TrainingUIAPIError(f"Элемент class schema #{index} должен быть объектом.")
        item = {
            "id": int(raw.get("id", raw.get("class_id", index))),
            "slug": str(raw.get("slug") or "").strip(),
            "name": str(raw.get("name") or raw.get("slug") or "").strip(),
            "color": str(raw.get("color") or "#808080").upper(),
            "priority": int(raw.get("priority") or 0),
        }
        if not item["slug"] or not item["name"]:
            raise TrainingUIAPIError(f"Некорректный элемент class schema #{index}.")
        schema.append(item)
    schema.sort(key=lambda item: int(item["id"]))
    if [item["id"] for item in schema] != list(range(1, len(schema) + 1)):
        raise TrainingUIAPIError("class schema должна использовать последовательные id от 1.")
    if output_channels != len(schema) + 1:
        raise TrainingUIAPIError(
            "output_channels checkpoint не соответствует class schema: "
            f"{output_channels} != {len(schema)} + 1."
        )
    return task, schema


def _export_class_schema_override(
    task: str,
    checkpoint_schema: list[dict[str, object]],
    override: list[dict[str, object]] | None,
) -> list[dict[str, object]]:
    if override is None:
        return checkpoint_schema
    if task == "binary":
        if override:
            raise TrainingUIAPIError("Для binary checkpoint схема типов должна быть пустой.")
        return []
    normalized = _normalize_export_class_schema(override)
    checkpoint_channels = [
        (int(item["id"]), str(item["name"]).strip().casefold())
        for item in checkpoint_schema
    ]
    override_channels = [
        (int(item["id"]), str(item["name"]).strip().casefold())
        for item in normalized
    ]
    if checkpoint_channels != override_channels:
        raise TrainingUIAPIError(
            "Новая схема типов меняет порядок или назначение каналов checkpoint."
        )
    return normalized


def _normalize_export_class_schema(
    raw_schema: list[dict[str, object]],
) -> list[dict[str, object]]:
    schema: list[dict[str, object]] = []
    for index, raw in enumerate(raw_schema, start=1):
        item = {
            "id": int(raw.get("id", raw.get("class_id", index))),
            "slug": str(raw.get("slug") or "").strip(),
            "name": str(raw.get("name") or raw.get("slug") or "").strip(),
            "color": str(raw.get("color") or "#808080").upper(),
            "priority": int(raw.get("priority") or 0),
        }
        if not item["slug"] or not item["name"]:
            raise TrainingUIAPIError(f"Некорректный элемент новой class schema #{index}.")
        schema.append(item)
    schema.sort(key=lambda item: int(item["id"]))
    if [item["id"] for item in schema] != list(range(1, len(schema) + 1)):
        raise TrainingUIAPIError(
            "Новая class schema должна использовать последовательные id от 1."
        )
    if len({str(item["slug"]) for item in schema}) != len(schema):
        raise TrainingUIAPIError("Новая class schema содержит повторяющиеся slug.")
    return schema


def _threshold_from_metadata(metadata: dict[str, object]) -> float:
    raw_threshold = metadata.get(
        "confidence_threshold",
        metadata.get("val_best_threshold"),
    )
    if raw_threshold is None:
        raise TrainingUIAPIError(
            "Checkpoint не содержит metadata.confidence_threshold или metadata.val_best_threshold."
        )
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


def _context_from_metadata_or_request(
    metadata: dict[str, object],
    request_context: int | None,
) -> tuple[int, str]:
    raw_context = metadata.get("inference_context")
    if raw_context is not None:
        parsed = _validate_optional_context(raw_context)
        assert parsed is not None
        return parsed, "checkpoint_metadata"
    if request_context is not None:
        return request_context, "request"
    return 0, "legacy_default"


def _validate_inference_window(sample_size: int, context: int) -> int:
    core_size = sample_size - 2 * context
    if core_size <= 0:
        raise TrainingUIAPIError("sample_size должен быть больше удвоенного context.")
    return core_size


def _export_segmentation_mask_onnx(
    *,
    model: object,
    input_channels: int,
    output_channels: int,
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

    class SegmentationMaskWrapper(torch.nn.Module):
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
            if output_channels == 1:
                return (torch.sigmoid(logits) > self.threshold_value).to(torch.uint8)
            probabilities = torch.softmax(logits, dim=1)
            confidence, labels = torch.max(probabilities, dim=1)
            labels = torch.where(
                (labels > 0) & (confidence < self.threshold_value),
                torch.zeros_like(labels),
                labels,
            )
            one_hot = torch.nn.functional.one_hot(
                labels,
                num_classes=output_channels,
            ).permute(0, 3, 1, 2)
            return one_hot[:, 1:, :, :].to(torch.uint8)

    if not hasattr(model, "eval"):
        raise TrainingUIAPIError("Checkpoint содержит модель без метода eval().")
    model.eval()
    wrapper = SegmentationMaskWrapper(model, threshold).eval()
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


def _export_binary_mask_onnx(
    *,
    model: object,
    input_channels: int,
    sample_size: int,
    threshold: float,
    onnx_path: Path,
) -> None:
    _export_segmentation_mask_onnx(
        model=model,
        input_channels=input_channels,
        output_channels=1,
        sample_size=sample_size,
        threshold=threshold,
        onnx_path=onnx_path,
    )


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


def _triton_config(
    model_name: str,
    input_channels: int,
    foreground_channels: int = 1,
    instance_kind: str = "KIND_CPU",
) -> str:
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
    dims: [ -1, {foreground_channels}, -1, -1 ]
  }}
]
instance_group [
  {{
    kind: {instance_kind}
    count: 1
  }}
]
"""


def _pipeline_yaml(
    model_name: str,
    sample_size: int,
    input_channels: int,
    class_schema: list[dict[str, object]] | None = None,
    context: int = 0,
    postprocess_config: dict[str, object] | None = None,
    resolution_m: float | None = None,
) -> str:
    core_size = _validate_inference_window(sample_size, context)
    if input_channels == 3:
        bands = ("RED", "GRN", "BLU")
    elif input_channels == 4:
        bands = ("RED", "GRN", "BLU", "NIR")
    else:
        raise TrainingUIAPIError(
            f"Экспорт поддерживает только 3- и 4-канальные модели, получено {input_channels}."
        )
    bands_yaml = "\n".join(f"        - {band}" for band in bands)
    schema = list(class_schema or [])
    labels = [str(item["slug"]) for item in schema] or ["mask"]
    outputs = [f"{label}.geojson" for label in labels] if schema else ["output.geojson"]
    labels_yaml = "\n".join(f"        - {label}" for label in labels)
    outputs_yaml = "\n".join(f"    - {output}" for output in outputs)
    vector_outputs_yaml = "\n".join(f"        - {Path(output).stem}" for output in outputs)
    segmentation_labels, mask_postprocess_yaml = _mask_postprocess_yaml(
        labels,
        postprocess_config,
    )
    segmentation_labels_yaml = "\n".join(
        f"        - {label}" for label in segmentation_labels
    )
    vector_postprocess_yaml = _vector_postprocess_yaml(labels, postprocess_config)
    resolution_yaml = (
        f"      crs: utm\n      res: [{float(resolution_m)}, {float(resolution_m)}]\n"
        if resolution_m is not None and float(resolution_m) > 0
        else ""
    )
    return f"""version: 0.1.4
config:
  _class: Compose
  inputs:
    - input.tif
  outputs:
{outputs_yaml}
  bricks:
    - _class: SplitRaster
      input: input
      input_ext: tif
      output:
{bands_yaml}
    - _class: Segmentation
      bounds: {context}
      sample_size:
        - {core_size}
        - {core_size}
      input_rasters:
{bands_yaml}
      output_labels:
{segmentation_labels_yaml}
{resolution_yaml}      nodata: 0
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
{mask_postprocess_yaml}
    - _class: VectorizeMasks
      input_rasters:
{labels_yaml}
      output_fcs:
{vector_outputs_yaml}
{vector_postprocess_yaml}
"""


def _validate_instance_kind(value: str) -> str:
    if value not in {"KIND_CPU", "KIND_GPU"}:
        raise TrainingUIAPIError("instance_kind должен быть KIND_CPU или KIND_GPU.")
    return value


def _validate_threshold(value: object) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingUIAPIError("Порог инференса должен быть числом от 0 до 1.") from exc
    if not 0.0 <= threshold <= 1.0:
        raise TrainingUIAPIError("Порог инференса должен быть числом от 0 до 1.")
    return threshold


def _positive_number(config: dict[str, object], key: str) -> float | None:
    value = config.get(key)
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _mask_postprocess_yaml(
    labels: list[str],
    config: dict[str, object] | None,
) -> tuple[list[str], str]:
    values = dict(config or {})
    operations: list[tuple[str, str, int]] = []
    min_objects = _positive_number(values, "postprocess.mask_min_object_pixels")
    if min_objects is not None:
        operations.append(("remove_small_objects", "min_size", int(min_objects)))
    min_holes = _positive_number(values, "postprocess.mask_min_hole_pixels")
    if min_holes is not None:
        operations.append(("remove_small_holes", "area_threshold", int(min_holes)))
    closing = _positive_number(values, "postprocess.binary_closing_radius")
    if closing is not None:
        operations.append(("binary_closing", "selem_size", int(closing)))
    if not operations:
        return labels, ""

    current_labels = [f"mlsystem2_raw_{index}" for index in range(1, len(labels) + 1)]
    segmentation_labels = list(current_labels)
    bricks: list[str] = []
    for operation_index, (operation, parameter, value) in enumerate(operations, start=1):
        output_labels = (
            labels
            if operation_index == len(operations)
            else [
                f"mlsystem2_stage_{operation_index}_{index}"
                for index in range(1, len(labels) + 1)
            ]
        )
        extra = ""
        bound = 1
        if operation == "binary_closing":
            extra = "      selem: disk\n"
            bound = max(1, value)
        bricks.append(
            "    - _class: MaskMorphology\n"
            f"      input_masks: [{', '.join(current_labels)}]\n"
            f"      out_masks: [{', '.join(output_labels)}]\n"
            f"      mask_operation: {operation}\n"
            f"{extra}"
            f"      {parameter}: {value}\n"
            "      sample_size: [2048, 2048]\n"
            f"      bound: {bound}"
        )
        current_labels = list(output_labels)
    return segmentation_labels, "\n".join(bricks)


def _vector_postprocess_yaml(
    labels: list[str],
    config: dict[str, object] | None,
) -> str:
    values = dict(config or {})
    nested: list[str] = []
    simplify = _positive_number(values, "postprocess.simplify_m")
    if simplify is not None:
        nested.append(f"        - _class: Simplify\n          rate: {simplify}")
    min_area = _positive_number(values, "postprocess.min_area_m2")
    if min_area is not None:
        nested.append(
            f"        - _class: FilterSmallObjects\n          min_area: {min_area}\n          area_tag: area"
        )
    min_hole_area = _positive_number(values, "postprocess.min_hole_area_m2")
    if min_hole_area is not None:
        nested.append(
            f"        - _class: RemoveSmallHoles\n          min_hole_area: {min_hole_area}"
        )
    if bool(values.get("postprocess.filter_compact_objects.enabled")):
        quotient = float(
            values.get("postprocess.filter_compact_objects.min_isoperimetric_quotient")
            or 0.25
        )
        ratio = float(values.get("postprocess.filter_compact_objects.max_bbox_ratio") or 3.5)
        nested.append(
            "        - _class: FilterCompactObjects\n"
            f"          min_isoperimetric_quotient: {quotient}\n"
            f"          max_bbox_ratio: {ratio}"
        )
    if not nested:
        return ""
    return "\n".join(
        "    - _class: UnifiedVectorProcessing\n"
        f"      input: {label}\n"
        f"      output: {label}\n"
        "      crs: utm\n"
        "      bricks:\n"
        + "\n".join(nested)
        for label in labels
    )


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def _write_json(path: Path, content: dict[str, object]) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _zip_directory(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir).as_posix())
