"""Реализация подготовки датасета."""

from __future__ import annotations

from mlsystem2.storage import api as storage_api
from mlsystem2.storage.contracts import StorageError

from ._footprints import build_placeholder_footprints
from ._split import object_count_balanced_split
from .contracts import (
    DatasetManifest,
    DatasetPreparationError,
    DatasetPreparationReport,
    DatasetPreparationRequest,
    DatasetPreparationResult,
    ObjectCountByScene,
    SceneRecord,
)


def prepare_dataset(request: DatasetPreparationRequest) -> DatasetPreparationResult:
    missing_inputs = [
        uri
        for uri in (request.scenes_file, request.annotation_file, request.default_class_dir)
        if not storage_api.exists(uri)
    ]
    if missing_inputs:
        report = _empty_failed_report(
            [f"Обязательный вход подготовки датасета отсутствует: {uri}" for uri in missing_inputs]
        )
        return DatasetPreparationResult(status="failed", manifest=None, report=report, errors=report.errors)

    try:
        scenes_payload = storage_api.read_json(request.scenes_file)
        annotations_payload = storage_api.read_json(request.annotation_file)
    except StorageError as exc:
        raise DatasetPreparationError("Не удалось прочитать входы подготовки датасета") from exc

    records = _scene_records(request, scenes_payload, annotations_payload)
    missing_images = [record.image_uri for record in records if not storage_api.exists(record.image_uri)]
    object_counts = [
        ObjectCountByScene(scene_id=record.scene_id, object_count=record.object_count)
        for record in records
    ]
    split = object_count_balanced_split(records, request.val_fraction)
    footprints = build_placeholder_footprints(records)

    status = "failed" if missing_images else "succeeded"
    errors = [f"Снимок отсутствует: {uri}" for uri in missing_images]
    footprints_artifact_uri = (
        f"{request.output_uri.rstrip('/')}/footprints.json" if request.output_uri is not None else None
    )
    report = DatasetPreparationReport(
        status=status,
        scenes_total=len(records),
        train_scenes=split.train_scenes,
        val_scenes=split.val_scenes,
        object_counts=object_counts,
        missing_files=missing_images,
        warnings=[
            "Границы сцен являются заглушками до миграции реального GIS-извлечения."
        ],
        errors=errors,
        footprints_artifact_uri=footprints_artifact_uri,
    )
    manifest = None
    if status == "succeeded":
        manifest = DatasetManifest(
            images_uri=request.images_uri,
            scenes=records,
            split=split,
            footprints=footprints,
            object_counts=object_counts,
            footprints_artifact_uri=footprints_artifact_uri,
        )

    return DatasetPreparationResult(status=status, manifest=manifest, report=report, errors=errors)


def _empty_failed_report(errors: list[str]) -> DatasetPreparationReport:
    return DatasetPreparationReport(
        status="failed",
        scenes_total=0,
        train_scenes=[],
        val_scenes=[],
        object_counts=[],
        missing_files=[],
        warnings=[],
        errors=errors,
        footprints_artifact_uri=None,
    )


def _scene_records(
    request: DatasetPreparationRequest,
    scenes_payload: dict[str, object],
    annotations_payload: dict[str, object],
) -> list[SceneRecord]:
    raw_scenes = scenes_payload.get("scenes", [])
    if not isinstance(raw_scenes, list):
        raise DatasetPreparationError("Данные сцен должны содержать список 'scenes'")

    records: list[SceneRecord] = []
    for index, raw_scene in enumerate(raw_scenes):
        if isinstance(raw_scene, str):
            scene_id = raw_scene
            image_uri = _join_uri(request.images_uri, f"{scene_id}.tif")
            annotation_uri = request.annotation_file
            object_count = _annotation_count(scene_id, annotations_payload)
        elif isinstance(raw_scene, dict):
            scene_id_value = raw_scene.get("scene_id", raw_scene.get("id", f"scene-{index}"))
            scene_id = str(scene_id_value)
            image_uri = str(raw_scene.get("image_uri") or _join_uri(request.images_uri, f"{scene_id}.tif"))
            annotation_value = raw_scene.get("annotation_uri")
            annotation_uri = str(annotation_value) if annotation_value is not None else request.annotation_file
            object_count_value = raw_scene.get("object_count")
            object_count = (
                int(object_count_value)
                if isinstance(object_count_value, int)
                else _annotation_count(scene_id, annotations_payload)
            )
        else:
            raise DatasetPreparationError("Каждая сцена должна быть строкой или объектом")

        records.append(
            SceneRecord(
                scene_id=scene_id,
                image_uri=image_uri,
                annotation_uri=annotation_uri,
                object_count=max(0, object_count),
            )
        )
    return records


def _annotation_count(scene_id: str, annotations_payload: dict[str, object]) -> int:
    raw = annotations_payload.get(scene_id)
    if isinstance(raw, list):
        return len(raw)
    if isinstance(raw, dict):
        objects = raw.get("objects")
        if isinstance(objects, list):
            return len(objects)
    return 0


def _join_uri(base: str, child: str) -> str:
    return f"{base.rstrip('/')}/{child}"
