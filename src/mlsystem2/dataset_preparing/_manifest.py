"""Чтение и строгая проверка схемы per-image multiclass-датасета."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .contracts import DatasetManifest, DatasetPreparationError


DATASET_MANIFEST_NAME = ".mlsystem2-dataset.json"
SCHEMA_VERSION_PROPERTY = "_mlsystem2_schema_version"
TASK_PROPERTY = "_mlsystem2_task"
CLASSES_PROPERTY = "_mlsystem2_classes"
ROLE_PROPERTY = "_mlsystem2_role"
CLASS_PROPERTY = "_mlsystem2_class"
ORIGIN_KEY_PROPERTY = "_mlsystem2_origin_key"


@dataclass(frozen=True, slots=True)
class MulticlassAnnotationCounts:
    positive: int
    hard_negative: int
    by_class: dict[str, int]


def load_dataset_manifest(
    annotations_dir_or_file: str | Path,
) -> DatasetManifest | None:
    """Загрузить манифест, если он присутствует рядом с per-image GeoJSON."""

    path = Path(annotations_dir_or_file)
    if path.is_dir() or path.suffix == "":
        path = path / DATASET_MANIFEST_NAME
    if not path.exists():
        return None
    if not path.is_file():
        raise DatasetPreparationError(f"Манифест датасета не является файлом: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise DatasetPreparationError(f"Не удалось прочитать манифест датасета: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetPreparationError(
            f"Манифест датасета содержит некорректный JSON: {path}: {exc}"
        ) from exc
    try:
        return DatasetManifest.model_validate(payload)
    except ValidationError as exc:
        raise DatasetPreparationError(
            f"Манифест датасета не соответствует схеме: {path}: {exc}"
        ) from exc


def validate_multiclass_annotation(
    annotation_file: str | Path,
    manifest: DatasetManifest,
) -> MulticlassAnnotationCounts:
    """Проверить повторённую схему и системные свойства одного GeoJSON."""

    path = Path(annotation_file)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetPreparationError(f"Не удалось прочитать GeoJSON-разметку: {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise DatasetPreparationError(f"GeoJSON должен быть FeatureCollection: {path}")
    if payload.get(SCHEMA_VERSION_PROPERTY) != manifest.schema_version:
        raise DatasetPreparationError(
            f"Версия схемы GeoJSON не совпадает с манифестом: {path}"
        )
    if payload.get(TASK_PROPERTY) != manifest.task:
        raise DatasetPreparationError(f"Тип задачи GeoJSON не совпадает с манифестом: {path}")
    raw_classes = payload.get(CLASSES_PROPERTY)
    if not isinstance(raw_classes, list):
        raise DatasetPreparationError(f"В GeoJSON отсутствует список классов: {path}")
    expected_classes = [item.model_dump(mode="json") for item in manifest.classes]
    try:
        actual_classes = [
            type(manifest.classes[0]).model_validate(item).model_dump(mode="json")
            for item in raw_classes
        ]
    except (ValidationError, TypeError) as exc:
        raise DatasetPreparationError(f"Некорректная схема классов в GeoJSON: {path}: {exc}") from exc
    if actual_classes != expected_classes:
        raise DatasetPreparationError(f"Список классов GeoJSON не совпадает с манифестом: {path}")

    features = payload.get("features")
    if not isinstance(features, list):
        raise DatasetPreparationError(f"GeoJSON должен содержать массив features: {path}")
    known_slugs = {item.slug for item in manifest.classes}
    by_class: Counter[str] = Counter()
    hard_negative = 0
    feature_ids: set[str] = set()
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict):
            raise DatasetPreparationError(f"Feature #{index} должен быть объектом: {path}")
        feature_id = feature.get("id")
        if feature_id is None or str(feature_id).strip() == "":
            raise DatasetPreparationError(f"У Feature #{index} отсутствует стабильный id: {path}")
        normalized_id = str(feature_id)
        if normalized_id in feature_ids:
            raise DatasetPreparationError(f"Повторяющийся Feature id {feature_id!r}: {path}")
        feature_ids.add(normalized_id)
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            raise DatasetPreparationError(f"properties Feature #{index} должен быть объектом: {path}")
        origin_key = properties.get(ORIGIN_KEY_PROPERTY)
        if not isinstance(origin_key, str) or not origin_key:
            raise DatasetPreparationError(
                f"У Feature #{index} отсутствует системный origin-key: {path}"
            )
        role = properties.get(ROLE_PROPERTY)
        class_slug = properties.get(CLASS_PROPERTY)
        if role == "positive":
            if class_slug not in known_slugs:
                raise DatasetPreparationError(
                    f"Feature #{index} содержит неизвестный класс {class_slug!r}: {path}"
                )
            by_class[str(class_slug)] += 1
        elif role == "hard_negative":
            if CLASS_PROPERTY in properties:
                raise DatasetPreparationError(
                    f"Hard negative Feature #{index} не должен содержать класс: {path}"
                )
            hard_negative += 1
        else:
            raise DatasetPreparationError(
                f"Feature #{index} содержит неизвестную роль {role!r}: {path}"
            )
    return MulticlassAnnotationCounts(
        positive=sum(by_class.values()),
        hard_negative=hard_negative,
        by_class={item.slug: by_class[item.slug] for item in manifest.classes},
    )


def manifest_path(annotations_dir: str | Path) -> Path:
    return Path(annotations_dir) / DATASET_MANIFEST_NAME


def canonical_json_hash_payload(value: Any) -> str:
    """Каноническое JSON-представление для стабильных хешей сборщика."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "DATASET_MANIFEST_NAME",
    "MulticlassAnnotationCounts",
    "canonical_json_hash_payload",
    "load_dataset_manifest",
    "manifest_path",
    "validate_multiclass_annotation",
]
