"""Proverka sloev i atomarnoe primenenie staging-rezultatov."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureRequest,
    QgsGeometry,
    QgsProject,
    QgsVectorDataProvider,
    QgsVectorLayer,
    QgsWkbTypes,
)


class LayerOperationError(RuntimeError):
    """Oshibka proverki ili zapisi v celevoi sloi."""


@dataclass(frozen=True)
class ApplyResult:
    """Itog idempotentnogo primeneniya staging."""

    added: int
    existing: int


def polygon_layers() -> list[QgsVectorLayer]:
    """Vernut dostupnye poligonalnye vektornye sloi proekta."""

    return sorted(
        [
            layer
            for layer in QgsProject.instance().mapLayers().values()
            if isinstance(layer, QgsVectorLayer)
            and layer.isValid()
            and QgsWkbTypes.geometryType(layer.wkbType()) == Qgis.GeometryType.Polygon
        ],
        key=lambda layer: layer.name().casefold(),
    )


def validate_target_layers(
    annotation_layer: QgsVectorLayer | None,
    hard_negative_layer: QgsVectorLayer | None,
) -> tuple[QgsVectorLayer, QgsVectorLayer]:
    """Proverit yavno vybrannye celi pered zapisyu."""

    if annotation_layer is None or hard_negative_layer is None:
        raise LayerOperationError("Выберите оба целевых слоя.")
    if annotation_layer.id() == hard_negative_layer.id():
        raise LayerOperationError("Слой разметки и hard_negative должны различаться.")
    for name, layer in (("разметки", annotation_layer), ("hard_negative", hard_negative_layer)):
        if not layer.isValid() or not isinstance(layer, QgsVectorLayer):
            raise LayerOperationError(f"Слой {name} удалён или недоступен.")
        if QgsWkbTypes.geometryType(layer.wkbType()) != Qgis.GeometryType.Polygon:
            raise LayerOperationError(f"Слой {name} должен быть полигональным.")
        if not layer.crs().isValid():
            raise LayerOperationError(f"У слоя {name} не задана CRS.")
        if layer.dataProvider() is None:
            raise LayerOperationError(f"Provider слоя {name} недоступен.")
        capabilities = layer.dataProvider().capabilities()
        capability_enum = getattr(Qgis, "VectorProviderCapability", QgsVectorDataProvider)
        if not capabilities & capability_enum.AddFeatures:
            raise LayerOperationError(f"Provider слоя {name} не разрешает добавление объектов.")
    return annotation_layer, hard_negative_layer


def apply_reviewed_candidates(
    candidate_layer: QgsVectorLayer,
    annotation_layer: QgsVectorLayer,
    hard_negative_layer: QgsVectorLayer,
    annotation_mapping: dict[str, str],
    hard_negative_mapping: dict[str, str],
) -> ApplyResult:
    """Dobavit vse raspredelennye obekty bez commita celevoi razmetki."""

    annotation_layer, hard_negative_layer = validate_target_layers(
        annotation_layer,
        hard_negative_layer,
    )
    if not candidate_layer.isValid():
        raise LayerOperationError("Слой кандидатов удалён во время сессии.")
    targets = {
        "annotation": (annotation_layer, annotation_mapping),
        "hard_negative": (hard_negative_layer, hard_negative_mapping),
    }
    existing_by_layer = {
        layer.id(): _existing_candidate_ids(layer, mapping)
        for layer, mapping in targets.values()
    }
    pending: list[tuple[QgsFeature, QgsVectorLayer, QgsFeature]] = []
    existing: list[tuple[QgsFeature, QgsVectorLayer, int]] = []
    for candidate in candidate_layer.getFeatures():
        status = str(candidate["review_status"] or "")
        if status in targets:
            target, mapping = targets[status]
        elif status == "exported":
            target_id = str(candidate["target_layer_id"] or "")
            target = next(
                (layer for layer, _ in targets.values() if layer.id() == target_id),
                None,
            )
            if target is None:
                continue
            mapping = annotation_mapping if target.id() == annotation_layer.id() else hard_negative_mapping
        else:
            continue
        target_feature = _target_feature(candidate_layer, candidate, target, mapping)
        candidate_id = str(candidate["candidate_id"] or "")
        existing_fid = existing_by_layer[target.id()].get(candidate_id)
        if existing_fid is None and not mapping.get("candidate_id"):
            existing_fid = _matching_feature_id(target, target_feature, mapping)
        if existing_fid is not None:
            existing.append((candidate, target, existing_fid))
        else:
            pending.append((candidate, target, target_feature))

    if not pending and not existing:
        return ApplyResult(added=0, existing=0)

    started_layers: list[QgsVectorLayer] = []
    command_layers: list[QgsVectorLayer] = []
    added: list[tuple[QgsFeature, QgsVectorLayer, int]] = []
    try:
        for layer in (annotation_layer, hard_negative_layer):
            if not layer.isEditable():
                if not layer.startEditing():
                    raise LayerOperationError(f"Слой «{layer.name()}» доступен только для чтения.")
                started_layers.append(layer)
            layer.beginEditCommand("Применить проверенную псевдоразметку")
            command_layers.append(layer)
        for candidate, target, target_feature in pending:
            if not target.addFeature(target_feature):
                raise LayerOperationError(
                    f"Не удалось добавить candidate_id={candidate['candidate_id']} в слой «{target.name()}»."
                )
            added.append((candidate, target, int(target_feature.id())))
        updates = _candidate_export_updates(candidate_layer, [*existing, *added])
        if updates and not candidate_layer.dataProvider().changeAttributeValues(updates):
            raise LayerOperationError("Не удалось сохранить состояние выгрузки в слое кандидатов.")
    except Exception:
        for layer in reversed(command_layers):
            layer.destroyEditCommand()
        for layer in started_layers:
            layer.rollBack()
        raise
    for layer in command_layers:
        layer.endEditCommand()
    candidate_layer.triggerRepaint()
    return ApplyResult(added=len(added), existing=len(existing))


# Chitaet uzhe zapasannye candidate_id iz yavno vybrannogo sloya.
def _existing_candidate_ids(
    layer: QgsVectorLayer,
    mapping: dict[str, str],
) -> dict[str, int]:
    field_name = mapping.get("candidate_id", "")
    if not field_name:
        return {}
    index = layer.fields().indexOf(field_name)
    if index < 0:
        raise LayerOperationError(
            f"Для слоя «{layer.name()}» сопоставьте существующее поле candidate_id."
        )
    request = QgsFeatureRequest().setSubsetOfAttributes([field_name], layer.fields())
    return {
        str(feature[index]): int(feature.id())
        for feature in layer.getFeatures(request)
        if feature[index] not in {None, ""}
    }


# Gotovit odin obekt s transformaciei CRS i sopostavleniem polei.
def _target_feature(
    candidate_layer: QgsVectorLayer,
    candidate: QgsFeature,
    target: QgsVectorLayer,
    mapping: dict[str, str],
) -> QgsFeature:
    geometry = QgsGeometry(candidate.geometry()).makeValid()
    transform = QgsCoordinateTransform(
        candidate_layer.crs(),
        target.crs(),
        QgsProject.instance(),
    )
    if geometry.transform(transform) != 0:
        raise LayerOperationError("Не удалось преобразовать CRS геометрии кандидата.")
    geometry = geometry.makeValid()
    if geometry.isNull() or geometry.isEmpty():
        raise LayerOperationError("После преобразования получена пустая геометрия.")
    target_is_multi = QgsWkbTypes.isMultiType(target.wkbType())
    if target_is_multi and not geometry.isMultipart():
        geometry.convertToMultiType()
    elif not target_is_multi and geometry.isMultipart():
        parts = geometry.asGeometryCollection()
        if len(parts) != 1:
            raise LayerOperationError(
                f"Слой «{target.name()}» не принимает составную геометрию."
            )
        geometry = parts[0]
    feature = QgsFeature(target.fields())
    feature.setGeometry(geometry)
    class_id = _candidate_attribute(candidate, "class_id")
    class_name = _candidate_attribute(candidate, "class_name") or class_id
    values: dict[str, Any] = {
        "class_id": class_id,
        "source": candidate["source_image_ids"],
        "confidence": candidate["confidence"],
        "candidate_id": candidate["candidate_id"],
        "model_version": candidate["model_version"],
    }
    for logical_name, field_name in mapping.items():
        if not field_name:
            continue
        index = target.fields().indexOf(field_name)
        if index < 0:
            raise LayerOperationError(
                f"В слое «{target.name()}» отсутствует поле «{field_name}»."
            )
        value = values.get(logical_name)
        if logical_name == "class_id" and _normalized_field_name(field_name) in {
            "classname",
            "class",
            "класс",
            "имякласса",
        }:
            value = class_name
        feature[index] = value
    return feature


def _matching_feature_id(
    layer: QgsVectorLayer,
    expected: QgsFeature,
    mapping: dict[str, str],
) -> int | None:
    """Найти уже добавленный объект, когда в целевой схеме нет candidate_id."""

    mapped_fields = [
        field_name
        for field_name in mapping.values()
        if field_name and layer.fields().indexOf(field_name) >= 0
    ]
    request = QgsFeatureRequest().setFilterRect(expected.geometry().boundingBox())
    for feature in layer.getFeatures(request):
        if not feature.geometry().equals(expected.geometry()):
            continue
        if all(_attributes_equal(feature[field], expected[field]) for field in mapped_fields):
            return int(feature.id())
    return None


def _candidate_attribute(candidate: QgsFeature, name: str) -> Any:
    index = candidate.fields().indexOf(name)
    return candidate[index] if index >= 0 else None


def _attributes_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return left == right or str(left) == str(right)


def _normalized_field_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


# Gotovit odin batch izmenenii audita kandidatov.
def _candidate_export_updates(
    candidate_layer: QgsVectorLayer,
    exported: list[tuple[QgsFeature, QgsVectorLayer, int]],
) -> dict[int, dict[int, Any]]:
    indexes = {
        name: candidate_layer.fields().indexOf(name)
        for name in ("review_status", "exported", "target_layer_id", "target_feature_id")
    }
    return {
        int(candidate.id()): {
            indexes["review_status"]: "exported",
            indexes["exported"]: True,
            indexes["target_layer_id"]: target.id(),
            indexes["target_feature_id"]: str(target_fid),
        }
        for candidate, target, target_fid in exported
    }


__all__ = [
    "ApplyResult",
    "LayerOperationError",
    "apply_reviewed_candidates",
    "polygon_layers",
    "validate_target_layers",
]
