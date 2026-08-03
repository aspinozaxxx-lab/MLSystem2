"""Postoyannaya sessiya proverki kandidatov v GeoPackage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qgis.PyQt.QtCore import QObject, pyqtSignal
from qgis.PyQt.QtGui import QColor
try:
    from qgis.PyQt.QtCore import QMetaType
except ImportError:
    QMetaType = None
try:
    from qgis.PyQt.QtCore import QVariant
except ImportError:
    QVariant = None
try:
    from qgis.PyQt.QtGui import QUndoStack
except ImportError:
    from qgis.PyQt.QtWidgets import QUndoStack
from qgis.core import (
    QgsFeature,
    QgsField,
    QgsFields,
    QgsJsonUtils,
    QgsProject,
    QgsRuleBasedRenderer,
    QgsSymbol,
    QgsVectorFileWriter,
    QgsVectorLayer,
)

from .contracts import validate_feature_collection
from .geometry_splitter import GeometrySplitError, SplitPart, split_geometry
from .undo_commands import ReviewStatusCommand, SplitCandidateCommand


_FIELD_NAMES = (
    "candidate_id",
    "parent_candidate_id",
    "job_id",
    "class_id",
    "confidence",
    "model_id",
    "model_version",
    "source_image_ids",
    "area_m2",
    "review_status",
    "reviewed_at",
    "exported",
    "target_layer_id",
    "target_feature_id",
    "original_order",
)
# Otdelyaet avtomaticheskoe vremya ot yavnogo vosstanovleniya NULL pri undo.
_AUTO_REVIEWED_AT = object()


class ReviewSessionError(RuntimeError):
    """Oshibka sozdaniya ili izmeneniya review session."""


class ReviewSession(QObject):
    """Upravlyaet GeoPackage, ocheredyu i edinoi istoriei undo."""

    changed = pyqtSignal()
    current_changed = pyqtSignal(object)

    # Privyazyvaet ochered i undo stack k postoyannomu sloyu.
    def __init__(self, layer: QgsVectorLayer, path: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.layer = layer
        self.layer_id = layer.id()
        self.path = path
        self.undo_stack = QUndoStack(self)
        self._filter = "new"
        self._sort = "original"
        self._min_area_m2 = 0.0
        self._min_confidence = 0.0
        self._current_feature_id: int | None = None
        self._apply_view_style()
        self._refresh_current()

    @classmethod
    def from_geojson(
        cls,
        payload: object,
        job_id: str,
        directory: Path,
        project: QgsProject | None = None,
    ) -> ReviewSession:
        """Sozdat ili prodolzhit sessiyu dlya servernogo job_id."""

        project = project or QgsProject.instance()
        validated = validate_feature_collection(payload)
        path = directory / f"pseudolabel_{job_id}.gpkg"
        uri = f"{path}|layername=candidates"
        if path.is_file():
            layer = QgsVectorLayer(uri, f"Кандидаты MLSystem2 — {job_id[:8]}", "ogr")
            if not layer.isValid() or any(
                layer.fields().indexOf(name) < 0 for name in _FIELD_NAMES
            ):
                raise ReviewSessionError("Сохранённая сессия повреждена или недоступна.")
        else:
            layer = _create_candidate_layer(validated, job_id, path)
        if project.mapLayer(layer.id()) is None:
            project.addMapLayer(layer)
        return cls(layer, path)

    @classmethod
    def open_existing(
        cls,
        path: Path,
        project: QgsProject | None = None,
    ) -> ReviewSession:
        """Otkryt nezavershennuyu sessiyu posle perezapuska."""

        project = project or QgsProject.instance()
        layer = QgsVectorLayer(
            f"{path}|layername=candidates",
            f"Кандидаты MLSystem2 — {path.stem}",
            "ogr",
        )
        if not layer.isValid() or any(layer.fields().indexOf(name) < 0 for name in _FIELD_NAMES):
            raise ReviewSessionError("Файл не является совместимой сессией MLSystem2.")
        project.addMapLayer(layer)
        return cls(layer, path)

    def close(self, *, remove_layer: bool = False) -> None:
        """Deaktivirovat istoriyu i pri neobhodimosti ubrat sloi iz proekta."""

        self.undo_stack.clear()
        self._current_feature_id = None
        if remove_layer and QgsProject.instance().mapLayer(self.layer_id) is not None:
            QgsProject.instance().removeMapLayer(self.layer_id)

    # Menyaet vidimuyu kategoriyu ocheredi.
    def set_filter(self, value: str) -> None:
        self._filter = value
        self._apply_view_style()
        self._refresh_current()

    # Menyaet stabilnyi poryadok ocheredi.
    def set_sort(self, value: str) -> None:
        self._sort = value
        self._refresh_current(keep_current=True)

    def set_thresholds(self, min_area_m2: float, min_confidence: float) -> None:
        """Синхронно отфильтровать очередь и отображаемые кандидаты."""

        self._min_area_m2 = max(0.0, float(min_area_m2))
        self._min_confidence = min(1.0, max(0.0, float(min_confidence)))
        self._apply_view_style()
        self._refresh_current(keep_current=True)

    def feature_ids(self) -> list[int]:
        """Vernut vidimuyu ochered s tekushchim filtrom i sortirovkoi."""

        features = [feature for feature in self.layer.getFeatures() if self._matches(feature)]
        if self._sort == "confidence_desc":
            features.sort(key=lambda item: (-(float(item["confidence"] or -1.0)), int(item.id())))
        elif self._sort == "area_desc":
            features.sort(key=lambda item: (-(float(item["area_m2"] or 0.0)), int(item.id())))
        else:
            features.sort(key=lambda item: (int(item["original_order"] or 0), int(item.id())))
        return [int(feature.id()) for feature in features]

    # Chitaet tekushchii obekt, esli on eshche dostupen.
    def current_feature(self) -> QgsFeature | None:
        if self._current_feature_id is None:
            return None
        feature = self.layer.getFeature(self._current_feature_id)
        return feature if feature.isValid() else None

    # Bezopasno chitaet obekt i proveriaet nalichie sloya v proekte.
    def feature(self, feature_id: int) -> QgsFeature:
        if QgsProject.instance().mapLayer(self.layer_id) is None:
            raise ReviewSessionError("Слой кандидатов удалён во время сессии.")
        feature = self.layer.getFeature(feature_id)
        if not feature.isValid():
            raise ReviewSessionError("Кандидат удалён из слоя во время сессии.")
        return feature

    # Schitaet obshchee i ostavsheesya chislo kandidatov.
    def counts(self) -> dict[str, int]:
        features = list(self.layer.getFeatures())
        return {
            "total": len(features),
            "new": sum(
                1
                for feature in features
                if str(feature["review_status"] or "new") == "new"
                and self._passes_thresholds(feature)
            ),
            "visible": len(self.feature_ids()),
        }

    # Vozvrashchaet poziciyu v tekushchem filtre.
    def position(self) -> tuple[int, int]:
        ids = self.feature_ids()
        if self._current_feature_id not in ids:
            return 0, len(ids)
        return ids.index(self._current_feature_id) + 1, len(ids)

    # Perehodit k sleduyushchemu kandidatu.
    def next(self) -> None:
        self._move(1)

    # Perehodit k predydushchemu kandidatu.
    def previous(self) -> None:
        self._move(-1)

    # Yavno delaet feature tekushchim.
    def select_feature(self, feature_id: int) -> None:
        if not self.feature(feature_id).isValid():
            return
        self._current_feature_id = feature_id
        self.current_changed.emit(self.current_feature())
        self.changed.emit()

    def classify(self, status: str) -> None:
        """Postavit atomarnuyu komandu klassifikacii v QUndoStack."""

        titles = {
            "annotation": "В разметку",
            "hard_negative": "В hard_negative",
            "discarded": "Выкинуть",
        }
        if status not in titles:
            raise ReviewSessionError("Неизвестный статус проверки.")
        feature = self.current_feature()
        if feature is None:
            raise ReviewSessionError("Нет текущего кандидата.")
        self.undo_stack.push(ReviewStatusCommand(self, int(feature.id()), status, titles[status]))

    def split_current(self, max_area_m2: float, min_area_m2: float) -> None:
        """Postavit razbienie tekushchego kandidata odnoi komandoi."""

        feature = self.current_feature()
        if feature is None:
            raise ReviewSessionError("Нет текущего кандидата.")
        try:
            parts = split_geometry(
                feature.geometry(),
                self.layer.crs(),
                str(feature["candidate_id"]),
                max_area_m2,
                min_area_m2,
            )
        except GeometrySplitError as exc:
            raise ReviewSessionError(str(exc)) from exc
        if len(parts) < 2:
            raise ReviewSessionError("Объект не требует разбиения при заданном пороге.")
        self.undo_stack.push(SplitCandidateCommand(self, int(feature.id()), parts))

    def split_large_candidates(self, max_area_m2: float, min_area_m2: float) -> int:
        """Razbit vse novye krupnye obekty pri yavno vklyuchennoi nastroike."""

        feature_ids = [
            int(feature.id())
            for feature in self.layer.getFeatures()
            if str(feature["review_status"] or "new") == "new"
            and float(feature["area_m2"] or 0.0) > max_area_m2
        ]
        count = 0
        for feature_id in feature_ids:
            self.select_feature(feature_id)
            self.split_current(max_area_m2, min_area_m2)
            count += 1
        return count

    # Vybirayet sleduyushchii obekt posle redo klassifikacii.
    def advance_after_action(self, feature_id: int) -> None:
        ids = self.feature_ids()
        if feature_id in ids:
            index = ids.index(feature_id)
            self._current_feature_id = ids[min(index + 1, len(ids) - 1)] if ids else None
        else:
            self._current_feature_id = ids[0] if ids else None
        self.current_changed.emit(self.current_feature())
        self.changed.emit()

    # Postoyanno menyaet dva polya audita odnim provider batch.
    def _set_review_status(
        self,
        feature_id: int,
        status: str,
        *,
        reviewed_at: object = _AUTO_REVIEWED_AT,
    ) -> None:
        if reviewed_at is _AUTO_REVIEWED_AT:
            reviewed_at = _now_text()
        updates = {
            self.layer.fields().indexOf("review_status"): status,
            self.layer.fields().indexOf("reviewed_at"): reviewed_at,
        }
        if not self.layer.dataProvider().changeAttributeValues({feature_id: updates}):
            raise ReviewSessionError("Не удалось сохранить статус кандидата.")
        self.layer.triggerRepaint()

    # Atomarno dobavlyaet vse chasti razbieniya v GeoPackage.
    def _apply_split(self, feature_id: int, parts: list[SplitPart]) -> list[int]:
        parent = self.feature(feature_id)
        fields = self.layer.fields()
        children: list[QgsFeature] = []
        for offset, part in enumerate(parts, start=1):
            child = QgsFeature(fields)
            for field_name in _FIELD_NAMES:
                child[field_name] = parent[field_name]
            child.setGeometry(part.geometry)
            child["candidate_id"] = part.candidate_id
            child["parent_candidate_id"] = part.parent_candidate_id
            child["area_m2"] = part.area_m2
            child["review_status"] = "new"
            child["reviewed_at"] = None
            child["exported"] = False
            child["target_layer_id"] = None
            child["target_feature_id"] = None
            child["original_order"] = int(parent["original_order"] or 0) * 10_000 + offset
            children.append(child)
        result = self.layer.dataProvider().addFeatures(children)
        success, added = result if isinstance(result, tuple) else (bool(result), children)
        if not success:
            raise ReviewSessionError("Не удалось добавить части в слой кандидатов.")
        try:
            self._set_review_status(feature_id, "split")
        except ReviewSessionError:
            self.layer.dataProvider().deleteFeatures([int(item.id()) for item in added])
            raise
        self.layer.updateExtents()
        self.changed.emit()
        return [int(item.id()) for item in added]

    # Udalyayet detei i vosstanavlivaet roditelya pri undo.
    def _undo_split(
        self,
        feature_id: int,
        child_feature_ids: list[int],
        old_status: str,
        old_reviewed_at: object,
    ) -> None:
        if child_feature_ids and not self.layer.dataProvider().deleteFeatures(child_feature_ids):
            raise ReviewSessionError("Не удалось отменить добавление частей.")
        self._set_review_status(feature_id, old_status, reviewed_at=old_reviewed_at)
        self.layer.updateExtents()
        self.changed.emit()

    # Proveriaet sootvetstvie tekushchemu filtru.
    def _matches(self, feature: QgsFeature) -> bool:
        status = str(feature["review_status"] or "new")
        return (self._filter == "all" or status == self._filter) and self._passes_thresholds(
            feature
        )

    def _passes_thresholds(self, feature: QgsFeature) -> bool:
        area = _numeric_attribute(feature["area_m2"])
        if area is None or area < self._min_area_m2:
            return False
        if self._min_confidence <= 0.0:
            return True
        confidence = _numeric_attribute(feature["confidence"])
        return confidence is not None and confidence >= self._min_confidence

    def _apply_view_style(self) -> None:
        _apply_status_style(
            self.layer,
            status_filter=self._filter,
            min_area_m2=self._min_area_m2,
            min_confidence=self._min_confidence,
        )

    # Ciklicheski sdvigaet ukazatel ocheredi.
    def _move(self, offset: int) -> None:
        ids = self.feature_ids()
        if not ids:
            self._current_feature_id = None
        elif self._current_feature_id not in ids:
            self._current_feature_id = ids[0]
        else:
            index = ids.index(self._current_feature_id)
            self._current_feature_id = ids[(index + offset) % len(ids)]
        self.current_changed.emit(self.current_feature())
        self.changed.emit()

    # Perestraivaet tekushchii ukazatel posle filtra ili sortirovki.
    def _refresh_current(self, *, keep_current: bool = False) -> None:
        ids = self.feature_ids()
        if not keep_current or self._current_feature_id not in ids:
            self._current_feature_id = ids[0] if ids else None
        self.current_changed.emit(self.current_feature())
        self.changed.emit()


# Sozdaet novyi GeoPackage iz proverennogo servernogo GeoJSON.
def _create_candidate_layer(payload: dict[str, Any], job_id: str, path: Path) -> QgsVectorLayer:
    memory = QgsVectorLayer("MultiPolygon?crs=EPSG:4326", "Кандидаты MLSystem2", "memory")
    fields = QgsFields()
    string_type = _field_type("QString", "String")
    double_type = _field_type("Double", "Double")
    bool_type = _field_type("Bool", "Bool")
    int_type = _field_type("Int", "Int")
    for name in (
        "candidate_id",
        "parent_candidate_id",
        "job_id",
        "class_id",
        "class_name",
        "model_id",
        "model_version",
        "source_image_ids",
        "review_status",
        "reviewed_at",
        "target_layer_id",
        "target_feature_id",
    ):
        fields.append(QgsField(name, string_type))
    fields.append(QgsField("confidence", double_type))
    fields.append(QgsField("area_m2", double_type))
    fields.append(QgsField("exported", bool_type))
    fields.append(QgsField("original_order", int_type))
    memory.dataProvider().addAttributes(fields)
    memory.updateFields()
    features: list[QgsFeature] = []
    for order, item in enumerate(payload["features"]):
        properties = item["properties"]
        geometry = QgsJsonUtils.geometryFromGeoJson(
            json.dumps(item["geometry"], ensure_ascii=False)
        )
        if geometry.isNull() or geometry.isEmpty():
            raise ReviewSessionError(f"У кандидата {order + 1} пустая геометрия.")
        geometry = geometry.makeValid()
        if not geometry.isMultipart():
            geometry.convertToMultiType()
        feature = QgsFeature(memory.fields())
        feature.setGeometry(geometry)
        values = {
            "candidate_id": properties.get("candidate_id"),
            "parent_candidate_id": properties.get("parent_candidate_id"),
            "job_id": properties.get("job_id") or job_id,
            "class_id": properties.get("class_id"),
            "class_name": properties.get("class_name"),
            "confidence": properties.get("confidence"),
            "model_id": properties.get("model_id"),
            "model_version": properties.get("model_version"),
            "source_image_ids": json.dumps(
                properties.get("source_image_ids") or [],
                ensure_ascii=False,
            ),
            "area_m2": properties.get("area_m2"),
            "review_status": "new",
            "reviewed_at": None,
            "exported": False,
            "target_layer_id": None,
            "target_feature_id": None,
            "original_order": order,
        }
        for name, value in values.items():
            feature[name] = value
        features.append(feature)
    result = memory.dataProvider().addFeatures(features)
    success = result[0] if isinstance(result, tuple) else bool(result)
    if not success:
        raise ReviewSessionError("Не удалось создать внутренний слой кандидатов.")
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = "candidates"
    options.actionOnExistingFile = QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteFile
    error = QgsVectorFileWriter.writeAsVectorFormatV3(
        memory,
        str(path),
        QgsProject.instance().transformContext(),
        options,
    )
    error_code = error[0] if isinstance(error, tuple) else error
    if error_code != QgsVectorFileWriter.WriterError.NoError:
        raise ReviewSessionError(f"Не удалось записать GeoPackage сессии: {error}.")
    layer = QgsVectorLayer(
        f"{path}|layername=candidates",
        f"Кандидаты MLSystem2 — {job_id[:8]}",
        "ogr",
    )
    if not layer.isValid():
        raise ReviewSessionError("Созданный GeoPackage не удалось открыть.")
    return layer


# Vozvrashchaet sovmestimyi tip polya Qt5/Qt6.
def _field_type(qmeta_name: str, qvariant_name: str):
    if QMetaType is not None:
        enum = getattr(QMetaType, "Type", QMetaType)
        if hasattr(enum, qmeta_name):
            return getattr(enum, qmeta_name)
    if QVariant is not None:
        return getattr(QVariant, qvariant_name)
    raise ReviewSessionError("Среда Qt не предоставляет типы полей.")


# Назначает стиль статусов и скрывает объекты вне активного фильтра.
def _apply_status_style(
    layer: QgsVectorLayer,
    *,
    status_filter: str = "new",
    min_area_m2: float = 0.0,
    min_confidence: float = 0.0,
) -> None:
    styles = {
        "new": ("Новый", (255, 215, 0)),
        "annotation": ("В разметку", (55, 180, 75)),
        "hard_negative": ("Хард-негатив", (230, 120, 30)),
        "discarded": ("Отклонён", (130, 130, 130)),
        "split": ("Разбит", (90, 90, 200)),
        "exported": ("Выгружен", (30, 150, 170)),
    }
    root_rule = QgsRuleBasedRenderer.Rule(None)
    for status, (label, rgb) in styles.items():
        if status_filter != "all" and status != status_filter:
            continue
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        symbol.setColor(QColor(*rgb, 110))
        expression = (
            f'"review_status" = \'{status}\' '
            f'AND coalesce("area_m2", 0) >= {min_area_m2:.12g}'
        )
        if min_confidence > 0.0:
            expression += (
                ' AND "confidence" IS NOT NULL '
                f'AND "confidence" >= {min_confidence:.12g}'
            )
        rule = QgsRuleBasedRenderer.Rule(symbol)
        rule.setLabel(label)
        rule.setFilterExpression(expression)
        root_rule.appendChild(rule)
    layer.setRenderer(QgsRuleBasedRenderer(root_rule))
    layer.triggerRepaint()


def _numeric_attribute(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Formiruet UTC-vremya audita v ISO 8601.
def _now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["ReviewSession", "ReviewSessionError"]
