"""Postoyannaya sessiya proverki kandidatov v GeoPackage."""

from __future__ import annotations

import json
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
from qgis.core import (
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsJsonUtils,
    QgsProject,
    QgsRuleBasedRenderer,
    QgsSymbol,
    QgsVectorFileWriter,
    QgsVectorLayer,
)

from .contracts import validate_feature_collection
from .geometry_splitter import GeometrySplitError, SplitPart, split_geometry


_OUTPUT_FIELD_NAMES = (
    "candidate_id",
    "parent_candidate_id",
    "job_id",
    "class_id",
    "class_name",
    "object_type_id",
    "object_type_slug",
    "object_type_name",
    "object_type_color",
    "confidence",
    "model_id",
    "model_version",
    "source_image_ids",
    "area_m2",
)


_FIELD_NAMES = (
    "candidate_id",
    "parent_candidate_id",
    "job_id",
    "class_id",
    "class_name",
    "object_type_id",
    "object_type_slug",
    "object_type_name",
    "object_type_color",
    "confidence",
    "model_id",
    "model_version",
    "source_image_ids",
    "area_m2",
    "review_status",
    "original_order",
)
_LEGACY_REQUIRED_FIELD_NAMES = tuple(
    name
    for name in _FIELD_NAMES
    if not name.startswith("object_type_")
)
_SESSION_LAYER_PROPERTY = "mlsystem2/review_session"


class ReviewSessionError(RuntimeError):
    """Oshibka sozdaniya ili izmeneniya review session."""


class ReviewSession(QObject):
    """Управляет локальным GeoPackage, фильтрацией и разбиением."""

    changed = pyqtSignal()
    current_changed = pyqtSignal(object)

    # Привязывает очередь кандидатов к постоянному локальному слою.
    def __init__(self, layer: QgsVectorLayer, path: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.layer = layer
        self.layer_id = layer.id()
        self.path = path
        self.layer.setCustomProperty(_SESSION_LAYER_PROPERTY, True)
        self._sort = "original"
        self._min_area_m2 = 0.0
        self._min_confidence = 0.0
        self._object_type_slug: str | None = None
        self._view_layer_ids: list[str] = []
        self._display_mode = "categorized"
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
                layer.fields().indexOf(name) < 0 for name in _LEGACY_REQUIRED_FIELD_NAMES
            ):
                raise ReviewSessionError("Сохранённая сессия повреждена или недоступна.")
        else:
            layer = _create_candidate_layer(validated, job_id, path)
        if project.mapLayer(layer.id()) is None:
            project.addMapLayer(layer)
        return cls(layer, path)

    def close(self, *, remove_layer: bool = False) -> None:
        """Deaktivirovat istoriyu i pri neobhodimosti ubrat sloi iz proekta."""

        self._current_feature_id = None
        project = QgsProject.instance()
        for layer_id in self._view_layer_ids:
            if project.mapLayer(layer_id) is not None:
                project.removeMapLayer(layer_id)
        self._view_layer_ids = []
        canonical_node = project.layerTreeRoot().findLayer(self.layer_id)
        if canonical_node is not None and not remove_layer:
            canonical_node.setItemVisibilityChecked(True)
        if remove_layer and QgsProject.instance().mapLayer(self.layer_id) is not None:
            QgsProject.instance().removeMapLayer(self.layer_id)

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

    def set_object_type_filter(self, slug: str | None) -> None:
        """Фильтровать единую очередь по подтипу, не меняя review-статусы."""

        normalized = str(slug or "").strip()
        self._object_type_slug = normalized or None
        self._apply_view_style()
        self._refresh_current(keep_current=True)

    def object_types(self) -> list[dict[str, Any]]:
        """Вернуть типы из канонического слоя в стабильном порядке."""

        result: dict[str, dict[str, Any]] = {}
        if self.layer.fields().indexOf("object_type_slug") < 0:
            return []
        for feature in self.layer.getFeatures():
            slug = str(feature["object_type_slug"] or "").strip()
            if not slug:
                continue
            result.setdefault(
                slug,
                {
                    "id": int(feature["object_type_id"] or 0),
                    "slug": slug,
                    "name": str(feature["object_type_name"] or slug),
                    "color": str(feature["object_type_color"] or "#F3C623"),
                },
            )
        return sorted(result.values(), key=lambda item: (item["id"], item["slug"]))

    def set_display_mode(self, mode: str) -> list[QgsVectorLayer]:
        """Переключить категоризированный слой или синхронизированные представления."""

        if mode not in {"categorized", "group"}:
            raise ReviewSessionError("Неизвестный режим отображения типов объектов.")
        self._display_mode = mode
        project = QgsProject.instance()
        for layer_id in self._view_layer_ids:
            if project.mapLayer(layer_id) is not None:
                project.removeMapLayer(layer_id)
        self._view_layer_ids = []
        root = project.layerTreeRoot()
        canonical_node = root.findLayer(self.layer_id)
        if canonical_node is not None:
            canonical_node.setItemVisibilityChecked(mode == "categorized")
        if mode == "categorized" or len(self.object_types()) < 2:
            return [self.layer]
        group_name = f"Типы объектов MLSystem2 — {self.layer.name()}"
        existing = root.findGroup(group_name)
        group = existing or root.addGroup(group_name)
        views: list[QgsVectorLayer] = []
        for item in self.object_types():
            view = QgsVectorLayer(self.layer.source(), str(item["name"]), "ogr")
            if not view.isValid():
                raise ReviewSessionError("Не удалось создать представление типа объекта.")
            view.setCustomProperty("mlsystem2/review_session_view", self.layer_id)
            view.setCustomProperty("mlsystem2/object_type_slug", str(item["slug"]))
            view.setCustomProperty("mlsystem2/object_type_color", str(item["color"]))
            project.addMapLayer(view, False)
            group.addLayer(view)
            self._view_layer_ids.append(view.id())
            views.append(view)
        self._apply_type_view_filters()
        return views

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

    # Считает активное и отображаемое число кандидатов.
    def counts(self) -> dict[str, int]:
        active = [
            feature
            for feature in self.layer.getFeatures()
            if str(feature["review_status"] or "new") == "new"
        ]
        return {
            "total": len(active),
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

    def split_current(self, max_area_m2: float) -> None:
        """Поставить разбиение текущего кандидата одной командой."""

        feature = self.current_feature()
        if feature is None:
            raise ReviewSessionError("Нет текущего кандидата.")
        try:
            parts = split_geometry(
                feature.geometry(),
                self.layer.crs(),
                str(feature["candidate_id"]),
                max_area_m2,
                0.0,
            )
        except GeometrySplitError as exc:
            raise ReviewSessionError(str(exc)) from exc
        if len(parts) < 2:
            raise ReviewSessionError("Объект не требует разбиения при заданном пороге.")
        child_feature_ids = self._apply_split(int(feature.id()), parts)
        if child_feature_ids:
            self.select_feature(child_feature_ids[0])

    def split_large_candidates(self, max_area_m2: float) -> int:
        """Разбить все отфильтрованные объекты крупнее заданного порога."""

        feature_ids = [
            feature_id
            for feature_id in self.feature_ids()
            if float(self.feature(feature_id)["area_m2"] or 0.0) > max_area_m2
        ]
        count = 0
        for feature_id in feature_ids:
            self.select_feature(feature_id)
            self.split_current(max_area_m2)
            count += 1
        return count

    def export_filtered_layer(self, name: str | None = None) -> QgsVectorLayer:
        """Создать временный слой только из объектов, прошедших текущие фильтры."""

        feature_ids = self.feature_ids()
        if not feature_ids:
            raise ReviewSessionError("Нет объектов, прошедших фильтры площади и уверенности.")
        crs = self.layer.crs().authid() or "EPSG:4326"
        base_name = name or "Отфильтрованные объекты MLSystem2"
        layer_name = _unique_layer_name(base_name)
        output = QgsVectorLayer(f"MultiPolygon?crs={crs}", layer_name, "memory")
        output_fields = QgsFields()
        for field_name in _OUTPUT_FIELD_NAMES:
            field_index = self.layer.fields().indexOf(field_name)
            if field_index >= 0:
                output_fields.append(QgsField(self.layer.fields()[field_index]))
        if not output.dataProvider().addAttributes(output_fields):
            raise ReviewSessionError("Не удалось создать поля нового слоя.")
        output.updateFields()
        features: list[QgsFeature] = []
        for feature_id in feature_ids:
            source = self.feature(feature_id)
            target = QgsFeature(output.fields())
            target.setGeometry(QgsGeometry(source.geometry()))
            for field in output.fields():
                target[field.name()] = source[field.name()]
            features.append(target)
        result = output.dataProvider().addFeatures(features)
        success = result[0] if isinstance(result, tuple) else bool(result)
        if not success:
            raise ReviewSessionError("Не удалось добавить объекты в новый слой.")
        output.updateExtents()
        QgsProject.instance().addMapLayer(output)
        return output

    # Postoyanno menyaet dva polya audita odnim provider batch.
    def _set_review_status(
        self,
        feature_id: int,
        status: str,
    ) -> None:
        updates = {
            self.layer.fields().indexOf("review_status"): status,
        }
        if not self.layer.dataProvider().changeAttributeValues({feature_id: updates}):
            raise ReviewSessionError("Не удалось сохранить статус кандидата.")
        self.layer.triggerRepaint()
        self._refresh_type_views()

    # Atomarno dobavlyaet vse chasti razbieniya v GeoPackage.
    def _apply_split(self, feature_id: int, parts: list[SplitPart]) -> list[int]:
        parent = self.feature(feature_id)
        fields = self.layer.fields()
        children: list[QgsFeature] = []
        for offset, part in enumerate(parts, start=1):
            child = QgsFeature(fields)
            for field_name in _FIELD_NAMES:
                if fields.indexOf(field_name) >= 0:
                    child[field_name] = parent[field_name]
            child.setGeometry(part.geometry)
            child["candidate_id"] = part.candidate_id
            child["parent_candidate_id"] = part.parent_candidate_id
            child["area_m2"] = part.area_m2
            child["review_status"] = "new"
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
        self._refresh_type_views()
        self.changed.emit()
        return [int(item.id()) for item in added]

    # Proveriaet sootvetstvie tekushchemu filtru.
    def _matches(self, feature: QgsFeature) -> bool:
        status = str(feature["review_status"] or "new")
        return status == "new" and self._passes_thresholds(feature)

    def _passes_thresholds(self, feature: QgsFeature) -> bool:
        if (
            self._object_type_slug is not None
            and self.layer.fields().indexOf("object_type_slug") >= 0
            and str(feature["object_type_slug"] or "") != self._object_type_slug
        ):
            return False
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
            min_area_m2=self._min_area_m2,
            min_confidence=self._min_confidence,
            object_type_slug=self._object_type_slug,
        )
        self._apply_type_view_filters()

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

    def _refresh_type_views(self) -> None:
        project = QgsProject.instance()
        for layer_id in self._view_layer_ids:
            layer = project.mapLayer(layer_id)
            if isinstance(layer, QgsVectorLayer):
                layer.dataProvider().reloadData()
                layer.triggerRepaint()

    def _apply_type_view_filters(self) -> None:
        """Синхронизировать групповые представления с общей очередью и фильтрами."""

        project = QgsProject.instance()
        for layer_id in self._view_layer_ids:
            layer = project.mapLayer(layer_id)
            if not isinstance(layer, QgsVectorLayer):
                continue
            slug = str(layer.customProperty("mlsystem2/object_type_slug", "")).strip()
            if not slug:
                continue
            layer.setSubsetString(
                _active_subset_expression(
                    slug=slug,
                    selected_slug=self._object_type_slug,
                    min_area_m2=self._min_area_m2,
                    min_confidence=self._min_confidence,
                )
            )
            _apply_status_style(layer)

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
    int_type = _field_type("Int", "Int")
    for name in (
        "candidate_id",
        "parent_candidate_id",
        "job_id",
        "class_id",
        "class_name",
        "object_type_slug",
        "object_type_name",
        "object_type_color",
        "model_id",
        "model_version",
        "source_image_ids",
        "review_status",
    ):
        fields.append(QgsField(name, string_type))
    fields.append(QgsField("confidence", double_type))
    fields.append(QgsField("object_type_id", int_type))
    fields.append(QgsField("area_m2", double_type))
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
            "object_type_id": properties.get("object_type_id"),
            "object_type_slug": properties.get("object_type_slug"),
            "object_type_name": properties.get("object_type_name"),
            "object_type_color": properties.get("object_type_color"),
            "confidence": properties.get("confidence"),
            "model_id": properties.get("model_id"),
            "model_version": properties.get("model_version"),
            "source_image_ids": json.dumps(
                properties.get("source_image_ids") or [],
                ensure_ascii=False,
            ),
            "area_m2": properties.get("area_m2"),
            "review_status": "new",
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


# Назначает стиль и скрывает объекты вне фильтров.
def _active_subset_expression(
    *,
    slug: str,
    selected_slug: str | None,
    min_area_m2: float,
    min_confidence: float,
) -> str:
    escaped = slug.replace("'", "''")
    expression = (
        f'"object_type_slug" = \'{escaped}\' '
        'AND "review_status" = \'new\' '
        f'AND coalesce("area_m2", 0) >= {min_area_m2:.12g}'
    )
    if min_confidence > 0.0:
        expression += (
            ' AND "confidence" IS NOT NULL '
            f'AND "confidence" >= {min_confidence:.12g}'
        )
    if selected_slug is not None and selected_slug != slug:
        expression += " AND 1 = 0"
    return expression


def _apply_status_style(
    layer: QgsVectorLayer,
    *,
    min_area_m2: float = 0.0,
    min_confidence: float = 0.0,
    object_type_slug: str | None = None,
) -> None:
    root_rule = QgsRuleBasedRenderer.Rule(None)
    base_expression = (
        '"review_status" = \'new\' '
        f'AND coalesce("area_m2", 0) >= {min_area_m2:.12g}'
    )
    if min_confidence > 0.0:
        base_expression += (
            ' AND "confidence" IS NOT NULL '
            f'AND "confidence" >= {min_confidence:.12g}'
        )
    if object_type_slug:
        escaped = object_type_slug.replace("'", "''")
        base_expression += f' AND "object_type_slug" = \'{escaped}\''
    object_types: dict[str, tuple[str, str]] = {}
    if layer.fields().indexOf("object_type_slug") >= 0:
        for feature in layer.getFeatures():
            slug = str(feature["object_type_slug"] or "").strip()
            if slug:
                object_types.setdefault(
                    slug,
                    (
                        str(feature["object_type_name"] or slug),
                        str(feature["object_type_color"] or "#F3C623"),
                    ),
                )
    if not object_types:
        object_types = {"": ("Прошёл фильтры", "#FFD700")}
    for slug, (name, color_value) in sorted(object_types.items()):
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        color = QColor(color_value)
        color.setAlpha(110)
        symbol.setColor(color)
        expression = base_expression
        if slug:
            escaped = slug.replace("'", "''")
            expression += f' AND "object_type_slug" = \'{escaped}\''
        rule = QgsRuleBasedRenderer.Rule(symbol)
        rule.setLabel(name)
        rule.setFilterExpression(expression)
        root_rule.appendChild(rule)
    layer.setRenderer(QgsRuleBasedRenderer(root_rule))
    layer.triggerRepaint()


def _numeric_attribute(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unique_layer_name(base_name: str) -> str:
    """Подобрать свободное имя нового слоя в текущем проекте."""

    project = QgsProject.instance()
    if not project.mapLayersByName(base_name):
        return base_name
    suffix = 2
    while project.mapLayersByName(f"{base_name} ({suffix})"):
        suffix += 1
    return f"{base_name} ({suffix})"


def is_review_session_layer(layer: object) -> bool:
    """Распознать слой сессии, включая сохранённые в старых проектах слои."""

    if not isinstance(layer, QgsVectorLayer):
        return False
    marker = str(layer.customProperty(_SESSION_LAYER_PROPERTY, "")).casefold()
    if marker in {"1", "true"}:
        return True
    source = layer.source().replace("\\", "/").casefold()
    return (
        layer.name().startswith("Кандидаты MLSystem2")
        and "/.mlsystem2/review_sessions/pseudolabel_" in source
        and "|layername=candidates" in source
        and all(layer.fields().indexOf(name) >= 0 for name in _LEGACY_REQUIRED_FIELD_NAMES)
    )

__all__ = ["ReviewSession", "ReviewSessionError", "is_review_session_layer"]
