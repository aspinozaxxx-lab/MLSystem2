"""Deterministicheskoe razbienie poligonov v metricheskoi CRS."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsProject,
    QgsRectangle,
)


class GeometrySplitError(RuntimeError):
    """Bezopasno razbit geometriyu ne udalos."""


@dataclass(frozen=True)
class SplitPart:
    """Odna proverennaya chast razbieniya."""

    candidate_id: str
    parent_candidate_id: str
    geometry: QgsGeometry
    area_m2: float


def split_geometry(
    geometry: QgsGeometry,
    source_crs: QgsCoordinateReferenceSystem,
    parent_candidate_id: str,
    max_area_m2: float,
    min_area_m2: float,
) -> list[SplitPart]:
    """Razbit geometriyu po dlinnoi storone bounding box."""

    if geometry.isNull() or geometry.isEmpty() or not source_crs.isValid():
        raise GeometrySplitError("Геометрия или её CRS некорректна.")
    if max_area_m2 <= 0 or min_area_m2 < 0 or min_area_m2 >= max_area_m2:
        raise GeometrySplitError("Пороги площади разбиения заданы некорректно.")
    source = geometry.makeValid()
    if source.isNull() or source.isEmpty():
        raise GeometrySplitError("Не удалось исправить исходную геометрию.")
    metric_crs = _local_equal_area_crs(source, source_crs)
    to_metric = QgsCoordinateTransform(source_crs, metric_crs, QgsProject.instance())
    to_source = QgsCoordinateTransform(metric_crs, source_crs, QgsProject.instance())
    metric = QgsGeometry(source)
    if metric.transform(to_metric) != 0:
        raise GeometrySplitError("Не удалось преобразовать геометрию в метрическую CRS.")
    original_area = metric.area()
    if original_area <= 0:
        raise GeometrySplitError("Площадь геометрии должна быть больше нуля.")
    queue = _polygon_parts(metric)
    accepted: list[QgsGeometry] = []
    iterations = 0
    while queue:
        part = queue.pop(0)
        if part.area() <= max_area_m2:
            accepted.append(part)
            continue
        children = _bisect(part)
        if len(children) < 2:
            raise GeometrySplitError("Полигон не удалось безопасно разделить.")
        queue.extend(children)
        iterations += 1
        if iterations > 10_000:
            raise GeometrySplitError("Превышено число итераций разбиения.")
    split_area = sum(part.area() for part in accepted)
    if abs(original_area - split_area) / original_area > 0.01:
        raise GeometrySplitError("При разбиении потеряно более 1% площади.")
    retained = [part for part in accepted if part.area() >= min_area_m2]
    if not retained:
        raise GeometrySplitError("Все части меньше минимальной допустимой площади.")
    retained.sort(key=_geometry_sort_key)
    result: list[SplitPart] = []
    for part in retained:
        area_m2 = part.area()
        restored = QgsGeometry(part)
        if restored.transform(to_source) != 0:
            raise GeometrySplitError("Не удалось вернуть часть в исходную CRS.")
        restored = restored.makeValid()
        if restored.isNull() or restored.isEmpty():
            raise GeometrySplitError("После преобразования получена пустая часть.")
        candidate_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"mlsystem2:split:{parent_candidate_id}:{bytes(restored.asWkb()).hex()}",
        )
        result.append(
            SplitPart(
                candidate_id=str(candidate_id),
                parent_candidate_id=parent_candidate_id,
                geometry=restored,
                area_m2=area_m2,
            )
        )
    return result


# Stroit lokalnuyu ravnoploshchadnuyu CRS po centru poligona.
def _local_equal_area_crs(
    geometry: QgsGeometry,
    source_crs: QgsCoordinateReferenceSystem,
) -> QgsCoordinateReferenceSystem:
    centroid = geometry.centroid()
    to_wgs84 = QgsCoordinateTransform(
        source_crs,
        QgsCoordinateReferenceSystem("EPSG:4326"),
        QgsProject.instance(),
    )
    if centroid.transform(to_wgs84) != 0:
        raise GeometrySplitError("Не удалось определить центр геометрии.")
    point = centroid.asPoint()
    definition = (
        f"+proj=laea +lat_0={point.y():.10f} +lon_0={point.x():.10f} "
        "+datum=WGS84 +units=m +no_defs"
    )
    crs = QgsCoordinateReferenceSystem()
    if not crs.createFromProj(definition):
        raise GeometrySplitError("Не удалось создать локальную равноплощадную CRS.")
    return crs


# Razvorachivaet sostavnuyu geometriyu v otdelnye poligony.
def _polygon_parts(geometry: QgsGeometry) -> list[QgsGeometry]:
    parts = geometry.asGeometryCollection() if geometry.isMultipart() else [geometry]
    return [part.makeValid() for part in parts if not part.isEmpty() and part.area() > 0]


# Rezet poligon dvumya pryamougolnikami po dlinnoi storone.
def _bisect(geometry: QgsGeometry) -> list[QgsGeometry]:
    bounds = geometry.boundingBox()
    if bounds.width() >= bounds.height():
        middle = (bounds.xMinimum() + bounds.xMaximum()) / 2.0
        rectangles = (
            QgsRectangle(bounds.xMinimum(), bounds.yMinimum(), middle, bounds.yMaximum()),
            QgsRectangle(middle, bounds.yMinimum(), bounds.xMaximum(), bounds.yMaximum()),
        )
    else:
        middle = (bounds.yMinimum() + bounds.yMaximum()) / 2.0
        rectangles = (
            QgsRectangle(bounds.xMinimum(), bounds.yMinimum(), bounds.xMaximum(), middle),
            QgsRectangle(bounds.xMinimum(), middle, bounds.xMaximum(), bounds.yMaximum()),
        )
    output: list[QgsGeometry] = []
    for rectangle in rectangles:
        clipped = geometry.intersection(QgsGeometry.fromRect(rectangle)).makeValid()
        output.extend(_polygon_parts(clipped))
    return output


# Formiruet deterministicheskii klyuch poryadka chastei.
def _geometry_sort_key(geometry: QgsGeometry) -> tuple[float, float, str]:
    bounds = geometry.boundingBox()
    return (bounds.xMinimum(), bounds.yMinimum(), bytes(geometry.asWkb()).hex())


__all__ = ["GeometrySplitError", "SplitPart", "split_geometry"]
