"""Геометрические брики экспортируемого шаблона в штатном процессе Geoalert."""

import json
import logging
import math
from pathlib import Path

import shapely
from shapely.ops import transform as transform_geometry

from ._compact_geometry import is_compact_polygon


_LOGGER = logging.getLogger(__name__)
_COVERAGE_PROFILE = "shared_edges_v1"


def register_vector_bricks() -> None:
    from pydantic import Field
    from urban import FilterSmallObjects as OriginalFilterSmallObjects
    from urban import VectorizeMasks as OriginalVectorizeMasks
    from urban.base.brick import PolygonProcessingBrick
    from urban.functional import io
    import rasterio

    class VectorizeMasks(OriginalVectorizeMasks):
        mark_boundary_objects: bool = False
        boundary_tolerance_pixels: float = Field(default=1.0, ge=0)
        boundary_tag: str = "_touches_raster_boundary"
        match_instance_edges: bool = False

        def __call__(self, path):
            if self.match_instance_edges and self.value_property_name == "instance_id":
                from gpdadapter import FeatureCollection

                # Общие вершины вставляются в родном CRS: репроекция ненодированного
                # T-стыка превращает общую прямую в разные ломаные соседних объектов.
                for raster, vector in zip(self.input_rasters, self.output_fcs, strict=True):
                    with rasterio.open(Path(path) / f"{raster}.tif") as source:
                        fc = FeatureCollection.from_features(
                            rasterio.features.dataset_features(source, bidx=1, geographic=False), source.crs)
                        if not fc.empty:
                            fc[:, self.value_property_name] = fc[:, "val"]
                            fc.drop("filename", axis=1, inplace=True)
                            fc.drop("val", axis=1, inplace=True)
                            fc[:, "_x_res"], fc[:, "_y_res"] = abs(source.res[0]), abs(source.res[1])
                            fc[:, "_crs"] = str(source.crs)
                            fc.geometry, diagnostic = _match_coverage_edges(list(fc.geometry))
                            _log_coverage("Подготовка общих границ", diagnostic)
                            if self.mark_boundary_objects:
                                inverse = ~source.transform
                                fc[:, self.boundary_tag] = fc.apply(lambda feature: _touches_boundary(
                                    feature.geometry, inverse, source.width, source.height,
                                    self.boundary_tolerance_pixels))
                        io.save_fc(fc, path, vector)
                return
            super().__call__(path)
            if not self.mark_boundary_objects:
                return
            for raster, vector in zip(self.input_rasters, self.output_fcs, strict=True):
                with rasterio.open(Path(path) / f"{raster}.tif") as source:
                    fc = io.read_fc(path, vector, crs=source.crs)
                    inverse = ~source.transform
                    fc[:, self.boundary_tag] = fc.apply(lambda feature: _touches_boundary(
                        feature.geometry, inverse, source.width, source.height,
                        self.boundary_tolerance_pixels,
                    ))
                    io.save_fc(fc, path, vector)

    class FilterSmallObjects(OriginalFilterSmallObjects):
        preserve_boundary_objects: bool = False
        boundary_tag: str = "_touches_raster_boundary"

        def process(self, fc):
            if not self.preserve_boundary_objects:
                return super().process(fc)
            fc = fc.filter(lambda feature: _boundary_flag(feature, self.boundary_tag)
                           or feature.geometry.area > self.min_area)
            if self.area_tag:
                fc[:, self.area_tag] = fc.geometry.area
            return fc

    class FilterCompactObjects(PolygonProcessingBrick):
        min_isoperimetric_quotient: float = Field(default=0.25, ge=0, le=1)
        max_bbox_ratio: float = Field(default=3.5, ge=1)
        preserve_boundary_objects: bool = True
        boundary_tag: str = "_touches_raster_boundary"

        def process(self, fc):
            return self._filter(fc, keep_compact=False)

        def _filter(self, fc, *, keep_compact):
            def keep(feature):
                geometry = feature.geometry
                if self.preserve_boundary_objects and _boundary_flag(feature, self.boundary_tag):
                    return True
                # Дефект геометрии не должен превращать весь расчёт F1 в ошибку.
                if geometry is None or geometry.is_empty or not geometry.is_valid or geometry.area <= 0:
                    return True
                return is_compact_polygon(geometry,
                    min_isoperimetric_quotient=self.min_isoperimetric_quotient,
                    max_bbox_ratio=self.max_bbox_ratio) == keep_compact

            return fc.filter(keep)

    class FilterNonCompactObjects(FilterCompactObjects):
        def process(self, fc):
            return self._filter(fc, keep_compact=True)

    class CoverageSimplify(PolygonProcessingBrick):
        """Упростить общие рёбра покрытия, сохранив внешний контур и пустоты."""

        rate: float = Field(default=0.5, gt=0, allow_inf_nan=False)

        def process(self, fc):
            def final_projection(geometries):
                # Точно тот же путь, что io.save_fc перед make_valid/explode.
                # Проверяем копию: не меняем CRS и свойства рабочей коллекции.
                projected = fc.copy()
                projected.geometry = geometries
                projected.to_crs("EPSG:4326", inplace=True)
                return list(projected.geometry)

            geometries, diagnostic = _simplify_coverage(
                list(fc.geometry), self.rate, final_projection=final_projection,
            )
            if diagnostic["status"] == "fallback":
                # Даже отказ от упрощения не разрешает SDK незаметно дробить
                # исходные объекты при заключительном make_valid/explode.
                try:
                    signatures = [_polygon_signature(geometry) for geometry in geometries]
                    projected = final_projection(geometries)
                    if (any(signature is None for signature in signatures)
                            or [_polygon_signature(geometry) for geometry in projected] != signatures
                            or not shapely.coverage_is_valid(projected)):
                        raise ValueError("unsafe_original_final_projection")
                except Exception as exc:
                    _log_coverage("Сохранение исходных границ запрещено", {
                        **diagnostic, "reason": "unsafe_original_final_projection",
                    })
                    raise RuntimeError(
                        "Исходные границы нельзя безопасно сохранить в EPSG:4326. "
                        "Обработка сцены остановлена, чтобы не дробить и не терять объекты."
                    ) from exc
                diagnostic.update(final_crs_validated=True, fallback_original_validated=True)
            fc.geometry = geometries
            _log_coverage("Совместное упрощение границ", diagnostic)
            return fc


def _coverage_supported():
    return (getattr(shapely, "geos_version", (0,)) >= (3, 12, 0)
            and callable(getattr(shapely, "coverage_is_valid", None))
            and callable(getattr(shapely, "coverage_simplify", None)))


def _coverage_diagnostic(geometries):
    return {"profile": _COVERAGE_PROFILE, "status": "unchanged", "reason": None,
            "features": len(geometries), "coordinates_before": int(shapely.count_coordinates(geometries)),
            "coordinates_after": int(shapely.count_coordinates(geometries))}


def _polygon_signature(geometry):
    if geometry is None or geometry.is_empty or not geometry.is_valid:
        return None
    if geometry.geom_type == "Polygon":
        return ("Polygon", len(geometry.interiors))
    if geometry.geom_type == "MultiPolygon":
        return ("MultiPolygon", tuple(sorted(len(part.interiors) for part in geometry.geoms)))
    return None


def _match_coverage_edges(geometries):
    """Вставить отсутствующие вершины T-стыков без перемещения исходных границ."""
    diagnostic = _coverage_diagnostic(geometries)
    if not geometries:
        return geometries, diagnostic
    if not _coverage_supported():
        return geometries, {**diagnostic, "status": "fallback", "reason": "coverage_unavailable"}
    if any(_polygon_signature(geometry) is None for geometry in geometries):
        return geometries, {**diagnostic, "status": "fallback", "reason": "invalid_geometry"}
    try:
        if shapely.coverage_is_valid(geometries):
            return geometries, diagnostic
        # Нодирование не чинит наложения: каждый полученный полигон обязан
        # топологически совпадать ровно с одним исходным объектом целиком.
        linework = shapely.unary_union([geometry.boundary for geometry in geometries])
        faces = list(shapely.get_parts(shapely.polygonize(shapely.get_parts(linework))))
        tree = shapely.STRtree(geometries)
        matched = [None] * len(geometries)
        for face in faces:
            indexes = tree.query(face.representative_point(), predicate="within")
            if not len(indexes):  # Отверстие или фон между объектами.
                continue
            if len(indexes) != 1:
                raise ValueError("overlapping_inputs")
            index = int(indexes[0])
            if (matched[index] is not None or not face.equals(geometries[index])
                    or _polygon_signature(face) != _polygon_signature(geometries[index])):
                raise ValueError("changed_object")
            matched[index] = face
        if any(geometry is None for geometry in matched) or not shapely.coverage_is_valid(matched):
            raise ValueError("unmatched_coverage")
        diagnostic.update(status="matched", coordinates_after=int(shapely.count_coordinates(matched)))
        return matched, diagnostic
    except Exception as exc:  # Неподходящий вход сохраняется, а не «исправляется» с потерей ID.
        reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        return geometries, {**diagnostic, "status": "fallback", "reason": reason}


def _simplify_coverage(geometries, tolerance, *, final_projection=None):
    """Visvalingam для общего покрытия; допуск — масштаб площади, не max distance."""
    diagnostic = {**_coverage_diagnostic(geometries), "tolerance_m": tolerance,
                  "simplify_boundary": False, "algorithm": "Visvalingam-Whyatt",
                  "effective_tolerance_m": None, "simplification_attempts": 0,
                  "final_crs_validated": False}
    if not geometries:
        return geometries, diagnostic
    if not _coverage_supported():
        return geometries, {**diagnostic, "status": "fallback", "reason": "coverage_unavailable"}
    if (not math.isfinite(tolerance) or tolerance <= 0
            or any(_polygon_signature(geometry) is None for geometry in geometries)):
        return geometries, {**diagnostic, "status": "fallback", "reason": "invalid_input"}
    try:
        if not shapely.coverage_is_valid(geometries):
            return geometries, {**diagnostic, "status": "fallback", "reason": "invalid_coverage"}
        signatures = [_polygon_signature(geometry) for geometry in geometries]
        original_union = shapely.coverage_union_all(geometries)
        for attempt in range(8 if final_projection is not None else 1):
            actual_tolerance = tolerance / (2 ** attempt)
            diagnostic["simplification_attempts"] = attempt + 1
            # Каждый повтор начинает с исходного покрытия и меняет допуск всех
            # общих рёбер вместе: нельзя вернуть лишь один «неудачный» полигон.
            result = list(shapely.coverage_simplify(
                geometries, actual_tolerance, simplify_boundary=False,
            ))
            if (len(result) != len(geometries)
                    or [_polygon_signature(geometry) for geometry in result] != signatures
                    or not shapely.coverage_is_valid(result)):
                raise ValueError("changed_topology")
            # Не допускаем расширение ни внешней границы, ни дыр nodata/фона.
            if not original_union.equals(shapely.coverage_union_all(result)):
                raise ValueError("changed_coverage_boundary")
            if final_projection is not None:
                projected = list(final_projection(result))
                if (len(projected) != len(result)
                        or [_polygon_signature(geometry) for geometry in projected] != signatures
                        or not shapely.coverage_is_valid(projected)):
                    continue
                diagnostic["final_crs_validated"] = True
            diagnostic.update(status="simplified", effective_tolerance_m=actual_tolerance,
                              coordinates_after=int(shapely.count_coordinates(result)))
            return result, diagnostic
        return geometries, {**diagnostic, "status": "fallback", "reason": "unsafe_final_projection"}
    except Exception as exc:
        reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        return geometries, {**diagnostic, "status": "fallback", "reason": reason}


def _log_coverage(operation, diagnostic):
    payload = json.dumps(diagnostic, ensure_ascii=False)
    try:
        from loguru import logger
    except ImportError:
        log = _LOGGER.warning if diagnostic["status"] == "fallback" else _LOGGER.info
        log("%s: %s", operation, payload)
    else:
        # Geoalert собирает именно loguru: успешное упрощение тоже видно в журнале.
        logger.log("WARNING" if diagnostic["status"] == "fallback" else "INFO", "{}: {}", operation, payload)


def _boundary_flag(feature, tag: str) -> bool:
    # Отсутствующее значение после объединения коллекций бывает NaN.
    value = feature.get(tag, False)
    return value is True or value == 1


def _touches_boundary(geometry, inverse, width: int, height: int, tolerance: float) -> bool:
    if geometry is None or geometry.is_empty:
        return False
    def to_pixels(x, y, z=None):
        return inverse.a * x + inverse.b * y + inverse.c, inverse.d * x + inverse.e * y + inverse.f

    min_x, min_y, max_x, max_y = transform_geometry(to_pixels, geometry).bounds
    return min_x <= tolerance or min_y <= tolerance or max_x >= width - tolerance or max_y >= height - tolerance
