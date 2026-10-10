"""Геометрические брики экспортируемого шаблона в штатном процессе Geoalert."""

from pathlib import Path

from shapely.ops import transform as transform_geometry

from ._compact_geometry import is_compact_polygon


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

        def __call__(self, path):
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
