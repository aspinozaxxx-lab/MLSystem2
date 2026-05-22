from gpdadapter import FeatureCollection
from typing import Optional
from functools import wraps
from loguru import logger
from ..constants import Tag, Shape
from .circle import simplify_geometry_with_circle
from .rectangle import simplify_geometry_with_rectangle
from .l_shape import simplify_geometry_with_l_shape
from .grid import simplify_geometry_with_grid_snap
from .dynamic_grid import simplify_geometry_with_dynamic_grid


def return_none_on_fail(f):  # TODO: we already have similar wrapper in measure.py
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return None
    return wrapper


SIMPLIFICATION_FNS = {
    Shape.CIRCLE: simplify_geometry_with_circle,
    Shape.RECTANGLE: simplify_geometry_with_rectangle,
    Shape.LSHAPE: simplify_geometry_with_l_shape,
    Shape.GRID_SNAP: simplify_geometry_with_grid_snap,
    Shape.DYN_GRID: simplify_geometry_with_dynamic_grid
}


def simplify_fc_with_shapes(
        fc: FeatureCollection,
        shape_type: str,
        min_iou: float = 0.8,
        max_hausdorff: float = 7.0,
        min_area: float = 0.,
        iou_confidence_tag: Optional[str] = None,
        func_params: dict = None
) -> FeatureCollection:
    if fc.empty:
        return fc
    if func_params is None:
        func_params = dict()
    if  fc.crs != fc.estimate_utm_crs():
        logger.warning(f"FC crs is wrong: {fc.crs} != {fc.estimate_utm_crs()}")

    if Tag.BLD_SHAPE_TYPE not in fc.columns:
        fc[:, Tag.BLD_SHAPE_TYPE] = None
    if Tag.BLD_SIMPL_IOU not in fc.columns:
        fc[:, Tag.BLD_SIMPL_IOU] = 0.
    if Tag.BLD_SIMPL_HDF not in fc.columns:
        fc[:, Tag.BLD_SIMPL_HDF] = 0.
    if iou_confidence_tag and iou_confidence_tag not in fc.columns:
        fc[:, iou_confidence_tag] = 0.

    for idx in range(len(fc)):
        # skip already simplified
        if fc[idx, Tag.BLD_SHAPE_TYPE] not in (Shape.UNKNOWN, None):
            continue

        try:
            simplified_geometry = SIMPLIFICATION_FNS[shape_type](fc[idx, 'geometry'], **func_params)
            if not simplified_geometry:
                continue

            if (simplified_geometry.iou > min_iou and simplified_geometry.hausdorff < max_hausdorff) \
                    or fc[idx, 'geometry'].area < min_area:
                fc[idx, 'geometry'] = simplified_geometry.simple_geometry
                fc[idx, Tag.BLD_SHAPE_TYPE] = simplified_geometry.simple_geometry_type
                fc[idx, Tag.BLD_SIMPL_IOU] = round(simplified_geometry.iou, 2)
                fc[idx, Tag.BLD_SIMPL_HDF] = round(simplified_geometry.hausdorff, 2)
                if iou_confidence_tag:
                    fc[idx, iou_confidence_tag] = round(simplified_geometry.iou, 2)
        except Exception as e:
            logger.warning(f"Geometry simplification with {shape_type} skipped: {e}")
    return fc
