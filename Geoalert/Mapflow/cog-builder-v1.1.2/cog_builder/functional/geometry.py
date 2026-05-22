import geojson
from loguru import logger
from typing import Mapping, Any
from .errors import CogInvalidAOI

geojson.geometry.DEFAULT_PRECISION = 15
SHAPELY_PRESENT = True
try:
    from shapely import geometry as shapely_geom
    from shapely.validation import explain_validity
    mapping = shapely_geom.mapping
except ImportError:
    SHAPELY_PRESENT = False
    logger.warning('Shapely not found, falling back to geojson validation')
    mapping = lambda x: x

def maybe_valid_geometry(geometry_json: Mapping[str, Any]):
    """
    validate geometry and transform to geojson object.
    args:
        mask_json_str: JSON-like dict representing geojson geometry,
                       like {'type': 'Polygon', 'coordinates': [[[...],[...],...]] }
    """
    try:
        geometry = validate_with_shapely(geometry_json)
    except ImportError:
        geometry = validate_with_geojson(geometry_json)
    return geometry


def validate_with_shapely(geometry_json: Mapping[str, Any]):
    from shapely import geometry as shapely_geom
    from shapely.validation import explain_validity
    logger.debug('Checking with shapely')
    try:
        geometry_shape = shapely_geom.shape(geometry_json)
    except Exception as e:
        raise CogInvalidAOI(aoi = geometry_json, reason = str(e))
    if not isinstance(geometry_shape, (shapely_geom.Polygon, shapely_geom.MultiPolygon)):
        raise CogInvalidAOI(aoi=geometry_json, reason = f"Provided mask must be a geometry object (Polygon, Multipolygon, etc.)"
                          f", got {type(geometry_shape)} instead")
    if not geometry_shape.is_valid:
        # try to fix it with shapely
        logger.debug('Trying to fix geometry')
        geometry_shape = geometry_shape.buffer(0)
        if not geometry_shape.is_valid:
            raise CogInvalidAOI(aoi = geometry_shape, reaeon = explain_validity(geometry_shape))
    return geometry_shape


def validate_with_geojson(geometry_json: Mapping[str, Any]):
    logger.debug('Shapely not present, checking with geojson')
    try:
        feature = geojson.Feature(geometry=geometry_json)
        geometry = feature.geometry
    except Exception as e:
        raise CogInvalidAOI(aoi = geometry_json, reason=str(e))
    if not isinstance(geometry, (geojson.Polygon, geojson.MultiPolygon)):
        raise CogInvalidAOI(f"Provided mask must be a geometry object (Polygon, Multipolygon, etc.)"
                          f", got {type(geometry)} instead")
    if not geometry.is_valid:
        raise CogInvalidAOI(aoi = geometry_json, reason = geometry.errors())

    return geometry_json