# TODO: deprecate this file in favor of utils/buildingutils

import numpy as np
import shapely
import shapely.affinity
from functools import wraps
from gpdadapter import FeatureCollection
from typing import Tuple, Any
from loguru import logger
from .utils.geomutils import return_empty_if_error, apply_buffer_if_invalid

# ================================== Public functions ==========================================


def roofs_to_footprints(roofs, meta):  # TODO: we never use it
    return _apply(roofs, meta, generate_footprint_from_roof)


def footprints_to_roofs(roofs, meta):  # TODO: we never use it
    return _apply(roofs, meta, generate_roof_from_footprint)


# ================================= Private generation helper functions =========================

def _valid(geometry: shapely.Polygon) -> shapely.Polygon:  # TODO: it is defined in multiple places
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    return geometry


def _xy_shift_by_height(height: float, azimuth: float, elevation: float) -> Tuple[float, float]:
    azimuth = azimuth - 180.
    azimuth = np.deg2rad(azimuth)
    elevation = np.deg2rad(elevation)
    proj_length = height / np.tan(elevation)
    x = proj_length * np.sin(azimuth)
    y = proj_length * np.cos(azimuth)
    return x, y


def _generate_shadow_end(roof: shapely.Polygon, height: float,
                         sun_azimuth: float, sun_elevation: float) -> shapely.Polygon:
    x, y = _xy_shift_by_height(height, sun_azimuth, sun_elevation)
    shadow_end = shapely.affinity.translate(roof, x, y)
    return shadow_end


def _fuse(polygon: shapely.Polygon, x: float, y: float) -> shapely.Polygon:
    # support for MultiPolygon
    if isinstance(polygon, shapely.geometry.MultiPolygon):
        fused_polygons = [_fuse(p, x, y) for p in polygon.geoms]
        geometry = shapely.geometry.MultiPolygon([])
        for p in fused_polygons:
            geometry = geometry.union(p)
        return geometry

    exterior_coords = list(polygon.exterior.coords)
    interiors_coords = [list(interior.coords) for interior in polygon.interiors]

    fused_polygons = []

    for i in range(len(exterior_coords) - 1):
        coords = sorted([exterior_coords[i], exterior_coords[i + 1]])

        crd_1 = [coords[0][0], coords[0][1]]
        crd_2 = [coords[0][0] + x, coords[0][1] + y]
        crd_3 = [coords[1][0] + x, coords[1][1] + y]
        crd_4 = [coords[1][0], coords[1][1]]

        p = shapely.geometry.Polygon([crd_1, crd_2, crd_3, crd_4])
        fused_polygons.append(p)

    for interior_coords in interiors_coords:
        for i in range(len(interior_coords) - 1):
            coords = sorted([interior_coords[i], interior_coords[i + 1]])

            crd_1 = [coords[0][0], coords[0][1]]
            crd_2 = [coords[0][0] + x, coords[0][1] + y]
            crd_3 = [coords[1][0] + x, coords[1][1] + y]
            crd_4 = [coords[1][0], coords[1][1]]

            p = shapely.geometry.Polygon([crd_1, crd_2, crd_3, crd_4])

            fused_polygons.append(p)

    geometry = shapely.geometry.MultiPolygon([])
    for p in fused_polygons:
        geometry = geometry.union(p)
    # geometry = shapely.geometry.MultiPolygon(fused_polygons)
    return geometry


def _to_numeric(x: Any, default: float = 0) -> float:
    try:
        output = float(x)
    except ValueError:
        output = default
    return output


def _apply(fc, meta, function):  # TODO: we never use it
    features = []

    for region in meta:
        for f in fc.intersection(region):

            if not f.properties:
                continue

            # print(_to_numeric(region.properties['sat_azimuth']),
            #       _to_numeric(region.properties['sat_elevation']))

            geom = function(f._geometry,
                            _to_numeric(f.properties['building_height']),
                            _to_numeric(region.properties['sat_azimuth']),
                            _to_numeric(region.properties['sat_elevation']))

            feature = geom
            features.append(feature)

    new_fc = FeatureCollection(features, crs=fc.crs)
    return new_fc


# ================================= Generation functions =========================================

def translate_polygon(p: shapely.Polygon, x: float, y: float) -> shapely.Polygon:
    if np.isnan(x) or np.isnan(y) or not p:
        return p
    return shapely.affinity.translate(p, x, y)

def generate_roof_from_footprint(footprint: shapely.Polygon, height: float,
                                 sat_azimuth: float, sat_elevation: float) -> shapely.Polygon:
    if (not isinstance(sat_azimuth, (int, float)) or not isinstance(sat_elevation, (int, float))
            or np.isnan(sat_azimuth) or np.isnan(sat_elevation)):
        logger.warning(f"Invalid values of sat_azimuth={sat_azimuth} or sat_elevation={sat_elevation}")
        return footprint
    if not isinstance(height, (int, float)) or np.isnan(height):
        logger.warning(f"Invalid value of height={height}")
        return footprint
    x, y = _xy_shift_by_height(height, sat_azimuth, sat_elevation)
    return translate_polygon(footprint, x, y)

def generate_footprint_from_roof(roof: shapely.Polygon, height: float,
                                 sat_azimuth: float, sat_elevation: float) -> shapely.Polygon:
    if (not isinstance(sat_azimuth, (int, float)) or not isinstance(sat_elevation, (int, float))
            or np.isnan(sat_azimuth) or np.isnan(sat_elevation)):
        logger.warning(f"Invalid values of sat_azimuth={sat_azimuth} or sat_elevation={sat_elevation}")
        return roof
    if not isinstance(height, (int, float)) or np.isnan(height):
        logger.warning(f"Invalid value of height={height}")
        return roof
    x, y = _xy_shift_by_height(height, sat_azimuth + 180, sat_elevation)
    return translate_polygon(roof, x, y)

@return_empty_if_error(shapely.geometry.Polygon)
@apply_buffer_if_invalid
def generate_wall_from_roof(roof: shapely.Polygon, height: float,
                            sat_azimuth: float, sat_elevation: float,
                            closing: float = 0.2, simplify: float = 0.2) -> shapely.Polygon:
    footprint = generate_footprint_from_roof(roof, height, sat_azimuth, sat_elevation)
    x, y = _xy_shift_by_height(height, sat_azimuth, sat_elevation)

    walls = _fuse(footprint, x, y)
    walls = _valid(walls)

    # subtract roof
    walls = walls.difference(roof)

    # simplify
    walls = walls.buffer(-closing).buffer(closing).simplify(simplify)
    return walls

@return_empty_if_error(shapely.geometry.Polygon)
@apply_buffer_if_invalid
def generate_wall_from_roof_by_x_y(roof: shapely.Polygon,
                                   x: float, y: float,
                                   closing: float = 0.2,
                                   simplify: float = 0.2) -> Tuple[shapely.Polygon, shapely.Polygon]:
    footprint = translate_polygon(roof, x, y)
    walls = _fuse(footprint, -x, -y)
    walls = _valid(walls)

    # subtract roof
    walls = walls.difference(roof)

    # simplify
    walls = walls.buffer(-closing).buffer(closing).simplify(simplify)
    return walls

def generate_wall_from_footprint(footprint: shapely.Polygon, height: float,
                                 sat_azimuth: float, sat_elevation: float,
                                 closing: float = 0.2, simplify: float = 0.2) -> shapely.Polygon:
    roof = generate_roof_from_footprint(footprint, height, sat_azimuth, sat_elevation)
    walls = generate_wall_from_roof(roof, height, sat_azimuth, sat_elevation, closing, simplify)
    return walls


def generate_shadow_from_footprint(footprint: shapely.Polygon, height: float,
                                   sun_azimuth: float, sun_elevation: float, sat_azimuth: float, sat_elevation: float,
                                   closing: float = 0.2, simplify: float = 0.2) -> shapely.Polygon:
    roof = generate_roof_from_footprint(footprint, height, sat_azimuth, sat_elevation)
    shadow = generate_shadow_from_roof(roof, height, sun_azimuth, sun_elevation, sat_azimuth, sat_elevation,
                                       closing, simplify)
    return shadow

def generate_shadow_and_wall_from_roof(roof: shapely.Polygon, height: float,
                                       sun_azimuth: float, sun_elevation: float,
                                       sat_azimuth: float, sat_elevation: float,
                                       closing: float = 0.2, simplify: float = 0.2,
                                       return_empty_on_error: bool = False) -> Tuple[shapely.Polygon, shapely.Polygon]:
    # generate wall
    try:
        wall = generate_wall_from_roof(roof, height, sat_azimuth, sat_elevation, closing, simplify)
    except shapely.errors.TopologicalError as e:
        if return_empty_on_error:
            wall = shapely.geometry.MultiPolygon()
        else:
            raise e

    # generate shadow
    try:
        footprint = generate_footprint_from_roof(roof, height, sat_azimuth, sat_elevation)
        shadow_end = _generate_shadow_end(footprint, height, sun_azimuth, sun_elevation)

        # validate geometry
        footprint = _valid(footprint)
        wall = _valid(wall)
        shadow_end = _valid(shadow_end)

        x, y = _xy_shift_by_height(height, sun_azimuth, sun_elevation)

        shadow = _fuse(footprint, x, y)
        shadow = _valid(shadow)

        shadow = shadow.union(shadow_end)
        shadow = _valid(shadow)

        shadow = shadow.difference(roof)
        shadow = _valid(shadow)

        shadow = shadow.difference(wall)
        shadow = _valid(shadow)

        shadow = shadow.buffer(-closing).buffer(closing).simplify(simplify)
    except shapely.errors.TopologicalError as e:
        if return_empty_on_error:
            shadow = shapely.geometry.MultiPolygon()
        else:
            raise e

    return shadow, wall


@wraps(generate_shadow_and_wall_from_roof)
def generate_shadow_from_roof(*args, **kwargs):
    return generate_shadow_and_wall_from_roof(*args, **kwargs)[0]
