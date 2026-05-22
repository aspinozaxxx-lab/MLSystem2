from shapely import Polygon, MultiPolygon, make_valid, GeometryCollection
from typing import Sequence, Union, Optional
from .angleutils import vector_from_height, vector_from_relative_positions
from ...base.defaults import SAT_ELEVATION_TAG, SUN_ELEVATION_TAG, SAT_AZIMUTH_TAG, SUN_AZIMUTH_TAG
import shapely.affinity


def failsafe(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(str(e))
            return Polygon()
    return wrapper


def move_primitive(coords: Sequence[Sequence[float]], vec: Sequence[float]):
    crd_1 = [coords[0][0], coords[0][1]]
    crd_2 = [coords[0][0] + vec[0], coords[0][1] + vec[1]]
    crd_3 = [coords[1][0] + vec[0], coords[1][1] + vec[1]]
    crd_4 = [coords[1][0], coords[1][1]]
    return Polygon([crd_1, crd_2, crd_3, crd_4])


def extrude(geom: Polygon, vec: Sequence[float]):
    """Extrudes a polygon in vec direction, returns None if fails"""
    exterior_coords = list(geom.exterior.coords)
    interiors_coords = [list(interior.coords) for interior in geom.interiors]
    fused_polygons = []
    for i in range(len(exterior_coords) - 1):
        fused_polygons.append(move_primitive(sorted([exterior_coords[i], exterior_coords[i + 1]]), vec))

    for interior_coords in interiors_coords:
        for i in range(len(interior_coords) - 1):
            fused_polygons.append(move_primitive(sorted([interior_coords[i], interior_coords[i + 1]]), vec))

    geometry = Polygon([])
    for p in fused_polygons:
        p = make_valid(p)
        if isinstance(p, (Polygon, MultiPolygon)):
            geometry = geometry.union(p)

    if isinstance(geometry, GeometryCollection):
        polygons = [p for p in geometry.geoms if isinstance(p, (Polygon, MultiPolygon))]
        geometry = MultiPolygon([])
        for p in polygons:
            geometry = geometry.union(p)

    return geometry


def generate_rooftop(footprint: Polygon, height: float, sat_azimuth: float, sat_elevation: float) -> Polygon:
    x, y = vector_from_height(sat_azimuth, sat_elevation, height)
    return shapely.affinity.translate(footprint, -x, -y)


def generate_wall_from_rooftop(rooftop: Polygon,
                               height: float,
                               sat_azimuth: float,
                               sat_elevation: float,
                               subtract_rooftop: bool = False,
                               subtract_buffer: float = 0.1) -> Optional[Polygon]:
    """Builds a wall Feature from rooftop and footprint, returns None if fails"""
    vec = vector_from_height(sat_azimuth, sat_elevation, height)

    wall = extrude(rooftop, vec)
    if wall is None:
        return None
    if subtract_rooftop:
        wall = wall.difference(rooftop.buffer(subtract_buffer))

    if isinstance(wall, GeometryCollection):
        polygons = [p for p in wall.geoms if isinstance(p, (Polygon, MultiPolygon))]
        wall = MultiPolygon([])
        for p in polygons:
            wall = wall.union(p)

    # remove interior
    if isinstance(wall, Polygon):
        wall = Polygon(wall.exterior.coords)
    return wall.buffer(0)


def generate_wall_from_footprint(footprint: Polygon,
                                 height: float,
                                 sat_azimuth: float,
                                 sat_elevation: float,
                                 subtract_rooftop: bool = False) -> Optional[Polygon]:
    """Builds a wall Feature from rooftop and footprint, returns None if fails"""
    rooftop = generate_rooftop(footprint, height, sat_azimuth, sat_elevation)
    return generate_wall_from_rooftop(rooftop, height, sat_azimuth, sat_elevation, subtract_rooftop)


def generate_shadow_from_footprint(footprint: Polygon,
                                   height: float,
                                   sun_azimuth: float,
                                   sun_elevation: float,
                                   subtract_footprint: bool = False,
                                   subtract_buffer: float = 0.1) -> Optional[Polygon]:
    """Builds a shadow Feature from height, sun_angles and footprint, returns None if fails"""
    vec = vector_from_height(sun_azimuth, sun_elevation, height)
    shadow = extrude(footprint, -vec)
    if shadow is None:
        return None

    if subtract_footprint:
        shadow = shadow.difference(footprint.buffer(subtract_buffer))

    if isinstance(shadow, shapely.geometry.collection.GeometryCollection):
        polygons = [p for p in shadow.geoms if isinstance(p, (Polygon, MultiPolygon))]
        shadow = MultiPolygon([])
        for p in polygons:
            shadow = shadow.union(p)

    # remove interior
    if isinstance(shadow, Polygon):
        shadow = Polygon(shadow.exterior.coords)
    return shadow.buffer(0)

