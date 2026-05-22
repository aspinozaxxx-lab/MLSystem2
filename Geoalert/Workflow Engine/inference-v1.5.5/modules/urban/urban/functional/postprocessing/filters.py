import shapely
import shapely.geometry as sg
from ._types import PolygonLike


def remove_small_holes(geometry: PolygonLike, min_hole_area: float) -> PolygonLike:
    """
    Remove small holes from given geometry (polygon)

    Args:
        geometry (Polygon): any given shapely Polygon
        min_hole_area: holes with area (m^2) less than `min_hole_area` will be removed

    Return:
        geometry (if not geometry is Polygon return origin geometry)
    """

    if isinstance(geometry, sg.MultiPolygon):
        return sg.MultiPolygon([remove_small_holes(g, min_hole_area) for g in geometry.geoms])

    if geometry.is_empty or not isinstance(geometry, shapely.Polygon) or \
       (isinstance(geometry, shapely.Polygon) and len(geometry.interiors) == 0):
        return geometry

    coords = sg.mapping(geometry)['coordinates']
    holes = [h for h in coords[1:] if sg.Polygon(h).area > min_hole_area]
    out_poly = sg.Polygon(shell=coords[0], holes=holes)

    return out_poly
