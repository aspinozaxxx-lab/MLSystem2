import warnings
import shapely
from gpdadapter import FeatureCollection
from shapely.geometry import Polygon, MultiPolygon
from typing import Union


def cut_ring_angles(ring, offset=0.25):
    """
    Cuts angles in shapely LinearRing, that is a part of a Polygon. Actually, will accept any sequence of points
    Every vertex after the cut is split into two new vertices that lie on the adjacent edges.

    Args:
        ring: a closed sequence of points, every point being a tuple of two coordinates.
        offset: a fraction of the edge, which is cut. Bigger offset means bigger radius of smoothing.
        Values vary from 0 to 0.5 however the optimal value is 0.25, which leads the square to converge to a circle

    Returns:
        a new sequence of points with the angles cut. The number of vertices is double of the initial
    """
    if len(ring) == 0:
        return ring
    if ring[0] != ring[-1]:
        raise ValueError('Ring must be a closed sequence, with first point equal to the last')
    if offset < 0 or offset > 0.5:
        raise ValueError(f"Offset must be in range from 0 to 0.5, got {offset} instead")

    # That is why we omit the point 0 when cutting angles,
    contour = ring[1:]

    new_contour = []
    for i, point in enumerate(contour):
        if i == len(contour) - 1:
            prev_p = contour[i - 1]
            next_p = contour[0]
        else:
            prev_p = contour[i - 1]
            next_p = contour[i + 1]
        new_angle = [[point[0] * (1 - offset) + prev_p[0] * offset, point[1] * (1 - offset) + prev_p[1] * offset],
                     [point[0] * (1 - offset) + next_p[0] * offset, point[1] * (1 - offset) + next_p[1] * offset]]
        new_contour += new_angle
    # In the end we make the sequence closed again
    new_contour.append(new_contour[0])

    return new_contour


def cut_polygon_angles(poly, offset=0.25):
    new_coords = [cut_ring_angles(poly.exterior.coords, offset)]
    # holes
    for ring in poly.interiors:
        new_coords.append(cut_ring_angles(ring.coords, offset))
    new_poly = Polygon(shell=new_coords[0], holes=new_coords[1:])
    return new_poly


def cut_feature_angles(feature: Union[shapely.Polygon, shapely.MultiPolygon],
                       offset: float = 0.25) -> Union[shapely.Polygon, shapely.MultiPolygon]:
    """
    Cuts every point of the feature
    Args:
        feature: shapely geometry
        offset: a fraction of the edge, which is cut. Bigger offset means bigger radius of smoothing.
        Values vary from 0 to 0.5 however the optimal value is 0.25, which leads the square to converge to a circle

    Returns:
        a new Feature with cut angles
    """
    # outer contour of the polygon
    if isinstance(feature, Polygon):
        return cut_polygon_angles(feature, offset)
    elif isinstance(feature, MultiPolygon):
        new_geom = [cut_polygon_angles(poly, offset) for poly in feature.geoms if isinstance(poly, Polygon)]
        return MultiPolygon(new_geom) if len(new_geom) > 1 else new_geom[0]
    else:
        raise ValueError('Only Polygon and MultiPolygon features are supported')


def smooth(fc: FeatureCollection, iterations=1, offset=0.25, allow_fails=True):
    """
    Smoothes the FeatureCollection, works just like
    `QGIS smooth tool:
    <https://docs.qgis.org/testing/en/docs/user_manual/processing_algs/qgis/vectorgeometry.html#smooth>`_.
    with the same parameters

    Args:
        fc: FeatureCollection, input
        iterations: number of iterations of the angle cutting.
        Typically, 3 iteration make are enough to make the angle look like a smooth curve to human eye
        offset:  a fraction of the edge, which is cut on every iteration
        allow_fails: if True, the polygons that failed to smooth, will be added to the output unchanged with a warning;
            else an exceptions is raised
    Returns:
        a new FeatureCollection with smoothed geometry
    """

    def iterative_cut_angles(g: Union[Polygon, MultiPolygon]) -> Union[Polygon, MultiPolygon]:
        for _ in range(iterations):
            try:
                g = cut_feature_angles(g, offset)
            except Exception as e:
                if allow_fails:
                    warnings.warn('Something wrong in smoothing, skipping a feature. Error: ' + str(e))
                else:
                    raise e
        return g
    fc.geometry = fc.geometry.map(iterative_cut_angles)
    return fc
