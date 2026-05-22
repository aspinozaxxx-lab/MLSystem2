import shapely
from shapely import LinearRing, Polygon
import numpy as np
from ..shapely_ext.angles import angle_PCA
from typing import Tuple
from .base import Shape
from ._simplified_geometry import SimplifiedGeometry
from .. import shapely_ext as se


def _snap_to_grid(coords: np.array, xgrid: np.array, ygrid: np.array) -> np.array:
    x, y = coords[:, 0], coords[:, 1]
    x = xgrid[np.argmin(np.abs((x[:, np.newaxis] - xgrid)), axis=1)]
    y = ygrid[np.argmin(np.abs((y[:, np.newaxis] - ygrid)), axis=1)]
    return np.stack((x, y), axis=1)


def _histogram(coords: np.array, window: int=2, max_iters: int=20) -> Tuple[np.array, np.array]:
    ybins = np.arange(np.min(coords[:, 1]) - window, np.max(coords[:, 1]) + window + 1, 1)
    xbins = np.arange(np.min(coords[:, 0]) - window, np.max(coords[:, 0]) + window + 1, 1)
    xstep = xbins[1] - xbins[0]
    ystep = ybins[1] - ybins[0]
    xhist = np.histogram(coords[:, 0], bins=xbins)[0]
    yhist = np.histogram(coords[:, 1], bins=ybins)[0]

    xgrid, ygrid = list(), list()

    i = 0
    while np.any(xhist) > 0 and i < max_iters:  # TODO: more sophisticated way to find maximums
        i += 1
        max_idx = np.argmax(xhist)
        avg_region = xbins[max_idx - window:max_idx + window + 1] + xstep / 2
        avg_weights = xhist[max_idx - window:max_idx + window + 1]
        xgrid.append(np.average(avg_region, weights=avg_weights))
        xhist[max_idx - window:max_idx + window + 1] = 0

    i = 0
    while np.any(yhist) > 0 and i < max_iters:
        i += 1
        max_idx = np.argmax(yhist)
        avg_region = ybins[max_idx - window:max_idx + window + 1] + ystep / 2
        avg_weights = yhist[max_idx - window:max_idx + window + 1]
        ygrid.append(np.average(avg_region, weights=avg_weights))
        yhist[max_idx - window:max_idx + window + 1] = 0

    return np.array(xgrid), np.array(ygrid)


def simplify_ring(
        linear_ring: LinearRing,
        min_area: float = 10,
        angle_pca_simplification: float = 0.5,
        angle_pca_segmentize: float = 0,
        angle_pca_min_vec_length: float = 3,
        angle_pca_cond_thr: float = 3,
        histogram_window=2,
        histogram_max_iters=20
):
    if shapely.Polygon(linear_ring).area < min_area:
        return shapely.oriented_envelope(linear_ring).exterior

    angle = angle_PCA(linear_ring, pre_simplification=angle_pca_simplification, segmentize=angle_pca_segmentize,
                      min_length=angle_pca_min_vec_length,
                      cond_thr=angle_pca_cond_thr)
    center = linear_ring.centroid
    coords = np.array(shapely.affinity.rotate(linear_ring, -angle, center).coords)
    xgrid, ygrid = _histogram(coords, window=histogram_window, max_iters=histogram_max_iters)
    snapped = _snap_to_grid(coords, xgrid, ygrid)
    simplified = shapely.affinity.rotate(Polygon(snapped), angle, center)
    return simplified


def dynamic_grid_simplification(
        polygon: Polygon,
        min_area: float = 10,
        angle_pca_simplification: float = 0.5,
        angle_pca_segmentize: float = 0,
        angle_pca_min_vec_length: float = 3,
        angle_pca_cond_thr: float = 3,
        histogram_window=2,
        histogram_max_iters=20,
        final_simplify: float = 0
) -> Polygon:
    if not isinstance(polygon, shapely.Polygon):
        raise ValueError('Only Polygons allowed! ')
    if not polygon:
        return shapely.Polygon()

    exterior = simplify_ring(polygon.exterior, min_area=min_area, angle_pca_simplification=angle_pca_simplification,
                             angle_pca_segmentize=angle_pca_segmentize,
                             angle_pca_min_vec_length=angle_pca_min_vec_length,
                             angle_pca_cond_thr=angle_pca_cond_thr, histogram_window=histogram_window,
                             histogram_max_iters=histogram_max_iters)
    holes = [simplify_ring(hole, min_area=min_area, angle_pca_simplification=angle_pca_simplification,
                           angle_pca_segmentize=angle_pca_segmentize,
                           angle_pca_min_vec_length=angle_pca_min_vec_length,
                           angle_pca_cond_thr=angle_pca_cond_thr, histogram_window=histogram_window,
                           histogram_max_iters=histogram_max_iters) for hole in polygon.interiors]

    return shapely.Polygon(exterior, holes).buffer(0).simplify(final_simplify)


def simplify_geometry_with_dynamic_grid(
        polygon: Polygon,
        min_area: float = 10,
        angle_pca_simplification: float = 0.5,
        angle_pca_segmentize: float = 0,
        angle_pca_min_vec_length: float = 3,
        angle_pca_cond_thr: float = 3,
        histogram_window=2,
        histogram_max_iters=20,
        final_simplify: float = 0) -> SimplifiedGeometry:
    simplified_poly = dynamic_grid_simplification(
        polygon, min_area=min_area,
        angle_pca_simplification=angle_pca_simplification,
        angle_pca_segmentize=angle_pca_segmentize,
        angle_pca_min_vec_length=angle_pca_min_vec_length,
        angle_pca_cond_thr=angle_pca_cond_thr,
        histogram_window=histogram_window,
        histogram_max_iters=histogram_max_iters,
        final_simplify=final_simplify)

    simplified_geometry = SimplifiedGeometry(
        origin_geometry=polygon,
        simple_geometry=simplified_poly,
        simple_geometry_type=Shape.DYN_GRID,
        iou=se.intersection_over_union(polygon, simplified_poly),
    )

    return simplified_geometry
