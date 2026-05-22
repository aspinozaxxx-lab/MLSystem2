import math
import copy
import shapely
import shapely.geometry
import shapely.affinity
from gpdadapter import FeatureCollection
from typing import Tuple

__all__ = ["correct_topology_by_edge_shift"]


def shift_point(
        point: shapely.geometry.Point,
        angle: float,
        distance: float,
):
    """Shift point in specified direction

    Args:
        point: point geometry
        angle: angle in radians
        distance: distance in any units

    """
    xoff = math.cos(angle) * distance
    yoff = math.sin(angle) * distance
    return shapely.affinity.translate(point, xoff, yoff)


def shift_edge(
        edge: Tuple[shapely.geometry.Point, shapely.geometry.Point],
        angle: float,
        distance: float,
):
    """Shift edge in specified direction

    Args:
        edge: tuple of two points
        angle: angle in radians
        distance: distance in any units

    """
    return (
        shift_point(edge[0], angle, distance),
        shift_point(edge[1], angle, distance),
    )


def get_inner_norm_angle(edge: Tuple[shapely.geometry.Point, shapely.geometry.Point]):
    """Return angle perpendicular to edge direction (normal angle)"""
    p1, p2 = edge
    x = p2.x - p1.x
    y = p2.y - p1.y
    angle = math.atan2(y, x)
    return angle - math.pi / 2


def shift_edge_perpendicular(edge: Tuple[shapely.geometry.Point, shapely.geometry.Point], distance: float):
    angle = get_inner_norm_angle(edge)
    return shift_edge(edge, angle, distance)


def point_coords(p):
    return p.x, p.y


def shift_polygon_edge(polygon, edge_number, distance):
    """Shift one edge of geometry in perpendicular direction (in direction which reduce geometry area)

    Args:
        polygon: geojson-like geometry
        edge_number:
        distance:
    """
    exterior_coords = list(polygon.exterior.coords)
    p1 = shapely.geometry.Point(exterior_coords[edge_number])
    p2 = shapely.geometry.Point(exterior_coords[edge_number + 1])
    p1_, p2_ = shift_edge_perpendicular((p1, p2), distance)
    exterior_coords[edge_number] = (p1_.x, p1_.y)
    exterior_coords[edge_number + 1] = (p2_.x, p2_.y)

    if edge_number == 0:
        exterior_coords[-1] = (p1_.x, p1_.y)

    if edge_number == len(exterior_coords) - 2:
        exterior_coords[0] = (p2_.x, p2_.y)

    p = shapely.geometry.Polygon(shell=exterior_coords).buffer(0)  # buffer fix validity for last point
    return p


def line_generator(line_ring: shapely.geometry.LinearRing):
    """Generate lines from Linear Ring (exterior of polygon)"""
    coords = list(line_ring.coords)
    for i in range(len(coords) - 1):
        line = shapely.geometry.LineString((coords[i], coords[i + 1]))
        yield line


def find_intersected_lines(
        geom1: shapely.geometry.Polygon,
        geom2: shapely.geometry.Polygon,
):
    """Return list of lines of first geometry which intersects with second geometry

    Args:
        geom1: Polygon
        geom2: Polygon

    Returns:
        result: List[
            (
                line_number: int,
                intersection_length: float,
                line_length: float,
            ),
        ]
    """
    result = []
    for i, line in enumerate(line_generator(geom1.exterior)):
        line_intersection = geom2.intersection(line)
        if not line_intersection.is_empty:
            result.append([i, line_intersection.length, line.length])
    return result


def remove_intersection_iteratively(
        polygon1: shapely.geometry.Polygon,
        polygon2: shapely.geometry.Polygon,
        distance_step: float = 1.,
        max_tries: int = 100,
):
    """Sequentially move all edges ito reduce intersection,
    stops when geometries are not intersects anymore.

    Args:
        polygon1: first geometry
        polygon2: second geometry
        distance_step: maximum step for one iteration
        max_tries: maximum number of iterations

    """
    intersection = polygon1.intersection(polygon2)

    # necessary to make polygons "canonical"
    # the bypass clockwise
    polygon1 = polygon1.buffer(0)
    polygon2 = polygon2.buffer(0)

    def calculate_weights(res1, res2):
        """calculate weights to decide distance for edge shifting"""
        weights1 = [(intersected_line_length / line_length)
                    for _, intersected_line_length, line_length in res1]

        weights2 = [(intersected_line_length / line_length)
                    for _, intersected_line_length, line_length in res2]
        weights = weights1 + weights2
        weights = [w / sum(weights) for w in weights]
        return weights[:len(res1)], weights[len(res1):]

    def move_edges(g, res, weights):
        """move edges of geometry"""
        for i, nw in enumerate(weights):
            distance = nw * distance_step
            line_number = res[i][0]
            g = shift_polygon_edge(g, line_number, distance)
        return g

    n_tries = 0

    while not intersection.is_empty:
        if n_tries > max_tries:
            break

        result_g1 = find_intersected_lines(polygon1, polygon2)
        result_g2 = find_intersected_lines(polygon2, polygon1)

        w1, w2 = calculate_weights(result_g1, result_g2)

        polygon1 = move_edges(polygon1, result_g1, w1)
        polygon2 = move_edges(polygon2, result_g2, w2)

        intersection = polygon1.intersection(polygon2)
        n_tries += 1

    return polygon1, polygon2


def get_polygon(feature: shapely.geometry.base.BaseGeometry) -> shapely.Polygon:
    """Convert feature into polygon"""
    if feature.geom_type == 'MultiPolygon' and len(feature.geoms) == 1:
        polygon = feature.geoms[0]
    elif feature.geom_type == 'Polygon':
        polygon = feature
    else:
        polygon = None
    return polygon


def correct_topology_by_edge_shift(fc: FeatureCollection, distance_step: float = 1.,
                                   correction_threshold: Tuple[float, float] = (0.5, 2.0)):
    """Correct FeatureCollection topology by removing intersections between polygons"""
    if fc.empty:
        return fc
    src_fc = copy.deepcopy(fc)
    for feature_idx in range(len(fc)):
        intersect_idxs = fc.query(fc[feature_idx, 'geometry'])
        if len(intersect_idxs) == 1:
            continue

        for intersect_idx in intersect_idxs:

            # skip same feature
            if intersect_idx == feature_idx:
                continue

            # skip not polygons
            # TODO: support for MultiPolygon
            polygon1 = get_polygon(fc[feature_idx, 'geometry'])
            polygon2 = get_polygon(fc[intersect_idx, 'geometry'])

            if polygon1 is None or polygon2 is None:
                continue

            # if intersection does not exist - skip
            if polygon1.intersection(polygon2).is_empty:
                continue

            try:
                polygon1, polygon2 = remove_intersection_iteratively(polygon1, polygon2, distance_step=distance_step)
                if polygon1.is_valid:
                    fc[feature_idx, 'geometry'] = polygon1
                if polygon2.is_valid:
                    fc[intersect_idx, 'geometry'] = polygon2
            except Exception as e:
                pass
        
    # This code have two reasons:
    # First - there we rebuild rtree index, which is affected by inplace geometry change in previous step
    # Second - sometimes correction preforms not good and affects items too much, in this case we
    # not change items, which area affected more than threshold value
    feature_list = []
    min_threshold, max_threshold = correction_threshold
    for src_f, new_f in zip(src_fc, fc):
        src_area = src_f.geometry[0].area
        new_area = new_f.geometry[0].area
        if src_area == 0 or new_area == 0:
            continue
        area_ratio = src_area / new_area
        if area_ratio > max_threshold or area_ratio < min_threshold:
            feature_list.append(src_f)
        else:
            feature_list.append(new_f)

    return FeatureCollection(feature_list, crs=fc.crs)


def remove_overlapped_features(fc: FeatureCollection,
                               max_area=0.0,
                               area_fraction_threshold=1.0) -> FeatureCollection:
    """
    Removes all the geometries below max_area which have
    more than area_fraction_threshold area intersected with other features
    if max_area == 0, all geometries are processed regardless of area
    """
    # TODO: reimplement faster and simpler with query
    if fc.empty:
        return fc
    new_feats = []
    for feat in fc:
        feat_area = feat.geometry[0].area
        remove = False
        if max_area and feat_area > max_area:
            new_feats.append(feat)
            continue
        # todo: check if the feature remains the same after selection (and we can use != for comparison)
        # we select all the larger intersecting geometries as candidates for dissolve
        candidates = [candidate for candidate in fc[fc.query(feat.geometry[0])]
                      if candidate.geometry[0] != feat.geometry[0]
                      and candidate.geometry[0].area > feat_area]
        for candidate in candidates:
            if feat.geometry[0].intersection(candidate.geometry[0]).area >= area_fraction_threshold*feat_area:
                remove = True
                break
        if not remove:
            new_feats.append(feat)
    return FeatureCollection(new_feats)


def correct_topology_by_subtraction(fc: FeatureCollection,
                                    buffer: float = 0.0,
                                    direction: int = 1) -> FeatureCollection:
    """
    Subtracts smaller polygon from bigger one if they intersect, to remove the intersection
    """
    if fc.empty:
        return fc
    if direction not in {-1, 1}:
        raise ValueError('Parameter direction in correct_topology_by_subtraction '
                         'must be eiter 1 (subtract smaller from bigger) or -1 (subtract bigger from smaller)')
    fc.sort(by='geometry', key=lambda x: x.map(lambda y: y.area), inplace=True, reverse=(direction > 0))
    idxs = fc.query(fc.geometry).T
    for idx in idxs:
        if idx[0] < idx[1]:
            fc[idx[0], 'geometry'] = shapely.make_valid(fc[idx[0], 'geometry'].difference(fc[idx[1], 'geometry']))
    return fc
