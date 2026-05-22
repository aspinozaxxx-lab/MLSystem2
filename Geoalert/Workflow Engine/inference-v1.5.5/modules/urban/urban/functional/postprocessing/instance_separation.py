import numpy as np
import shapely
import shapely.geometry
import shapely.affinity
import shapely.ops
from gpdadapter import FeatureCollection
from sklearn.cluster import KMeans
from .shapely_ext import (
    voronoi_regions_from_coords, 
    extend_line, 
    get_angle_for_line, 
    split_to_lines,
)
from .constants import Shape
from typing import Tuple, List
import geopandas as gpd

__all__ = ["split_instances"]


MIN_ROW_DISTANCE = 6.
MIN_COL_DISTANCE = 5.
CLOSING_BUFFER = 2.1

# ===============================================================================
# GRID SEPARATION HELP FUNCTIONS
# ===============================================================================


def get_rotation_angle(shape: shapely.geometry.Polygon) -> float:
    """Calculate angle to rotate geometry to horizontal (take most l)"""
    lines = sorted(split_to_lines(shape), key=lambda x: x.length)
    longest_line = list(lines)[-1]
    angle = get_angle_for_line(longest_line)
    return angle


def close_shape_if_needed(shape: shapely.geometry.Polygon, buffer: float = 2):
    """Make morphological closing operation for geometry if it is not rectangular, possibly remove small parts"""
    if len(list(shape.exterior.coords)) != 5:
        return shape.buffer(-buffer).buffer(buffer, join_style=2)  # join_style preserve 90 degrees angles
    else:
        return shape


def get_row_cut_y_coordinates(points: List[shapely.geometry.Point], init_n_rows: int = 5) -> List[float]:
    """Separate points by clustering over Y coordinate (rows)
    Return middle points for sorted cluster centers (Y coordinates for row separation)
    """
    ys = [list(p.coords)[0][1] for p in points]

    n_rows = min(init_n_rows, len(ys))
    for _ in range(n_rows):
        res = KMeans(n_clusters=n_rows, n_init=3).fit(np.array(ys)[:, None])
        centers = np.sort(res.cluster_centers_.ravel())
        distances = centers[1:] - centers[:-1]
        if np.all(distances > MIN_ROW_DISTANCE):
            break
        else:
            n_rows -= 1
            
    return centers[:-1] + distances / 2


def get_cols_cut_coords(points: List[shapely.geometry.Point]) -> List[float]:
    """Separate points by X coordinate, take middle between each of two nearest points"""
    xs = [list(p.coords)[0][0] for p in points]
    xs = sorted(xs)
    new_xs = []
    for x in xs:
        if not new_xs:
            new_xs.append(x)
        elif abs(x - new_xs[-1]) > MIN_COL_DISTANCE:
            new_xs.append(x)
        else:
            new_xs[-1] = (x + new_xs[-1]) / 2
    centers = np.array(new_xs)
    distances = centers[1:] - centers[:-1]
    return centers[:-1] + distances / 2


def select_points(
    shapes: shapely.MultiPolygon,
    points: List[shapely.geometry.Point]
    ) -> List[List[shapely.geometry.Point]]:
    """Select for each shape points inside shape"""
    return [[p for p in points if shape.contains(p)] for shape in shapes.geoms]


def cut_shape_to_rows(shape: shapely.geometry.Polygon, y_coords: List[float]) -> shapely.geometry.MultiPolygon:
    """Cut geometry along X axis with provided Y coordinates (make rows)"""
    x_min, _, x_max, _ = shape.bounds
    lines = [shapely.geometry.LineString([[x_min - 1, y], [x_max + 1, y]]) for y in y_coords]
    new_shape = shape
    for line in lines:
        new_shape = shapely.ops.split(new_shape, line)
        new_shape = shapely.geometry.MultiPolygon(new_shape)
        
    if isinstance(new_shape, shapely.geometry.Polygon):
        new_shape = shapely.geometry.MultiPolygon([new_shape])
    
    # flatten multipolygons
    polygons = []
    for s in new_shape.geoms:
        s = close_shape_if_needed(s, buffer=CLOSING_BUFFER)
        if isinstance(s, shapely.geometry.MultiPolygon):
            ps = [p for p in s.geoms]
        else:
            ps = [s]
        polygons.extend(ps)

    new_shape = shapely.geometry.MultiPolygon(polygons)
    return new_shape


def cut_shape_to_cols(shape: shapely.geometry.Polygon, x_coords: List[float]) -> shapely.geometry.MultiPolygon:
    """Cut geometry (one ROW!) along Y axis with provided X coordinates (make columns)"""
    _, y_min, _, y_max = shape.bounds
    lines = [shapely.geometry.LineString([[x, y_min - 1], [x, y_max + 1]]) for x in x_coords]
    new_shape = shape
    for line in lines:
        new_shape = shapely.ops.split(new_shape, line)
        new_shape = shapely.geometry.MultiPolygon(new_shape)
    
    if isinstance(new_shape, shapely.geometry.Polygon):
        new_shape = shapely.geometry.MultiPolygon([new_shape])
    return new_shape


def cut_row_shape_to_cols(shapes: shapely.geometry.MultiPolygon,
                          points: List[List[shapely.geometry.Point]]) -> shapely.geometry.MultiPolygon:
    """Cut geometry (all ROWs one by none!) along Y axis with provided X coordinates (make columns)"""
    new_shapes = []
    for row, row_points in zip(shapes.geoms, points):
        coords = get_cols_cut_coords(row_points)
        shape = cut_shape_to_cols(row, coords)
        new_shapes.extend([s for s in shape.geoms])
    
    return shapely.geometry.MultiPolygon(new_shapes)


def calculate_num_rows_correction(shapes: shapely.geometry.MultiPolygon,
                                  points: List[List[shapely.geometry.Point]]) -> int:
    """Search for rows with high column h/w ratio -> delete this rows"""
    if len(shapes.geoms) == 1:
        return 0
    
    reduce_rows = 0
    for row, row_points in zip(shapes.geoms, points):
        x_min, y_min, x_max, y_max = row.bounds
        h, w = abs(y_max - y_min), abs(x_max - x_min)
        w = w / max(1, len(row_points))  # num points is a number of columns in row
        max_dim, min_dim = max(h, w), min(h, w)
        ratio = max_dim / min_dim

        if ratio > 5.:
            reduce_rows += 1
        
    return reduce_rows


def cut_shape_with_points(
    shape: shapely.geometry.Polygon, 
    points: List[shapely.geometry.Point],
    ) -> List[shapely.geometry.Polygon]:
    """Split geometry by rectangular grid according to provided points"""

    # rotate to horizontal
    rotation_point = shape.centroid
    angle = get_rotation_angle(shape)
    shape = shapely.affinity.rotate(shape, - angle, origin=rotation_point)
    points = [shapely.affinity.rotate(p, - angle, origin=rotation_point) for p in points]
    
    # perform cut operations
    init_n_rows = int(len(points) ** 0.5) + 1
    coords = get_row_cut_y_coordinates(points, init_n_rows=init_n_rows)
    row_cut_shapes = cut_shape_to_rows(shape, coords)
    points_for_rows = select_points(row_cut_shapes, points)

    # check if we need to reduce number of rows and recalculate if needed
    n_rows = len(row_cut_shapes.geoms)
    reduce_rows = calculate_num_rows_correction(row_cut_shapes, points_for_rows)
    if reduce_rows:
        n_rows = max(1, n_rows - reduce_rows)
        coords = get_row_cut_y_coordinates(points, init_n_rows=n_rows)
        row_cut_shapes = cut_shape_to_rows(shape, coords)
        points_for_rows = select_points(row_cut_shapes, points)

    cut_shape = cut_row_shape_to_cols(row_cut_shapes, points_for_rows)
    
    # rotate back
    cut_shape = shapely.affinity.rotate(cut_shape, angle, origin=rotation_point)
    polygons = [shape for shape in cut_shape.geoms]
    return polygons


# ===============================================================================
# CORNER CASE - 2 POINTS
# ===============================================================================

def cut_polygon_with_two_points(
    polygon: shapely.geometry.Polygon, 
    points: Tuple[shapely.geometry.Point, shapely.geometry.Point],
) -> shapely.geometry.GeometryCollection:
    """Cut polygon with line perpendicular to two provided points"""

    p1, p2 = points
    extension = polygon.length
    
    p1, p2 = extend_line(p1, p2, extension)
    
    extended_line = shapely.geometry.LineString(coordinates=[p1, p2])
    rotated_line = shapely.affinity.rotate(extended_line, 90, origin="centroid")
    splitted_polygons = shapely.ops.split(polygon, rotated_line)
    
    return splitted_polygons


# ===============================================================================
# CORNER CASE - 2 POINTS
# ===============================================================================

def split_instances(
    fc: FeatureCollection,
    markers_fc: FeatureCollection,
    max_average_area: float = 300.,
    flat_multipolygons=True, min_area=0.
) -> FeatureCollection:

    new_features = FeatureCollection()
    shape_types = (Shape.RECTANGLE, Shape.GRID_SNAP, Shape.LSHAPE)
    # TODO: refactor/optimize
    for feature in fc:

        point_features: gpd.GeoSeries = markers_fc[markers_fc.query(feature.geometry[0])].geometry.centroid

        num_points = len(point_features)
        polygon_area = feature.geometry[0].area
        average_area = polygon_area / max(num_points, 1)

        if average_area > max_average_area:
            new_features.append(feature)
            continue

        if feature.properties.get("shape_type") in shape_types and len(point_features) > 1:
            points = point_features.tolist()
            splitted_polygons = cut_shape_with_points(feature.geometry[0], points)
            features = FeatureCollection({'geometry': splitted_polygons}, crs=fc.crs)
            for k, v in feature.properties.items():
                features[:, k] = v
            new_features.append(features)

        elif len(point_features) == 2:
            points = point_features.geometry
            splitted_polygons = cut_polygon_with_two_points(feature.geometry[0], points)
            features = FeatureCollection({'geometry': splitted_polygons}, crs=fc.crs)
            for k, v in feature.properties.items():
                features[:, k] = v
            new_features.append(features)

        # voronoi separation
        elif len(point_features) > 2:
            points = [list(p.coords) for p in point_features]
            points = np.array(points).squeeze(1)
            polygon = feature.geometry[0]
            splitted_polygons, _, _ = voronoi_regions_from_coords(
                points, polygon, farpoints_max_extend_factor=1000,
            )

            features = FeatureCollection()
            for geom in splitted_polygons:
                # split the multipolygons if they appear in results, also deleting too small areas
                if flat_multipolygons and isinstance(geom, shapely.geometry.MultiPolygon):
                    features.append(FeatureCollection({'geometry': [p for p in geom.geoms if p.area > min_area]},
                                                      crs=feature.crs))
                else:
                    features.append(FeatureCollection({'geometry': [geom]}, crs=feature.crs))
            for k, v in feature.properties.items():
                features[:, k] = v
            new_features.append(features)

        # without any separation
        else:
            new_features.append(feature)
    new_features.dropna(inplace=True)
    return new_features
