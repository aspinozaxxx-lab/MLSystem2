import pandas as pd
import pyproj
import numpy as np
from tqdm import tqdm
from shapely.geometry import LineString, Polygon, Point
from gpdadapter import FeatureCollection, concatenate
from typing import Callable, Union, Tuple, List, Optional, Sequence, Dict, Set
from shapely.geometry.base import BaseGeometry

# ------------------------------------------------------------------------------------------------
#                               Operations with LineStings
# ------------------------------------------------------------------------------------------------


def index_points(geoms: Sequence[BaseGeometry],
                 endnodes_only: bool = False) -> Dict[Tuple[float, float], Set[int]]:
    """
    Create a dictionary with keys - all points from fc, values - corresponding lines indexes
    
    Args:
        geoms: Sequence[BaseGeometry]
        endnodes_only (bool) - if True, only endpoints  will be indexed
    
    Returns:
        points_index (dict {point:set(line_indexes)})
    """

    points_index = dict()
    
    for i, coords in enumerate(geoms):
        
        if isinstance(coords, BaseGeometry):
            coords = list(coords.coords)
            
        if endnodes_only:
            coords = [coords[0], coords[-1]]
        
        for node in coords:
            if node not in points_index.keys():
                points_index[node] = set()
            
            points_index[node].add(i)
    
    return points_index


def find_double_points(geoms: Sequence[LineString],
                       num_lines: int = 2,
                       endnodes_only: bool = True) -> Dict[Tuple[float, float], List[LineString]]:
    
    """
    Find all points from fc that appears in 'num_lines' lines
    
    Args:
        geoms: Sequence[BaseGeometry] with LineString geometry
        num_lines (int): number of point appearance
        endnodes_only (bool): if True - only endnodes will be considered
    
    Returns:
        double_p (dict): {point(tuple): [lines]}
    """
    
    points = index_points(geoms, endnodes_only)
    double_p = {k: [geoms[i] for i in v] for k, v in points.items() if len(v) > num_lines-1}
    return double_p


def get_angle(coords: Union[list, LineString]) -> float:
    
    """
    Calculate angle (slope) from line coords
    
    Args:
        coords (Union[list, Feature, LineString]): coordinates
    
    Returns:
        angle (float): angle in degrees
    """
    
    if isinstance(coords, LineString):
        coords = list(coords.coords)
        
    angle = np.arctan((coords[-1][1] - coords[0][1]) /
                      (coords[-1][0] - coords[0][0] + 1e-7))
    
    angle = angle * 180 / 3.14
    angle = angle + 180 if angle < 0 else angle
    
    return angle


# ------------------------------------------------------------------------------------------------
#                                         Filters
# ------------------------------------------------------------------------------------------------


def remove_duplicates(fc: FeatureCollection) -> FeatureCollection:  # TODO: not used
    """
    Removes duplicate lines from fc
    Args:
        fc: FeatureCollection with LineString geometry
    
    Returns:
        new_fc: FeatureCollection with removed duplicates
    """
    return fc.drop_duplicates('geometry')


def filter_by_dist(point1, min_dist: float = 5):
    
    """
    Returns True if distance between two points < min_dist
    """
    
    def filter_(point2):
        return np.linalg.norm([point1[0] - point2[0], point1[1] - point2[1]]) < min_dist
    
    return filter_


def filter_short_lines(fc: FeatureCollection, min_length=30, min_connections=2, endnodes_only=False) -> Callable:
    
    """
    Filter short lines, this also checks if line connected to other lines
    and does not remove it, if number of connections >= min_connections
    
    Args:
        fc:  FeatureCollection
        min_length (float) - min length of line to be filtered
        min_connections (int) - lines, connected to >= lines will not be filtered
        endnodes_only (bool) - if True, connections will be checked only at line endnodes
    
    Returns:
        filter_(feature)
    """
    
    points_idx = index_points(fc.geometry, endnodes_only)
    
    def filter_(row: pd.Series):
        feature = row.geometry
        if feature.length < min_length:
            
            coords = feature.coords
            if endnodes_only:
                coords = (coords[0], coords[-1])
            
            # count number of connections
            connections = 0
            for node in coords:
                connections += (len(points_idx[node]) > 1)
                if connections >= min_connections:
                    return True
                
            return False
        return True
    
    return filter_


# ------------------------------------------------------------------------------------------------
#                                 LineStrings snapping
# ------------------------------------------------------------------------------------------------


def snap_2_lines(point: Tuple[float, float], lines: Tuple[LineString, LineString]):
    
    """
    Snap two lines at given point
    
    Args:
        point (tuple): point at which snap lines
        lines (tuple(list, list)): lines for snapping
    
    Returns:
        new_line (LineString): snapped line
    """
    
    line_1, line_2 = list(lines[0].coords), list(lines[1].coords)
    
    # orient lines and remove same point
    right = line_1[1:] if line_1.index(point) == 0 else line_1[::-1][1:]
    left = line_2[::-1] if line_2.index(point) == 0 else line_2
    
    # concatenate coordinates
    new_line = np.concatenate((left, right))
    
    return LineString(new_line)


def create_new_line(point, lines, min_angle: float = 30, remain_first: bool = False):
    
    """
    Snap lines (if possible) at given point
    It checks min angle difference between given lines, if it < min_angle - creates new line
    
    Args:
        point (tuple): point at which snap lines
        lines (tuple(list, ...)): lines for snapping
        min_angle (float): min difference in lines angles to snap
        remain_first (bool): if True - first line from lines will remain in snapped lines
                                
    Returns:
        if lines snapped:
            new_line (LineString): snapped line
            to_drop (list): lines to drop from fc
        else:
            None
            []
    """
        
    if remain_first:
        
        # if fist line remains, check angles between it and other lines
        base_angle = get_angle(lines[0])
        angles_diff = [abs(base_angle - get_angle(line)) for line in lines[1:]]
        
        # indexes of line with min angle diff and first line
        merge_idx = [0, np.argmin(angles_diff) + 1]
        
    else:
        
        # check angles diff between all lines
        angles = list(map(get_angle, lines))
        rng = range(len(angles))
        angles_diff = np.array([[abs(angles[i]-angles[j]) for i in rng if i != j] for j in rng])
        
        # indexes of two lines with min angles diff
        merge_idx = np.where(angles_diff == angles_diff.min())[0][:2]
    
    if np.min(angles_diff) < min_angle:
        
        # lines for snapping
        to_snap = [lines[i] for i in merge_idx]
        
        # create new line
        new_line = snap_2_lines(point, to_snap)
        
        return new_line, to_snap
    
    return None, []


def snap_lines(fc: FeatureCollection, min_angle: float = 40, verbose: bool = True) -> FeatureCollection:
    
    """
    Snap lines with same endnodes (lines that were broken into pieces after skeletonization)
    
    Args:
        fc: FeatureCollection
        min_angle (float): min difference in lines angles (degrees) to snap
        verbose (bool): verbose
    Returns:
        new_fc: FeatureCollection with snapped lines
    """
    
    # find endnodes that appears in more than two lines
    double_p = find_double_points(fc.geometry)
    new_features, drop_features = [], []
    
    # iterate over double points
    for point, lines in tqdm(double_p.items(), disable=(not verbose)):
        
        # check if lines were already snapped
        lines = [line for line in lines if line not in drop_features]
        
        # check that more than two lines for snapping
        if len(lines) < 2:
            continue
        
        # create new line from candidates
        new_line, to_drop = create_new_line(point, lines, min_angle)
        drop_features.extend(to_drop)
        
        if new_line is None:
            continue
        
        # check endnodes of new line have intersection with other lines
        endnodes = [new_line.coords[0], new_line.coords[-1]]
        while endnodes[0] in double_p.keys() or endnodes[1] in double_p.keys():
            
            # iterate over new line endnodes
            for i, p in enumerate(endnodes):
                try:
                    
                    # try to find new lines for snapping
                    snapping_lines = [list(new_line.coords)]
                    snapping_lines.extend([line for line in double_p[p] if line not in drop_features])
                    assert len(snapping_lines) >= 2
                    
                    # snap lines (if possible) and create new line with new endnodes
                    candidate, to_drop = create_new_line(p, snapping_lines, min_angle, remain_first=True)
                    drop_features.extend(to_drop)
                        
                    new_line = LineString(candidate.coords)
                    endnodes = [new_line.coords[0], new_line.coords[-1]]
                            
                except (KeyError, AttributeError, AssertionError):
                    # if new line cannot be snapped
                    endnodes[i] = None

        new_features.append(new_line)
    
    # create FeatureCollection from snapped lines
    new_fc = FeatureCollection(new_features, crs=fc.crs)
    drop_fc = FeatureCollection([f for f in fc.geometry if f not in drop_features], crs=fc.crs)
    new_fc.append(drop_fc)
    new_fc.drop_duplicates('geometry', inplace=True)
    return new_fc


# ------------------------------------------------------------------------------------------------
#                                 Crossroads refinement
# ------------------------------------------------------------------------------------------------


def parse_crossroads_centers(fc: FeatureCollection,
                             num_lines: int = 3,
                             snap_dist: float = 10,
                             endnodes_only: bool = True) -> Tuple[List[Tuple[float, float]], List[List[LineString]]]:
    
    """
    Generate center of crossroads from given fc.
    We assume that some crossroads are broken into >-< (two points, in each
    two lines connected and some line at center that should be removed).
    We parse all points in which 'num_lines' lines ends, then check if
    distance between some of these points < 'snap_dist'. Lines, that ends in
    such points we define as broken crossroad and recalculate center for them.
    
    Args:
        fc: FeatureCollection with LineString geometry
        num_lines (int): number of lines that should end in one point
        snap_dist (float): distance between two points to snap them into crossroad
        endnodes_only (bool) - if True, connections will be checked only at line endnodes
        
    Returns:
        centers (list) - list of new crossroads
        lines_to_move (list) - lines that should be moved to each center
    """
    
    centers, lines_to_move = [], []
    
    # lines with same endnodes
    dp_lines = find_double_points(fc.geometry, num_lines, endnodes_only)

    for point in dp_lines.keys():
        
        # find close intersecting endnodes to given point
        check = filter_by_dist(point, snap_dist)
        intersecting_and_close = list(filter(check, dp_lines.keys()))

        if len(intersecting_and_close) > 1:
            
            # calculate center of crossroad
            points = np.array(intersecting_and_close)
            center: Tuple[float, float] = (points[:, 0].sum() / points.shape[0], points[:, 1].sum() / points.shape[0])
                    
            if center not in centers:
                centers.append(center)
                
                lines = []
                for p in intersecting_and_close:
                    lines.extend(dp_lines[p])
                lines_to_move.append(lines)
                
    return centers, lines_to_move


def snap_crossroads_centers(centers: List[Tuple[float, float]], lines_to_move: List[List[LineString]],
                            snap_dist: float = 10) -> Tuple[List[Tuple[float, float]], List[List[LineString]]]:
    
    """
    Recalculate center for crossroads that close to each other.
    It checks distance between all centers and if < 'snap_dist', recalculate
    new center for them.
    
    Args:
        centers (list[tuple, ...]): centers of crossroads
        lines_to_move (list): coords of lines, that should be moved to each center
        snap_dist (float): distance between centers to snap them into new one
    
    Returns:
        new_centers (list) - list of new crossroads
        new_lines_to_move (list) - lines that should be moved to each center
    """
    
    new_centers, new_lines_to_move = [], []
    for i in range(len(centers)):

        point = centers[i]
        check = filter_by_dist(point, snap_dist)
        intersecting_and_close = list(map(check, centers))

        if len(intersecting_and_close) == 1:
            
            new_center = tuple(list(centers[i]))
            new_lines = lines_to_move[i].copy()

        else:
            
            nonzero = np.where(np.array(intersecting_and_close))[0]
            points = [centers[i] for i in nonzero]
            points = np.array(points)
            new_center = (points[:, 0].sum() / points.shape[0], points[:, 1].sum() / points.shape[0])

            new_lines = []
            for j in nonzero:
                for line in lines_to_move[j]:
                    if line not in new_lines:
                        new_lines.append(line)

        if new_center not in new_centers:
            new_centers.append(new_center)
            new_lines_to_move.append(new_lines)
    
    return new_centers, new_lines_to_move


def find_closest_node(point: Tuple[float, float], line: Union[LineString, Sequence[Tuple]]):
    
    """
    Find the closest node index line to given point.
    It checks only first and last nodes.
    
    Args:
        point (tuple): point coords
        line (list[tuple, ...]): line cords
        
    Returns:
        idx (int): index of the closest node
    """
    
    norms = []
    if isinstance(line, LineString):
        line = list(line.coords)
    for i in (0, -1):
        norms.append(np.linalg.norm((abs(point[0] - line[i][0]),
                                     abs(point[1] - line[i][1]))))

    idx = -np.argmin(norms)
    return idx


def move_to_center(center: Tuple[float, float], lines: Sequence[Union[LineString, Sequence[Tuple]]]):
    
    """
    Move lines to center.
    Removes one point from each line and replace it with center
    
    Args:
        center (tuple): coords of center
        lines ([list, ...]): lines to move
        
    Returns:
        new_lines ([list, ...]): coords of new lines
    """
    
    new_lines = []
    for line in lines:
        
        # find the closest point indx to center
        closest = find_closest_node(center, line)

        if isinstance(line, LineString):
            line = list(line.coords)
        # replace 
        new_line = line.copy()
        new_line[closest] = tuple(list(center))
        new_lines.append(new_line)
            
    return new_lines


def filter_moved_lines(new_lines: list, drop_lines: list, fc: FeatureCollection,
                       min_length: float = 10,
                       min_connections: int = 2, endnodes_only: bool = True) -> FeatureCollection:
    """
    After fixing crossroads some short useless lines will appear. They cannot
    be removed during crossroads snapping, since they may represent interest
    for the global road network. There are two aspects for filtering - line
    length and number of another lines connected to it. Only lines from
    'new_lines' can be removed, all other features from fc will be untouched
    Args:
        new_lines (list): moved lines after crossroads snapping
        drop_lines (list): lines that should be dropped from fc (old crossroads)
        fc: original FeatureCollection
        min_length (float) - min length of line to be filtered
        min_connections (int) - lines, with number of connected (to other lines)
                                   nodes >= min_connections will not be filtered
        endnodes_only (bool) - if True, connections will be checked only at line endnodes
    Returns:
        new_fc: filtered FeatureCollection with fixed crossroads
    """

    # only fixed crossroads
    new_fc = FeatureCollection([LineString(line) for line in new_lines], crs=fc.crs)
    # aware of duplicates
    new_fc.drop_duplicates('geometry', inplace=True)

    # features from original fc except fixed crossroads
    old_fc = FeatureCollection([feature for feature in fc.geometry if feature not in drop_lines],
                               crs=fc.crs)

    # concatenate new and old features to obtain road network
    to_filter = concatenate((new_fc, old_fc))
    
    # filter only new features
    new_fc = new_fc.filter(filter_short_lines(to_filter, min_length=min_length, min_connections=min_connections,
                           endnodes_only=endnodes_only))
    # concatenate new (filtered) and old features to obtain result
    new_fc = concatenate((new_fc, old_fc))
    
    return new_fc


def snap_crossroads(fc: FeatureCollection, num_lines: int = 3, snap_dist: float = 5, merge_dist: float = 10,
                    min_length: float = 10, min_connections: int = 2,
                    endnodes_only: bool = True, verbose: bool = True) -> FeatureCollection:
    
    """
    Fix crossroads that were broken into several parts after skeletonization.
    We can detect such crossroads as groups of lines that ends at same point,
    usually there are two points at each of which two (or more) lines ends.
    There is also another short line between this two points. This function
    will remove this line and snap all other lines to its center.
    Basically, it replace >-< with ><.
    
    Args:
        fc: FeatureCollection with LineString geometry
        num_lines (int): number of lines ending at one point for crossroad search
        snap_dist (float): max distance between points of broken crossroad
        merge_dist (float): crossroads at this distance will merge into one
        min_length (float): min line length (for filtering)
        min_connections (int): min lines connected to given (for filtering)
        endnodes_only (bool): search crossroads only at lines endnodes
        verbose (bool): verbose
    Returns:
        new_fc: same as original fc, but with fixed crossroads
    """
    
    # parse crossroads coordinates
    centers, lines_to_move = parse_crossroads_centers(fc, num_lines, snap_dist, endnodes_only)
    
    # snap close crossroads
    centers, lines_to_move = snap_crossroads_centers(centers, lines_to_move, merge_dist)

    num_crossroads = len(centers)
    new_lines, drop_lines = [], []

    i = 0
    with tqdm(total=num_crossroads, disable=(not verbose)) as pbar:
        while i < num_crossroads:
            
            # move lines to center
            center, lines = centers[i], lines_to_move[i]
            
            # create new lines, moved lines to drop
            new = move_to_center(center, lines)
            drop = lines.copy()
            
            i += 1
            #if i == num_crossroads:   WTF???
            #    continue
            
            # if moved line appears later we replace it with new line
            for j, next_to_move in enumerate(lines_to_move[i:]):
                for n, line in enumerate(next_to_move):
                    if line in drop:
                        idx = drop.index(line)
                        lines_to_move[i+j][n] = new[idx]
                        drop_lines.append(line)
                        new.pop(drop.index(line))
                        drop.remove(line)
            
            # aware of duplicates
            new_lines.extend(new)
            drop_lines.extend(drop)
            pbar.update()
    
    # filter result
    new_fc = filter_moved_lines(new_lines, drop_lines, fc, min_length, min_connections, True)
    
    return new_fc


# ------------------------------------------------------------------------------------------------
#                                 Gaps Closing (lines merging)
# ------------------------------------------------------------------------------------------------


def draw_perpendicular(point: tuple, slope: float, length: float = 50) -> tuple:
    """
    Draw perpendicular at given point with given slope.
    Args:
        point (tuple): point coords
        slope (float): slope of line to be perpendicular with
        length (float): length of perpendicular
    
    Returns:
        (list, list) - coords of perpendicular
    """
    dy = np.sqrt((length/2)**2 / (slope**2+1))
    dx = -slope * dy
    p1 = [point[0] + dx, point[1] + dy]
    p2 = [point[0] - dx, point[1] - dy]
    return p1, p2


def draw_buffer(line: LineString, perpendicular: tuple, length: float, figure: str = 'rectangle') -> Polygon:
    """
    Draw buffer at given line.
    Line here is base for future buffer. We draw perpendiculars at each it
    endnode (they will be parallel to the road) and then build either rectangle
    or triangle based on its points.
    Args:
        line (LineString): base of future buffer
        perpendicular:
        length (float): buffer length
        figure (str): either 'rectangle' or 'triangle'
    Returns:
        poly (Polygon): buffer
    """
    
    # draw perpendiculars (parallels to road) for each endnode
    # length is doubled since we draw in two direction
    p1 = perpendicular
    slope = (p1[1][1] - p1[0][1] + 1e-5) / (p1[1][0] - p1[0][0] + 1e-5)
    p3 = draw_perpendicular(p1[0], slope, length*2)
    p4 = draw_perpendicular(p1[1], slope, length*2)
    
    if figure == 'rectangle':
        
        # 4 points are form rectangle - both p1 endnodes 
        # and one of each p3 and p4 endnodes
        poly = Polygon((p1[1], p1[0], p3[0], p4[0], p1[1]))
        
        # if polygon intersects original line
        # we need to choose other p3 and p4 endnodes
        if poly.buffer(-0.001).intersects(line):
            poly = Polygon((p1[1], p1[0], p3[-1], p4[-1], p1[1]))

    elif figure == 'triangle':
        
        # vertex of triangle is the mid-point of p3 and p4 endnodes
        vertex = ((p3[0][0] + p4[0][0])/2, (p3[0][1] + p4[0][1])/2)
        poly = Polygon((p1[1], p1[0], vertex, p1[1]))
        
        if poly.buffer(-0.001).intersects(line):
            vertex = ((p3[-1][0] + p4[-1][0])/2, (p3[-1][1] + p4[-1][1])/2)
            poly = Polygon((p1[1], p1[0], vertex, p1[1]))

    else:
        raise ValueError(f'figure must be either rectangle or triangle, got {figure}')
    return poly


def create_buffer(line: LineString, length: float = 50, width: float = 10, figure: str = 'rectangle') -> list:
    """
    Create buffer at both line endnodes.
    Buffer itself is either triangle or rectangle with slope as corresponding
    line tail.
    Args:
        line (LineString): line for which create buffer
        length (float): buffer length in crs units
        width (float): buffer width in crs units
        figure (str): either 'rectangle' or 'triangle'
    Returns:
        buffer ([Polygons]): list of two buffers for each endnode

    """
    
    coords = np.array(line.coords)
    
    # draw perpendicular at each endnode (base line for each buffer)
    slope = (coords[0][1] - coords[1][1] + 1e-5) / (coords[0][0] - coords[1][0] + 1e-5)
    perpendicular_1 = draw_perpendicular(coords[0], slope, length=width)

    slope = (coords[-1][1] - coords[-2][1] + 1e-5) / (coords[-1][0] - coords[-2][0] + 1e-5)
    perpendicular_2 = draw_perpendicular(coords[-1], slope, length=width)
        
    buffer = []
    for perpendicular in [perpendicular_1, perpendicular_2]:
        
        # draw polygon and revert it if it intersects original line
        poly = draw_buffer(line, perpendicular, length, figure)

        buffer.append(poly)
    
    return buffer


def check_intersection(geometry, mode='endnodes'):
    
    """
    Check intersection
    
    Args:
        geometry:  shapely.geometry or Feature to check intersection with
        mode (str) - how to check intersection:
                     if 'all' - simple intersection,
                     if 'tails' - check only tails of broken line,
                     if 'endnodes' - check only line endnodes.
    Returns:
        check (function)
    """
    
    if mode not in ['all', 'tails', 'endnodes']:
        raise ValueError("'mode' should be one of ['all', 'tails', 'endnodes']")
    
    def check(linestring):
        
        """
        Args:
            linestring - LineString or Feature with LineString geom, can be
                                                   Polygon if mode is 'all'
        Returns:
            True if geometry intersects linestring
        """
        
        if mode == 'all':
            return linestring.intersects(geometry)
        
        elif mode == 'tails':
            
            coords = list(linestring.coords)
            tail_1 = LineString(coords[0:2])
            tail_2 = LineString(coords[::-1][0:2])
            return tail_1.intersects(geometry) or tail_2.intersects(geometry)
        
        elif mode == 'endnodes':
            
            coords = list(linestring.coords)
            point_1 = Point(coords[0])
            point_2 = Point(coords[-1])
            return point_1.within(geometry) or point_2.within(geometry)
        
    return check


def check_if_link_intersects_fc(fc: FeatureCollection, max_intersections: int = 2) -> Callable:
    
    """
    Check if line intersects any feature from fc.
    It could appear that new merged line will intersect other lines from fc,
    that will violate the integrity of the road network, so it is better
    not to merge such lines.
    
    Args:
        fc: FeatureCollection to check for intersections
        max_intersections (int): max intersections, default in 2, since link line connects two lines
    
    """
        
    def filter_(link):
        
        check = check_intersection(LineString(link), 'all')
        is_intersects = list(fc.geometry.map(check))
        
        return sum(is_intersects) <= max_intersections
    
    return filter_


def filter_by_link_angle(original_ls: LineString,
                         candidates: List[LineString],
                         links: List[LineString],
                         max_angle: float = 10) -> Optional[List]:
    
    """
    Calculate slope of link line and select most appropriate
    candidate for merging.
    
    Args:
        original_ls (LineString): line to merge with
        candidates ([LineStrings]): candidates for merging
        links ([LineStrings]): link lines for each candidate
        max_angle(float): max diff in slope of link and merging lines
    
    Returns:
        if candidate exists:
            best (list): coords of most appropriate line for merging
        else:
            None
    """
    
    # mean angle of original line and candidates
    base_angles = [(get_angle(original_ls) + get_angle(candidate)) / 2 for candidate in candidates]
    
    # angles of link lines
    link_angles = list(map(get_angle, links))
    
    # difference between link angle and base angle
    angles_diff = [abs(base - link) for base, link in zip(base_angles, link_angles)]
    
    # select best (if possible)
    idx = np.where(np.array(angles_diff) < max_angle)[0]
    sorted_idx = np.argsort(idx)
    
    best = candidates[idx[sorted_idx[0]]] if len(idx) > 0 else None
    
    return best


def check_tails_angle(original_ls: List, angle_diff: float = 5) -> Callable:
    
    """
    Check slope difference of original line and candidate for merging.
    Both lines could be broken into several segments, we check angles
    only at the last segments (tails).
    
    Args:
        original_ls (LineString): original line to merge with
        angle_diff (float): max diff in tails slope in degrees
    
    Returns:
        check (function)
    """
    
    def check(candidate):
        
        """
        Args:
            candidate (LineString): candidate for merging
        
        Returns:
            True if candidate and original lines slope < angle_diff,
                                                     otherwise False
        """
        
        original_coords = np.array(original_ls)
        candidate_coords = np.array(candidate.coords)
        
        # find closes points from lines endnodes by minimal norm
        norms = [np.linalg.norm(abs(original_coords[(0, -1), :][i] - candidate_coords[(0, -1), :][j]))
                 for i in range(2) for j in range(2)]

        min_norm = np.argmin(norms)
        i_min, j_min = np.unravel_index(min_norm, [2, 2])
        
        # orient lines to start from the closest point
        original_coords = original_coords if i_min == 0 else original_coords[::-1]
        candidate_coords = candidate_coords if j_min == 0 else candidate_coords[::-1]
        
        # calculate tails slopes
        base_angle = get_angle(original_coords[:2])
        proposed_angle = get_angle(candidate_coords[:2])

        return abs(base_angle - proposed_angle) < angle_diff
    return check


def merge_2_ls(line_1, line_2, return_link=False):
    
    """
    Merge two lines into one.
    
    Args:
        line_1 (LineString): line to merge
        line_2 (LineString): line to merge
        return_link (bool): if True will return only link line
    
    Returns:
        new_ls (LineString): merged line if return_link is False,
                                        otherwise only link line

    """
    
    # coordinates
    line_1 = np.array(line_1.coords)
    line_2 = np.array(line_2.coords)
        
    # find points with min distance
    norms = [np.linalg.norm(abs(line_1[(0, -1), :][i] - line_2[(0, -1), :][j]))
             for i in range(2) for j in range(2)]
    
    min_norm = np.argmin(norms)
    i_min, j_min = np.unravel_index(min_norm, [2, 2])
        
    # orient lines
    left = line_1[::-1] if i_min == 0 else line_1
    right = line_2 if j_min == 0 else line_2[::-1]
    
    # merge into one line
    if return_link:
        new_ls = (left[-1], right[0])
    else:
        new_ls = np.concatenate([left, right])
        
    return LineString(new_ls)


def find_ls_to_merge(feature: LineString, fc: FeatureCollection,
                     fc_buff: FeatureCollection, angle: float = 5,
                     buff_length: float = 150, buff_width: float = 5,
                     buff_type: str = 'rectangle', intersection_mode: str = 'endnodes') -> Optional[LineString]:
    
    """
    Search for best candidate for merging.
    Selection of candidate is done in several steps:
    1) Find lines that intersects buffer of original line and have similar
    slope at tails; 2) New line that will connect merging lines must not cross
    other lines from fc and have similar slope.
    
    Args:
        feature (LineString): line for which to look for candidates
        fc: FeatureCollection from which to look for candidates
        fc_buff: bufferized fc to look for intersections
        angle (float): max diff in lines slope
        buff_length (float): buffer length in crs units
        buff_width (float): buffer width in crs units
        buff_type (str): buffer type - either rectangle or triangle
        intersection_mode (str): how check intersection (with buffer)
    
    Returns:
        if candidate found:
            ls_to_merge (LineString): appropriate candidate for merging
        else:
            None

    """
    
    # create buffer from both endnodes
    buffer = create_buffer(feature, buff_length, buff_width, buff_type)
    ls_to_merge = set()
    
    # we iterate over buffer and line tails
    for buff, line in zip(buffer, [list(feature.coords)[0:2], list(feature.coords)[::-1][0:2]]):
        
        # find features from fc that intersects buffer of tail
        find_intersecting = check_intersection(buff, intersection_mode)
        proposed_features = list(filter(find_intersecting, fc.geometry))
        
        # select candidates with similar slope
        check_angle = check_tails_angle(line, angle)
        proposed_features = list(filter(check_angle, proposed_features))
        
        ls_to_merge.update(proposed_features)
    
    # remove original feature from list
    ls_to_merge = [f for f in ls_to_merge if f != feature]
    
    # create link lines for further filtering
    link_lines = [merge_2_ls(feature, candidate, True) for candidate in ls_to_merge]
    
    # select candidates for which link lines does not intersect lines from fc_buff
    filter_by_link_intersection = check_if_link_intersects_fc(fc_buff)
    filter_idxs = list(map(filter_by_link_intersection, link_lines))
    filter_idxs = np.where(filter_idxs)[0]
    
    ls_to_merge = [ls_to_merge[i] for i in filter_idxs]
    link_lines = [link_lines[i] for i in filter_idxs]
    
    # select candidates for which slope of link line similar
    # if there are several candidates remains - choose best
    ls_to_merge = filter_by_link_angle(feature, ls_to_merge, link_lines, angle)
    
    if ls_to_merge is not None:
        ls_to_merge = LineString(ls_to_merge)
    
    return ls_to_merge


def close_gaps(
    fc: FeatureCollection,
    angle: float = 10,
    buff_length: float = 150,
    buff_width: float = 12,
    buff_type: str = 'rectangle',
    intersection_mode: str = 'endnodes',
    simplify_param: float = 2,
    intersect_buff: float = 3,
    verbose: bool = True
) -> FeatureCollection:
    
    """
    Close small gaps in road network.
    We create two buffers (for both endnodes) for each line and check for other
    lines endnodes intersects this buffer and have similar slope.
    
    Args:
        fc: FeatureCollection with LineString geometry
        angle (float): max difference in lines angles for merging in degrees
        buff_length (float): buffer length in crs units
        buff_width (float): buffer width in crs units
        intersection_mode (str): intersection_mode
        buff_type (str): buffer type - either rectangle or triangle
        simplify_param (float): tolerance param for merged line simplification
        intersect_buff (float): buffer param for fc to avoid intersecting and
                                              duplicating lines after merging
        verbose (bool): verbose
    
    Returns:
        new_fc: FeatureCollection with closed gaps
    """
    
    new_fc = list()
    
    # bufferize fc one time
    fc_buff = fc.buffer(intersect_buff)
    merged_ls = []
    
    for feature in tqdm(fc.geometry, disable=(not verbose)):
        feature = LineString(feature.coords)
        simplify = False
        
        if feature in merged_ls:
            continue
        
        # find (best) line that intersects original one
        to_merge = find_ls_to_merge(feature, fc, fc_buff, angle, buff_length,
                                    buff_width, buff_type, intersection_mode)
        
        # skip if line was already merged
        if to_merge in merged_ls:
            new_fc.append(feature)
            merged_ls.append(feature)
            continue
        
        merged_ls.extend([feature, to_merge])
        
        while to_merge is not None:
            
            # merge lines
            feature = merge_2_ls(feature, to_merge)
            merged_ls.append(feature)
            
            # merged line could be broken, so we will simplify
            # it after merge all possible line
            simplify = True
            
            # find (best) line that intersects merged one
            to_merge = find_ls_to_merge(feature, fc, fc_buff, angle, buff_length,
                                        buff_width, buff_type, intersection_mode)
            
            # break if candidate is already merged
            if to_merge in merged_ls:
                break

            merged_ls.append(to_merge)
        
        # simplify after merging
        if simplify and simplify_param > 0:
            feature = feature.simplify(simplify_param)
            simplify = False

        new_fc.append(feature)

    return FeatureCollection({'geometry': new_fc}, crs=fc.crs)


# ------------------------------------------------------------------------------------------------
#                                           Other
# ------------------------------------------------------------------------------------------------


LATLON_CRS = 'EPSG:4326'


def meters_to_degrees(meters, crs):
    
    """
    Convert utm meters to degrees
    
    Args:
        meters (float): distance in crs units
        crs: projection
    
    Returns:
        degrees
    """
    
    x1, y1 = 0, 0
    x2, y2 = meters, 0
    transformer = pyproj.Transformer.from_crs(crs, LATLON_CRS)
    lat1, lon1 = transformer.transform(x1, y1)
    lat2, lon2 = transformer.transform(x2, y2)
    
    degrees = np.linalg.norm((abs(lat1-lat2), abs(lon1-lon2)))
    
    return degrees
