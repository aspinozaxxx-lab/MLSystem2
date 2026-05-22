# TODO: move this to utils/angleutils
import math
import shapely
import numpy as np
import shapely.geometry as sg
from typing import List, Tuple
from .split import split_to_lines

_Point = Tuple[float, float]

__all__ = ['get_angle_for_line', 'get_angles_for_polygon']


def get_angle_for_line(line: sg.LineString) -> float:
    """Calculate line angle, line should contain only 2 points"""
    x1, y1 = line.coords[0]
    x2, y2 = line.coords[1]
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def get_angles_for_polygon(polygon: sg.Polygon, sort: bool = True) -> List[float]:
    """Return positive line angles (% 180) in degrees for all lines in geometry,
    could be sorted in descending order according to line length"""
    lines = split_to_lines(polygon)
    lines = sorted(lines, key=lambda x: x.length, reverse=True) if sort else lines
    angles = [get_angle_for_line(l) % 180 for l in lines]
    return angles


def angle_envelope(geometry: shapely.Polygon) -> float:
    envelope = geometry.oriented_envelope.exterior
    vec1 = np.array(envelope.coords[0]) - np.array(envelope.coords[1])
    return np.rad2deg(np.arctan2(vec1[1], vec1[0]))


def angle_PCA(linear_ring,
              pre_simplification: float = 0.5,
              segmentize: float = 0,
              min_length: float = 3,
              cond_thr: float = 3):
    if pre_simplification:
        linear_ring = linear_ring.simplify(pre_simplification)
    if segmentize:
        linear_ring = shapely.segmentize(linear_ring, segmentize)
    c = np.array(linear_ring.coords)
    vectors = c - np.roll(c, 1, axis=0)

    if min_length:
        vectors = vectors[np.linalg.norm(vectors, axis=1) > min_length]

    if len(vectors) < 3:
        return 0

    try:
        cov_matrix = np.cov(vectors, rowvar=False)
        if np.linalg.cond(cov_matrix) < cond_thr:
            longest = vectors[np.argmax(np.linalg.norm(vectors, axis=1))]
            return np.rad2deg(np.arctan2(longest[1], longest[0]))

        eigval, eigvec = np.linalg.eig(cov_matrix)
        eigvec = (eigvec * eigval).T
        return np.rad2deg(np.arctan2(eigvec[0, 1], eigvec[0, 0]))
    except:
        longest = vectors[np.argmax(np.linalg.norm(vectors, axis=1))]
        return np.rad2deg(np.arctan2(longest[1], longest[0]))
