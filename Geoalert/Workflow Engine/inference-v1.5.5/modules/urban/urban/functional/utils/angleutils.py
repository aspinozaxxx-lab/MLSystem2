"""Low-level utils to work with vectors and angles (mostly for meta-angles computations)"""

import numpy as np
from typing import Sequence
from shapely import Polygon


def azimuth_guard(az: float):
    """Ensure azimuth is in range [-180,180)"""
    return (az-180) % 360 - 180


def elevation_guard(el: float):
    if not 0 < el <= 90:
        raise ValueError(f'Elevation is {el}')
    return el


def angle_diff(a: float, b: float) -> float:
    """Angle difference within (-180, 180)"""
    return azimuth_guard(b - a)


def invert_angle(a: float) -> float:
    """Inverted angle (opposite direction)"""
    return azimuth_guard(a-180)


def azimuth_from_vector(vec) -> float:
    """Calculates azimuth in degrees (-180-180) from vector"""
    return azimuth_guard(90 - np.rad2deg(np.arctan2(vec[1], vec[0])))


def elevation_from_vector(vec, height: float):
    """Calculates elevation in degrees (0-90) from vector and height"""
    return np.rad2deg(np.arctan(height/np.linalg.norm(vec)))


def vector_from_relative_positions(poly1: Polygon, poly2: Polygon):
    """
    Returns shift vector polygons relative positions
    Args:
        poly1: footprint geometry
        poly2: rooftop geometry
    Returns:
        shift vector(x, y) as np.array
    """
    return (np.array(poly1.centroid.coords) - np.array(poly2.centroid.coords))[0]


def vector_from_angle(angle: float) -> np.ndarray:
    """
    Returns vector with norm=1 and given angle
    Args:
        angle: from y-axis in degrees clockwise
    Returns:
        (x, y) as np.array, norm=1
    """
    angle = np.deg2rad(azimuth_guard(90 - angle))
    return np.array((np.cos(angle), np.sin(angle)))


def vector_from_height(azimuth: float, elevation: float, height: float = 1) -> np.ndarray:
    """
    Returns wall vector from building height and satellite angles
    Args:
        height: building height
        azimuth: from y-axis in degrees clockwise
        elevation: sat elevation
    Returns:
        (x, y) as np.array, norm=1
    """
    azimuth = np.deg2rad(azimuth_guard(azimuth))
    proj_length = height / np.tan(np.deg2rad(elevation_guard(elevation)))
    x = proj_length * np.sin(azimuth)
    y = proj_length * np.cos(azimuth)
    return np.array((x, y))


def height_from_vector(vec: Sequence, elevation: float) -> float:
    """
    Returns building height from wall vector
    Args:
        vec: np.array or tuple, (x, y)
        elevation: sat elevation
    Returns:
        height
    """
    return np.linalg.norm(vec) * np.tan(np.deg2rad(elevation_guard(elevation)))


def angle_between_vectors(x: np.array, y: np.array) -> float:
    """Returns angle between two vectors in degrees"""
    return np.rad2deg(np.arccos(np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y))))


def rotate_vector(vec: np.ndarray, angle: float) -> np.array:
    angle = np.deg2rad(angle)
    return np.array([vec[0] * np.cos(angle) - vec[1] * np.sin(angle),
                     vec[0] * np.sin(angle) + vec[1] * np.cos(angle)])


def mean_angle(angles: Sequence[float]) -> float:
    if not angles:
        raise ValueError('Got empty angles sequence in mean_angle()')
    if len(angles) == 1:
        return angles[0]
    return azimuth_from_vector(np.mean(np.array([vector_from_angle(a) for a in angles]), axis=0))


def median_angle(angles: Sequence[float]) -> float:
    if not angles:
        raise ValueError('Got empty angles sequence in median_angle()')
    if len(angles) == 1:
        return angles[0]
    return azimuth_from_vector(np.median(np.array([vector_from_angle(a) for a in angles]), axis=0))
