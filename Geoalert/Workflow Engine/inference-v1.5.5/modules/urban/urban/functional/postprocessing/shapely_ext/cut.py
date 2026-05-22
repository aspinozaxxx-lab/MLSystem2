import shapely
import shapely.geometry as sg
from typing import Union
from .angles import get_angle_for_line


def strip_line(line: sg.LineString, distance: Union[float, int]) -> Union[sg.LineString, sg.Point]:
    """Cut pieces from both ends of LineString"""
    if line.length < distance * 2:
        return line.centroid

    # if line is longer than 2 distances
    angle = get_angle_for_line(line)
    line = shapely.affinity.rotate(line, - angle)
    x1, y1 = line.coords[0]
    x2, y2 = line.coords[1]
    if x1 > x2:
        x1 -= distance
        x2 += distance
    else:
        x1 += distance
        x2 -= distance
    line = shapely.geometry.LineString([[x1, y1], [x2, y2]])
    line = shapely.affinity.rotate(line, angle)
    return line
