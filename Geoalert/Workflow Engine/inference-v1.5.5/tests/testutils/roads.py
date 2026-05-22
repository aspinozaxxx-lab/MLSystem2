from typing import Tuple, Sequence, Final
import numpy as np
from dataclasses import dataclass, field
from shapely import LineString, Polygon
from urban.functional.utils.angleutils import azimuth_from_vector, vector_from_angle


ROAD_BUFFER: Final[float] = 2


@dataclass
class Road:
    start: Tuple[float, float]
    end: Tuple[float, float]

    @property
    def line(self):
        return LineString((self.start, self.end))

    @property
    def polygon(self):
        return self.line.buffer(ROAD_BUFFER)

    @property
    def angle(self):
        return azimuth_from_vector(np.array((self.start, self.end)))

    @classmethod
    def from_angle_and_offset(cls, aoi: Polygon, angle: float, offset: float):
        minx, miny, maxx, maxy = aoi.bounds
        start = minx, miny + offset
        end = np.array(start) + vector_from_angle(angle) * (maxx - minx) * (maxy - miny)
        line = LineString((start, end)).intersection(aoi)
        return cls(line.coords[0], line.coords[-1])

    def as_feature(self, geometry='polygon') -> dict:
        return {'geometry': self.__getattribute__(geometry)}


def random_roads_within_aoi(aoi: Polygon,
                            n: int,
                            angles_range: Tuple[float, float],
                            offsets_range: Tuple[float, float]) -> Sequence[Road]:
    # TODO: avoid roads and buildings intersections
    return [Road.from_angle_and_offset(aoi, np.random.uniform(*angles_range),
                                       np.random.uniform(*offsets_range)) for _ in range(n)]
