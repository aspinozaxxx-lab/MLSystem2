from gpdadapter import FeatureCollection, show_fcs
from typing import Tuple, Sequence
import numpy as np
from dataclasses import dataclass, field
from shapely import Polygon
from urban.base.defaults import SAT_AZIMUTH_TAG, SAT_ELEVATION_TAG, SUN_ELEVATION_TAG, SUN_AZIMUTH_TAG

@dataclass
class AOI:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    sat_azimuth: float = field(default=None)
    sat_elevation: float = field(default=None)
    sun_azimuth: float = field(default=None)
    sun_elevation: float = field(default=None)

    @property
    def polygon(self):
        return Polygon.from_bounds(self.xmin, self.ymin, self.xmax, self.ymax)

    def as_feature(self, with_angels: bool = False) -> dict:
        if with_angels:
            return {'geometry': self.polygon, SAT_AZIMUTH_TAG: self.sat_azimuth, SAT_ELEVATION_TAG: self.sat_elevation,
                    SUN_AZIMUTH_TAG: self.sun_azimuth, SUN_ELEVATION_TAG: self.sun_elevation}
        return {'geometry': self.polygon}


def random_meta(n_aois: int = 1,
                aoi_size: Tuple[int, int] = (100, 100),
                gap_coef: float = 1.5,
                sat_azimuth_range: Tuple[float, float] = (-180, 180),
                sat_elevation_range: Tuple[float, float] = (40, 60),
                sun_azimuth_range: Tuple[float, float] = (-180, 180),
                sun_elevation_range: Tuple[float, float] = (40, 60)) -> Sequence[AOI]:
    height = int(np.floor(np.sqrt(n_aois)))
    width = height + (n_aois - height ** 2)
    grid = np.stack(np.meshgrid(np.arange(0, width), np.arange(0, height))).reshape(2, -1).T * aoi_size * gap_coef
    return [AOI(xy[0], xy[1], xy[0] + aoi_size[0], xy[1] + aoi_size[1],
                np.random.uniform(*sat_azimuth_range), np.random.uniform(*sat_elevation_range),
                np.random.uniform(*sun_azimuth_range), np.random.uniform(*sun_elevation_range)) for xy in grid]


if __name__ == '__main__':
    meta = random_meta(3, aoi_size=(200, 200), sun_azimuth_range=(45, 46), sun_elevation_range=(45, 46),
                       sat_azimuth_range=(-46, -45), sat_elevation_range=(45, 46))
    show_fcs(FeatureCollection([aoi.polygon for aoi in meta], crs='EPSG:3857'))

