import shapely
from typing import Tuple, Sequence, Final
import numpy as np
import shapely.affinity
from shapely import Polygon
from urban.functional.utils.buildingsutils import generate_rooftop, generate_wall_from_rooftop,\
    generate_shadow_from_footprint
from urban.base.defaults import DEFINITIVE_HEIGHT_TAG, BUILDING_CLASS_TAG
from urban.functional.postprocessing.constants import Shape, Tag
from dataclasses import dataclass, field
from .utils import create_triangle_polygon, random_points, building_shapes_generators


DEFAULT_BUILDING_HEIGHT: Final[float] = 20  # meters
DEFAULT_BUILDING_SHAPE: Final[str] = Shape.RECTANGLE
DEFAULT_BUILDING_CREATION_PARAMS: Final[dict] = {'width': 10, 'height': 10}
CNT_BUFFER: Final[float] = 2


@dataclass
class Building:
    center: Tuple[float, float]
    shape_type: str = field(default=Shape.RECTANGLE)
    height: float = field(default=DEFAULT_BUILDING_HEIGHT)
    bld_class: str = field(default='101')
    rotation: float = field(default=0)
    sat_azimuth: float = field(default=None)
    sat_elevation: float = field(default=None)
    sun_azimuth: float = field(default=None)
    sun_elevation: float = field(default=None)
    floor_count_max: int = field(default=0)
    area_total: int = field(default=0)
    floor_count_min: int = field(default=0)
    is_alarm: int = field(default=0)
    quarters_count: int = field(default=0)
    living_quarters_count: int = field(default=0)
    area_residential: int = field(default=0)
    generator_kwargs: dict = field(default_factory=lambda: {k: v for k, v in DEFAULT_BUILDING_CREATION_PARAMS})

    @property
    def footprint(self) -> Polygon:
        return building_shapes_generators[self.shape_type](self.center, self.rotation, **self.generator_kwargs)

    @property
    def rooftop(self) -> Polygon:
        return generate_rooftop(self.footprint, self.height, self.sat_azimuth, self.sat_elevation)

    @property
    def wall(self) -> Polygon:
        return generate_wall_from_rooftop(self.rooftop, self.height, self.sat_azimuth, self.sat_elevation)

    @property
    def shadow(self) -> Polygon:
        return generate_shadow_from_footprint(self.footprint, self.height, self.sun_azimuth, self.sun_elevation)

    @property
    def contour(self) -> Polygon:
        return shapely.intersection(self.rooftop.exterior.buffer(CNT_BUFFER), self.rooftop)

    @property
    def wall_markup(self) -> Polygon:
        return create_triangle_polygon(list(self.footprint.exterior.coords)[0], self.height,
                                       self.sat_azimuth, self.sat_elevation)

    @property
    def shadow_markup(self) -> Polygon:
        return create_triangle_polygon(list(self.footprint.exterior.coords)[0], self.height,
                                       self.sun_azimuth, self.sun_elevation)

    def as_feature(self, geometry='footprint') -> dict:
        return {'geometry': self.__getattribute__(geometry),
                DEFINITIVE_HEIGHT_TAG: self.height,
                Tag.BLD_SHAPE_TYPE: self.shape_type,
                BUILDING_CLASS_TAG: self.bld_class
                }

    def zkh_point(self, shift_from_center: Tuple[float, float] = (0, 0), **override_zkh_params) -> dict:
        point = {'geometry': shapely.affinity.translate(self.footprint.centroid, *shift_from_center),
                 'floor_count_max': self.floor_count_max,
                 'area_total': self.area_total,
                 'floor_count_min': self.floor_count_min,
                 'is_alarm': self.is_alarm,
                 'quarters_count': self.quarters_count,
                 'living_quarters_count': self.living_quarters_count,
                 'area_residential': self.area_residential
                 }
        if override_zkh_params:
            point.update(override_zkh_params)
        return point


def random_building(center: Tuple[float, float],
                    height_range: Tuple[float, float],
                    rotation_range: Tuple[float, float],
                    shapes: Sequence[str],
                    sat_azimuth: float,
                    sat_elevation: float,
                    sun_azimuth: float,
                    sun_elevation: float,
                    **shape_generator_kwargs
                    ) -> Building:
    return Building(center=center,
                    shape_type=np.random.choice(shapes),
                    height=np.random.uniform(*height_range),
                    rotation=np.random.uniform(*rotation_range),
                    sat_azimuth=sat_azimuth,
                    sat_elevation=sat_elevation,
                    sun_azimuth=sun_azimuth,
                    sun_elevation=sun_elevation,
                    generator_kwargs=shape_generator_kwargs)


def random_buildings_within_aoi(aoi: Polygon,
                                n: int,
                                height_range: Tuple[float, float],
                                rotation_range: Tuple[float, float],
                                shapes: Sequence[str],
                                sat_azimuth: float,
                                sat_elevation: float,
                                sun_azimuth: float,
                                sun_elevation: float,
                                **shape_generator_kwargs
                                ) -> Sequence[Building]:
    centers = random_points(n, aoi)
    return [random_building(c, height_range, rotation_range, shapes,
                            sat_azimuth=sat_azimuth,
                            sat_elevation=sat_elevation,
                            sun_azimuth=sun_azimuth,
                            sun_elevation=sun_elevation, **shape_generator_kwargs) for c in centers]
