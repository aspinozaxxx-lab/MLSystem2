from typing import Final, Sequence, Dict
import shapely
from .meta import AOI
from .buildings import Building
from .roads import Road
from .parcels import Parcel
from .forest import Forest
from urban.functional.postprocessing.constants import Shape
from gpdadapter import FeatureCollection
import rasterio
import rasterio.features as rasterio_features
import numpy as np
from .defaults import SemanticClass, RED_COLORMAP, GREEN_COLORMAP, BLUE_COLORMAP

SYNTHETIC_DATA_PATH: Final[str] = 'tests/test_data/synthetic'

SEMANTIC_FILENAMES: Final[Dict[SemanticClass, str]] = {
    SemanticClass.FOOTPRINT: 'fp',
    SemanticClass.ROOFTOP: 'rt',
    SemanticClass.WALL: 'wl',
    SemanticClass.SHADOW: 'sh',
    SemanticClass.ROAD: 'roads',
    SemanticClass.BLD_CONTOUR: 'cnt',
    SemanticClass.FOREST: 'forest',
}

ROADS_NAME: Final[str] = 'roads'
WLL_NAME: Final[str] = 'walls_labels'
SHL_NAME: Final[str] = 'shadows_labels'
META_NAME: Final[str] = 'meta'
ZKH_NAME: Final[str] = '100zkh'
OSM_ROADS_NAME: Final[str] = '301-osm'
OSM_BUILDINGS_NAME: Final[str] = '100-osm'
RASTER_INPUT_FILENAME: Final[str] = 'input'

VEC_EXT: Final[str] = '.geojson'
RAST_EXT: Final[str] = '.tif'
DEFAULT_CRS: Final[str] = 'EPSG:3857'

default_sat_azimuth = 45
default_sat_elevation = 45
default_sun_azimuth = -45
default_sun_elevation = 45

default_aois = [AOI(0, 0, 2000, 2000, default_sat_azimuth, default_sat_elevation,
                    default_sun_azimuth, default_sun_elevation)]

default_buildings = [
    Building(center=(500, 500), shape_type=Shape.CIRCLE, height=30, bld_class='101', rotation=10,
             sat_azimuth=default_sat_azimuth, sat_elevation=default_sat_elevation,
             sun_azimuth=default_sun_azimuth, sun_elevation=default_sun_elevation,
             generator_kwargs=dict(radius=50, n_vertices=20)),

    Building(center=(1500, 500), shape_type=Shape.RECTANGLE, height=40, bld_class='102', rotation=0,
             sat_azimuth=default_sat_azimuth, sat_elevation=default_sat_elevation,
             sun_azimuth=default_sun_azimuth, sun_elevation=default_sun_elevation,
             generator_kwargs=dict(polygon_width=100, polygon_height=100)),

    Building(center=(500, 1500), shape_type=Shape.LSHAPE, height=50, bld_class='103', rotation=45,
             sat_azimuth=default_sat_azimuth, sat_elevation=default_sat_elevation,
             sun_azimuth=default_sun_azimuth, sun_elevation=default_sun_elevation,
             generator_kwargs=dict(polygon_width=100, polygon_height=200, corner_x=50, corner_y=150)),
]


default_roads = [
    Road((200, 0), (200, 2000)),
    Road((1000, 0), (1000, 2000)),
    Road((0, 1000), (2000, 1000)),
]

default_parcels = [
    Parcel(center=(100, 100), score=1, generator_kwargs=dict(polygon_width=50, polygon_height=50)),
    Parcel(center=(400, 400), score=1,  generator_kwargs=dict(polygon_width=50, polygon_height=50)),
    Parcel(center=(100, 100), score=0.5,  generator_kwargs=dict(polygon_width=60, polygon_height=60)),
    Parcel(center=(400, 400), score=0.5,  generator_kwargs=dict(polygon_width=60, polygon_height=60))
]

default_forest = [
    Forest(center=(100, 100), height=2, generator_kwargs=dict(polygon_width=50, polygon_height=50)),
    Forest(center=(150, 100), height=4, generator_kwargs=dict(polygon_width=50, polygon_height=50)),
    Forest(center=(100, 150), height=6, generator_kwargs=dict(polygon_width=50, polygon_height=50)),
    Forest(center=(150, 150), height=8, generator_kwargs=dict(polygon_width=50, polygon_height=50)),
]


def generate_meta(aois: Sequence[AOI] = default_aois, with_angles: bool = False):
    return FeatureCollection([m.as_feature(with_angles) for m in aois], crs=DEFAULT_CRS)


def generate_footprints(buildings: Sequence[Building] = default_buildings):
    return FeatureCollection([b.as_feature() for b in buildings], crs=DEFAULT_CRS)


def generate_rooftops(buildings: Sequence[Building] = default_buildings):
    return FeatureCollection([b.as_feature('rooftop') for b in buildings], crs=DEFAULT_CRS)


def generate_walls(buildings: Sequence[Building] = default_buildings):
    return FeatureCollection([b.as_feature('wall') for b in buildings], crs=DEFAULT_CRS)


def generate_shadows(buildings: Sequence[Building] = default_buildings):
    return FeatureCollection([b.as_feature('shadow') for b in buildings], crs=DEFAULT_CRS)


def generate_cnt(buildings: Sequence[Building] = default_buildings):
    return FeatureCollection([b.as_feature('contour') for b in buildings], crs=DEFAULT_CRS)


def generate_wl_markup(buildings: Sequence[Building] = default_buildings):
    return FeatureCollection([b.as_feature('wall_markup') for b in buildings], crs=DEFAULT_CRS)


def generate_sh_markup(buildings: Sequence[Building] = default_buildings):
    return FeatureCollection([b.as_feature('shadow_markup') for b in buildings], crs=DEFAULT_CRS)


def generate_zkh(buildings: Sequence[Building] = default_buildings):
    return FeatureCollection([b.zkh_point() for b in buildings], crs=DEFAULT_CRS)


def generate_roads(roads: Sequence[Road] = default_roads):
    return FeatureCollection([r.as_feature() for r in roads], crs=DEFAULT_CRS)


def generate_osm_roads(roads: Sequence[Road] = default_roads):
    return FeatureCollection([r.as_feature('line') for r in roads], crs=DEFAULT_CRS)


def generate_osm_buildings(buildings: Sequence[Building] = default_buildings):
    return FeatureCollection([b.as_feature('footprint') for b in buildings], crs=DEFAULT_CRS)


def generate_parcels(parcels: Sequence[Parcel] = default_parcels):
    return FeatureCollection([b.as_feature() for b in parcels], crs=DEFAULT_CRS)

def generate_forest(forest: Sequence[Forest] = default_forest):
    return FeatureCollection([b.as_feature() for b in forest], crs=DEFAULT_CRS)


def _rasterize(fc: FeatureCollection, height, width, transform) -> np.ndarray:
    if len(fc) > 0:
        mask = rasterio_features.rasterize([(geom, 1) for geom in fc.geometry],
                                           out_shape=(height, width),
                                           transform=transform,
                                           dtype=np.uint8)
    else:
        mask = np.zeros((height, width), dtype='uint8')
    return mask


def generate_raster(raster_filename: str, aoi: shapely.Polygon,
                    vectors: Dict[SemanticClass, FeatureCollection], res: float = 0.6):
    xmin, ymin, xmax, ymax = aoi.bounds
    width = int(np.ceil((xmax - xmin) / res))
    height = int(np.ceil((ymax - ymin) / res))
    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)
    mask = np.zeros((height, width, 3), dtype='uint8')

    for cls, fc in vectors.items():

        # red channel - semantic
        _mask = _rasterize(fc, height, width, transform)
        mask[:, :, 0][_mask > 0] = RED_COLORMAP[cls]

        # green channel - categorical properties
        if cls in GREEN_COLORMAP.keys():
            property_name = list(GREEN_COLORMAP[cls].keys())[0]
            for property_value, green_value in GREEN_COLORMAP[cls][property_name].items():
                fc_cls = fc[fc[:, property_name] == property_value]
                _mask = _rasterize(fc_cls, height, width, transform)
                mask[:, :, 1][_mask > 0] = green_value

        # blue channel - integer properties
        if cls in BLUE_COLORMAP.keys():
            property_name = BLUE_COLORMAP[cls]
            values_set = set(fc[:, property_name].to_list())
            for value in values_set:
                _mask = _rasterize(fc[fc[:, property_name] == value], height, width, transform)
                mask[:, :, 2][_mask > 0] = value

    with rasterio.open(raster_filename, 'w', driver='GTiff',
                       height=height, width=width, count=3, dtype='uint8',
                       crs=DEFAULT_CRS, transform=transform) as dst:
        dst.write(mask[:, :, 0], 1)
        dst.write(mask[:, :, 1], 2)
        dst.write(mask[:, :, 2], 3)
