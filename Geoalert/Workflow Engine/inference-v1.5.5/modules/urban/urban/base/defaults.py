from typing import Final, Dict
import shapely
from pydantic import Field
from typing_extensions import Annotated

# constants
UNDEFINED_HEIGHT: Final[float] = -1
DEFAULT_HEIGHT: Final[float] = 4
MIN_HEIGHT: Final[float] = 3
HEIGHT_DECIMALS: Final[int] = 0
CONFIDENCE_DECIMALS: Final[int] = 3
SW_CONFIDENCE_THRESHOLD = 0.1
EMB_CONFIDENCE_THRESHOLD = 0.02
DEFAULT_SAMPLE_SIZE: Final[tuple] = (1024, 1024)
DEFAULT_BOUNDS: Final[int] = 256
DEFAULT_RESIZE_INTERPOLATION: Final[str] = 'bilinear'

# tags
SW_HEIGHT_TAG: Final[str] = '_sw_height'
SW_CONFIDENCE_TAG: Final[str] = '_sw_confidence'
REGR_HEIGHT_TAG: Final[str] = '_regr_height'
EMB_HEIGHT_TAG: Final[str] = '_emb_height'
EMB_CONFIDENCE_TAG: Final[str] = '_emb_confidence'
DEFINITIVE_HEIGHT_TAG: Final[str] = 'building_height'
AREA_HEIGHT_TAG: Final[str] = '_area_height'
AREA_HEIGHT_CONFIDENCE_TAG: Final[str] = '_area_height_confidence'
DEFINITIVE_HEIGHT_SOURCE_TAG: Final[str] = '_height_from'
BUILDING_CLASS_TAG: Final[str] = 'class_id'
PARCEL_CONFIDENCE_TAG: Final[str] = 'score'  # for NSPD parcel models, to keep the original behavior

SUN_AZIMUTH_TAG: Final[str] = 'sun_azimuth'
SUN_ELEVATION_TAG: Final[str] = 'sun_elevation'
SAT_AZIMUTH_TAG: Final[str] = 'sat_azimuth'
SAT_ELEVATION_TAG: Final[str] = 'sat_elevation'

angle_tags: Final[tuple] = (SUN_AZIMUTH_TAG, SUN_ELEVATION_TAG, SAT_AZIMUTH_TAG, SAT_ELEVATION_TAG)


# types

GEOM_TYPES: Final[Dict[str, type]] = {
    'polygon': shapely.Polygon,
    'linestring': shapely.LineString,
    'point': shapely.Point
}

AOI_FILENAME: Final[str] = 'aoi'
VECTOR_EXT: Final[str] = '.geojson'
