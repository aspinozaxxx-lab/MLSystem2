"""This describes semantic classes in synthetic raster data and its colormap.
First (RED) channel in RGB raster encodes semantic class, other two (GREEN and BLUE) encodes properties
like building class, height, etc. which may be different for different classes"""

from typing import Final, Dict, Hashable
from enum import Enum
from urban.base.defaults import DEFINITIVE_HEIGHT_TAG, BUILDING_CLASS_TAG


class SemanticClass(Enum):
    SHADOW = 1
    WALL = 2
    ROOFTOP = 3
    BLD_CONTOUR = 4
    ROAD = 5
    FOREST = 6
    FOOTPRINT = 7


def invert_colormap(d: Dict):
    return {v: k for k, v in d.items()}


# Semantic class : value
RED_COLORMAP: Final[Dict[SemanticClass, int]] = {
    SemanticClass.SHADOW: 1,
    SemanticClass.WALL: 100,
    SemanticClass.ROOFTOP: 200,
    SemanticClass.BLD_CONTOUR: 255,
    SemanticClass.ROAD: 2,
    SemanticClass.FOREST: 254
}

# Green for categorical values
# Semantic class: {property name: {category: value}}
GREEN_COLORMAP: Final[Dict[SemanticClass, Dict[str, Dict[Hashable, int]]]] = {
    SemanticClass.ROOFTOP: {
        BUILDING_CLASS_TAG: {
            '101': 1,
            '102': 2,
            '103': 3,
            '104': 4,
            '105': 5
        }
    }
}

# Blue for integer values
# Semantic class: property name
BLUE_COLORMAP: Final[Dict[SemanticClass, str]] = {
    SemanticClass.ROOFTOP: DEFINITIVE_HEIGHT_TAG,
    SemanticClass.FOREST: DEFINITIVE_HEIGHT_TAG
}

MOCK_PIPELINES_PATH: Final[str] = 'tests/test_pipelines/mock_models_tests'
