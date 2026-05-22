import numpy as np
from tests.testutils.defaults import GREEN_COLORMAP, SemanticClass
from urban.base.defaults import BUILDING_CLASS_TAG


CLASSES_MAP = {
    '101': 1,
    '102': 2,
    '103': 3,
    '104': 4,
    '105': 5
}


class Model:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == (1, 4, 1024, 1024)
        x = x[0, 1]
        out = np.zeros((1024, 1024), np.uint8)
        for key in CLASSES_MAP.keys():
            out[(x == GREEN_COLORMAP[SemanticClass.ROOFTOP][BUILDING_CLASS_TAG][key])] = CLASSES_MAP[key]
        return np.expand_dims(out, 0)
