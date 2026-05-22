import numpy as np
from tests.testutils.defaults import RED_COLORMAP, SemanticClass
from typing import Final, Dict

SWRC_MODEL_OUTPUT_MAP: Final[Dict[SemanticClass, int]] = {
    SemanticClass.SHADOW: 1,
    SemanticClass.WALL: 2,
    SemanticClass.ROOFTOP: 3,
    SemanticClass.BLD_CONTOUR: 4
}


class Model:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == (1, 3, 1024, 1024)
        x = x[0, 0]
        out = np.zeros((1024, 1024), np.uint8)
        for cls in SWRC_MODEL_OUTPUT_MAP.keys():
            out[x == RED_COLORMAP[cls]] = SWRC_MODEL_OUTPUT_MAP[cls]
        return np.expand_dims(out, 0)
