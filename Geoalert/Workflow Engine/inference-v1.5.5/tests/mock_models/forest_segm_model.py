import numpy as np
from tests.testutils.defaults import RED_COLORMAP, SemanticClass
from typing import Final, Dict


class Model:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == (3, 1504, 1504)
        out = (x[0] == RED_COLORMAP[SemanticClass.FOREST]).astype(np.uint8)
        return np.expand_dims(out, 0)
