import numpy as np
from tests.testutils.defaults import RED_COLORMAP, SemanticClass


class Model:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == (3, 1024+256, 1024+256)
        x = x[0]
        out = np.zeros((1024+256, 1024+256), np.uint8)
        out[(x == RED_COLORMAP[SemanticClass.ROAD])] = 1
        return np.expand_dims(out, 0)
