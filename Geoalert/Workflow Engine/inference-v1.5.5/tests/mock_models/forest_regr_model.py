import numpy as np
from tests.testutils.defaults import RED_COLORMAP, SemanticClass


class Model:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == (3, 736+150*2, 736+150*2)
        x = x[2]*(x[0] == RED_COLORMAP[SemanticClass.FOREST])
        return np.expand_dims(x, 0)
