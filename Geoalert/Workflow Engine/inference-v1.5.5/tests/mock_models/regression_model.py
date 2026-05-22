import numpy as np
from tests.testutils.defaults import RED_COLORMAP, SemanticClass


class Model:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == (1, 4, 512, 512)
        height = x[0, 2, 255, 255]  # central point
        return np.array((height,)).astype(float)
