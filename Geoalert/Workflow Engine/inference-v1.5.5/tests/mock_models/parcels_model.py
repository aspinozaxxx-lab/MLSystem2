import numpy as np


class Model:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.shape == (3, 628*3, 628*3)
        masks = [x[ch] for ch in (0, 1, 2) if np.any(x[ch] > 0)]
        scores = [np.max(mask) for mask in masks]
        return masks, scores
