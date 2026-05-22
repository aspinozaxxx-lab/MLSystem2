"""Low level math utils"""

import numpy as np
from typing import Tuple


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def normalize(x: float, from_range: Tuple[float, float], to_range: Tuple[float, float]) -> float:
    return (x - from_range[0]) / (from_range[1] - from_range[0]) * (to_range[1] - to_range[0]) + to_range[0]
