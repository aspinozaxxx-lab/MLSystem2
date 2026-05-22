from scipy.signal import convolve2d
import numpy as np


def remove_diagonal_connectivity(x: np.ndarray) -> np.ndarray:
    """Replace diagonal ones to zeros:

    Case 1:

        [[0, 1],   --->  [[0, 0],
         [1, 0]]          [0, 0]]

    Case 2:

        [[1, 0],   --->  [[0, 0],
         [0, 1]]          [0, 0]]

    Args:
        x: np.array binary image

    Returns:
        result: image with removed diagonal connectivity

    """
    assert x.ndim == 2

    x = x.astype('uint8')

    kernel_1 = np.array([[1, -1], [-1, 1]], dtype='int8')
    kernel_2 = kernel_1[::-1]

    connection_1 = (convolve2d(x, kernel_1, mode='same', boundary='wrap') == 2).astype('uint8')
    connection_2 = (convolve2d(x, kernel_2, mode='same', boundary='wrap') == 2).astype('uint8')

    result = x.copy()
    result *= (1 - connection_1)
    result *= (1 - np.roll(connection_2, -1, axis=0))

    return result
