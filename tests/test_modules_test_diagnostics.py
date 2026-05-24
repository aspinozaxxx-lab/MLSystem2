from __future__ import annotations

import numpy as np

from mlsystem2.cli.modules_test import _is_black_tile, _mask_edge


def test_modules_test_black_detector_marks_all_zero_tile_black() -> None:
    image = np.zeros((4, 1024, 1024), dtype=np.float32)

    assert _is_black_tile(image) is True


def test_modules_test_black_detector_uses_interior_grid_not_only_boundary() -> None:
    image = np.zeros((4, 1024, 1024), dtype=np.float32)
    image[:, 512, 512] = 1.0

    assert _is_black_tile(image) is False


def test_modules_test_mask_edge_marks_only_contour() -> None:
    mask = np.zeros((1, 8, 8), dtype=np.float32)
    mask[:, 2:6, 2:6] = 1.0

    edge = _mask_edge(mask)

    assert edge.sum() == 12
    assert edge[2, 2]
    assert not edge[3, 3]
