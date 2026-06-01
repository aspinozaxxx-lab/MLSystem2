from __future__ import annotations

import pytest

from mlsystem2.cli.tiling_test_for_black import _tile_quality_flags


def test_tile_quality_flags_marks_only_fully_black_tiles() -> None:
    torch = pytest.importorskip("torch")
    images = torch.zeros((3, 2, 4, 4), dtype=torch.float32)
    images[1, 0, 0, 0] = 0.5
    images[2, 1, 2, 2] = -2.0

    black_flags, nonfinite_flags = _tile_quality_flags(images, eps=1e-6)

    assert black_flags == [True, False, False]
    assert nonfinite_flags == [False, False, False]


def test_tile_quality_flags_reports_nonfinite_as_problem() -> None:
    torch = pytest.importorskip("torch")
    images = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
    images[0, 0, 0, 0] = float("nan")
    images[1, 0, 0, 0] = 1.0
    images[1, 0, 1, 1] = float("inf")

    black_flags, nonfinite_flags = _tile_quality_flags(images, eps=1e-6)

    assert black_flags == [True, False]
    assert nonfinite_flags == [True, True]
