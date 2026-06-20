from __future__ import annotations

from mlsystem2.tile_preparation import contracts


def test_tile_preparation_contracts_all_is_exact() -> None:
    assert list(contracts.__all__) == [
        "HARD_NEGATIVE_LABEL",
        "TileClassAnnotation",
        "TileDataloaderRequest",
        "TilePreparationError",
        "TileSplitRequest",
    ]
