from __future__ import annotations

from mlsystem2.tile_preparation import contracts


def test_tile_preparation_contracts_all_is_exact() -> None:
    assert list(contracts.__all__) == [
        "TileClassAnnotation",
        "TileDataloaderRequest",
        "TilePreparationError",
        "TileSplitRequest",
    ]
