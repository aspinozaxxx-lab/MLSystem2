from __future__ import annotations

from mlsystem2.dataset_preparing import contracts


def test_dataset_preparing_contracts_all_is_exact() -> None:
    assert list(contracts.__all__) == [
        "DatasetClassAnnotation",
        "DatasetClassRequest",
        "DatasetPreparationError",
        "DatasetPreparationRequest",
        "PreparedDataset",
        "DatasetSceneReport",
        "DatasetPreparationReport",
        "DatasetPreparationResult",
    ]


def test_removed_dataset_preparing_dto_are_absent() -> None:
    for name in ("SceneFootprint", "DatasetManifest", "DatasetSplit", "ObjectCountByScene"):
        assert not hasattr(contracts, name)
