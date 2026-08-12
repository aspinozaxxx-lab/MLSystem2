from __future__ import annotations

from mlsystem2.dataset_preparing import contracts


def test_dataset_preparing_contracts_all_is_exact() -> None:
    assert list(contracts.__all__) == [
        "DatasetClassAnnotation",
        "DatasetClassDefinition",
        "DatasetClassRequest",
        "DatasetManifest",
        "DatasetPreparationError",
        "DatasetPreparationRequest",
        "DatasetSourceRevision",
        "PreparedDataset",
        "DatasetSceneReport",
        "DatasetPreparationReport",
        "DatasetPreparationResult",
        "PreparedScene",
        "ResolvedSceneImage",
        "SceneImageResolution",
        "SceneImageResolutionRequest",
    ]


def test_removed_dataset_preparing_dto_are_absent() -> None:
    for name in ("SceneFootprint", "DatasetSplit", "ObjectCountByScene"):
        assert not hasattr(contracts, name)
