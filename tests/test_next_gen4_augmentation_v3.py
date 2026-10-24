"""Защита валидности только нового профиля без изменения прежних аугментаций."""

import pickle

import numpy as np
import pytest

from mlsystem2.tile_preparation._augmentations import build_notebook_augmentations
from mlsystem2.tile_preparation.contracts import TilePreparationError
from mlsystem2.training_ui_api._templates import initial_templates, sanitize_template_config
from mlsystem2.training_ui_api import _service


VARIANT = "next_gen4_augmentation_v3"


def sample(channels=4):
    image = np.full((64, 96, channels), 25, np.uint8)
    image[..., -1] = 17
    valid = np.ones((64, 96), np.uint8)
    valid[8:16, 12:24] = 0
    valid[:, :3] = 0
    image[valid == 0] = 0
    mask = np.full(valid.shape, 2, np.int64)
    mask[:, 40:50] = -1
    negative = np.zeros_like(valid)
    negative[20:30, 50:70] = 1
    return {"image": image, "mask": mask, "valid": valid, "masks": [negative]}


@pytest.mark.parametrize("channels", [3, 4])
def test_unsafe_photometry_reverts_only_photo_and_all_masks_stay_synchronous(channels):
    A = pytest.importorskip("albumentations")
    original = sample(channels)
    transform = build_notebook_augmentations(42, masked=True, preserve_validity=True)
    transform.geometry = A.Compose([A.HorizontalFlip(p=1)], additional_targets={"valid": "mask"}, save_applied_params=True)
    transform.photometry = A.Compose([A.RandomBrightnessContrast(brightness_limit=(-.5, -.5), contrast_limit=(0, 0), p=1)], save_applied_params=True)
    expected = transform.geometry(**original)
    actual = transform(**original)
    np.testing.assert_array_equal(actual["image"], expected["image"])
    for key in ("mask", "valid", "masks"):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual["augmentation_guard"] == {"geometry_reverted": False, "photometry_reverted": True}
    assert np.array_equal(np.any(actual["image"] != 0, axis=-1), actual["valid"] > 0)
    assert [name for name, _ in actual["applied_transforms"]] == ["HorizontalFlip"]
    np.testing.assert_array_equal(original["image"], sample(channels)["image"])


def test_safe_photo_is_applied_without_filling_nodata():
    A = pytest.importorskip("albumentations")
    original = sample()
    transform = build_notebook_augmentations(42, masked=True, preserve_validity=True)
    transform.geometry = A.Compose([], additional_targets={"valid": "mask"}, save_applied_params=True)
    transform.photometry = A.Compose([A.RandomBrightnessContrast(brightness_limit=(.1, .1), contrast_limit=(0, 0), p=1)], save_applied_params=True)
    actual = transform(**original)
    valid = original["valid"] > 0
    assert (actual["image"][valid] > original["image"][valid]).all()
    assert not actual["image"][~valid].any()
    np.testing.assert_array_equal(actual["valid"], original["valid"])
    assert not any(actual["augmentation_guard"].values())


def test_mask_equality_is_checked_not_only_valid_pixel_count():
    A = pytest.importorskip("albumentations")
    original = sample()
    transform = build_notebook_augmentations(42, masked=True, preserve_validity=True)
    transform.geometry = A.Compose([], additional_targets={"valid": "mask"}, save_applied_params=True)
    class SwapValidity:
        def __call__(self, *, image):
            image[0, 4] = 0
            image[0, 0] = 25
            return {"image": image, "applied_transforms": []}
    transform.photometry = SwapValidity()
    actual = transform(**original)
    np.testing.assert_array_equal(actual["image"], original["image"])
    assert actual["augmentation_guard"]["photometry_reverted"]


def test_unsafe_geometry_reverts_image_and_all_masks_without_retry():
    A = pytest.importorskip("albumentations")
    original = sample()
    transform = build_notebook_augmentations(42, masked=True, preserve_validity=True)
    class BrokenGeometry:
        calls = 0
        def __call__(self, **values):
            self.calls += 1
            return {**values, "image": np.zeros_like(values["image"]), "applied_transforms": [("Affine", {})]}
    transform.geometry = BrokenGeometry()
    transform.photometry = A.Compose([], save_applied_params=True)
    actual = transform(**original)
    for key in ("image", "mask", "valid", "masks"):
        np.testing.assert_array_equal(actual[key], original[key])
    assert transform.geometry.calls == 1
    assert actual["augmentation_guard"] == {"geometry_reverted": True, "photometry_reverted": False}


def test_empty_window_and_invalid_source_mask_are_handled_explicitly():
    pytest.importorskip("albumentations")
    original = sample()
    transform = build_notebook_augmentations(42, masked=True, preserve_validity=True)
    original["image"][:] = 0
    with pytest.raises(TilePreparationError, match="Исходная valid-mask"):
        transform(**original)
    original["valid"][:] = 0
    actual = transform(**original)
    assert not actual["valid"].any() and not actual["image"].any()


@pytest.mark.parametrize("channels", [3, 4])
def test_seed_and_spawn_pickle_preserve_exact_result(channels):
    pytest.importorskip("albumentations")
    transform = build_notebook_augmentations(57, masked=True, preserve_validity=True)
    restored = pickle.loads(pickle.dumps(transform))
    first = transform(**sample(channels))
    actual = restored(**sample(channels))
    for key in ("image", "mask", "valid", "masks"):
        np.testing.assert_array_equal(first[key], actual[key])
    restored.set_random_seed(57)
    repeated = restored(**sample(channels))
    np.testing.assert_array_equal(repeated["image"], first["image"])


@pytest.mark.parametrize("architecture", [f"smp_segformer_b{value}" for value in range(4)])
@pytest.mark.parametrize("size", [512, 768, 1024, 1536])
def test_new_profile_available_independently_of_existing_defaults(architecture, size):
    config = sanitize_template_config({"train.pipeline_variant": VARIANT, "tile_preparation.tile_size": size}, architecture=architecture)
    assert config["train.pipeline_variant"] == VARIANT and config["tile_preparation.stride"] == size
    _service._validate_training_pipeline_variant({**config, "dataset.task": "binary", "dataset.imagery_type": "kanopus", "train.input_channels": 4}, architecture)
    template = next(item for item in initial_templates() if item["architecture"] == architecture)
    assert VARIANT in template["config_schema"]["fields"][0]["options"]
    assert template["config_schema"]["pipeline_defaults"][VARIANT]["train.pipeline_variant"] == VARIANT
    assert "valid-mask" in template["config_schema"]["pipeline_descriptions"][VARIANT]
    assert template["default_config"]["train.pipeline_variant"] == "legacy"


@pytest.mark.parametrize("architecture", [f"smp_segformer_b{value}" for value in range(4)])
def test_checkpoint_preserves_policy_and_global_normalization(tmp_path, architecture):
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    from mlsystem2.models.api import create_model, load_checkpoint, save_checkpoint
    from mlsystem2.models.contracts import ModelSpec, LoadCheckpointRequest, SaveCheckpointRequest, ModelsError
    torch.set_num_threads(2)
    parameters = {"pipeline_variant": VARIANT, "augmentation_policy": "validity_guard_v3",
        "preprocessing": {"mode": "global_minmax", "low": [0.0] * 4, "high": [255.0] * 4,
            "epsilon": 1e-6, "nodata": 0.0, "mask_output_nodata": True,
            "source": "retained_train_windows", "valid_pixel_count": 100, "window_count": 4}}
    spec = ModelSpec(name=architecture, input_channels=4, output_channels=1, parameters=parameters)
    model = create_model(spec)
    path = str(tmp_path / "best.pt")
    save_checkpoint(SaveCheckpointRequest(model=model, checkpoint_uri=path))
    loaded = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=path, map_location="cpu"))
    assert loaded.model.spec.parameters["augmentation_policy"] == "validity_guard_v3"
    assert loaded.model.spec.parameters["preprocessing"] == parameters["preprocessing"]
    model.model.eval()
    loaded.model.model.eval()
    image = torch.full((1, 4, 32, 32), 30.0)
    image[..., :3, :] = 0
    with torch.no_grad():
        torch.testing.assert_close(model.model(image), loaded.model.model(image), rtol=0, atol=0)
    incompatible = {**parameters, "augmentation_policy": "неизвестная"}
    with pytest.raises(ModelsError, match="политику защищённых"):
        create_model(spec.model_copy(update={"parameters": incompatible}))
