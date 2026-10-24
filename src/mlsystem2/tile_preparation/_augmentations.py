"""Внутренние пресеты аугментаций тайлов."""

from __future__ import annotations


import hashlib

import numpy as np


def apply_augmentations(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    nodata_pixels: np.ndarray,
    nodata: object,
    level: int,
    seed: int,
    sample_index: int,
    return_nodata: bool = False,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if level <= 0:
        if return_nodata:
            return np.ascontiguousarray(image), np.ascontiguousarray(mask), nodata_pixels, False
        return np.ascontiguousarray(image), np.ascontiguousarray(mask), False

    rng = np.random.default_rng(seed + sample_index)
    image, mask, nodata_pixels, augmented = _geometric(
        image,
        mask,
        nodata_pixels,
        rng,
    )

    if level >= 2:
        image = _photometric(image, rng, valid_pixels=~nodata_pixels if return_nodata else None)
        augmented = True

    image = np.clip(image, 0.0, 255.0)
    image[:, nodata_pixels] = nodata
    if return_nodata:
        return np.ascontiguousarray(image), np.ascontiguousarray(mask), nodata_pixels, augmented
    return np.ascontiguousarray(image), np.ascontiguousarray(mask), augmented


def apply_next_gen_augmentations(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    nodata_pixels: np.ndarray,
    nodata: object,
    level: int,
    seed: int,
    sample_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Применить воспроизводимые аугментации ко всем синхронным маскам."""

    if level <= 0:
        return (
            np.ascontiguousarray(image),
            np.ascontiguousarray(mask),
            np.ascontiguousarray(nodata_pixels),
            False,
        )

    digest = hashlib.sha256(f"{seed}\0{sample_key}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big", signed=False))
    image, mask, nodata_pixels, augmented = _geometric(
        image,
        mask,
        nodata_pixels,
        rng,
    )
    if level >= 2:
        image = _photometric(image, rng, valid_pixels=~nodata_pixels)
        augmented = True
    if level >= 3:
        image = _spectral(image, rng, valid_pixels=~nodata_pixels)
        augmented = True

    image = np.clip(image, 0.0, 255.0)
    image[:, nodata_pixels] = nodata
    return (
        np.ascontiguousarray(image),
        np.ascontiguousarray(mask),
        np.ascontiguousarray(nodata_pixels),
        augmented,
    )


def _geometric(
    image: np.ndarray,
    mask: np.ndarray,
    nodata_pixels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    augmented = False
    if rng.random() < 0.5:
        image = np.flip(image, axis=2)
        mask = np.flip(mask, axis=_mask_horizontal_axis(mask))
        nodata_pixels = np.flip(nodata_pixels, axis=1)
        augmented = True
    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=_mask_vertical_axis(mask))
        nodata_pixels = np.flip(nodata_pixels, axis=0)
        augmented = True

    rotations = int(rng.integers(0, 4))
    if rotations:
        image = np.rot90(image, rotations, axes=(1, 2))
        mask = np.rot90(mask, rotations, axes=_mask_rotation_axes(mask))
        nodata_pixels = np.rot90(nodata_pixels, rotations, axes=(0, 1))
        augmented = True
    return image, mask, nodata_pixels, augmented


def _photometric(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    valid_pixels: np.ndarray | None = None,
) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    original = image
    value_scale = _image_value_scale(image, valid_pixels)
    contrast = float(rng.uniform(0.85, 1.15))
    brightness = float(rng.uniform(-0.08, 0.08)) * value_scale
    image = image * contrast + brightness

    if rng.random() < 0.5:
        noise = rng.normal(0.0, 0.02 * value_scale, size=image.shape).astype(np.float32)
        image = image + noise
    if rng.random() < 0.3:
        image = (
            _mean_blur(image)
            if valid_pixels is None
            else _masked_mean_blur(image, valid_pixels)
        )
    if valid_pixels is not None:
        image = np.where(valid_pixels[None, :, :], image, original)
    return image.astype(np.float32, copy=False)


def _spectral(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    valid_pixels: np.ndarray | None = None,
) -> np.ndarray:
    original = image.astype(np.float32, copy=False)
    value = np.clip(image.astype(np.float32, copy=False) / 255.0, 0.0, 1.0)
    gamma = float(rng.uniform(0.9, 1.1))
    gains = rng.uniform(0.9, 1.1, size=(value.shape[0], 1, 1)).astype(np.float32)
    transformed = np.power(value, gamma).astype(np.float32, copy=False) * gains * 255.0
    if valid_pixels is not None:
        transformed = np.where(valid_pixels[None, :, :], transformed, original)
    return transformed


def _image_value_scale(
    image: np.ndarray,
    valid_pixels: np.ndarray | None = None,
) -> float:
    selected = image if valid_pixels is None else image[:, valid_pixels]
    finite = selected[np.isfinite(selected)]
    if finite.size == 0:
        return 1.0

    value_scale = float(finite.max() - finite.min())
    if value_scale > 0.0:
        return value_scale
    return max(float(np.max(np.abs(finite))), 1.0)


def _mean_blur(image: np.ndarray) -> np.ndarray:
    channels, height, width = image.shape
    padded = np.pad(image, ((0, 0), (1, 1), (1, 1)), mode="edge")
    blurred = np.zeros((channels, height, width), dtype=np.float32)
    for y_shift in range(3):
        for x_shift in range(3):
            blurred += padded[:, y_shift : y_shift + height, x_shift : x_shift + width]
    return blurred / 9.0


def _masked_mean_blur(image: np.ndarray, valid_pixels: np.ndarray) -> np.ndarray:
    channels, height, width = image.shape
    valid = valid_pixels.astype(np.float32, copy=False)
    weighted = image * valid[None, :, :]
    padded_values = np.pad(weighted, ((0, 0), (1, 1), (1, 1)), mode="edge")
    padded_valid = np.pad(valid, ((1, 1), (1, 1)), mode="edge")
    value_sum = np.zeros((channels, height, width), dtype=np.float32)
    valid_count = np.zeros((height, width), dtype=np.float32)
    for y_shift in range(3):
        for x_shift in range(3):
            value_sum += padded_values[
                :, y_shift : y_shift + height, x_shift : x_shift + width
            ]
            valid_count += padded_valid[
                y_shift : y_shift + height, x_shift : x_shift + width
            ]
    blurred = value_sum / np.maximum(valid_count[None, :, :], 1.0)
    return np.where(valid_pixels[None, :, :], blurred, image)


def _mask_horizontal_axis(mask: np.ndarray) -> int:
    return 1 if mask.ndim == 2 else 2


def _mask_vertical_axis(mask: np.ndarray) -> int:
    return 0 if mask.ndim == 2 else 1


def _mask_rotation_axes(mask: np.ndarray) -> tuple[int, int]:
    return (0, 1) if mask.ndim == 2 else (1, 2)


class _ValidityPreservingAugmentations:
    """Сохранить valid-mask геометрии, отменяя только небезопасный блок преобразований."""

    def __init__(self, geometry, photometry, seed: int, *, reject_empty_geometry: bool = False) -> None:
        self.geometry = geometry
        self.photometry = photometry
        self.reject_empty_geometry = reject_empty_geometry
        self.set_random_seed(seed)

    def set_random_seed(self, seed: int) -> None:
        self.geometry.set_random_seed(seed)
        self.photometry.set_random_seed(seed)

    @staticmethod
    def _validity_matches(image: np.ndarray, valid: np.ndarray) -> bool:
        return bool(np.all(np.isfinite(image[valid])) and np.all(np.any(image[valid] != 0, axis=-1)))

    def __call__(self, **sample):
        from .contracts import TilePreparationError

        original = sample["image"]
        if original.dtype != np.uint8 or "valid" not in sample:
            raise TilePreparationError("Защищённые аугментации требуют uint8 и исходную valid-mask")
        if not self._validity_matches(original, sample["valid"].astype(bool)):
            raise TilePreparationError("Исходная valid-mask содержит невалидные пиксели изображения")
        geometric = self.geometry(**sample)
        valid = geometric["valid"].astype(bool)
        geometry_reverted = not self._validity_matches(geometric["image"], valid)
        if self.reject_empty_geometry and np.any(sample["valid"]) and not np.any(valid):
            geometry_reverted = True
        if geometry_reverted:
            geometric = {**sample, "applied_transforms": []}
            valid = sample["valid"].astype(bool)
        baseline = geometric["image"].copy()
        baseline[~valid] = 0
        photometric = self.photometry(image=baseline.copy())
        photometry_reverted = not self._validity_matches(photometric["image"], valid)
        result = baseline if photometry_reverted else photometric["image"]
        result[~valid] = 0
        return {
            **geometric,
            "image": np.ascontiguousarray(result),
            "applied_transforms": [
                *geometric["applied_transforms"],
                *(photometric["applied_transforms"] if not photometry_reverted else []),
            ],
            "augmentation_guard": {"geometry_reverted": geometry_reverted, "photometry_reverted": photometry_reverted},
        }


def build_notebook_augmentations(
    seed: int, *, object_instances: bool = False, masked: bool = False, preserve_validity: bool = False,
    context_crop_size: int | None = None,
):
    """Преобразования uint8 до min-max, с общей геометрией изображения и маски."""
    import math
    import albumentations as A
    import cv2

    geometry = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_px=(-32, 32), rotate=(-15, 15),
                 border_mode=cv2.BORDER_CONSTANT if object_instances or masked else cv2.BORDER_REFLECT_101, p=0.4),
    ]
    if context_crop_size is not None:
        if not preserve_validity or context_crop_size <= 0:
            from .contracts import TilePreparationError

            raise TilePreparationError("Случайные вырезки требуют защиты valid-mask и положительного размера")
        # Только пиксели уже выбранного train-окна; соседние окна не читаются.
        geometry.insert(0, A.RandomResizedCrop(
            size=(context_crop_size, context_crop_size), scale=(0.25, 1.0),
            ratio=(1.0, 1.0), interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST, p=0.5,
        ))
    photometry = [
        A.OneOf([
            # Albumentations 2 принимает долю стандартного отклонения вместо var_limit.
            A.GaussNoise(std_range=(math.sqrt(5) / 255, math.sqrt(30) / 255), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
    ]
    targets = {"instances": "mask", "valid": "mask", "ambiguous": "mask"} if object_instances else {"valid": "mask"} if masked or preserve_validity else None
    if preserve_validity:
        return _ValidityPreservingAugmentations(
            A.Compose(geometry, seed=seed, strict=True, save_applied_params=True, additional_targets=targets),
            A.Compose(photometry, seed=seed, strict=True, save_applied_params=True), seed,
            reject_empty_geometry=context_crop_size is not None,
        )
    return A.Compose(geometry + photometry, seed=seed, strict=True, save_applied_params=True, additional_targets=targets)
