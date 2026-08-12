"""Внутренние пресеты аугментаций тайлов."""

from __future__ import annotations

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
) -> tuple[np.ndarray, np.ndarray, bool]:
    if level <= 0:
        return np.ascontiguousarray(image), np.ascontiguousarray(mask), False

    rng = np.random.default_rng(seed + sample_index)
    image, mask, nodata_pixels, augmented = _geometric(
        image,
        mask,
        nodata_pixels,
        rng,
    )

    if level >= 2:
        image = _photometric(image, rng)
        augmented = True

    image = np.clip(image, 0.0, 255.0)
    image[:, nodata_pixels] = nodata
    return np.ascontiguousarray(image), np.ascontiguousarray(mask), augmented


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


def _photometric(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    value_scale = _image_value_scale(image)
    contrast = float(rng.uniform(0.85, 1.15))
    brightness = float(rng.uniform(-0.08, 0.08)) * value_scale
    image = image * contrast + brightness

    if rng.random() < 0.5:
        noise = rng.normal(0.0, 0.02 * value_scale, size=image.shape).astype(np.float32)
        image = image + noise
    if rng.random() < 0.3:
        image = _mean_blur(image)
    return image.astype(np.float32, copy=False)


def _image_value_scale(image: np.ndarray) -> float:
    finite = image[np.isfinite(image)]
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


def _mask_horizontal_axis(mask: np.ndarray) -> int:
    return 1 if mask.ndim == 2 else 2


def _mask_vertical_axis(mask: np.ndarray) -> int:
    return 0 if mask.ndim == 2 else 1


def _mask_rotation_axes(mask: np.ndarray) -> tuple[int, int]:
    return (0, 1) if mask.ndim == 2 else (1, 2)
