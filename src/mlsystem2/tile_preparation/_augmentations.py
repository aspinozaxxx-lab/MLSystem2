"""Внутренние пресеты аугментаций тайлов."""

from __future__ import annotations

import numpy as np


def apply_augmentations(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    level: int,
    seed: int,
    sample_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if level <= 0:
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    rng = np.random.default_rng(seed + sample_index)
    image, mask = _geometric(image, mask, rng)

    if level >= 2:
        image = _photometric(image, rng)
    if level >= 3:
        image = _cutout(image, rng)

    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def _geometric(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = np.flip(image, axis=2)
        mask = np.flip(mask, axis=2)
    if rng.random() < 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)

    rotations = int(rng.integers(0, 4))
    if rotations:
        image = np.rot90(image, rotations, axes=(1, 2))
        mask = np.rot90(mask, rotations, axes=(1, 2))
    return image, mask


def _photometric(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    contrast = float(rng.uniform(0.85, 1.15))
    brightness = float(rng.uniform(-0.08, 0.08))
    image = image * contrast + brightness

    gamma = float(rng.uniform(0.85, 1.15))
    image = np.power(np.clip(image, 0.0, 1.0), gamma).astype(np.float32, copy=False)

    if rng.random() < 0.5:
        image = image + rng.normal(0.0, 0.02, size=image.shape).astype(np.float32)
    if rng.random() < 0.3:
        image = _mean_blur(image)
    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def _mean_blur(image: np.ndarray) -> np.ndarray:
    channels, height, width = image.shape
    padded = np.pad(image, ((0, 0), (1, 1), (1, 1)), mode="edge")
    blurred = np.zeros((channels, height, width), dtype=np.float32)
    for y_shift in range(3):
        for x_shift in range(3):
            blurred += padded[:, y_shift : y_shift + height, x_shift : x_shift + width]
    return blurred / 9.0


def _cutout(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    _, height, width = image.shape
    max_h = max(1, height // 4)
    max_w = max(1, width // 4)
    cut_h = int(rng.integers(1, max_h + 1))
    cut_w = int(rng.integers(1, max_w + 1))
    y = int(rng.integers(0, height - cut_h + 1))
    x = int(rng.integers(0, width - cut_w + 1))
    image = image.copy()
    image[:, y : y + cut_h, x : x + cut_w] = 0.0
    return image
