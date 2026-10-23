"""Внутренние вычисления параметров next-gen конвейера."""

from __future__ import annotations

from typing import Any


IMAGENET_MEAN_RGB_RED_NIR = [0.485, 0.456, 0.406, 0.485]
IMAGENET_STD_RGB_RED_NIR = [0.229, 0.224, 0.225, 0.229]


def global_minmax_parameters(histogram: dict[str, object] | None, channels: int) -> dict[str, object]:
    """Зафиксировать общие границы только по допустимым train-окнам."""
    if not isinstance(histogram, dict) or histogram.get("source") != "retained_train_windows":
        raise ValueError("Глобальная нормализация требует статистику только train-тайлов")
    counts = histogram.get("counts")
    if not isinstance(counts, list) or len(counts) != channels:
        raise ValueError("Число каналов статистики глобальной нормализации не совпадает с моделью")
    low, high = [], []
    for channel in counts:
        if not isinstance(channel, list) or len(channel) != 256 or any(value < 0 for value in channel):
            raise ValueError("Некорректная статистика глобальной нормализации")
        occupied = [index for index, value in enumerate(channel) if value > 0]
        if not occupied:
            raise ValueError("Train не содержит допустимых пикселей для глобальной нормализации")
        low.append(float(occupied[0]))
        high.append(float(occupied[-1]))
    return {"mode": "global_minmax", "epsilon": 1e-6, "nodata": 0.0,
            "mask_output_nodata": True, "low": low, "high": high,
            "source": "retained_train_windows", "valid_pixel_count": int(histogram["valid_pixel_count"]),
            "window_count": int(histogram["window_count"])}


def preprocessing_parameters(
    mode: str,
    histogram: dict[str, object] | None,
) -> dict[str, object]:
    base: dict[str, object] = {"mode": mode, "nodata": 0.0}
    if mode == "scale_255":
        return base
    if mode == "imagenet_rgb_red_nir":
        return {
            **base,
            "mean": list(IMAGENET_MEAN_RGB_RED_NIR),
            "std": list(IMAGENET_STD_RGB_RED_NIR),
        }
    if mode != "robust_percentile":
        raise ValueError(f"Неизвестный preprocessing next_gen: {mode}")
    if not isinstance(histogram, dict):
        raise ValueError("robust_percentile требует histogram train pixels")
    raw_counts = histogram.get("counts")
    if not isinstance(raw_counts, list) or len(raw_counts) != 4:
        raise ValueError("robust_percentile требует histogram четырёх каналов")
    low = [_histogram_percentile(counts, 0.02) for counts in raw_counts]
    high = [_histogram_percentile(counts, 0.98) for counts in raw_counts]
    high = [max(low_value + 1.0, high_value) for low_value, high_value in zip(low, high)]
    return {
        **base,
        "low": low,
        "high": high,
        "percentiles": [0.02, 0.98],
        "valid_pixel_count": int(histogram.get("valid_pixel_count") or 0),
        "histogram_source": histogram.get("source"),
    }


def _histogram_percentile(raw_counts: Any, quantile: float) -> float:
    if not isinstance(raw_counts, list) or len(raw_counts) != 256:
        raise ValueError("Некорректная 256-bin histogram")
    counts = [max(0, int(value)) for value in raw_counts]
    total = sum(counts)
    if total <= 0:
        raise ValueError("Histogram не содержит valid pixels")
    target = max(1, int(total * quantile + 0.5))
    cumulative = 0
    for value, count in enumerate(counts):
        cumulative += count
        if cumulative >= target:
            return float(value)
    return 255.0
