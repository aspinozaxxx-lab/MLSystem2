"""Отдельный аудит окон, почти целиком заполненных объектами, без подбора порога."""

from __future__ import annotations


class DenseWindowMetrics:
    """Микро-счётчики объединения объектов внутри исходных полных окон."""

    def __init__(self) -> None:
        self.groups = {
            str(level): {
                "window_count": 0,
                "thresholds": {str(t): {"tp": 0, "fp": 0, "fn": 0} for t in (0.5, 0.9)},
            }
            for level in (0.9, 0.99)
        }

    def observe(self, torch, probabilities, masks, valid_pixels, class_negatives=None) -> None:
        # Бинарная вероятность объекта либо сумма вероятностей всех не-фоновых классов.
        if masks.ndim == 4:
            masks = masks[:, 0]
        if valid_pixels is None:
            valid = torch.ones_like(masks, dtype=torch.bool)
        else:
            valid = valid_pixels[:, 0] if valid_pixels.ndim == 4 else valid_pixels
        if class_negatives is not None:
            valid = valid & ~class_negatives.any(dim=1)
        true = masks > 0
        valid_counts = valid.flatten(1).sum(1)
        positive_counts = (true & valid).flatten(1).sum(1)
        area = int(masks.shape[-2] * masks.shape[-1])
        for level, group in self.groups.items():
            selected = (valid_counts >= 0.99 * area) & (positive_counts >= float(level) * valid_counts)
            window_count = int(selected.sum().item())
            group["window_count"] += window_count
            if not window_count:
                continue
            usable, expected, scores = valid[selected], true[selected], probabilities[selected]
            for threshold, counts in group["thresholds"].items():
                predicted = scores > float(threshold)
                counts["tp"] += int((predicted & expected & usable).sum().item())
                counts["fp"] += int((predicted & ~expected & usable).sum().item())
                counts["fn"] += int((~predicted & expected & usable).sum().item())

    def snapshot(self) -> dict:
        groups = {}
        for level, group in self.groups.items():
            thresholds = {}
            for threshold, counts in group["thresholds"].items():
                tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
                thresholds[threshold] = {
                    **counts,
                    "precision": tp / (tp + fp) if tp + fp else None,
                    "recall": tp / (tp + fn) if tp + fn else None,
                    "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                }
            groups[level] = {"window_count": group["window_count"], "thresholds": thresholds}
        return {"target": "foreground_union", "comparison": "strict_more", "min_valid_fraction": 0.99, "groups": groups}
