"""Счётчики ошибок фона без сохранения изображений и без повторного инференса."""

from __future__ import annotations

from mlsystem2.tile_preparation.contracts import AdaptiveSamplingController, SamplingFeedback


def adaptive_controller(loader: object) -> AdaptiveSamplingController | None:
    seen = set()
    while loader is not None and id(loader) not in seen:
        seen.add(id(loader))
        sampler = getattr(loader, "sampler", None)
        if isinstance(sampler, AdaptiveSamplingController):
            return sampler
        loader = getattr(loader, "loader", None)
    return None


class BackgroundErrors:
    def __init__(self) -> None:
        self.counts = SamplingFeedback().model_dump()

    def observe(self, torch, logits, masks, valid_pixels, class_negatives=None) -> list[float | None]:
        with torch.no_grad():
            expected = masks[:, 0] > 0 if masks.ndim == 4 else masks > 0
            valid = valid_pixels[:, 0] if valid_pixels.ndim == 4 else valid_pixels
            if class_negatives is not None:
                valid = valid & ~class_negatives.any(dim=1)
            if logits.shape[1] == 2:
                # Ровно тот же argmax, что в binary validation, включая почти равные float32 logits.
                predicted = logits.detach()[:, 1] > logits.detach()[:, 0]
            else:
                confidence, labels = logits.detach().softmax(dim=1).max(dim=1)
                predicted = (labels > 0) & (confidence > 0.5)
            valid_count = valid.flatten(1).sum(1)
            positive = (expected & valid).flatten(1).sum(1)
            tp = (predicted & expected & valid).flatten(1).sum(1)
            fp = (predicted & ~expected & valid).flatten(1).sum(1)
            rows = torch.stack((valid_count, positive, tp, fp), dim=1).cpu().tolist()
        scores = []
        for usable, positive, tp, fp in rows:
            self.counts["positive_pixels"] += positive
            self.counts["true_positive_pixels"] += tp
            if usable and not positive:
                self.counts["empty_tiles"] += 1
                self.counts["empty_pixels"] += usable
                self.counts["false_positive_pixels"] += fp
                self.counts["false_positive_tiles"] += int(fp >= max(4, (usable + 9999) // 10000))
                scores.append(fp / usable)
            else:
                scores.append(None)
        return scores

    def feedback(self) -> SamplingFeedback:
        return SamplingFeedback(**self.counts)
