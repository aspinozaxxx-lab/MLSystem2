"""Воспроизводимая ротация фона и ограниченное управление его долей в next-gen3."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import random

from .contracts import SamplingFeedback, TilePreparationError


class AdaptiveBackgroundSampler:
    """Все индексы принадлежат только уже выделенной train-части."""

    POLICY = {
        "version": "adaptive_background_v1", "positive_pixels_min": 1,
        "background_per_positive": 8, "background_floor": 256,
        "background_per_scene": 32, "positive_repeats_max": 4,
        "epoch_draws_per_informative_max": 20, "background_fraction_min": 0.25,
        "hard_fraction_max": 0.25, "initial_background_fraction": 0.5,
        "background_fraction_max": 0.9, "fraction_step": 0.05,
        "ema_alpha": 0.3, "warmup_epochs": 2,
        "feedback_threshold": 0.5, "empty_pixel_fpr_high": 0.001,
        "empty_tile_fpr_high": 0.1, "recall_drop_guard": 0.05,
        "recall_reference_requires_controlled_background": True,
        "false_alarm_pixels_min": 4, "false_alarm_valid_fraction": 0.0001,
        "split_warning": "Оконное разбиение 60/20/20 не является независимой территориальной оценкой",
    }

    def __init__(self, tiles: list[dict], *, seed: int) -> None:
        policy = self.POLICY
        self.seed = seed
        self.epoch = 0
        self.positive = [i for i, t in enumerate(tiles) if t["positive"] and t["valid_pixels"]]
        # Частично отрицательная multiclass-разметка не является подтверждённым фоном.
        partial = [i for i, t in enumerate(tiles) if not t["positive"] and t.get("partial_negative") and t["valid_pixels"]]
        self.informative = self.positive + partial
        self.background = [i for i, t in enumerate(tiles) if not t["positive"] and not t.get("partial_negative") and t["valid_pixels"]]
        if not self.positive:
            raise TilePreparationError("next-gen3: в train нет ни одного допустимого пикселя объекта")
        self._background_set = set(self.background)
        self._declared_hard = {i for i in self.background if tiles[i].get("hard_negative")}
        self._groups: dict[str, list[int]] = defaultdict(list)
        self._cursors: dict[str, int] = {}
        for i in self.background:
            self._groups[str(tiles[i]["group"])].append(i)
        for key, values in self._groups.items():
            random.Random(self._seed("group", key)).shuffle(values)
            self._cursors[key] = 0
        scenes = len({str(t["scene_id"]) for t in tiles if t["valid_pixels"]})
        p = len(self.informative)
        self.excluded_invalid = len(tiles) - len(self.informative) - len(self.background)
        self.background_cap = min(len(self.background), max(
            policy["background_per_positive"] * p, policy["background_floor"], policy["background_per_scene"] * scenes,
        ))
        # Очень редкие объекты: большой резерв фона не должен требовать десятков повторов объекта
        # или эпохи с 99% фона. Резерв ротируется между более короткими эпохами.
        self.num_samples = min(p + self.background_cap, policy["epoch_draws_per_informative_max"] * p)
        self._min_bg = min(self.background_cap, max(
            1, math.ceil(policy["background_fraction_min"] * self.num_samples),
            self.num_samples - policy["positive_repeats_max"] * p,
        )) if self.background else 0
        self._max_bg = min(self.background_cap, self.num_samples - p,
                           math.floor(policy["background_fraction_max"] * self.num_samples))
        # Если фона мало, берём весь, не выбрасывая положительные примеры.
        if len(self.background) < p:
            self._min_bg = self._max_bg = len(self.background)
        self._bg_count = min(self._max_bg, max(self._min_bg, round(policy["initial_background_fraction"] * self.num_samples)))
        self._scores: dict[int, float] = {}
        self._ema: dict[str, float] = {}
        self._best_recall: float | None = None
        self._reason = "Прогрев"
        self._history: list[dict] = []
        self._selected: dict = {}
        self._plan: list[tuple[int, int, int]] = []

    def _seed(self, *parts: object) -> int:
        value = "\0".join(map(str, (self.seed, *parts)))
        return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if epoch == self.epoch:
            return
        if epoch != self.epoch + 1:
            raise TilePreparationError("next-gen3: эпохи сэмплера должны идти последовательно")
        self.epoch = epoch
        rng = random.Random(self._seed("epoch", epoch))
        hard = sorted(
            (i for i in self.background if i in self._declared_hard or self._scores.get(i, 0) > 0),
            key=lambda i: (-self._scores.get(i, 0), self._seed("hard", epoch, i)),
        )[:math.floor(self._bg_count * self.POLICY["hard_fraction_max"])]
        background = list(hard)
        chosen = set(background)
        groups = sorted(self._groups)
        rng.shuffle(groups)
        # Не более одного тайла зоны за проход; непрерывные курсоры обходят весь фон.
        while len(background) < self._bg_count:
            progressed = False
            for key in groups:
                values = self._groups[key]
                for _ in range(len(values)):
                    cursor = self._cursors[key]
                    i = values[cursor % len(values)]
                    self._cursors[key] = cursor + 1
                    if i not in chosen:
                        background.append(i)
                        chosen.add(i)
                        progressed = True
                        break
                if len(background) == self._bg_count:
                    break
            if not progressed:
                raise TilePreparationError("next-gen3: недостаточно уникальных фоновых тайлов")
        foreground = []
        needed = self.num_samples - len(background)
        while len(foreground) < needed:
            cycle = list(self.informative)
            rng.shuffle(cycle)
            foreground.extend(cycle[:needed - len(foreground)])
        indices = foreground + background
        rng.shuffle(indices)
        self._plan = [(i, epoch, draw) for draw, i in enumerate(indices)]
        repetitions = Counter(foreground)
        self._selected = {
            "epoch": epoch, "draws": len(indices), "foreground_draws": len(foreground),
            "background_draws": len(background), "background_fraction": len(background) / len(indices),
            "unique_foreground": len(repetitions), "unique_background": len(chosen),
            "positive_repeats_max_observed": max(repetitions.values(), default=0),
            "hard_background_draws": len(hard),
            "background_groups": len({str(k) for k, v in self._groups.items() if any(i in chosen for i in v)}),
            "selection_sha256": hashlib.sha256(
                ",".join(map(str, indices)).encode()
            ).hexdigest(),
        }

    def __iter__(self):
        if not self._plan:
            self.set_epoch(1)
        return iter(self._plan)

    def observe_train(self, indices: list[int], scores: list[float | None]) -> None:
        if len(indices) != len(scores):
            raise TilePreparationError("next-gen3: число индексов не совпадает с числом оценок фона")
        for i, score in zip(indices, scores):
            if i not in self._background_set or score is None:
                continue
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise TilePreparationError("next-gen3: некорректная оценка ошибки фона")
            alpha = self.POLICY["ema_alpha"]
            self._scores[i] = (1 - alpha) * self._scores.get(i, score) + alpha * score

    def update_feedback(self, feedback: SamplingFeedback) -> None:
        if not self._plan:
            raise TilePreparationError("next-gen3: обратная связь получена до выбора эпохи")
        policy = self.POLICY
        if self._history and self._history[-1]["epoch"] == self.epoch:
            raise TilePreparationError("next-gen3: повторная обратная связь одной эпохи")
        values = {}
        if feedback.empty_pixels and feedback.empty_tiles:
            values["empty_pixel_fpr"] = feedback.false_positive_pixels / feedback.empty_pixels
            values["empty_tile_fpr"] = feedback.false_positive_tiles / feedback.empty_tiles
        if feedback.positive_pixels:
            values["foreground_recall"] = feedback.true_positive_pixels / feedback.positive_pixels
        for key, value in values.items():
            self._ema[key] = (1 - policy["ema_alpha"]) * self._ema.get(key, value) + policy["ema_alpha"] * value
        delta = 0
        self._reason = "Доля сохранена: ошибки в допустимой полосе"
        if self.epoch <= policy["warmup_epochs"]:
            self._reason = "Прогрев: первые две эпохи без корректировки"
        elif len(values) != 3:
            self._reason = "Доля сохранена: в validation недостаточно фона или объектов"
        elif self._best_recall is not None and self._ema["foreground_recall"] < self._best_recall - policy["recall_drop_guard"]:
            delta = -1
            self._reason = "Уменьшение фона: сработала защита полноты объектов"
        elif self._ema["empty_pixel_fpr"] > policy["empty_pixel_fpr_high"] or self._ema["empty_tile_fpr"] > policy["empty_tile_fpr_high"]:
            delta = 1
            self._reason = "Увеличение фона: ложные срабатывания на пустых тайлах"
        elif self._ema["empty_pixel_fpr"] < policy["empty_pixel_fpr_high"] / 2 and self._ema["empty_tile_fpr"] < policy["empty_tile_fpr_high"] / 2:
            delta = -1
            self._reason = "Уменьшение фона: устойчиво мало ложных срабатываний"
        # Почти сплошной foreground даёт искусственную полноту около 1, но не является
        # рабочей точкой. Такой прогрев не должен навсегда блокировать борьбу с FP.
        if (len(values) == 3 and self._ema["empty_pixel_fpr"] <= policy["empty_pixel_fpr_high"]
                and self._ema["empty_tile_fpr"] <= policy["empty_tile_fpr_high"]):
            self._best_recall = max(self._best_recall or 0, self._ema["foreground_recall"])
        before = self._bg_count
        step = max(1, round(policy["fraction_step"] * self.num_samples))
        self._bg_count = min(self._max_bg, max(self._min_bg, before + delta * step))
        if delta and self._bg_count == before:
            self._reason += "; достигнут предел квоты или повторов"
        self._history.append({
            **self._selected, "feedback": feedback.model_dump(), "rates": values,
            "ema": dict(self._ema), "next_background_draws": self._bg_count,
            "next_background_fraction": self._bg_count / self.num_samples,
            "reason": self._reason,
        })

    def snapshot(self, *, include_state: bool = False) -> dict:
        result = {
            "policy": dict(self.POLICY), "seed": self.seed,
            "pool": {"positive": len(self.positive), "informative": len(self.informative),
                     "background": len(self.background), "background_cap": self.background_cap,
                     "excluded_invalid": self.excluded_invalid,
                     "background_draws_min": self._min_bg, "background_draws_max": self._max_bg,
                     "rare_object_budget_limited": self.num_samples < len(self.informative) + self.background_cap,
                     "draws_per_epoch": self.num_samples, "background_groups": len(self._groups)},
            "current": self._history[-1] if self._history else dict(self._selected),
        }
        if include_state:
            result.update(history=list(self._history), state={
                "epoch": self.epoch, "next_background_draws": self._bg_count,
                "cursors": dict(self._cursors), "train_scores": dict(self._scores),
                "ema": dict(self._ema), "best_recall": self._best_recall,
            })
        return result
