"""Публичные контракты подготовки тайлов."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TilePreparationError(RuntimeError):
    """Ошибка подготовки тайлов."""


HARD_NEGATIVE_LABEL = -1


class SamplingFeedback(BaseModel):
    """Счётчики полной validation на фиксированном пороге 0,5, без идентификаторов тайлов."""

    model_config = ConfigDict(extra="forbid")

    empty_tiles: int = Field(default=0, ge=0)
    false_positive_tiles: int = Field(default=0, ge=0)
    empty_pixels: int = Field(default=0, ge=0)
    false_positive_pixels: int = Field(default=0, ge=0)
    positive_pixels: int = Field(default=0, ge=0)
    true_positive_pixels: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if (self.false_positive_tiles > self.empty_tiles
                or self.false_positive_pixels > self.empty_pixels
                or self.true_positive_pixels > self.positive_pixels):
            raise ValueError("Счётчики ошибок и верных обнаружений превышают соответствующий объём проверки")
        return self


@runtime_checkable
class AdaptiveSamplingController(Protocol):
    """Управление train-сэмплером; validation передаёт только агрегированные счётчики."""

    def set_epoch(self, epoch: int) -> None: ...

    def observe_train(self, indices: list[int], scores: list[float | None]) -> None: ...

    def update_feedback(self, feedback: SamplingFeedback) -> None: ...

    def snapshot(self, *, include_state: bool = False) -> dict: ...


class TileClassAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: int = Field(gt=0)
    slug: str
    name: str
    annotation_file: str | Path
    hard_negative_annotation_file: str | Path | None = None
    priority: int = 0


class TileClassDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_id: int = Field(gt=0)
    slug: str = Field(min_length=1)
    name: str = Field(min_length=1)
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    priority: int = 0


class TileSceneSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    image_path: str | Path
    annotation_file: str | Path | None = None
    footprint_file: str | Path | None = None
    parent_scene_id: str | None = None
    zone_id: str | None = None
    region_geometry: dict | None = None
    region_window: tuple[int, int, int, int] | None = None


class TileSplitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    val_fraction: float = Field(gt=0.0, lt=1.0)
    test_fraction: float = Field(default=0.0, ge=0.0, lt=1.0)
    seed: int = 42
    strategy: Literal["window_random", "scene_fold", "scene_groups"] = "window_random"
    validation_fold: int = Field(default=0, ge=0)
    spatial_purge: bool = False

    @model_validator(mode="after")
    def validate_fractions(self) -> Self:
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("Сумма долей validation и test должна быть меньше 1")
        return self


class TileDataloaderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenes: list[TileSceneSource] = Field(min_length=1)
    annotation_file: str | Path | None = None
    hard_negative_annotation_file: str | Path | None = None
    class_annotations: list[TileClassAnnotation] = Field(default_factory=list)
    classes: list[TileClassDefinition] = Field(default_factory=list)
    batch_size: int = Field(gt=0)
    mode: Literal["train", "val", "test"]
    tile_split: TileSplitRequest | None = None
    max_batches_per_epoch: int | None = Field(default=None, gt=0)
    include_object_instances: bool = False
    pipeline_variant: Literal["legacy", "next_gen", "next_gen2", "next_gen3", "next_gen4", "next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "next_gen4_consistency_v5", "next_gen4_consistency_v6", "object_f1", "object_f1_v2"] = "legacy"
    collect_band_histogram: bool = False
    input_channels: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_annotation_mode(self) -> Self:
        if self.pipeline_variant in {"next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "next_gen4_consistency_v5", "next_gen4_consistency_v6"} and self.collect_band_histogram and self.mode != "train":
            raise ValueError("Статистика глобальной нормализации вычисляется только на train")
        has_per_image = any(item.annotation_file is not None for item in self.scenes)
        if has_per_image and not all(item.annotation_file is not None for item in self.scenes):
            raise ValueError("Все per-image сцены должны содержать annotation_file")
        has_global_binary = self.annotation_file is not None
        has_legacy_multiclass = bool(self.class_annotations)
        has_per_image_multiclass = has_per_image and bool(self.classes)
        mode_count = sum((has_global_binary, has_legacy_multiclass, has_per_image))
        if mode_count != 1:
            raise ValueError(
                "TileDataloaderRequest должен задавать один режим: глобальный binary, "
                "legacy multiclass или per-image"
            )
        if has_per_image and (
            self.annotation_file is not None
            or self.hard_negative_annotation_file is not None
            or has_legacy_multiclass
        ):
            raise ValueError(
                "Per-image TileDataloaderRequest не смешивается с глобальной разметкой"
            )
        if self.classes and not has_per_image:
            raise ValueError("classes допустимы только для per-image разметки")
        if has_legacy_multiclass and self.hard_negative_annotation_file is not None:
            raise ValueError(
                "hard_negative_annotation_file задается в TileDataloaderRequest только для binary режима"
            )
        if self.include_object_instances and (
            (self.mode != "val" and self.pipeline_variant not in {"object_f1", "object_f1_v2"}) or has_legacy_multiclass or has_per_image_multiclass
        ):
            raise ValueError(
                "include_object_instances поддерживается только для binary val loader"
            )
        if self.pipeline_variant == "next_gen":
            if self.tile_split is None or self.tile_split.strategy != "scene_fold":
                raise ValueError("next_gen loader требует tile_split strategy=scene_fold")
            if self.mode == "val" and self.max_batches_per_epoch is not None:
                raise ValueError("next_gen val loader не допускает ограничение числа batch")
        if self.pipeline_variant in {"object_f1", "object_f1_v2"}:
            if self.tile_split is None or self.tile_split.strategy != "scene_groups" or self.tile_split.val_fraction != 0.2 or self.tile_split.test_fraction != 0.2:
                raise ValueError("object f1 требует разбиение групп снимков 60/20/20")
            if has_legacy_multiclass or has_per_image_multiclass or self.input_channels not in (3, 4):
                raise ValueError("object f1 требует binary и вход RGB или RGB+NIR")
            if not self.include_object_instances or self.max_batches_per_epoch is not None:
                raise ValueError("object f1 требует маски объектов и полные эпохи")
        elif self.pipeline_variant in {"next_gen2", "next_gen3", "next_gen4", "next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "next_gen4_consistency_v5", "next_gen4_consistency_v6"}:
            if self.tile_split is None or self.tile_split.strategy != "window_random":
                raise ValueError("next-gen2 требует разбиение по тайлам")
            if self.tile_split.spatial_purge or self.tile_split.test_fraction != 0.2 or self.tile_split.val_fraction != 0.2:
                raise ValueError("next-gen2 требует случайное разбиение 60/20/20 без исключения пересечений")
            if self.max_batches_per_epoch is not None:
                raise ValueError("next-gen2 не допускает ограничение числа пакетов")
        elif self.mode == "test" or (self.tile_split and self.tile_split.test_fraction):
            raise ValueError("Внутренний test split доступен только в next-gen2")
        return self


__all__ = [
    "AdaptiveSamplingController",
    "HARD_NEGATIVE_LABEL",
    "SamplingFeedback",
    "TileClassAnnotation",
    "TileClassDefinition",
    "TileDataloaderRequest",
    "TilePreparationError",
    "TileSceneSource",
    "TileSplitRequest",
]
