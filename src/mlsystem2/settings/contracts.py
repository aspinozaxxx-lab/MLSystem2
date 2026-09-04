"""Публичные контракты настроек."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SettingsError(RuntimeError):
    """Ошибка загрузки или валидации настроек."""


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_root: str
    scratch_root: str
    logs_root: str
    cleanup_scratch_after_mlflow_log: bool


class DatasetClassSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str
    name: str
    scenes_file: str
    annotation_file: str
    hard_negative_annotation_file: str | None = None
    priority: int = 0


class DatasetSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    images_dir: str
    scenes_file: str | None = None
    annotation_file: str | None = None
    hard_negative_annotation_file: str | None = None
    annotations_dir: str | None = None
    classes: list[DatasetClassSettings] = Field(default_factory=list)
    val_fraction: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_dataset_mode(self) -> Self:
        has_legacy_binary_paths = (
            self.scenes_file is not None
            or self.annotation_file is not None
            or self.hard_negative_annotation_file is not None
        )
        has_per_image_binary = self.annotations_dir is not None
        has_classes = bool(self.classes)
        mode_count = sum((has_legacy_binary_paths, has_per_image_binary, has_classes))
        if mode_count != 1:
            raise ValueError(
                "dataset должен задавать ровно один режим: classes, "
                "scenes_file + annotation_file или annotations_dir"
            )
        if has_classes:
            _validate_unique_values([item.slug for item in self.classes], "slug")
            _validate_unique_values([item.name for item in self.classes], "name")
            return self
        if has_per_image_binary:
            if not self.annotations_dir:
                raise ValueError("annotations_dir не должен быть пустым")
            return self
        if not self.scenes_file or not self.annotation_file:
            raise ValueError("binary dataset должен задавать scenes_file и annotation_file")
        return self

    @property
    def is_multiclass(self) -> bool:
        return bool(self.classes)


class TilePreparationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tile_size: int = Field(gt=0)
    context: int = Field(default=0, ge=0)
    stride: int = Field(gt=0)
    num_workers: int = Field(default=16, ge=0)
    prefetch_epochs: float = Field(default=2.0, gt=0.0)
    seed: int = 42
    augmentation_level: int = Field(default=0, ge=0, le=3)
    positive_factor: float = Field(default=0.5, ge=0.0, le=1.0)
    hard_negative_factor: float = Field(default=0.0, ge=0.0, le=1.0)
    background_factor: float = Field(ge=0.0, le=1.0)
    val_positive_factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    class_balance: bool = False

    @model_validator(mode="before")
    @classmethod
    def resolve_background_factor(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "background_factor" in data:
            return data
        resolved = dict(data)
        positive_factor = float(resolved.get("positive_factor", 0.5))
        hard_negative_factor = float(resolved.get("hard_negative_factor", 0.0))
        resolved["background_factor"] = 1.0 - positive_factor - hard_negative_factor
        return resolved

    @model_validator(mode="after")
    def validate_tile_settings(self) -> Self:
        if self.tile_size <= 2 * self.context:
            raise ValueError("tile_size должен быть больше удвоенного context")
        if self.stride > self.tile_size:
            raise ValueError("stride должен быть меньше или равен tile_size")
        factor_sum = self.positive_factor + self.hard_negative_factor + self.background_factor
        if abs(factor_sum - 1.0) > 1e-6:
            raise ValueError(
                "Сумма positive_factor, hard_negative_factor и background_factor должна быть равна 1.0"
            )
        if (
            self.positive_factor == 0.0
            and self.hard_negative_factor == 0.0
            and self.background_factor == 0.0
        ):
            raise ValueError("Хотя бы один tile factor должен быть больше 0")
        return self


class TrainSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["binary", "multiclass"] = "binary"
    quality_metric: Literal["pixel", "objects"] = "pixel"
    pipeline_variant: Literal["legacy", "next_gen"] = "legacy"
    model_name: str
    input_channels: int = Field(default=4, gt=0)
    output_channels: int = Field(default=1, gt=0)
    pretrained: bool = False
    initial_checkpoint_uri: str | None = None
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    device: str = "cuda"
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    loss: Literal["bce_dice", "focal_dice", "focal_tversky", "cross_entropy", "cross_entropy_dice"]
    focal_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    pos_weight: float = Field(default=1.0, gt=0.0)
    background_weight: float = Field(default=1.0, gt=0.0)
    hard_negative_weight: float = Field(default=1.0, gt=0.0)
    tversky_alpha: float = Field(default=0.4, gt=0.0)
    tversky_beta: float = Field(default=0.6, gt=0.0)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    early_stopping_patience: int = Field(gt=0)
    max_train_batches_per_epoch: int | None = Field(default=None, gt=0)
    max_val_batches_per_epoch: int | None = Field(default=None, gt=0)
    max_training_time_sec: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_task_loss(self) -> Self:
        multiclass_losses = {"cross_entropy", "cross_entropy_dice"}
        if self.task == "multiclass" and self.loss not in multiclass_losses:
            raise ValueError("multiclass train требует loss=cross_entropy или cross_entropy_dice")
        if self.task == "binary" and self.loss in multiclass_losses:
            raise ValueError("binary train не поддерживает multiclass loss")
        if self.task != "binary" and self.quality_metric == "objects":
            raise ValueError("Объектовая метрика качества поддерживается только для binary train")
        return self


class NextGenSettings(BaseModel):
    """Параметры альтернативного конвейера обучения."""

    model_config = ConfigDict(extra="forbid")

    validation_fold: int = Field(default=0, ge=0)
    normalization: Literal[
        "scale_255",
        "imagenet_rgb_red_nir",
        "robust_percentile",
    ] = "scale_255"
    validation_interval_epochs: int = Field(default=5, gt=0)
    threshold_mode: Literal["fixed", "optimize"] = "optimize"
    evaluate_gaussian_blend: bool = False


class InferenceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint_uri: str | None = None
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    batch_size: int = Field(default=1, gt=0)
    device: str = "cuda"


class MLflowSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    tracking_uri: str
    experiment_name: str


class SystemSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime: RuntimeSettings
    dataset: DatasetSettings
    tile_preparation: TilePreparationSettings
    train: TrainSettings
    next_gen: NextGenSettings = Field(default_factory=NextGenSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    mlflow: MLflowSettings

    @model_validator(mode="after")
    def validate_dataset_train_consistency(self) -> Self:
        if self.dataset.is_multiclass:
            if self.train.task != "multiclass":
                raise ValueError("dataset.classes требует train.task=multiclass")
            expected_channels = len(self.dataset.classes) + 1
            if self.train.output_channels != expected_channels:
                raise ValueError(
                    "multiclass output_channels должен быть равен "
                    f"len(dataset.classes) + 1: ожидается {expected_channels}"
                )
        elif self.dataset.annotations_dir is not None:
            if self.train.task == "binary" and self.train.output_channels != 1:
                raise ValueError("binary per-image dataset требует output_channels=1")
            if self.train.task == "multiclass" and self.train.output_channels < 3:
                raise ValueError("multiclass per-image dataset требует минимум 3 output_channels")
        elif self.train.task != "binary":
            raise ValueError("binary dataset требует train.task=binary")
        if self.train.pipeline_variant == "next_gen":
            if self.train.task != "binary":
                raise ValueError("next_gen v1 поддерживает только binary train")
            if self.train.model_name not in {"segformer_b0", "smp_segformer_b0"}:
                raise ValueError(
                    "next_gen v1 поддерживает только segformer_b0 и smp_segformer_b0"
                )
            if self.train.input_channels != 4 or self.train.output_channels != 1:
                raise ValueError(
                    "next_gen v1 требует четыре входных канала и один выходной канал"
                )
            if self.train.pretrained and self.train.model_name != "segformer_b0":
                raise ValueError(
                    "pretrained-веса next_gen поддерживаются только для HF segformer_b0"
                )
            if self.train.max_val_batches_per_epoch is not None:
                raise ValueError(
                    "next_gen требует полную validation: max_val_batches_per_epoch должен быть null"
                )
            if self.next_gen.evaluate_gaussian_blend and (
                self.tile_preparation.tile_size != 512
                or self.tile_preparation.stride != 256
            ):
                raise ValueError("Gaussian A/B требует tile_size=512 и stride=256")
        return self


def _validate_unique_values(values: list[str], field_name: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        joined = ", ".join(sorted(duplicates))
        raise ValueError(f"dataset.classes должен иметь уникальные {field_name}: {joined}")


__all__ = [
    "DatasetClassSettings",
    "DatasetSettings",
    "InferenceSettings",
    "MLflowSettings",
    "NextGenSettings",
    "RuntimeSettings",
    "SettingsError",
    "SystemSettings",
    "TilePreparationSettings",
    "TrainSettings",
]
