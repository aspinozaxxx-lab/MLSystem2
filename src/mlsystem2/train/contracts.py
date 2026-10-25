"""Публичные контракты обучения."""

from __future__ import annotations

from typing import Any, Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlsystem2.models.contracts import ModelHandle
from mlsystem2.inference.contracts import ObjectSeparationConfig


class TrainError(RuntimeError):
    """Ошибка обучения."""


class TrainClassDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(gt=0)
    slug: str = Field(min_length=1)
    name: str = Field(min_length=1)
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    priority: int = 0


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["binary", "multiclass"] = "binary"
    quality_metric: Literal["pixel", "objects"] = "pixel"
    pipeline_variant: Literal["legacy", "next_gen", "next_gen2", "next_gen3", "next_gen4", "next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "object_f1", "object_f1_v2"] = "legacy"
    class_weights: list[float] = Field(default_factory=list)
    object_separation: ObjectSeparationConfig | None = None
    calibrate_object_threshold: bool = True
    validation_interval_epochs: int = Field(default=1, gt=0)
    threshold_mode: Literal["fixed", "optimize"] = "optimize"
    evaluate_gaussian_blend: bool = False
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    seed: int = 42
    inference_context: int = Field(default=0, ge=0)
    device: str
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    loss: Literal["bce_dice", "focal_dice", "focal_tversky", "cross_entropy", "cross_entropy_dice", "cross_entropy_tversky"]
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
    diagnostic_checkpoint_interval_sec: int | None = Field(default=None, gt=0)
    class_slugs: list[str] = Field(default_factory=list)
    class_schema: list[TrainClassDefinition] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_task_loss(self) -> Self:
        if self.object_separation is not None and (
            self.pipeline_variant not in {"object_f1", "object_f1_v2"}
            or self.object_separation.profile != self.pipeline_variant
        ):
            raise ValueError("Профиль разделения объектов должен совпадать с конвейером обучения")
        if self.pipeline_variant == "object_f1" and self.object_separation is not None:
            raise ValueError("Исторический object f1 использует неизменные параметры разделения")
        if self.pipeline_variant in {"object_f1", "object_f1_v2"} and self.quality_metric != "objects":
            raise ValueError("object f1 требует объектовую метрику")
        multiclass_losses = {"cross_entropy", "cross_entropy_dice"}
        if self.task == "multiclass" and self.loss not in multiclass_losses | {"cross_entropy_tversky"}:
            raise ValueError("multiclass train требует loss=cross_entropy или cross_entropy_dice")
        if self.pipeline_variant in {"next_gen2", "next_gen3", "next_gen4", "next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "object_f1", "object_f1_v2"}:
            if self.loss != "cross_entropy_tversky":
                raise ValueError("next-gen2 требует CrossEntropy + Tversky")
            if self.pipeline_variant in {"object_f1", "object_f1_v2"} and self.task != "binary":
                raise ValueError("object f1 требует бинарную разметку")
            expected_weights = (3 if self.pipeline_variant == "object_f1_v2" else 2) if self.task == "binary" else 1 + len(self.class_schema or self.class_slugs)
            if len(self.class_weights) != expected_weights or any(
                not (0 < value < float("inf")) for value in self.class_weights
            ):
                raise ValueError("next-gen2 требует положительные конечные веса всех классов, включая фон")
            if (
                self.max_train_batches_per_epoch is not None
                or self.max_val_batches_per_epoch is not None
                or self.inference_context != 0
                or self.threshold != 0.5
                or self.validation_interval_epochs != 1
            ):
                raise ValueError("next-gen2 требует полные эпохи, порог 0.5 и контекст 0")
        elif self.loss == "cross_entropy_tversky":
            raise ValueError("CrossEntropy + Tversky доступен только в next-gen2")
        elif self.task == "binary" and self.loss in multiclass_losses:
            raise ValueError("binary train не поддерживает multiclass loss")
        if self.task != "binary" and self.quality_metric == "objects":
            raise ValueError("Объектовая метрика качества поддерживается только для binary train")
        if self.task == "multiclass":
            if not self.class_schema and self.class_slugs:
                self.class_schema = [
                    TrainClassDefinition(
                        id=index,
                        slug=slug,
                        name=slug,
                        color="#808080",
                    )
                    for index, slug in enumerate(self.class_slugs, start=1)
                ]
            if not self.class_schema:
                raise ValueError("multiclass train требует непустую схему классов")
            ids = sorted(item.id for item in self.class_schema)
            if ids != list(range(1, len(self.class_schema) + 1)):
                raise ValueError("class_schema должен использовать последовательные id от 1")
            if len({item.slug for item in self.class_schema}) != len(self.class_schema):
                raise ValueError("class_schema должен содержать уникальные slug")
            self.class_slugs = [
                item.slug for item in sorted(self.class_schema, key=lambda item: item.id)
            ]
        elif self.class_schema or self.class_slugs:
            raise ValueError("binary train не должен содержать class_schema")
        if self.pipeline_variant == "next_gen":
            if self.task != "binary":
                raise ValueError("next_gen v1 поддерживает только binary train")
            if self.max_val_batches_per_epoch is not None:
                raise ValueError("next_gen требует полную validation без max_val_batches")
        return self


class EpochMetrics(BaseModel):
    adaptive_sampling: dict = Field(default_factory=dict)
    val_dense_window_metrics: dict[str, Any] = Field(default_factory=dict)
    train_context_consistency: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def fill_quality_metric_compatibility(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        resolved = dict(data)
        if resolved.get("validation_performed", True) is False:
            return resolved
        resolved.setdefault(
            "val_quality_f1",
            resolved.get("val_best_threshold_pixel_f1", 0.0),
        )
        resolved.setdefault(
            "val_quality_precision",
            resolved.get("val_best_threshold_precision", 0.0),
        )
        resolved.setdefault(
            "val_quality_recall",
            resolved.get("val_best_threshold_recall", 0.0),
        )
        resolved.setdefault(
            "val_best_pixel_threshold",
            resolved.get("val_best_threshold", 0.0),
        )
        resolved.setdefault(
            "val_best_threshold_pixel_precision",
            resolved.get("val_best_threshold_precision", 0.0),
        )
        resolved.setdefault(
            "val_best_threshold_pixel_recall",
            resolved.get("val_best_threshold_recall", 0.0),
        )
        for key in (
            "val_best_threshold",
            "val_best_threshold_pixel_f1",
            "val_best_threshold_precision",
            "val_best_threshold_recall",
        ):
            resolved.setdefault(key, 0.0)
        return resolved

    epoch: int = Field(ge=0)
    validation_performed: bool = True
    train_region_loss: float | None = None
    train_boundary_loss: float | None = None
    val_region_loss: float | None = None
    val_boundary_loss: float | None = None
    train_loss: float = Field(ge=0.0)
    val_loss: float | None = Field(default=None, ge=0.0)
    quality_metric: Literal["pixel", "objects"] = "pixel"
    val_quality_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    val_quality_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    val_quality_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    val_best_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    val_best_pixel_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    val_best_threshold_pixel_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    val_best_threshold_pixel_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    val_best_threshold_pixel_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    val_best_threshold_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    val_best_threshold_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    val_best_threshold_object_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    val_best_threshold_object_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    val_best_threshold_object_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    val_macro_pixel_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    val_macro_pixel_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    val_macro_pixel_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    val_macro_pixel_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    val_micro_pixel_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    val_micro_pixel_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    val_micro_pixel_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    val_fixed_0_5_pixel_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    val_fixed_0_5_pixel_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    val_fixed_0_5_pixel_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    val_foreground_pixel_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    val_foreground_pixel_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    val_foreground_pixel_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    val_per_class_metrics: list[dict[str, Any]] = Field(default_factory=list)
    val_multiclass_threshold_sweep: dict[str, dict[str, Any]] = Field(default_factory=dict)
    val_metric_warnings: list[str] = Field(default_factory=list)
    val_per_scene_metrics: list[dict[str, Any]] = Field(default_factory=list)
    val_object_separation: dict[str, Any] = Field(default_factory=dict)
    val_object_threshold_sweep: list[dict[str, Any]] = Field(default_factory=list)
    learning_rate: float | None = Field(default=None, ge=0.0)
    epoch_time_sec: float = Field(ge=0.0)


class CheckpointArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str
    label: str


class TrainProgressEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epoch: int = Field(ge=0)
    message: str
    metrics: EpochMetrics | None = None


@runtime_checkable
class TrainProgressSink(Protocol):
    def __call__(self, event: TrainProgressEvent) -> None: ...


class TrainRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model: ModelHandle
    train_loader: object
    val_loader: object
    test_loader: object | None = None
    dense_train_loader: object | None = None
    config: TrainConfig
    checkpoint_dir: str
    sample_size: int | None = Field(default=None, gt=0)
    run_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_inference_window(self) -> Self:
        if self.dense_train_loader is not None and self.config.pipeline_variant not in {"next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3"}:
            raise ValueError("Аудит плотных train-окон предусмотрен только для next-gen4-augmentation-v4")
        context = self.config.inference_context
        if context and self.sample_size is None:
            raise ValueError("sample_size обязателен при ненулевом inference_context")
        if self.sample_size is not None and self.sample_size <= 2 * context:
            raise ValueError("sample_size должен быть больше удвоенного inference_context")
        return self


class TrainResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history: list[EpochMetrics]
    epochs_total: int = Field(ge=0)
    training_time_sec: float = Field(ge=0.0)
    best_checkpoint_path: str | None = None
    final_checkpoint_path: str | None = None
    artifacts: list[CheckpointArtifact] = Field(default_factory=list)
    task: Literal["binary", "multiclass"] = "binary"
    class_schema: list[TrainClassDefinition] = Field(default_factory=list)
    best_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    stopped_early: bool = False
    diagnostics: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "CheckpointArtifact",
    "EpochMetrics",
    "TrainConfig",
    "TrainClassDefinition",
    "TrainError",
    "TrainProgressEvent",
    "TrainProgressSink",
    "TrainRequest",
    "TrainResult",
]
