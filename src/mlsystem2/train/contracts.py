"""Публичные контракты обучения."""

from __future__ import annotations

from typing import Any, Literal, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlsystem2.models.contracts import ModelHandle


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
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    seed: int = 42
    inference_context: int = Field(default=0, ge=0)
    device: str
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    loss: Literal["bce_dice", "focal_dice", "focal_tversky", "cross_entropy", "cross_entropy_dice"]
    focal_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    pos_weight: float = Field(default=1.0, gt=0.0)
    hard_negative_weight: float = Field(default=1.0, gt=0.0)
    tversky_alpha: float = Field(default=0.4, gt=0.0)
    tversky_beta: float = Field(default=0.6, gt=0.0)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    early_stopping_patience: int = Field(gt=0)
    max_train_batches_per_epoch: int | None = Field(default=None, gt=0)
    max_val_batches_per_epoch: int | None = Field(default=None, gt=0)
    max_training_time_sec: int | None = Field(default=None, gt=0)
    class_slugs: list[str] = Field(default_factory=list)
    class_schema: list[TrainClassDefinition] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_task_loss(self) -> Self:
        multiclass_losses = {"cross_entropy", "cross_entropy_dice"}
        if self.task == "multiclass" and self.loss not in multiclass_losses:
            raise ValueError("multiclass train требует loss=cross_entropy или cross_entropy_dice")
        if self.task == "binary" and self.loss in multiclass_losses:
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
        return self


class EpochMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def fill_quality_metric_compatibility(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        resolved = dict(data)
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
        return resolved

    epoch: int = Field(ge=0)
    train_loss: float = Field(ge=0.0)
    val_loss: float = Field(ge=0.0)
    quality_metric: Literal["pixel", "objects"] = "pixel"
    val_quality_f1: float = Field(default=0.0, ge=0.0, le=1.0)
    val_quality_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    val_quality_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    val_best_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    val_best_pixel_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    val_best_threshold_pixel_f1: float = Field(default=0.0, ge=0.0, le=1.0)
    val_best_threshold_pixel_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    val_best_threshold_pixel_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    val_best_threshold_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    val_best_threshold_recall: float = Field(default=0.0, ge=0.0, le=1.0)
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
    val_foreground_pixel_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    val_foreground_pixel_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    val_foreground_pixel_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    val_per_class_metrics: list[dict[str, Any]] = Field(default_factory=list)
    val_multiclass_threshold_sweep: dict[str, dict[str, Any]] = Field(default_factory=dict)
    val_metric_warnings: list[str] = Field(default_factory=list)
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
    config: TrainConfig
    checkpoint_dir: str
    sample_size: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_inference_window(self) -> Self:
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
