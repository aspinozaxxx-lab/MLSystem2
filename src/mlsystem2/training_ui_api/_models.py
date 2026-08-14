"""ORM-модели БД training UI API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from ._database import Base


def _json_type():
    return JSON().with_variant(JSONB, "postgresql")


class DatasetClassRow(Base):
    __tablename__ = "dataset_classes"
    __table_args__ = (
        CheckConstraint(
            "imagery_type IN ('kanopus', 'ortho')",
            name="ck_dataset_classes_imagery_type",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(180), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(240), unique=True)
    quality_metric: Mapped[str] = mapped_column(String(32), default="pixel")
    imagery_type: Mapped[str] = mapped_column(String(32), default="kanopus")
    primary_dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("datasets.id", ondelete="SET NULL"),
        nullable=True,
    )
    primary_dataset_locked: Mapped[bool] = mapped_column(Boolean, default=False)
    primary_training_result_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("training_results.id", ondelete="SET NULL", use_alter=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class DatasetRow(Base):
    __tablename__ = "datasets"
    __table_args__ = (
        UniqueConstraint("source_type", "source_path", name="uq_datasets_source"),
        UniqueConstraint("class_id", "name", name="uq_datasets_class_name"),
        Index("ix_datasets_class_id", "class_id"),
        Index("ix_datasets_deleted_at", "deleted_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(180), unique=True, index=True)
    class_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("dataset_classes.id", ondelete="RESTRICT"),
    )
    name: Mapped[str] = mapped_column(String(240))
    source_type: Mapped[str] = mapped_column(String(32), default="mlmarkup")
    source_path: Mapped[str] = mapped_column(String(1024))
    config_revision: Mapped[int] = mapped_column(Integer, default=1)
    legacy_version: Mapped[bool] = mapped_column(Boolean, default=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class DatasetEditorDraftRow(Base):
    __tablename__ = "dataset_editor_drafts"
    __table_args__ = (
        UniqueConstraint(
            "dataset_key",
            "annotation_name",
            "username",
            name="uq_dataset_editor_drafts_owner_scene",
        ),
        Index("ix_dataset_editor_drafts_dataset_owner", "dataset_key", "username"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_key: Mapped[str] = mapped_column(String(180))
    annotation_name: Mapped[str] = mapped_column(String(512))
    username: Mapped[str] = mapped_column(String(180))
    base_revision: Mapped[str] = mapped_column(String(128))
    geojson: Mapped[dict[str, Any]] = mapped_column(_json_type())
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class TrainingTemplateRow(Base):
    __tablename__ = "training_templates"
    __table_args__ = (
        UniqueConstraint("architecture", "dataset_key", name="uq_training_templates_architecture_dataset"),
        Index("ix_training_templates_architecture", "architecture"),
        Index("ix_training_templates_dataset_key", "dataset_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    architecture: Mapped[str] = mapped_column(String(96))
    dataset_key: Mapped[str | None] = mapped_column(String(180), nullable=True)
    dataset_name: Mapped[str | None] = mapped_column(String(240), nullable=True)
    parent_template_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("training_templates.id"),
        nullable=True,
    )
    display_name: Mapped[str] = mapped_column(String(160))
    config_schema: Mapped[dict[str, Any]] = mapped_column(_json_type())
    default_config: Mapped[dict[str, Any]] = mapped_column(_json_type())
    baseline_default_config: Mapped[dict[str, Any]] = mapped_column(_json_type())
    source: Mapped[str] = mapped_column(String(32))
    baseline_source: Mapped[str] = mapped_column(String(32))
    source_mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    baseline_source_mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    parent_template: Mapped["TrainingTemplateRow | None"] = relationship(remote_side=[id])


class InferenceTemplateRow(Base):
    __tablename__ = "inference_templates"
    __table_args__ = (
        UniqueConstraint("architecture", "dataset_key", name="uq_inference_templates_architecture_dataset"),
        Index("ix_inference_templates_architecture", "architecture"),
        Index("ix_inference_templates_dataset_key", "dataset_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    architecture: Mapped[str] = mapped_column(String(96))
    dataset_key: Mapped[str | None] = mapped_column(String(180), nullable=True)
    dataset_name: Mapped[str | None] = mapped_column(String(240), nullable=True)
    parent_template_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("inference_templates.id"),
        nullable=True,
    )
    display_name: Mapped[str] = mapped_column(String(160))
    config_schema: Mapped[dict[str, Any]] = mapped_column(_json_type())
    default_config: Mapped[dict[str, Any]] = mapped_column(_json_type())
    baseline_default_config: Mapped[dict[str, Any]] = mapped_column(_json_type())
    source: Mapped[str] = mapped_column(String(32))
    baseline_source: Mapped[str] = mapped_column(String(32))
    source_mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    baseline_source_mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    parent_template: Mapped["InferenceTemplateRow | None"] = relationship(remote_side=[id])


class StoredFileRow(Base):
    __tablename__ = "stored_files"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(48), index=True)
    original_name: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str | None] = mapped_column(String(160), nullable=True)
    path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    object_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CustomDatasetRow(Base):
    __tablename__ = "custom_datasets"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(180))
    scenes_file_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("stored_files.id"))
    annotation_file_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("stored_files.id"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    scenes_file: Mapped[StoredFileRow] = relationship(foreign_keys=[scenes_file_id])
    annotation_file: Mapped[StoredFileRow] = relationship(foreign_keys=[annotation_file_id])


class QueueControlRow(Base):
    __tablename__ = "queue_controls"

    queue_name: Mapped[str] = mapped_column(String(32), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class AutomationControlRow(Base):
    __tablename__ = "automation_controls"

    key: Mapped[str] = mapped_column(String(32), primary_key=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class AutomationRuleRow(Base):
    __tablename__ = "automation_rules"
    __table_args__ = (
        UniqueConstraint("dataset_key", "architecture", name="uq_automation_rules_dataset_architecture"),
        Index("ix_automation_rules_dataset_key", "dataset_key"),
        Index("ix_automation_rules_architecture", "architecture"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_key: Mapped[str] = mapped_column(String(180))
    architecture: Mapped[str] = mapped_column(String(96))
    training_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    pseudo_markup_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type: Mapped[str] = mapped_column(String(32), index=True)
    source: Mapped[str] = mapped_column(String(32), default="manual", index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    queue_position: Mapped[int] = mapped_column(Integer, index=True)
    automation_rule_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("automation_rules.id"),
        nullable=True,
    )
    dataset_key: Mapped[str | None] = mapped_column(String(180), nullable=True, index=True)
    dataset_version: Mapped[str | None] = mapped_column(String(160), nullable=True, index=True)
    dataset_name: Mapped[str] = mapped_column(String(240))
    training_dataset_name: Mapped[str | None] = mapped_column(String(240), nullable=True)
    inference_dataset_name: Mapped[str | None] = mapped_column(String(240), nullable=True)
    model_name: Mapped[str] = mapped_column(String(160))
    architecture: Mapped[str] = mapped_column(String(96))
    tile_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dedup_key: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        unique=True,
        index=True,
    )
    mlflow_experiment_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mlflow_experiment_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    mlflow_run_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    config: Mapped[dict[str, Any]] = mapped_column(_json_type())
    custom_dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("custom_datasets.id"),
        nullable=True,
    )
    process_pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tmp_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    custom_dataset: Mapped[CustomDatasetRow | None] = relationship()
    automation_rule: Mapped[AutomationRuleRow | None] = relationship()


class TrainingResultRow(Base):
    __tablename__ = "training_results"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String(32), default="manual", index=True)
    automation_rule_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("automation_rules.id"),
        nullable=True,
    )
    dataset_key: Mapped[str | None] = mapped_column(String(180), nullable=True, index=True)
    dataset_version: Mapped[str | None] = mapped_column(String(160), nullable=True, index=True)
    class_key: Mapped[str] = mapped_column(String(180), index=True)
    class_display_name: Mapped[str] = mapped_column(String(240))
    architecture: Mapped[str] = mapped_column(String(96))
    model_name: Mapped[str] = mapped_column(String(160))
    quality_metric: Mapped[str] = mapped_column(String(32), default="pixel")
    task: Mapped[str] = mapped_column(String(32), default="binary")
    class_schema: Mapped[list[dict[str, Any]]] = mapped_column(_json_type(), default=list)
    training_metrics: Mapped[dict[str, Any]] = mapped_column(_json_type(), default=dict)
    f1_score: Mapped[float | None] = mapped_column(nullable=True)
    epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    mlflow_run_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), ForeignKey("jobs.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class PseudoMarkupResultRow(Base):
    __tablename__ = "pseudo_markup_results"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String(32), default="manual", index=True)
    automation_rule_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("automation_rules.id"),
        nullable=True,
    )
    dataset_key: Mapped[str | None] = mapped_column(String(180), nullable=True, index=True)
    dataset_version: Mapped[str | None] = mapped_column(String(160), nullable=True, index=True)
    training_result_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("training_results.id"),
        nullable=True,
    )
    class_key: Mapped[str] = mapped_column(String(180), index=True)
    source_dataset_name: Mapped[str] = mapped_column(String(240))
    image_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scenes_file_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("stored_files.id"),
        nullable=True,
    )
    geojson_file_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("stored_files.id"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(32), index=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), ForeignKey("jobs.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    training_result: Mapped[TrainingResultRow | None] = relationship()
    automation_rule: Mapped[AutomationRuleRow | None] = relationship()
    scenes_file: Mapped[StoredFileRow | None] = relationship(foreign_keys=[scenes_file_id])
    geojson_file: Mapped[StoredFileRow | None] = relationship(foreign_keys=[geojson_file_id])


class TestSampleRow(Base):
    __tablename__ = "test_samples"
    __table_args__ = (
        Index("ix_test_samples_dataset_key", "dataset_key"),
        Index("ix_test_samples_class_dataset", "class_key", "dataset_key"),
        Index("ix_test_samples_created_at", "created_at"),
        Index(
            "uq_test_samples_primary_class_key",
            "class_key",
            unique=True,
            postgresql_where=text("is_primary"),
            sqlite_where=text("is_primary = 1"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(180))
    dataset_key: Mapped[str] = mapped_column(String(180))
    dataset_name: Mapped[str] = mapped_column(String(240))
    dataset_version: Mapped[str | None] = mapped_column(String(160), nullable=True)
    class_key: Mapped[str] = mapped_column(String(180))
    class_name: Mapped[str] = mapped_column(String(240))
    dataset_short_name: Mapped[str] = mapped_column(String(240))
    quality_metric: Mapped[str] = mapped_column(String(32), default="pixel")
    task: Mapped[str] = mapped_column(String(32), default="binary")
    class_schema: Mapped[list[dict[str, Any]]] = mapped_column(_json_type(), default=list)
    evaluation_metrics: Mapped[dict[str, Any]] = mapped_column(_json_type(), default=dict)
    tile_width: Mapped[int] = mapped_column(Integer)
    tile_height: Mapped[int] = mapped_column(Integer)
    image_count: Mapped[int] = mapped_column(Integer)
    requested_object_count: Mapped[int] = mapped_column(Integer)
    actual_object_count: Mapped[int] = mapped_column(Integer)
    territory_count: Mapped[int] = mapped_column(Integer)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    warnings: Mapped[list[str]] = mapped_column(_json_type(), default=list)
    content_revision: Mapped[int] = mapped_column(Integer, default=1)
    evaluated_revision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metric_status: Mapped[str] = mapped_column(String(32), default="unavailable")
    object_iou_threshold: Mapped[float] = mapped_column(default=0.5)
    pixel_precision: Mapped[float | None] = mapped_column(nullable=True)
    pixel_recall: Mapped[float | None] = mapped_column(nullable=True)
    pixel_f1: Mapped[float | None] = mapped_column(nullable=True)
    pixel_true_positive: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    pixel_false_positive: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    pixel_false_negative: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    object_precision: Mapped[float | None] = mapped_column(nullable=True)
    object_recall: Mapped[float | None] = mapped_column(nullable=True)
    object_f1: Mapped[float | None] = mapped_column(nullable=True)
    object_true_positive: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    object_false_positive: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    object_false_negative: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    evaluation_pseudo_result_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("pseudo_markup_results.id", ondelete="SET NULL"),
        nullable=True,
    )
    evaluation_training_result_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("training_results.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    evaluation_job_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    evaluation_inference_template_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("inference_templates.id", ondelete="SET NULL"),
        nullable=True,
    )
    evaluation_inference_template_version: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    evaluation_inference_config_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    evaluation_evaluator_version: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    evaluation_threshold: Mapped[float | None] = mapped_column(nullable=True)
    evaluation_model_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    evaluation_markup_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    evaluation_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    evaluation_pseudo_result: Mapped[PseudoMarkupResultRow | None] = relationship()
    evaluation_training_result: Mapped[TrainingResultRow | None] = relationship(
        foreign_keys=[evaluation_training_result_id]
    )
    evaluation_job: Mapped[JobRow | None] = relationship(foreign_keys=[evaluation_job_id])
    evaluation_inference_template: Mapped[InferenceTemplateRow | None] = relationship(
        foreign_keys=[evaluation_inference_template_id]
    )
    tiles: Mapped[list["TestSampleTileRow"]] = relationship(
        back_populates="sample",
        cascade="all, delete-orphan",
        order_by="TestSampleTileRow.tile_index",
    )


class TestSampleTileRow(Base):
    __tablename__ = "test_sample_tiles"
    __table_args__ = (
        UniqueConstraint("test_sample_id", "tile_index", name="uq_test_sample_tiles_sample_index"),
        Index("ix_test_sample_tiles_sample_id", "test_sample_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    test_sample_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("test_samples.id", ondelete="CASCADE"),
    )
    tile_index: Mapped[int] = mapped_column(Integer)
    source_name: Mapped[str] = mapped_column(String(512))
    territory: Mapped[str] = mapped_column(String(512))
    object_count: Mapped[int] = mapped_column(Integer)
    class_object_counts: Mapped[dict[str, int]] = mapped_column(_json_type(), default=dict)
    evaluation_metrics: Mapped[dict[str, Any]] = mapped_column(_json_type(), default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    pixel_f1: Mapped[float | None] = mapped_column(nullable=True)
    object_f1: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    sample: Mapped[TestSampleRow] = relationship(back_populates="tiles")


class TestSampleBatchRow(Base):
    __tablename__ = "test_sample_batches"
    __table_args__ = (
        UniqueConstraint("active_slot", name="uq_test_sample_batches_active_slot"),
        Index("ix_test_sample_batches_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    active_slot: Mapped[int | None] = mapped_column(Integer, nullable=True, default=1)
    tile_size: Mapped[int] = mapped_column(Integer)
    min_image_count: Mapped[int] = mapped_column(Integer)
    image_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    items: Mapped[list["TestSampleBatchItemRow"]] = relationship(
        back_populates="batch",
        cascade="all, delete-orphan",
        order_by="TestSampleBatchItemRow.position",
    )


class TestSampleBatchItemRow(Base):
    __tablename__ = "test_sample_batch_items"
    __table_args__ = (
        UniqueConstraint("batch_id", "position", name="uq_test_sample_batch_items_position"),
        UniqueConstraint("batch_id", "dataset_key", name="uq_test_sample_batch_items_dataset"),
        Index("ix_test_sample_batch_items_batch_id", "batch_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("test_sample_batches.id", ondelete="CASCADE"),
    )
    position: Mapped[int] = mapped_column(Integer)
    dataset_key: Mapped[str] = mapped_column(String(180))
    dataset_name: Mapped[str] = mapped_column(String(240))
    dataset_version: Mapped[str | None] = mapped_column(String(160), nullable=True)
    class_key: Mapped[str] = mapped_column(String(180))
    class_name: Mapped[str] = mapped_column(String(240))
    dataset_short_name: Mapped[str] = mapped_column(String(240))
    min_object_count: Mapped[int] = mapped_column(Integer)
    metric: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    pool_tile_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pool_object_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sample_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("test_samples.id", ondelete="SET NULL"),
        nullable=True,
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    batch: Mapped[TestSampleBatchRow] = relationship(back_populates="items")
    sample: Mapped[TestSampleRow | None] = relationship()


class TrainingResultTestMetricRow(Base):
    __tablename__ = "training_result_test_metrics"
    __table_args__ = (
        Index("ix_training_result_test_metrics_sample_id", "sample_id"),
        Index("ix_training_result_test_metrics_status", "status"),
    )

    training_result_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("training_results.id", ondelete="CASCADE"),
        primary_key=True,
    )
    sample_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("test_samples.id", ondelete="SET NULL"),
        nullable=True,
    )
    sample_revision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="unavailable")
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("jobs.id", ondelete="SET NULL"),
        nullable=True,
    )
    precision: Mapped[float | None] = mapped_column(nullable=True)
    recall: Mapped[float | None] = mapped_column(nullable=True)
    f1: Mapped[float | None] = mapped_column(nullable=True)
    true_positive: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    false_positive: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    false_negative: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    object_precision: Mapped[float | None] = mapped_column(nullable=True)
    object_recall: Mapped[float | None] = mapped_column(nullable=True)
    object_f1: Mapped[float | None] = mapped_column(nullable=True)
    object_true_positive: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    object_false_positive: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    object_false_negative: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(_json_type(), default=dict)
    inference_template_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("inference_templates.id", ondelete="SET NULL"),
        nullable=True,
    )
    inference_template_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    inference_config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    threshold: Mapped[float | None] = mapped_column(nullable=True)
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    training_result: Mapped[TrainingResultRow] = relationship()
    sample: Mapped[TestSampleRow | None] = relationship()
    job: Mapped[JobRow | None] = relationship()
