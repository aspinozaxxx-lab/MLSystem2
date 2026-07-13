"""ORM-модели БД training UI API."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from ._database import Base


def _json_type():
    return JSON().with_variant(JSONB, "postgresql")


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
        Index("ix_test_samples_class_variant", "class_key", "variant_key"),
        Index("ix_test_samples_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(180))
    dataset_key: Mapped[str] = mapped_column(String(180))
    dataset_name: Mapped[str] = mapped_column(String(240))
    dataset_version: Mapped[str | None] = mapped_column(String(160), nullable=True)
    class_key: Mapped[str] = mapped_column(String(180))
    class_name: Mapped[str] = mapped_column(String(240))
    variant_key: Mapped[str] = mapped_column(String(180))
    variant_name: Mapped[str] = mapped_column(String(240))
    tile_width: Mapped[int] = mapped_column(Integer)
    tile_height: Mapped[int] = mapped_column(Integer)
    image_count: Mapped[int] = mapped_column(Integer)
    requested_object_count: Mapped[int] = mapped_column(Integer)
    actual_object_count: Mapped[int] = mapped_column(Integer)
    territory_count: Mapped[int] = mapped_column(Integer)
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
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    sample: Mapped[TestSampleRow] = relationship(back_populates="tiles")
