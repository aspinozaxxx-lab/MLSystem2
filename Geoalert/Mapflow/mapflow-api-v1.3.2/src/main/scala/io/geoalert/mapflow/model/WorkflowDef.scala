package io.geoalert.mapflow.model

import java.time.Instant
import java.util.UUID

import cats.syntax.option._

import io.geoalert.mapflow.util.WorkflowDefParser

case class WorkflowDef(
    id: UUID,
    name: String,
    description: Option[String],
    weId: Long,
    weName: String,
    yml: String,
    created: Instant,
    updated: Instant,
    archived: Boolean,
    isDefault: Boolean,
  ) {
  private val summaryEither = WorkflowDefParser.parseYml(yml)

  if (summaryEither.isLeft)
    throw new IllegalStateException(
      "Unable to parse Workflow Definition YML",
      summaryEither.swap.getOrElse(new RuntimeException()),
    )

  val workflowDefSummary: WorkflowDefSummary = summaryEither.getOrElse(throw new RuntimeException())

  val pricePerSqKm: Option[Double] = workflowDefSummary.pricePerSqKm.some
}

case class CreateWorkflowDefInput(
    projectId: Option[UUID],
    name: String,
    description: Option[String],
    file: Option[Upload],
    ymlString: Option[String],
    pricePerSqKm: Option[Double],
    isDefault: Option[Boolean],
  )

case class UpdateWorkflowDefInput(
    id: UUID,
    projectId: Option[UUID],
    name: Option[String],
    description: Option[String],
    file: Option[Upload],
    ymlString: Option[String],
    pricePerSqKm: Option[Double],
    isDefault: Option[Boolean],
  )
