package io.geoalert.mapflow.rest.json

import java.time.Instant
import java.util.UUID

import cats.syntax.option._

import io.geoalert.mapflow.model.WorkflowDef

case class BlockConfigJson(
    name: String,
    displayName: String,
    optional: Boolean,
  )
case class WorkflowDefJson(
    id: UUID,
    name: String,
    description: Option[String],
    created: Instant,
    updated: Instant,
    blocks: Seq[BlockConfigJson],
  )

object WorkflowDefJson {
  def apply(wd: WorkflowDef) = new WorkflowDefJson(
    wd.id,
    wd.name,
    wd.description,
    wd.created,
    wd.updated,
    wd.workflowDefSummary
      .blocks
      .map(bp => BlockConfigJson(bp.name, bp.displayName, bp.optional)),
  )
}
