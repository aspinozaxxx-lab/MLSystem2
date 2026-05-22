package io.geoalert.mapflow.model

import java.time.Instant
import java.util.UUID

import geotrellis.vector.Geometry
import geotrellis.vector.Projected

case class Workflow(
    id: UUID,
    aoiId: UUID,
    workflowDef: WorkflowDef,
    externalId: Option[String],
    geometry: Projected[Geometry],
    area: Long,
    status: Status,
    completionDate: Option[Instant],
    requiredAction: Option[RequiredAction],
    locked: Boolean,
    lockedAt: Option[Instant],
    failedCount: Int,
  )

case class WorkflowSummary(
    id: UUID,
    aoiId: UUID,
    externalId: Option[String],
    status: Status,
    completionDate: Option[Instant],
  )

object WorkflowSummary {
  def apply(wf: Workflow): WorkflowSummary =
    WorkflowSummary(wf.id, wf.aoiId, wf.externalId, wf.status, wf.completionDate)
}
