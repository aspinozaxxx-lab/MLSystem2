package io.geoalert.mapflow.rest.json

import java.time.Instant
import java.util.UUID

import io.geoalert.mapflow.model.Progress
import io.geoalert.mapflow.model.Project
import io.geoalert.mapflow.model.User

case class ProjectJson(
    id: UUID,
    name: String,
    description: Option[String],
    // Deprecated. Will be removed in API 2
    progress: Progress,
    // Deprecated. Will be removed in API 2
    aoiCount: Int,
    // Deprecated.  Use area instead. Will be removed in API 2
    aoiArea: Long,
    area: Long,
    // Deprecated. Will be removed in API 2
    user: UserJson,
    isDefault: Boolean,
    created: Instant,
    updated: Instant,
    workflowDefs: List[WorkflowDefJson],
  )
case class CreateProjectInputJson(name: String, description: Option[String])

case class UpdateProjectInputJson(name: Option[String], description: Option[String])

object ProjectJson {
  def apply(
      project: Project,
      user: User,
      processedArea: Long,
    ) = new ProjectJson(
    project.id,
    project.name,
    project.description,
    project.progress,
    project.aoiCount,
    project.area,
    project.area,
    UserJson(user, processedArea),
    project.isDefault,
    project.created,
    project.updated,
    project.workflowDefs.map(WorkflowDefJson(_)),
  )
}
