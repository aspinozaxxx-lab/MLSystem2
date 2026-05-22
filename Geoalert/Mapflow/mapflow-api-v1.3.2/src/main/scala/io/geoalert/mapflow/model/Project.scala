package io.geoalert.mapflow.model

import java.time.Instant
import java.util.UUID

case class Project(
    id: UUID,
    name: String,
    description: Option[String],
    progress: Progress,
    aoiCount: Int,
    area: Long,
    userId: UUID,
    isDefault: Boolean,
    created: Instant,
    updated: Instant,
    workflowDefs: List[WorkflowDef],
    archived: Boolean,
    defaultWds: Boolean,
  )

case class ProjectBrief(
    id: UUID,
    name: String,
    description: Option[String],
    created: Instant,
    updated: Instant,
    user: UserBrief,
    progress: Progress,
  )

case class UserBrief(
    id: UUID,
    email: String,
    name: Option[String],
    preferredUsername: Option[String],
    avantpostUserId: Option[UUID],
  )

case class CreateProjectInput(
    name: String,
    description: Option[String],
    addDefaultWds: Option[Boolean],
  )

case class UpdateProjectInput(
    id: UUID,
    name: Option[String],
    description: Option[String],
  )
