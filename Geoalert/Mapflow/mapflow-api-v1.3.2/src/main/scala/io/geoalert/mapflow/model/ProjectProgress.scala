package io.geoalert.mapflow.model

import java.util.UUID

case class ProjectProgress(
    id: UUID,
    status: Status,
    percentCompleted: Long,
    estimate: Option[Long],
  )
