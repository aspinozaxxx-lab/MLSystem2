package io.geoalert.mapflow.model

import java.time.Instant
import java.util.UUID

case class ProcessingProgress(
    id: UUID,
    createdAt: Instant,
    status: Status,
    totalArea: Long,
    completedArea: Long,
    percentCompleted: Long,
    estimate: Option[Long],
  )
