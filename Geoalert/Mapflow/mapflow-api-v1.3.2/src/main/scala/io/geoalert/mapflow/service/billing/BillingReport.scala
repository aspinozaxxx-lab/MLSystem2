package io.geoalert.mapflow.service.billing

import java.time.Instant
import java.util.UUID

case class BillingReport(
    startDate: Instant,
    endDate: Instant,
    processings: List[ProcessingReport],
  )

case class ProcessingReport(
    id: UUID,
    email: String,
    name: String,
    area: Long,
    cost: Long,
    completionDate: Option[Instant],
    system: String,
    archived: Boolean,
  )
