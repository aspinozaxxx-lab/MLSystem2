package io.geoalert.mapflow.rest.json

import java.util.UUID

import io.geoalert.mapflow.model.TeamMembership

case class DataProviderJson(
    id: UUID,
    name: String,
    displayName: String,
    previewUrl: Option[String],
  )

case class DataProviderPriceJson(zoom: Int, priceSqKm: Double)

case class UserStatusJson(
    email: String,
    processedArea: Long,
    remainingArea: Long,
    remainingCredits: Long,
    areaLimit: Long,
    maxAoisPerProcessing: Int,
    memoryLimit: Long,
    billingType: String,
    models: List[WorkflowDefJson],
    teams: List[TeamMembership],
    dataProviders: Seq[DataProviderJson],
    reviewWorkflowEnabled: Boolean,
    isAdmin: Boolean,
  )
