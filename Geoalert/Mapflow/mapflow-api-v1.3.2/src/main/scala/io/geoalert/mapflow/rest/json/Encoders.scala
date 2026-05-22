package io.geoalert.mapflow.rest.json

import io.circe.Encoder
import io.circe.generic.auto._
import io.circe.generic.semiauto.deriveEncoder

import io.geoalert.mapflow.model.GetSkyWatchMetaInput
import io.geoalert.mapflow.model.Location
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.model.SkyWatchAnswer
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.model.TeamMemberRole.TeamMemberRole
import io.geoalert.mapflow.model.TileJson
import io.geoalert.mapflow.service.billing.BillingReport

import geotrellis.vector._

trait Encoders {
  // TODO: Move encoders to companion objects
  implicit val statusEncoder: Encoder[Status] =
    Encoder.encodeString.contramap(_.repr)

  implicit val processingEncoder: Encoder[ProcessingJson] =
    deriveEncoder[ProcessingJson]

  implicit val createAndRunProcessingJsonEncoder: Encoder[CreateAndRunProcessingJson] =
    deriveEncoder[CreateAndRunProcessingJson]

  implicit val updateProcessingJsonEncoder: Encoder[UpdateProcessingInputJson] =
    deriveEncoder[UpdateProcessingInputJson]

  implicit val projectJsonEncoder: Encoder[ProjectJson] =
    deriveEncoder[ProjectJson]

  implicit val userJsonEncoder: Encoder[UserJson] =
    deriveEncoder[UserJson]

  implicit val billingReportEncoder: Encoder[BillingReport] =
    deriveEncoder[BillingReport]

  implicit val roleEncoder: Encoder[Role] =
    Encoder.encodeString.contramap(_.repr)

  implicit val userStatusEncoder: Encoder[UserStatusJson] =
    deriveEncoder[UserStatusJson]

  implicit val tileEncoder: Encoder[TileJson] =
    deriveEncoder[TileJson]

  implicit val aoiEncoder: Encoder[AoiJson] =
    deriveEncoder[AoiJson]

  implicit val workflowDefEncoder: Encoder[WorkflowDefJson] =
    deriveEncoder[WorkflowDefJson]

  implicit val getSkyWatchMetaInputEncoder: Encoder[GetSkyWatchMetaInput] =
    deriveEncoder[GetSkyWatchMetaInput]

  implicit val locationEncoder: Encoder[Location] =
    deriveEncoder[Location]

  implicit val skyWatchAnswerEncoder: Encoder[SkyWatchAnswer] =
    deriveEncoder[SkyWatchAnswer]

  implicit val imageCatalogResponseJsonEncoder: Encoder[ImageCatalogResponseJson] =
    deriveEncoder[ImageCatalogResponseJson]

  implicit val imageCatalogRequestJsonEncoder: Encoder[ImageCatalogRequestJson] =
    deriveEncoder[ImageCatalogRequestJson]

  implicit val imageJsonEncoder: Encoder[ImageJson] =
    deriveEncoder[ImageJson]

  implicit val teamMemberJsonEncoder: Encoder[TeamMemberJson] =
    deriveEncoder[TeamMemberJson]

  implicit val memberRoleEncoder: Encoder[TeamMemberRole] =
    Encoder.encodeString.contramap(_.toString)

  implicit val createProcessingRatingJsonEncoder: Encoder[CreateProcessingRatingJson] =
    deriveEncoder[CreateProcessingRatingJson]
}
