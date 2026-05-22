package io.geoalert.mapflow.model

import java.time.Instant
import java.util.UUID

import io.circe.Decoder
import io.circe.Encoder
import io.circe.Json
import io.circe.generic.JsonCodec
import io.geoalert.mapflow.Config.externalUrl
import io.geoalert.mapflow.model.DataSource.DataSource
import io.geoalert.mapflow.model.SourceType.SourceType
import io.geoalert.mapflow.repo.BlockParameters
import io.geoalert.mapflow.repo.ProcessingReviewDto
import io.geoalert.mapflow.repo.RatingDto

import geotrellis.vector.Extent

case class Processing(
    id: UUID,
    projectId: UUID,
    vectorLayer: VectorLayer,
    rasterLayer: RasterLayer,
    dataProvider: Option[DataProvider],
    workflowDef: WorkflowDef,
    name: String,
    description: Option[String],
    bbox: Extent,
    aoiCount: Int,
    area: Long,
    progress: Progress,
    reviewStatus: Option[ProcessingReviewDto],
    params: ProcessingParams,
    blocks: Seq[BlockParameters],
    meta: Json,
    messages: List[Message],
    sourceType: Option[SourceType],
    source: Option[DataSource],
    cost: Option[Long],
    rating: Option[Rating],
    created: Instant,
    updated: Instant,
    archived: Boolean,
    projectName: Option[String],
    email: String,
    user: UserBrief,
  ) {
  private val linkToMap =
    s"$externalUrl/projects/$projectId/processings/$id"
  def toCSVField: List[String] =
    List(
      projectName,
      name,
      email,
      area.toString,
      cost.map(_.toString),
      created.toString,
      progress.status.repr,
      progress.percentCompleted.toString,
      archived.toString,
      dataProvider.map(_.name),
      linkToMap,
    )

  def toProcessingDetails: ProcessingDetailsJson =
    ProcessingDetailsJson(
      projectName,
      name,
      email,
      area.toString,
      cost.map(_.toString),
      created.toString,
      progress.completionDate,
      progress.status.repr,
      progress.percentCompleted.toString,
      archived.toString,
      dataProvider.map(_.name),
      linkToMap,
    )
}

@JsonCodec
case class ProcessingDetailsJson(
    projectName: Option[String],
    name: String,
    email: String,
    area: String,
    cost: Option[String],
    created: String,
    completionDate: Option[Instant],
    status: String,
    percentCompleted: String,
    archived: String,
    dataProvider: Option[String],
    linkToMap: String,
  )

object Processing {
  val CsvHeaders: List[String] = List(
    "Project name",
    "Processing name",
    "Email",
    "Area",
    "Cost",
    "Created",
    "Status",
    "Percent completed",
    "is Archived",
    "Source type",
    "Provider name",
    "Link to the map",
  )
}

case class Rating(rating: Int, feedback: Option[String])

sealed abstract class ReviewStatus(val repr: String)

object ReviewStatus {
  case object Accepted extends ReviewStatus("ACCEPTED")
  case object NotAccepted extends ReviewStatus("NOT_ACCEPTED")
  case object Refunded extends ReviewStatus("REFUNDED")
  case object InReview extends ReviewStatus("IN_REVIEW")

  def fromString(name: String): ReviewStatus = name match {
    case "ACCEPTED" => Accepted
    case "NOT_ACCEPTED" => NotAccepted
    case "REFUNDED" => Refunded
    case "IN_REVIEW" => InReview
    case _ => sys.error(s"Invalid ReviewStatus code: $name")
  }

  implicit val reviewStatusDecoder: Decoder[ReviewStatus] =
    Decoder.decodeString.map(a => ReviewStatus.fromString(a))

  implicit val reviewStatusEncoder: Encoder[ReviewStatus] =
    Encoder.encodeString.contramap(_.repr)
}

object Rating {
  def apply(dto: RatingDto): Rating = Rating(dto.rating, dto.feedback)
}

@JsonCodec
case class BlockParametersInput(name: String, enabled: Boolean)

case class CreateProcessingInput(
    projectId: UUID,
    vectorLayerId: Option[UUID],
    rasterLayerId: Option[UUID],
    workflowDefId: UUID,
    dataProviderId: Option[UUID],
    name: Option[String],
    description: Option[String],
    cost: Long = 0L,
    partitionSize: Option[Double] = None,
    params: Option[ProcessingParams] = None,
    meta: Option[Json] = None,
    source: Option[DataSource] = None,
    sourceType: Option[SourceType] = None,
    blocks: Option[Seq[BlockParametersInput]] = None,
    url: Option[String] = None,
  )

case class UpdateProcessingInput(
    processingId: UUID,
    projectId: Option[UUID] = None,
    vectorLayerId: Option[UUID] = None,
    rasterLayerId: Option[UUID] = None,
    workflowDefId: Option[UUID] = None,
    name: Option[String] = None,
    description: Option[String] = None,
    cost: Option[Long] = None,
    meta: Option[Json] = None,
  )
