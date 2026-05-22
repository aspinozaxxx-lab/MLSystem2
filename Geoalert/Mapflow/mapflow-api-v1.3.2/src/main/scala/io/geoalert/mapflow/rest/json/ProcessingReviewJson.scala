package io.geoalert.mapflow.rest.json

import java.time.Instant
import java.time.temporal.ChronoUnit

import cats.syntax.option._
import io.circe.Encoder
import io.circe.generic.auto._
import io.circe.generic.semiauto.deriveEncoder

import io.geoalert.mapflow.DefaultProcessingReviewConfig
import io.geoalert.mapflow.MiscConfig
import io.geoalert.mapflow.model.ReviewStatus
import io.geoalert.mapflow.repo.ProcessingReviewDto

import geotrellis.vector.io.json.JsonFeatureCollection

case class ProcessingReviewInputJson(comment: String, features: Option[JsonFeatureCollection])

object ProcessingReviewInputJson {
  implicit val processingReviewInputJsonEncoder: Encoder[ProcessingReviewInputJson] =
    deriveEncoder[ProcessingReviewInputJson]
}

case class ProcessingReviewJson(reviewStatus: ReviewStatus, inReviewUntil: Option[Instant])
case class ProcessingReviewDetailsJson(
    reviewStatus: ReviewStatus,
    inReviewUntil: Option[Instant],
    comment: Option[String],
    featuresUri: Option[String],
  )

object ProcessingReviewDetailsJson extends DefaultProcessingReviewConfig with MiscConfig {
  def apply(dto: ProcessingReviewDto): ProcessingReviewDetailsJson =
    ProcessingReviewDetailsJson(
      dto.reviewStatus,
      if (dto.reviewStatus == ReviewStatus.InReview)
        dto.updated.plus(autoConfirmProcessingsInterval.toSeconds, ChronoUnit.SECONDS).some
      else
        none,
      dto.comment,
      if (dto.hasFeatures)
        s"$externalUrl/rest/processings/${dto.processingId}/review_features".some
      else
        none,
    )

  implicit val processingReviewDetailsJsonEncoder: Encoder[ProcessingReviewDetailsJson] =
    deriveEncoder[ProcessingReviewDetailsJson]
}

object ProcessingReviewJson extends DefaultProcessingReviewConfig {
  def apply(dto: ProcessingReviewDto): ProcessingReviewJson =
    ProcessingReviewJson(
      dto.reviewStatus,
      if (dto.reviewStatus == ReviewStatus.InReview)
        dto.updated.plus(autoConfirmProcessingsInterval.toSeconds, ChronoUnit.SECONDS).some
      else
        none,
    )

  implicit val processingReviewJsonEncoder: Encoder[ProcessingReviewJson] =
    deriveEncoder[ProcessingReviewJson]
}
