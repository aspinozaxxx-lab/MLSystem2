package io.geoalert.mapflow.repo

import java.time.Instant
import java.util.UUID

import cats.syntax.option._
import doobie.ConnectionIO
import doobie.postgres.implicits._

case class RatingDto(
    processingId: UUID,
    rating: Int,
    feedback: Option[String],
  )

object ProcessingRateRepo
    extends GenericRepo[RatingDto](
      "processing_rating",
      Seq(
        "processing_id",
        "rating",
        "feedback",
        "created",
      ),
      "processing_id",
    ) {
  def create(
      processingId: UUID,
      rating: Int,
      feedback: Option[String],
    ): ConnectionIO[UUID] = {
    val columns = Seq(
      ("rating" -> rating).some,
      feedback.map("feedback" -> _),
      ("created" -> Instant.now()).some,
      ("updated" -> Instant.now()).some,
    ).flatten.toMap

    create(columns, processingId.some)
  }

  def update(
      processingId: UUID,
      rating: Option[Int],
      feedback: Option[String],
    ): ConnectionIO[Unit] = {
    val columns = Seq(
      rating.map("rating" -> _),
      feedback.map("feedback" -> _).getOrElse("feedback" -> None).some,
      ("updated" -> Instant.now()).some,
    ).flatten.toMap

    updateById(columns, processingId)
  }

  def getRate(processingId: UUID): ConnectionIO[Option[RatingDto]] =
    getOneById(processingId)

  def listRatings(processingIds: Seq[UUID]): ConnectionIO[Map[UUID, RatingDto]] =
    getAllByIds(processingIds)
      .map(_.map(dto => dto.processingId -> dto).toMap)
}
