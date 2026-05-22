package io.geoalert.mapflow.repo

import java.time.Instant
import java.util.UUID

import scala.collection.immutable.List

import cats.data.NonEmptySeq
import cats.syntax.applicative._
import cats.syntax.option._
import doobie.ConnectionIO
import doobie.implicits._
import doobie.implicits.legacy.instant._
import doobie.postgres.implicits._
import doobie.util.fragment.Fragment.const
import doobie.util.fragments
import io.geoalert.mapflow.implicits.Postgres._
import io.geoalert.mapflow.model.ReviewStatus

import geotrellis.vector._
import geotrellis.vector.io.json.GeoJson
import geotrellis.vector.io.json.JsonFeatureCollection

case class ProcessingReviewDto(
    processingId: UUID,
    reviewStatus: ReviewStatus,
    comment: Option[String],
    created: Instant,
    updated: Instant,
    hasFeatures: Boolean,
  )

object ProcessingReviewRepo
    extends GenericRepo[ProcessingReviewDto](
      "processing_review",
      Seq(
        "processing_id",
        "review_status",
        "comment",
        "created",
        "updated",
      ),
      "processing_id",
    ) {
  def get(processingId: UUID): ConnectionIO[Option[ProcessingReviewDto]] = {
    val sql = const(
      s"SELECT ${columns.mkString(",")}, features IS NOT NULL AS hasFeatures FROM $dbSchema.$table WHERE "
    ) ++
      fr"processing_id = $processingId"

    sql.query[ProcessingReviewDto].to[List].map(a => a.headOption)
  }

  def create(processingId: UUID): ConnectionIO[UUID] = {
    val columns = Seq(
      ("review_status" -> ReviewStatus.InReview.repr).some,
      ("created" -> Instant.now()).some,
      ("updated" -> Instant.now()).some,
    ).flatten.toMap

    create(columns, processingId.some)
  }

  def update(
      processingId: UUID,
      reviewStatus: ReviewStatus,
      comment: Option[String] = None,
      features: Option[JsonFeatureCollection] = None,
    ): ConnectionIO[Unit] = {
    val columns = Seq(
      ("review_status" -> reviewStatus.repr).some,
      comment.map("comment" -> _),
      features.map("features" -> _.asJson),
      ("updated" -> Instant.now()).some,
    ).flatten.toMap

    updateById(columns, processingId)
  }

  def listProcessingsInReviewBefore(timestamp: Instant): ConnectionIO[List[UUID]] = {
    val status: ReviewStatus = ReviewStatus.InReview
    val sql = const(
      s"SELECT processing_id FROM $dbSchema.$table"
    ) ++ fr" WHERE review_status = $status AND updated < $timestamp"

    sql.query[UUID].to[List]
  }

  def listReviewStatuses(processingIds: Seq[UUID]): ConnectionIO[Map[UUID, ProcessingReviewDto]] =
    NonEmptySeq.fromSeq(processingIds) match {
      case Some(ids) =>
        val sql = const(
          s"SELECT ${columns.mkString(",")}, features IS NOT null AS hasFeatures FROM $dbSchema.$table WHERE "
        ) ++
          fragments.in(fr"processing_id", ids)
        for {
          dtos <- sql.query[ProcessingReviewDto].to[List]
        } yield dtos.map(dto => dto.processingId -> dto).toMap
      case None =>
        Map[UUID, ProcessingReviewDto]().pure[ConnectionIO]
    }

  def getProcessingReviewFeatures(
      processingId: UUID
    ): ConnectionIO[Option[JsonFeatureCollection]] = {
    val sql = const(s"SELECT features FROM $dbSchema.$table WHERE ") ++
      fr"processing_id = $processingId"

    for {
      opt <- sql.query[Option[String]].unique
      fcOpt = opt.map(str => GeoJson.parse[JsonFeatureCollection](str))
    } yield fcOpt
  }
}
