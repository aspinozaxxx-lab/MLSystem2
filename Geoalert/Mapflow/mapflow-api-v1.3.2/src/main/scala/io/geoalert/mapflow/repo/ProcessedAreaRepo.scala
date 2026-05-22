package io.geoalert.mapflow.repo

import java.time.Instant
import java.util.UUID

import cats.data.NonEmptySeq
import cats.implicits.toFunctorOps
import cats.syntax.applicative._
import cats.syntax.option._
import doobie.ConnectionIO
import doobie.Fragment._
import doobie.Fragments._
import doobie.implicits._
import doobie.implicits.legacy.instant._
import doobie.postgres.implicits._

case class ProcessedAreaDto(
    processing_id: UUID,
    userId: UUID,
    area: Long,
    created: Instant,
    updated: Instant,
  )

object ProcessedAreaRepo
    extends GenericRepo[ProcessedAreaDto](
      "processed_area",
      Seq(
        "processing_id",
        "user_id",
        "area",
        "created",
        "updated",
        "hold",
      ),
      "processing_id",
    ) {
  def getProcessedAreaByUserId(userId: UUID): ConnectionIO[Long] = {
    val sql = fr"SELECT SUM(area) FROM " ++  const(s"$dbSchema.$table") ++
      fr"WHERE user_id = $userId GROUP BY user_id"

    sql.query[Long].option.map(_.getOrElse(0))
  }

  def getProcessedAreaByUserIds(userIds: Seq[UUID]): ConnectionIO[Map[UUID, Long]] =
    NonEmptySeq.fromSeq(userIds) match {
      case Some(value) =>
        val sql = fr"SELECT user_id, SUM(area) FROM " ++ const(s"$dbSchema.$table") ++
          fr"WHERE " ++
          in(fr"user_id", value) ++
          fr"GROUP BY user_id"

        sql.query[(UUID, Long)].to[List].map(_.toMap)
      case None => Map[UUID, Long]().pure[ConnectionIO]
    }

  def holdProcessedArea(
      processingId: UUID,
      userId: UUID,
      area: Long,
    ): ConnectionIO[Unit] = {
    val fields = Seq(
      Some("processing_id" -> processingId),
      Some("user_id" -> userId),
      Some("area" -> area),
      Some("created" -> Instant.now()),
      Some("updated" -> Instant.now()),
      Some("hold" -> true),
    ).flatten.toMap

    createOrIgnore(fields, processingId.some).void
  }

  def debitProcessedArea(processingId: UUID): ConnectionIO[Unit] = {
    val fields = Seq(
      Some("hold" -> false),
      Some("updated" -> Instant.now()),
    ).flatten.toMap

    update(fields, fr"processing_id = $processingId".some)
  }

  def deleteProcessedArea(processingId: UUID): ConnectionIO[Unit] = {
    val sql = const(s"DELETE FROM $dbSchema.$table WHERE") ++
      fr"processing_id = $processingId"

    sql.update.run.map { _ => }
  }
}
