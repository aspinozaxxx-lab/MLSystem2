package io.geoalert.mapflow.repo

import java.time.Instant
import java.util.UUID
import cats.data.NonEmptySeq
import cats.syntax.option._
import doobie.ConnectionIO
import doobie.Fragments._
import doobie.implicits._
import doobie.implicits.legacy.instant._
import doobie.postgres.implicits._
import doobie.util.fragment.Fragment.const
import io.geoalert.mapflow.Config.dbSchema

object BillingRepo {
  case class ReportDto(
      processingId: UUID,
      email: String,
      name: String,
      area: Long,
      cost: Long,
      completionDate: Option[Instant],
      meta: Map[String, String],
      archived: Boolean,
    )

  // TODO: Filter incomplete processings
  def listCompleteAois(
      startDate: Option[Instant],
      endDate: Option[Instant],
      userIds: Seq[UUID],
    ): ConnectionIO[List[ReportDto]] = {
    val select =
      const(s"""
        SELECT prc.id, app_user.email, prc.name, prc.area, prc.cost, prc.completion_date, prc.meta, prc.archived
        FROM
        (SELECT $dbSchema.processing.*,
        	 	bool_and(workflow.status = 2) AS completed,
        	 	max(workflow.status_update_date) AS completion_date,
        	 	sum(workflow.area) AS area
        	FROM $dbSchema.processing
        	LEFT JOIN $dbSchema.aoi a ON a.processing_id = processing.id
        	LEFT JOIN $dbSchema.workflow ON workflow.aoi_id = a.id
        	GROUP BY $dbSchema.processing.id) prc
        JOIN $dbSchema.project ON prc.project_id = project.id
        JOIN $dbSchema.app_user ON project.user_id = app_user.id
        """)

    val sql = select ++
      fr" WHERE " ++
      andOpt(
        startDate.map(value => fr"prc.completion_date >= $value"),
        endDate.map(value => fr"prc.completion_date <= $value"),
        fr"prc.completed IS TRUE".some,
        (NonEmptySeq.fromSeq(userIds) match {
          case Some(value) => in(fr"project.user_id", value)
          case None => fr"FALSE"
        }).some,
      )

    sql.query[ReportDto].to[List]
  }
}
