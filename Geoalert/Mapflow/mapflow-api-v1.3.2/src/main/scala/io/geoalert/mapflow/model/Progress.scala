package io.geoalert.mapflow.model

import java.time.Instant

import scala.util.Try

import io.geoalert.mapflow.model.Status.Ok
import sangria.macros.derive.GraphQLDescription

case class Progress(
    status: Status,
    percentCompleted: Int,
    details: List[ProgressDetail],
    completionDate: Option[Instant],
    @GraphQLDescription("Return in seconds")
    estimate: Option[Long],
  )

object Progress {
  def lastUpdateDate(status: Status, details: List[ProgressDetail]): Option[Instant] =
    if (status == Status.Ok) {
      val dates = details.flatMap(_.statusUpdateDate)
      if (dates.nonEmpty)
        Some(dates.max)
      else
        None
    }
    else
      None

  private def calculateEstimate(percentCompleted: Int, createdAt: Option[Instant]): Long = {
    val progressTime =
      createdAt.fold(0L)(startDate => Instant.now().getEpochSecond - startDate.getEpochSecond)
    Try(progressTime * (100 - percentCompleted) / percentCompleted).getOrElse(0L)
  }

  def apply(
      details: List[ProgressDetail],
      area: Long,
      createdAt: Option[Instant],
    ): Progress = {
    val status = Status.fromProgressDetails(details)
    val okArea = details.filter(_.status == Ok).map(_.area).sum
    val percent = if (area == 0) 0 else 100 min (okArea * 100 / area).toInt

    val lastStatusUpdate = lastUpdateDate(status, details)
    val calculatedETA = if (status == Status.InProgress) calculateEstimate(percent, createdAt) else 0L
    val estimate = Option.unless(calculatedETA == 0)(calculatedETA)
    Progress(status, percent, details, lastStatusUpdate, estimate)
  }
}

case class ProgressDetail(
    status: Status,
    count: Int,
    area: Long,
    statusUpdateDate: Option[Instant],
  )

object ProgressDetail {
  def apply(
      status: Int,
      count: Int,
      area: Long,
      statusUpdateDate: Option[Instant],
    ): ProgressDetail =
    ProgressDetail(Status(status), count, area, statusUpdateDate)
}
