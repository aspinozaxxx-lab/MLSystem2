package io.geoalert.mapflow.model

import java.time.Instant
import java.util.UUID

import _root_.io.geoalert.mapflow.model.Status.statuses
import io.circe.Json

import io.geoalert.mapflow.exception.InternalServerError
import io.geoalert.mapflow.rest.json.Decoders

import geotrellis.vector._

case class MessageParameter(key: String, value: String)

case class Message(
    code: String,
    parameters: List[MessageParameter],
    message: String,
  )

object Message extends Decoders {
  def fromJson(json: Option[Json]): List[Message] = {
    val a = json.getOrElse(Json.arr())
    val b = a.as[List[Message]]
    b.getOrElse(throw InternalServerError("Error decoding processing messages from DB"))
  }
}

case class Aoi(
    id: UUID,
    processingId: UUID,
    progress: Progress,
    messages: List[Message],
    geometry: Projected[Geometry],
    area: Long,
    completionDate: Option[Instant],
    vrtUri: Option[String]
  )

case class AoiList(aois: List[Aoi], hasMore: Boolean)

case class AoiStats(
    count: Int,
    area: Long,
    bbox: Projected[Geometry],
  )

object AoiStats {
  def apply(aoiSummary: AoiSummary): AoiStats =
    AoiStats(aoiSummary.count, aoiSummary.area, aoiSummary.bbox.toPolygon().withSRID(4326))
}

case class AoiFilter(
    ids: Option[List[UUID]] = None,
    processingIds: Option[List[UUID]] = None,
    statuses: Option[List[Status]] = None,
    geometry: Option[Projected[Geometry]],
  )

sealed abstract class AoiSortField(val repr: String, val field: String) {
  override def toString: String = repr
}

object AoiSortField {
  private val statusField: String = {
    val whenClauses = statuses.sortBy(_.repr).zipWithIndex.map {
      case (s, i) => s"WHEN a.status = ${s.intVal} THEN $i"
    }
    s"CASE ${whenClauses.mkString(" ")} ELSE 999 END"
  }

  case object Area extends AoiSortField("area", "a.area")
  case object PercentCompleted extends AoiSortField("percentCompleted", "a.percent_completed")
  case object Status extends AoiSortField("status", statusField)
}

case class AoiSortEntry(field: AoiSortField, desc: Option[Boolean])

case class AoiSummary(
    count: Int,
    area: Long,
    bbox: Extent,
  )

case class CreateAoisFromGeometryInput(processingId: UUID, geometry: Projected[Geometry])

case class CreateAoisFromFileInput(processingId: UUID, file: Upload)
