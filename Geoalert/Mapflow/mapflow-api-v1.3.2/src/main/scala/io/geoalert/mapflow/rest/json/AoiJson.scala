package io.geoalert.mapflow.rest.json

import java.util.UUID

import io.geoalert.mapflow.model.Aoi
import io.geoalert.mapflow.model.Status

import geotrellis.vector.Geometry

case class MessageJson(code: String, parameters: Map[String, String])

case class AoiJson(
    id: UUID,
    status: Status,
    percentCompleted: Int,
    geometry: Geometry,
    area: Long,
    messages: List[MessageJson],
    vrtUri: Option[String],
  )

object AoiJson {
  def apply(aoi: Aoi): AoiJson =
    AoiJson(
      aoi.id,
      aoi.progress.status,
      aoi.progress.percentCompleted,
      aoi.geometry.geom,
      aoi.area,
      aoi.messages.map(m => MessageJson(m.code, m.parameters.map(mp => (mp.key, mp.value)).toMap)),
      aoi.vrtUri,
    )
}
