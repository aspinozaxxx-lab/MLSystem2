package io.geoalert.mapflow.model

import java.util.UUID

import io.geoalert.mapflow.Config._

case class VectorLayer(
    id: UUID,
    externalId: String,
    name: String,
  ) {
  def tileJsonUrl: String = s"$vectorTileServerUrl/api/layers/$externalId.json"

  def tileUrl: String = s"$vectorTileServerUrl/api/layers/$externalId/tiles/{z}/{x}/{y}.vector.pbf"
}
