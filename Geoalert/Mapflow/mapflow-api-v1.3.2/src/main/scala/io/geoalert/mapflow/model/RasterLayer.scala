package io.geoalert.mapflow.model

import java.util.UUID

import io.geoalert.mapflow.Config._

case class RasterLayer(id: UUID, uri: String) {
  def tileJsonUrl: String = s"$rasterTileServerUrl/api/v0/cogs/tiles.json?uri=$uri"

  def tileUrl: String = s"$rasterTileServerUrl/api/v0/cogs/tiles/{z}/{x}/{y}.png?uri=$uri"
}
