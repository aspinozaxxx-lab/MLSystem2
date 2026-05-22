package io.geoalert.mapflow.rest.json

import java.util.UUID

import io.geoalert.mapflow.model.RasterLayer

case class RasterLayerJson(
    id: UUID,
    tileJsonUrl: String,
    tileUrl: String,
  )

object RasterLayerJson {
  def apply(rasterLayer: RasterLayer): RasterLayerJson =
    RasterLayerJson(rasterLayer.id, rasterLayer.tileJsonUrl, rasterLayer.tileUrl)
}
