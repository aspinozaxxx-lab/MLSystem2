package io.geoalert.mapflow.rest.json

import java.util.UUID

import io.geoalert.mapflow.model.VectorLayer

case class VectorLayerJson(
    id: UUID,
    name: String,
    tileJsonUrl: String,
    tileUrl: String,
  )

object VectorLayerJson {
  def apply(vectorLayer: VectorLayer): VectorLayerJson =
    VectorLayerJson(vectorLayer.id, vectorLayer.name, vectorLayer.tileJsonUrl, vectorLayer.tileUrl)
}
