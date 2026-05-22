package io.geoalert.mapflow.providers.maxar

import java.time.Instant

import io.geoalert.mapflow.rest.json.ImageJson

import geotrellis.vector.Geometry

case class MaxarFeatureMetadata(
    acquisitionDate: Instant,
    source: String,
    productType: String,
    cloudCover: Double,
    offNadirAngle: Double,
    colorBandOrder: String,
    groundSampleDistance: Double,
    legacyId: String,
  )

case class MaxarFeature(
    id: String,
    geometry: Geometry,
    metadata: MaxarFeatureMetadata,
  ) {
  def toImageJson: ImageJson =
    ImageJson(
      id,
      geometry,
      metadata.groundSampleDistance,
      metadata.acquisitionDate,
      metadata.productType,
      metadata.source,
      metadata.colorBandOrder,
      metadata.cloudCover,
      metadata.offNadirAngle,
      None,
      None,
      None,
    )
}
