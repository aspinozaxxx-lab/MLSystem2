package io.geoalert.mapflow.rest.json

import java.time.Instant

import geotrellis.vector.Geometry

case class ImageCatalogResponseJson(images: List[ImageJson])

case class ImageJson(
    id: String,
    footprint: Geometry,
    pixelResolution: Double,
    acquisitionDate: Instant,
    productType: String,
    sensor: String,
    colorBandOrder: String,
    cloudCover: Double,
    offNadirAngle: Double,
    previewType: Option[String],
    previewUrl: Option[String],
    providerName: Option[String],
  )
