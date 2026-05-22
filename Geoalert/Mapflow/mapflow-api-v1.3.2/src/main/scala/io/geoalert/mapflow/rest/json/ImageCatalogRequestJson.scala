package io.geoalert.mapflow.rest.json

import java.time.Instant

import geotrellis.vector.Geometry

case class ImageCatalogRequestJson(
    aoi: Geometry,
    acquisitionDateFrom: Option[Instant],
    acquisitionDateTo: Option[Instant],
    minResolution: Option[Double],
    maxResolution: Option[Double],
    maxCloudCover: Option[Double],
    minOffNadirAngle: Option[Double],
    maxOffNadirAngle: Option[Double],
    minAoiIntersectionPercent: Option[Double],
  )
