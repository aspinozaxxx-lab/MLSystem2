package io.geoalert.mapflow.providers.maxar

import java.time.Instant

import geotrellis.vector.Geometry

case class MaxarCatalogRequest(
    aoi: Option[Geometry] = None,
    acquisitionDateFrom: Option[Instant] = None,
    acquisitionDateTo: Option[Instant] = None,
    minResolution: Option[Double] = None,
    maxResolution: Option[Double] = None,
    maxCloudCover: Option[Double] = None,
    minOffNadirAngle: Option[Double] = None,
    maxOffNadirAngle: Option[Double] = None,
    minAoiIntersectionPercent: Option[Double] = None,
    featureId: Option[String] = None,
  )
