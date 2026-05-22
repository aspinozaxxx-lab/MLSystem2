package io.geoalert.mapflow.rest.json

import java.util.UUID

import geotrellis.vector.Geometry

case class CalculateCostInput(
    wdId: UUID,
    geometry: Option[Geometry],
    areaSqKm: Option[Double],
    params: Option[Map[String, String]],
    meta: Option[Map[String, String]],
    blocks: Option[Seq[BlockParametersJson]],
  )
