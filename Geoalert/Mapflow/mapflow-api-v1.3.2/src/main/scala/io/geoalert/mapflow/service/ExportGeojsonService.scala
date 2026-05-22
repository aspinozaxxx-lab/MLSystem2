package io.geoalert.mapflow.service

import akka.NotUsed
import akka.http.scaladsl.model.HttpEntity.ChunkStreamPart
import akka.stream.scaladsl.Source

import geotrellis.vector.Geometry
import geotrellis.vector.Projected

trait ExportGeojsonService {
  def exportLayer(
      externalLayerId: String,
      geometries: Option[Source[Projected[Geometry], NotUsed]] = None,
    ): Source[ChunkStreamPart, Any]
}

object ExportGeojsonService {
  def apply(): ExportGeojsonService = new ExportGeojsonServiceProduction()
}
