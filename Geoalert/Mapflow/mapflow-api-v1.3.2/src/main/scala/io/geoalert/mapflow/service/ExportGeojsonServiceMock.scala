package io.geoalert.mapflow.service

import akka.NotUsed
import akka.http.scaladsl.model.HttpEntity
import akka.stream.scaladsl.Source

import geotrellis.vector.Geometry
import geotrellis.vector.Projected

class ExportGeojsonServiceMock extends ExportGeojsonService {
  override def exportLayer(
      externalLayerId: String,
      geometries: Option[Source[Projected[Geometry], NotUsed]],
    ): Source[HttpEntity.ChunkStreamPart, Any] =
    Source.fromIterator(() => List(HttpEntity.ChunkStreamPart("{}")).iterator)
}
