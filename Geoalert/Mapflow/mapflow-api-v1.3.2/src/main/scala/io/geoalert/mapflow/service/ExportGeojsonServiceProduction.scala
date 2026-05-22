package io.geoalert.mapflow.service

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor
import scala.util.Failure
import scala.util.Success

import akka.NotUsed
import akka.http.scaladsl.Http
import akka.http.scaladsl.client.RequestBuilding.Post
import akka.http.scaladsl.model.HttpEntity.ChunkStreamPart
import akka.http.scaladsl.model.HttpRequest
import akka.http.scaladsl.model.StatusCodes
import akka.stream.scaladsl.Source
import akka.util.ByteString
import com.typesafe.scalalogging.LazyLogging
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._

import io.geoalert.mapflow.AkkaSystem._
import io.geoalert.mapflow.Config._
import io.geoalert.mapflow.exception.InternalServerError

import geotrellis.vector._

class ExportGeojsonServiceProduction extends ExportGeojsonService with LazyLogging {
  implicit val executionContext: ExecutionContextExecutor = ExecutionContext.global

  private val flow = Http().superPool[Unit]()

  private val fcStart = ByteString("""{"type": "FeatureCollection","features":[""")
  private val fcEnd = ByteString("]}")

  def exportLayer(
      externalLayerId: String,
      geometries: Option[Source[Projected[Geometry], NotUsed]],
    ): Source[ChunkStreamPart, Any] =
    geometries match {
      case None => exportLayerById(externalLayerId)
      case Some(geometries) => exportByGeometries(geometries, externalLayerId)
    }
  def exportLayerById(externalLayerId: String): Source[ChunkStreamPart, Any] = {
    val requests = Source.single(
      Post(
        s"$vectorProcessorUrl/api/v0/collections/$externalLayerId/export?wrapInFeatureCollection=false"
      )
    )
    pipeRequestResults(requests)
  }

  def exportByGeometries(
      geoms: Source[Projected[Geometry], NotUsed],
      externalLayerId: String,
    ): Source[ChunkStreamPart, Any] = {
    val requests = geoms.map { g =>
      Post(
        s"$vectorProcessorUrl/api/v0/collections/$externalLayerId/export?wrapInFeatureCollection=false",
        g.geom,
      )
    }
    pipeRequestResults(requests)
  }

  private def pipeRequestResults(
      requests: Source[HttpRequest, NotUsed]
    ): Source[ChunkStreamPart, Any] = {
    val concatenator = new Concatenator

    requests
      .map((_, ()))
      .via(flow)
      .zipWithIndex
      .map {
        case ((Success(r), _), i) =>
          logger.debug(
            s"Received response from vector processor, status code ${r.status.intValue()}. "
          )
          if (i % 100 == 0)
            logger.debug(s"So far received ${i + 1} responses in this export session.")

          if (r.status.isSuccess()) r.entity.dataBytes
          else if (r.status == StatusCodes.NotFound) {
            r.entity.discardBytes()
            Source.empty[ByteString]
          }
          else {
            r.entity.discardBytes()
            throw InternalServerError(
              s"Vector processor responded with status code ${r.status.intValue()}"
            )
          }
        case ((Failure(e), _), _) => throw e
      }
      .flatMapConcat(s => s.filter(_.nonEmpty).zipWithIndex)
      .statefulMapConcat(() => concatenator.concat)
      .prepend(Source.single(fcStart))
      .concat(Source.single(fcEnd))
      .map(ChunkStreamPart(_))
  }

  class Concatenator {
    @volatile
    var open = false

    val delimiter = ByteString(", ")

    def concat(elem: (ByteString, Long)) = (open, elem) match {
      case (true, (bs, 0)) => List(delimiter, bs)
      case (_, (bs, _)) =>
        open = true
        List(bs)
    }
  }
}
