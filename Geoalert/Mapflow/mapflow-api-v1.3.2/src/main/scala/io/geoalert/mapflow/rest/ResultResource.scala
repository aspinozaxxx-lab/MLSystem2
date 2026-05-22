package io.geoalert.mapflow.rest

import java.util.UUID

import scala.concurrent.duration._
import scala.util.Failure
import scala.util.Success

import akka.http.scaladsl.model.ContentType
import akka.http.scaladsl.model.ContentTypes
import akka.http.scaladsl.model.HttpEntity
import akka.http.scaladsl.model.MediaType
import akka.http.scaladsl.model.headers.RawHeader
import akka.http.scaladsl.server.Directives
import akka.http.scaladsl.server.Route
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import doobie.implicits._

import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.service.Services

object ResultResource extends Directives with Authorization with RestImplicits with Services {
  private val downloadHeaders =
    List(RawHeader("Content-Disposition", "attachment; filename=features.geojson"))

  private val pbfContentType = ContentType(
    MediaType.customBinary("application", "x-protobuf", MediaType.NotCompressible)
  )
  
  def loadPbfTile: Route = (path(
    "tiles" / "aoi" / IntNumber / IntNumber / IntNumber ~ ".vector.pbf"
  ) & get & parameters(Symbol("processingId").as[UUID])) { (z, x, y, processingId) =>
    onComplete(layerService.getMvtLayer(processingId, x, y, z).transact(xa).unsafeToFuture()) {
      case Success(bytes) => complete(HttpEntity(pbfContentType, bytes))
      case Failure(ex) => failWith(ex)
    }
  }

  def footprint: Route =
    (path("tiles" / "aoi.json") & get & parameters(Symbol("processingId").as[UUID])) {
      processingId =>
        toComplete(layerService.getTileJson(processingId))
    }

  def routes: Route = concat(
    loadPbfTile,
    footprint,
  )
}
