package io.geoalert.vectortileserver.rest

import akka.http.scaladsl.model.{ContentType, HttpEntity, MediaType}
import akka.http.scaladsl.server.{Directives, Route}
import io.geoalert.vectortileserver.{LayerService, TileParams}
import scala.util.{Failure, Success}
import doobie.syntax.connectionio._

object TileResource extends Directives with RestImplicits {
  private val pbfContentType = ContentType(
    MediaType.customBinary("application", "x-protobuf", MediaType.NotCompressible)
  )

  def vectorTile: Route = (path("layers" / JavaUUID / "tiles" / IntNumber / IntNumber / IntNumber ~ ".vector.pbf") &
    get & rejectEmptyResponse) { (layerId, z, x, y) =>
    (parameterMap & pathEndOrSingleSlash) { params =>
      val pbf = LayerService.getTile(layerId, x, y, z, TileParams(params))

      onComplete(pbf.transact(xa).unsafeToFuture()) {
        case Success(p) => complete(HttpEntity(pbfContentType, p))
        case Failure(ex) => failWith(ex)
      }
    }
  }

  def routes: Route = vectorTile

}
