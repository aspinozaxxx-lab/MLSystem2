package io.geoalert.vectortileserver.rest

import akka.http.scaladsl.server.{Directives, Route}
import io.geoalert.vectortileserver.{LayerService, TileJsonSupport, TileParams}
import io.circe.generic.auto._

object TileJsonResource extends Directives with TileJsonSupport with RestImplicits {
  def tilesJson: Route = (path("layers" / JavaUUID ~ ".json") & parameterMap & pathEndOrSingleSlash) { (layerId, params) =>
    val tileJson = LayerService.getTileJson(layerId, TileParams(params))

    toComplete(tileJson)
  }

  def routes: Route = tilesJson
}
