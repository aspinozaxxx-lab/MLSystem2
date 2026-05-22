package io.geoalert.mapflow.rest

import akka.http.scaladsl.server.Directives
import akka.http.scaladsl.server.Route
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import io.circe.generic.auto.exportDecoder

import io.geoalert.mapflow.rest.json.ImageCatalogRequestJson
import io.geoalert.mapflow.service.Services

import geotrellis.vector._

object CatalogResource extends Directives with Authorization with RestImplicits with Services {
  def searchMeta: Route =
    (path("catalog" / "meta") & post & authorized & entity(as[ImageCatalogRequestJson])) {
      (user, input) =>
        toComplete(catalogService.searchMeta(input)(user))
    }

  def fetchMeta: Route = (path("catalog" / "meta" / Segment) & get & authorized) {
    (imageId, user) =>
      toComplete(catalogService.fetchMeta(imageId)(user))
  }

  val routes: Route = concat(
    searchMeta,
    fetchMeta,
  )
}
