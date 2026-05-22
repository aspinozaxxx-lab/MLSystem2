package io.geoalert.vectortileserver.rest

import akka.http.scaladsl.server.{Directives, Route}

object RestRoute extends Directives {
  val routes: Route = pathPrefix("api") {
    concat(TileResource.routes, TileJsonResource.routes, HealthcheckResource.routes)
  }
}
