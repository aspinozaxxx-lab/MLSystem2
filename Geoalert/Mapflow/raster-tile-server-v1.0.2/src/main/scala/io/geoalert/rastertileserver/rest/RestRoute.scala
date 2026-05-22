package io.geoalert.rastertileserver.rest

import akka.http.scaladsl.server.{Directives, Route}

object RestRoute extends Directives {
  val routes : Route = pathPrefix("api" / "v0") {
    concat(
      CogResource.routes,
      HealthcheckResource.routes,
    )
  }

}
