package io.geoalert.vectortileserver.rest

import akka.http.scaladsl.server.{Directives, Route}
import io.geoalert.vectortileserver.LayerService

object HealthcheckResource extends Directives with RestImplicits {
  def heartbeat: Route = (path("heartbeat") & get) {
    toComplete(LayerService.countLayers())
  }

  def liveness: Route = (path("heartbeat" / "lite") & get) {
    complete("OK")
  }


  def routes: Route = concat(heartbeat, liveness)
}
