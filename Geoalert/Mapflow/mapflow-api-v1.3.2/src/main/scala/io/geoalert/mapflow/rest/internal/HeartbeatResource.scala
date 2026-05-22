package io.geoalert.mapflow.rest.internal

import akka.http.scaladsl.server.Directives
import akka.http.scaladsl.server.Route

import io.geoalert.mapflow.rest.RestImplicits
import io.geoalert.mapflow.service.Services

object HeartbeatResource extends Directives with RestImplicits with Services {
  def readiness: Route = (path("heartbeat") & get) {
    toComplete(healthCheckService.heartbeat())
  }
  def liveness: Route = (path("heartbeat" / "lite") & get) {
    complete("OK")
  }

  val routes: Route = concat(liveness, readiness)
}
