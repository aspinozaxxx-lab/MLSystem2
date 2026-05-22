package io.geoalert.mapflow.rest

import akka.http.scaladsl.server.Directives
import akka.http.scaladsl.server.Route

import io.geoalert.mapflow.Config.wmApiVersion
import io.geoalert.mapflow.service.Services

object SystemResource extends Directives with RestImplicits with Services {

  /** @return maximum supported API version. This version is used for compatibility check by QGIS plugin.
    */
  def apiVersion: Route = (path("version") & get) {
    complete(wmApiVersion)
  }

  val routes: Route = apiVersion
}
