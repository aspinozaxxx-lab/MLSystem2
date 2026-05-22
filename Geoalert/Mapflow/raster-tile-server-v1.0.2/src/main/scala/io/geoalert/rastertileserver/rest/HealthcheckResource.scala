package io.geoalert.rastertileserver.rest

import akka.http.scaladsl.server.{Directives, Route}
import io.prometheus.client.CollectorRegistry
import io.prometheus.client.exporter.common.TextFormat

import java.io.StringWriter

object HealthcheckResource extends Directives {
  def heartbeat: Route = (path("heartbeat") & get) {
    complete("OK")
  }

  def metrics:Route = (path("metrics") & get) {
    val writer = new StringWriter()
    try {
      TextFormat.write004(writer, CollectorRegistry.defaultRegistry.metricFamilySamples())
      complete(writer.toString)
    } finally {
      writer.close()
    }
  }

  def routes: Route = concat(heartbeat, metrics)
}
