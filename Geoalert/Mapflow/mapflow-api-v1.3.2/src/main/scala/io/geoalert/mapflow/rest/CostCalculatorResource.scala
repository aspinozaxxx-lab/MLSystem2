package io.geoalert.mapflow.rest

import akka.http.scaladsl.server.Directives
import akka.http.scaladsl.server.Route
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import io.circe.generic.auto.exportDecoder
import io.geoalert.mapflow.rest.json.CalculateCostInput
import io.geoalert.mapflow.service.Services

import geotrellis.vector._

object CostCalculatorResource
    extends Directives
       with Authorization
       with RestImplicits
       with Services {
  def estimateCost: Route =
    (path("processing" / "cost") & post & authorized & entity(as[CalculateCostInput])) {
      (user, input) =>
        toComplete(costCalculatorService.estimateCost(input)(user))
    }

  val routes: Route = concat(estimateCost)
}
