package io.geoalert.mapflow.rest

import scala.concurrent.duration._

import akka.http.scaladsl.server.Directives
import akka.http.scaladsl.server.Route

import io.geoalert.mapflow.rest.internal.BillingResource
import io.geoalert.mapflow.rest.internal.HeartbeatResource
import io.geoalert.mapflow.rest.internal.InternalUserResource

object RestRoute extends Directives {
  val publicApiRoutes: Route = (pathPrefix("rest") & withRequestTimeout(60.minutes)) {
    concat(
      ProjectResource.routes,
      ProcessingResource.routes,
      SystemResource.routes,
      UserResource.routes,
      MetaResource.routes,
      ResultResource.routes,
      CostCalculatorResource.routes,
      CatalogResource.routes,
      ConfigurationResource.routes,
      TeamResource.routes,
      ProcessingReviewResource.routes,
    )
  }

  val internalApiRoutes: Route = (pathPrefix("api" / "v0") & withRequestTimeout(60.minutes)) {
    concat(
      HeartbeatResource.routes,
      InternalUserResource.routes,
      BillingResource.routes,
    )
  }
}
