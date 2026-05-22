package io.geoalert.mapflow.rest.internal

import java.time.Instant

import akka.http.scaladsl.server.Directives
import akka.http.scaladsl.server.Route

import io.geoalert.mapflow.rest.Authorization
import io.geoalert.mapflow.rest.RestImplicits
import io.geoalert.mapflow.service.Services

object BillingResource extends Directives with Authorization with RestImplicits with Services {
  def getUserBilling: Route = (path("users" / Segment / "billing") & get & authorized & parameters(
    Symbol("start_date").as[String].optional,
    Symbol("end_date").as[String].optional,
  )) { (email, user, startDate, endDate) =>
    val result = for {
      user <- userService.getUserByEmail(email)(user)
      report <- billingReportService.generateBillingReport(
        email,
        startDate.map(Instant.parse),
        endDate.map(Instant.parse),
      )(user)
    } yield report

    toComplete(result)
  }

  val routes: Route = concat(
    getUserBilling
  )
}
