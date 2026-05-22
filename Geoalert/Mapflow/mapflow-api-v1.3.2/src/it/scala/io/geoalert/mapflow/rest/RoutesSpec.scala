package io.geoalert.mapflow.rest

import akka.http.scaladsl.model.ContentTypes.`application/json`
import akka.http.scaladsl.model.StatusCodes
import akka.http.scaladsl.model.headers.HttpOriginRange
import akka.http.scaladsl.model.headers.`Access-Control-Allow-Origin`
import akka.http.scaladsl.model.headers.`Content-Type`
import akka.http.scaladsl.testkit.ScalatestRouteTest

import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.HttpServer

class RoutesSpec extends DbIntegrationTest with ScalatestRouteTest {
  describe("/rest/version") {
    it("should return API version") {
      Get("/rest/version") ~> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
        responseAs[String] should be("1")
      }
    }
  }

  describe("/api/v0/heartbeat") {
    it("should return \"OK\"") {
      Get("/api/v0/heartbeat") ~> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
        responseAs[String] should be("\"OK\"")
      }
    }
  }

  describe("/rest/config/keycloak.json") {
    it("should return \"OK\"") {
      Get("/rest/config/keycloak.json") ~> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
        header[`Content-Type`] should be(Some(`Content-Type`(`application/json`)))
        header[`Access-Control-Allow-Origin`] should be(
          Some(`Access-Control-Allow-Origin`(HttpOriginRange.*))
        )
      }
    }
  }

  describe("/unknown/resource route") {
    it("Should return 404") {
      Get("/unknown/resource") ~> HttpServer.route ~> check {
        status should be(StatusCodes.NotFound)
      }
    }
  }
}
