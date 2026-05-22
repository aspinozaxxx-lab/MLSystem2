package io.geoalert.mapflow.rest

import scala.concurrent.duration._
import scala.io.Source

import akka.http.scaladsl.model.HttpHeader
import akka.http.scaladsl.model.StatusCodes
import akka.http.scaladsl.testkit.RouteTestTimeout
import akka.http.scaladsl.testkit.ScalatestRouteTest
import akka.testkit.TestDuration
import cats.syntax.option._
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import doobie.implicits._
import io.circe.generic.auto.exportDecoder

import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.HttpServer
import io.geoalert.mapflow.model.ReviewStatus
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.rest.json.ProcessingJson
import io.geoalert.mapflow.rest.json.ProcessingReviewDetailsJson
import io.geoalert.mapflow.rest.json.ProcessingReviewInputJson
import io.geoalert.mapflow.rest.json.ProcessingReviewJson
import io.geoalert.mapflow.service.Services
import io.geoalert.mapflow.util.ProcessingUtil
import io.geoalert.mapflow.util.TestAuthenticationUtils
import io.geoalert.mapflow.util.UserUtil

import geotrellis.vector.io.json.GeoJson
import geotrellis.vector.io.json.JsonFeatureCollection

class ProcessingReviewRoutesSpec extends DbIntegrationTest with ScalatestRouteTest with Services {
  implicit val timeout: RouteTestTimeout = RouteTestTimeout(5.seconds.dilated)

  describe("Processing Review Workflow") {
    it("should not confirm transaction until the results were reviewed") {
      val user = UserUtil.createUser("31d33710-07d2-40a3-b27a-a31995d07595", reviewWorkflowEnabled = true.some)
      val processing = ProcessingUtil.createProcessing(user)
      ProcessingUtil.completeProcessing(processing)(user)

      val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(user)
      Get(s"/rest/processings/${processing.id}") ~> addHeader(auth) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
        val json = responseAs[ProcessingJson]

        json.reviewStatus should matchPattern {
          case Some(ProcessingReviewJson(ReviewStatus.InReview, Some(_))) =>
        }
      }
    }

    it("user should be able to accept processing") {
      val user = UserUtil.createUser("31d33710-07d2-40a3-b27a-a31995d07595", reviewWorkflowEnabled = true.some)
      val processing = ProcessingUtil.createProcessing(user)
      ProcessingUtil.completeProcessing(processing)(user)

      val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(user)
      Put(s"/rest/processings/${processing.id}/acceptation") ~> addHeader(
        auth
      ) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
        val res = responseAs[String]

        res should be("OK")
      }
      Get(s"/rest/processings/${processing.id}") ~> addHeader(auth) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
        val json = responseAs[ProcessingJson]

        json.reviewStatus should matchPattern {
          case Some(ProcessingReviewJson(ReviewStatus.Accepted, None)) =>
        }
      }
    }

    it("user should be reject processings with a comment and feature collection") {
      val user = UserUtil.createUser("31d33710-07d2-40a3-b27a-a31995d07595", reviewWorkflowEnabled = true.some)
      val processing = ProcessingUtil.createProcessing(user)
      ProcessingUtil.completeProcessing(processing)(user)

      val json =
        Source.fromResource("five_geometries.geojson").getLines().toList.reduce(_ + "\n" + _)
      val features = GeoJson.parse[JsonFeatureCollection](json)
      val input = ProcessingReviewInputJson("Some buildings weren't detected", features.some)

      val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(user)
      Put(s"/rest/processings/${processing.id}/rejection", input) ~> addHeader(
        auth
      ) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
      }

      Get(s"/rest/processings/${processing.id}/review") ~> addHeader(
        auth
      ) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.OK)

        val review = responseAs[ProcessingReviewDetailsJson]
        review should matchPattern {
          case ProcessingReviewDetailsJson(
                 ReviewStatus.NotAccepted,
                 None,
                 Some("Some buildings weren't detected"),
                 Some(_),
               ) =>
        }

        Get(review.featuresUri.get) ~> addHeader(auth) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)

          val review = responseAs[JsonFeatureCollection]

          review.asJson should be(features.asJson)
        }
      }
    }

    it("administrator should be able to refund") {
      val user = UserUtil.createUser("31d33710-07d2-40a3-b27a-a31995d07595", reviewWorkflowEnabled = true.some)
      val processing = ProcessingUtil.createProcessing(user)
      ProcessingUtil.completeProcessing(processing)(user)

      val input = ProcessingReviewInputJson("Some buildings weren't detected", none)

      val authUser: HttpHeader = TestAuthenticationUtils.authorizationHeader(user)
      Put(s"/rest/processings/${processing.id}/rejection", input) ~> addHeader(
        authUser
      ) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
      }

      val res = reviewService
        .refund(processing.id)(UserUtil.admin)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      res should be("OK")

    }
  }
}
