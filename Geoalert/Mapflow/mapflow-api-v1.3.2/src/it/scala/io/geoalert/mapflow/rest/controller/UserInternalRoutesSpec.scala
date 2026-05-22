package io.geoalert.mapflow.rest.controller

import scala.concurrent.duration._

import akka.http.scaladsl.model.HttpHeader
import akka.http.scaladsl.model.StatusCodes
import akka.http.scaladsl.testkit.RouteTestTimeout
import akka.http.scaladsl.testkit.ScalatestRouteTest
import akka.testkit.TestDuration
import cats.syntax.option._
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import io.circe.generic.auto.exportDecoder
import io.circe.generic.auto.exportEncoder
import io.geoalert.mapflow.Config.defaultAreaLimit
import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.HttpServer
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.rest.json.CreateUserInputJson
import io.geoalert.mapflow.rest.json.UpdateUserInputJson
import io.geoalert.mapflow.rest.json.UserJson
import io.geoalert.mapflow.rest.json.UserStatusJson
import io.geoalert.mapflow.util.TestAuthenticationUtils
import io.geoalert.mapflow.util.UserUtil

class UserInternalRoutesSpec extends DbIntegrationTest with ScalatestRouteTest {
  implicit val timeout: RouteTestTimeout = RouteTestTimeout(30.seconds.dilated)
//  describe("User management API") {
//    it("should return user by login") {
//      val auth: HttpHeader = TestAuthenticationUtils.authorizationHeaderApiKey()
//
//      val user = UserUtil.createUser("97228df6-5dd9-482e-8e9a-8bd98067b21e")
//
//      Get(s"/api/v0/users/${user.email}") ~> addHeader(auth) ~!> HttpServer.route ~> check {
//        status should be(StatusCodes.OK)
//        val response = responseAs[UserJson]
//
//        response.email should be(user.email)
//        response.role should be(Role.User)
//      }
//    }
//
//    it("should create user") {
//      val auth: HttpHeader = TestAuthenticationUtils.authorizationHeaderApiKey()
//
//      val input =
//        CreateUserInputJson("user-x@example.com", "pA55w0rd".some, 1L.some, 1L.some, 1L.some)
//
//      Post(s"/api/v0/users", input) ~> addHeader(auth) ~!> HttpServer.route ~> check {
//        status should be(StatusCodes.OK)
//        val response = responseAs[UserJson]
//
//        response.email should be(input.email)
//        response.role should be(Role.User)
//        response.areaLimit should be(1L)
//      }
//    }
//
//    it("should update user") {
//      val auth: HttpHeader = TestAuthenticationUtils.authorizationHeaderApiKey()
//
//      UserUtil.createUser("user-x@example.com")
//
//      val input =
//        UpdateUserInputJson("user-x@example.com", "password".some, 2L.some, 2L.some, 2L.some)
//
//      Put(s"/api/v0/users/${input.email}", input) ~> addHeader(auth) ~!> HttpServer.route ~> check {
//        status should be(StatusCodes.OK)
//        val response = responseAs[UserJson]
//
//        response.email should be(input.email)
//        response.role should be(Role.User)
//        response.areaLimit should be(2L)
//      }
//    }
//
//    it("should return user status by login") {
//      val auth: HttpHeader = TestAuthenticationUtils.authorizationHeaderApiKey()
//
//      val user = UserUtil.createUser("97228df6-5dd9-482e-8e9a-8bd98067b21e")
//
//      Get(s"/api/v0/users/${user.email}/status") ~> addHeader(auth) ~!> HttpServer.route ~> check {
//        status should be(StatusCodes.OK)
//        val response = responseAs[UserStatusJson]
//
//        response.email should be(user.email)
//        response.remainingArea should be(defaultAreaLimit)
//      }
//    }
//  }
}
