package io.geoalert.mapflow.rest

import java.time.Instant
import java.util.UUID

import akka.http.scaladsl.model.StatusCodes
import akka.http.scaladsl.testkit.ScalatestRouteTest
import cats.syntax.option._
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import doobie.implicits._
import io.circe.generic.auto.exportDecoder
import io.geoalert.mapflow.Config
import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.HttpServer
import io.geoalert.mapflow.model.BillingType
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.rest.json.UserStatusJson
import io.geoalert.mapflow.service.Services
import io.geoalert.mapflow.service.avanpost.AvanpostClient.AvanpostTestAdminId
import io.geoalert.mapflow.service.avanpost.AvanpostClient.AvanpostTestAdminId2
import io.geoalert.mapflow.util.TestAuthenticationUtils
import io.geoalert.mapflow.util.UserUtil

class AuthorizationSpec extends DbIntegrationTest with ScalatestRouteTest with Services {
  describe("Synchronize DB user with JWT token") {
    it("should create new user after the first login with JWT token") {
      val tokenUser = 
        UUID.randomUUID(),
        AvanpostTestAdminId.toString,
        Role.Admin,
        Config.defaultAreaLimit,
        Config.defaultAoiAreaLimit,
        BillingType.Area,
        Instant.now(),
        Instant.now(),
        0,
        Config.defaultMemoryLimit,
        Config.maxAoisPerProcessing,
        List(),
        none,
        reviewWorkflowEnabled = false,
        none,
        none,
        none,
      )

      val auth = TestAuthenticationUtils.authorizationHeader(tokenUser)

      Get("/rest/user/status") ~> addHeader(auth) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
        val userStatus = responseAs[UserStatusJson]

        userStatus.email should be(AvanpostTestAdminId.toString)
        userStatus.areaLimit should be(userStatus.areaLimit)
        userStatus.memoryLimit should be(Config.defaultMemoryLimit)

        val createdUser = userService
          .getUsers(None, Seq(AvanpostTestAdminId.toString).some, None)(UserUtil.admin)
          .transact(xa)
          .unsafeRunSync()
          .head

        createdUser.role should be(Role.Admin)
      }
    }

    it("should update existing user role after login with JWT token") {
      val user = UserUtil.createUser(
        AvanpostTestAdminId2.toString,
        Role.Admin.some,
        areaLimit = 1_000_000L.some,
        aoiAreaLimit = 1_000_000L.some,
      )

      val auth = TestAuthenticationUtils.authorizationHeader(user)

      Get("/rest/user/status") ~> addHeader(auth) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
        val userStatus = responseAs[UserStatusJson]

        userStatus.email should be(AvanpostTestAdminId2.toString)
        userStatus.areaLimit should be(userStatus.areaLimit)
        userStatus.memoryLimit should be(Config.defaultMemoryLimit)

        val createdUser = userService
          .getUsers(None, Seq(AvanpostTestAdminId2.toString).some, None)(UserUtil.admin)
          .transact(xa)
          .unsafeRunSync()
          .head

        createdUser.role should be(Role.Admin)
      }
    }
  }
}
