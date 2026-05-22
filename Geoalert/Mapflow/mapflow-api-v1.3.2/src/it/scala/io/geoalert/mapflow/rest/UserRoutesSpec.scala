package io.geoalert.mapflow.rest

import scala.concurrent.duration._

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
import io.geoalert.mapflow.model.BillingType
import io.geoalert.mapflow.model.CreateTeamInput
import io.geoalert.mapflow.model.TeamMemberRole
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.rest.json.UserJson
import io.geoalert.mapflow.rest.json.UserStatusJson
import io.geoalert.mapflow.rest.json.WorkflowDefJson
import io.geoalert.mapflow.service.Services
import io.geoalert.mapflow.util.TestAuthenticationUtils
import io.geoalert.mapflow.util.UserUtil
import io.geoalert.mapflow.util.WorkflowDefUtil

class UserRoutesSpec extends DbIntegrationTest with ScalatestRouteTest with Services {
  implicit val timeout: RouteTestTimeout = RouteTestTimeout(30.seconds.dilated)

  describe("User resource") {
    it("should return user status for authorized user") {
      val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

      val defaultWdId = WorkflowDefUtil.createWd()
      WorkflowDefUtil.createWd(isDefault = false)

      Get("/rest/user/status") ~> addHeader(auth) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
        val userStatus = responseAs[UserStatusJson]

        userStatus.email should be(UserUtil.regularUser.email)
        userStatus.remainingArea should be(50_000_000L)
        userStatus.areaLimit should be(50_000_000L)
        userStatus.memoryLimit should be(1_000_000_000L)
        userStatus.maxAoisPerProcessing should be(10)
        userStatus.models.map(_.id) should be(List(defaultWdId))
        userStatus.billingType should be(BillingType.None.repr)
      }
    }

    describe("Linking WDs to user") {
      it("should be able to link and unlink custom WD to/from a user") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        val customWdId = WorkflowDefUtil.createWd(isDefault = false)
        Get(s"/rest/users/${UserUtil.regularUser.id}/models") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
          val models = responseAs[List[WorkflowDefJson]]
          models.map(_.id) should be(List())
        }

        Post(s"/rest/users/${UserUtil.regularUser.id}/models/$customWdId") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
        }

        Get(s"/rest/users/${UserUtil.regularUser.id}/models") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
          val models = responseAs[List[WorkflowDefJson]]
          models.map(_.id) should be(List(customWdId))
        }

        Delete(s"/rest/users/${UserUtil.regularUser.id}/models/$customWdId") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
        }

        Get(s"/rest/users/${UserUtil.regularUser.id}/models") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
          val models = responseAs[List[WorkflowDefJson]]
          models.map(_.id) should be(List())
        }
      }
    }
    describe("Team accounts should share WDs") {
      it("should be able to see WDs linked to a team owner") {
        val owner = UserUtil.createUser("0a8f9ebb-1d8e-4144-b454-ad3cee552ec9")
        // val member = UserUtil.createUser("b6a716a5-8689-4aef-8efe-895d6e565037")
        val customWdId = WorkflowDefUtil.createWd(isDefault = false)

        workflowDefService
          .linkWorkflowDefToUser(customWdId, owner.id)(UserUtil.admin)
          .transact(xa)
          .unsafeRunSync()

        val team = teamService
          .createTeam(CreateTeamInput("A-Team"))(UserUtil.admin)
          .transact(xa)
          .unsafeRunSync()

        teamService
          .linkUserToTeam(
            team.id,
            owner.email,
            TeamMemberRole.OWNER,
            None,
            None,
            None,
            failToLinkExistingUser = true,
          )(UserUtil.admin)
          .transact(xa)
          .unsafeRunSync()

        teamService
          .linkUserToTeam(
            team.id,
            "b6a716a5-8689-4aef-8efe-895d6e565037",
            TeamMemberRole.MEMBER,
            None,
            5_000L.some,
            None,
            failToLinkExistingUser = true,
          )(UserUtil.admin)
          .transact(xa)
          .unsafeRunSync()

        val auth: HttpHeader =
          TestAuthenticationUtils.authorizationHeader(UserUtil.getUserByEmail("b6a716a5-8689-4aef-8efe-895d6e565037"))
        Get("/rest/user/status") ~> addHeader(auth) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
          val userStatus = responseAs[UserStatusJson]

          userStatus.email should be("b6a716a5-8689-4aef-8efe-895d6e565037")
          userStatus.remainingArea should be(5_000L)
          userStatus.areaLimit should be(5_000L)
          userStatus.memoryLimit should be(1_000_000_000)
          userStatus.models.map(_.id) should be(List(customWdId))
        }
      }
    }
//    describe("Internal user routes") {
//      it("should allow API user see all users") {
//        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeaderApiKey()
//
//        UserUtil.createUser("1@example.com")
//        UserUtil.createUser("2@example.com")
//        UserUtil.createUser("3@example.com")
//        UserUtil.createUser("4@example.com")
//        UserUtil.createUser("5@example.com")
//
//        Get("/api/v0/users?offset=2&limit=3") ~> addHeader(auth) ~!> HttpServer.route ~> check {
//          status should be(StatusCodes.OK)
//
//          val users = responseAs[List[UserJson]]
//
//          users.map(_.email) should be(List("3@example.com", "4@example.com", "5@example.com"))
//        }
//      }
//
//      it("should archive user") {
//        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeaderApiKey()
//
//        val user = UserUtil.createUser("97228df6-5dd9-482e-8e9a-8bd98067b21e")
//
//        Delete(s"/api/v0/users/${user.email}") ~> addHeader(auth) ~!> HttpServer.route ~> check {
//          status should be(StatusCodes.OK)
//
//          val res = responseAs[String]
//          res should be("OK")
//
//          val userOpt = userService
//            .getUser(user.id)(UserUtil.admin)
//            .toOption
//            .value
//            .transact(xa)
//            .unsafeRunSync()
//          userOpt should be(None)
//        }
//      }
//
//      it("should link paid data providers to a user") {
//        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeaderApiKey()
//
//        val user = UserUtil.createUser("97228df6-5dd9-482e-8e9a-8bd98067b21e")
//
//        Put(s"/api/v0/users/${user.email}/data_providers") ~> addHeader(
//          auth
//        ) ~!> HttpServer.route ~> check {
//          status should be(StatusCodes.OK)
//
//          val res = responseAs[String]
//          res should be("OK")
//          val usr = userService
//            .getUser(user.id)(UserUtil.admin)
//            .rethrowT
//            .transact(xa)
//            .unsafeRunSync()
//
//          usr.availableDataProviders.map(_.name).toSet should be(
//            Set("securewatch", "vivid", "Mapbox", "arcgis_world_imagery")
//          )
//        }
//      }
//    }
  }
}
