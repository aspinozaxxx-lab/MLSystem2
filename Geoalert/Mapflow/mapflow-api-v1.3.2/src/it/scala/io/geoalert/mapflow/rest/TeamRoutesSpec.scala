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
import io.circe.generic.auto.exportEncoder
import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.HttpServer
import io.geoalert.mapflow.model.CreateTeamInput
import io.geoalert.mapflow.model.TeamMemberRole
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.rest.json.MembershipJson
import io.geoalert.mapflow.rest.json.TeamMemberJson
import io.geoalert.mapflow.rest.json.WorkflowDefJson
import io.geoalert.mapflow.service.Services
import io.geoalert.mapflow.util.TestAuthenticationUtils
import io.geoalert.mapflow.util.UserUtil
import io.geoalert.mapflow.util.WorkflowDefUtil

class TeamRoutesSpec extends DbIntegrationTest with ScalatestRouteTest with Services {
  implicit val timeout: RouteTestTimeout = RouteTestTimeout(5.seconds.dilated)

  describe("Team routes") {
    it("owner should list team members") {
      val team = teamService
        .createTeam(CreateTeamInput("A-Team"))(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      val owner =
        UserUtil.createUser("97228df6-5dd9-482e-8e9a-8bd98067b21e", areaLimit = 100_000L.some)
      teamService
        .linkUserToTeam(
          team.id,
          "97228df6-5dd9-482e-8e9a-8bd98067b21e",
          TeamMemberRole.OWNER,
          None,
          100_000L.some,
          1000L.some,
          failToLinkExistingUser = false,
        )(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(owner)

      Get(s"/rest/team/${team.id}/member") ~> addHeader(auth) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
        val members = responseAs[List[TeamMemberJson]]

        members should matchPattern {
          case List(
                 TeamMemberJson(
                   _,
                   "97228df6-5dd9-482e-8e9a-8bd98067b21e",
                   TeamMemberRole.OWNER,
                   None,
                   0,
                   100_000L,
                   100_000L,
                   0,
                   0,
                   1000,
                 )
               ) =>
        }
      }
    }

    it("owner should be able to link and unlink users") {
      val team = teamService
        .createTeam(CreateTeamInput("A-Team"))(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      val owner =
        UserUtil.createUser("97228df6-5dd9-482e-8e9a-8bd98067b21e", areaLimit = 100_000L.some)
      teamService
        .linkUserToTeam(
          team.id,
          "97228df6-5dd9-482e-8e9a-8bd98067b21e",
          TeamMemberRole.OWNER,
          None,
          None,
          None,
          failToLinkExistingUser = false,
        )(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(owner)

      Post(
        s"/rest/team/${team.id}/member/b6a716a5-8689-4aef-8efe-895d6e565037",
        MembershipJson(None, None, None, None),
      ) ~> addHeader(auth) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
      }

      Get(s"/rest/team/${team.id}/member") ~> addHeader(auth) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
        val members = responseAs[List[TeamMemberJson]]

        members.sortBy(_.email) should matchPattern {
          case List(
                 TeamMemberJson(
                   _,
                   "97228df6-5dd9-482e-8e9a-8bd98067b21e",
                   TeamMemberRole.OWNER,
                   None,
                   0,
                   100_000L,
                   100_000L,
                   0,
                   0,
                   0,
                 ),
                 TeamMemberJson(
                   _,
                   "b6a716a5-8689-4aef-8efe-895d6e565037",
                   TeamMemberRole.MEMBER,
                   None,
                   0,
                   100_000L,
                   50_000_000L,
                   0,
                   0,
                   0,
                 ),
               ) =>
        }
      }

      Delete(s"/rest/team/${team.id}/member/b6a716a5-8689-4aef-8efe-895d6e565037") ~> addHeader(
        auth
      ) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
      }

      Get(s"/rest/team/${team.id}/member") ~> addHeader(auth) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
        val members = responseAs[List[TeamMemberJson]]

        members should matchPattern {
          case List(
                 TeamMemberJson(
                   _,
                   "97228df6-5dd9-482e-8e9a-8bd98067b21e",
                   TeamMemberRole.OWNER,
                   None,
                   0,
                   100_000L,
                   100_000L,
                   0,
                   0,
                   0,
                 )
               ) =>
        }
      }
    }

    it("regular user cannot see teams") {
      val team = teamService
        .createTeam(CreateTeamInput("A-Team"))(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      teamService
        .linkUserToTeam(
          team.id,
          "97228df6-5dd9-482e-8e9a-8bd98067b21e",
          TeamMemberRole.OWNER,
          None,
          None,
          None,
          failToLinkExistingUser = false,
        )(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      val member = UserUtil.createUser("b6a716a5-8689-4aef-8efe-895d6e565037")
      teamService
        .linkUserToTeam(
          team.id,
          "b6a716a5-8689-4aef-8efe-895d6e565037",
          TeamMemberRole.MEMBER,
          None,
          None,
          None,
          failToLinkExistingUser = false,
        )(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      val authRegular: HttpHeader =
        TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)
      Get(s"/rest/team/${team.id}/member") ~> addHeader(authRegular) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.NotFound)
      }

      val authMember: HttpHeader = TestAuthenticationUtils.authorizationHeader(member)
      Get(s"/rest/team/${team.id}/member") ~> addHeader(authMember) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.NotFound)
      }
    }

    it("only one owner can be linked to a team") {
      val team = teamService
        .createTeam(CreateTeamInput("A-Team"))(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      teamService
        .linkUserToTeam(
          team.id,
          "97228df6-5dd9-482e-8e9a-8bd98067b21e",
          TeamMemberRole.OWNER,
          None,
          None,
          None,
          failToLinkExistingUser = false,
        )(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.admin)
      Post(
        s"/rest/team/${team.id}/member/b6a716a5-8689-4aef-8efe-895d6e565037",
        MembershipJson(TeamMemberRole.OWNER.some, None, None, None),
      ) ~> addHeader(auth) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.BadRequest)
      }
    }

    it("user can be linked to a single team only") {
      val teamA = teamService
        .createTeam(CreateTeamInput("A-Team"))(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      val teamB = teamService
        .createTeam(CreateTeamInput("B-Team"))(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      teamService
        .linkUserToTeam(
          teamA.id,
          "97228df6-5dd9-482e-8e9a-8bd98067b21e",
          TeamMemberRole.OWNER,
          None,
          None,
          None,
          failToLinkExistingUser = false,
        )(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.admin)
      Post(
        s"/rest/team/${teamB.id}/member/97228df6-5dd9-482e-8e9a-8bd98067b21e",
        MembershipJson(None, None, None, None),
      ) ~> addHeader(auth) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.BadRequest)
      }
    }

    it("user in a team shares owner's WDs") {
      val teamA = teamService
        .createTeam(CreateTeamInput("A-Team"))(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      val owner = UserUtil.createUser("97228df6-5dd9-482e-8e9a-8bd98067b21e")
      val member = UserUtil.createUser("b6a716a5-8689-4aef-8efe-895d6e565037")

      teamService
        .linkUserToTeam(
          teamA.id,
          owner.email,
          TeamMemberRole.OWNER,
          None,
          None,
          None,
          failToLinkExistingUser = false,
        )(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      teamService
        .linkUserToTeam(
          teamA.id,
          member.email,
          TeamMemberRole.MEMBER,
          None,
          None,
          None,
          failToLinkExistingUser = false,
        )(owner)
        .transact(xa)
        .unsafeRunSync()

      val wdId = WorkflowDefUtil.createWd(isDefault = false)
      workflowDefService
        .linkWorkflowDefToUser(wdId, owner.id)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(member)
      Get(s"/rest/users/${member.id}/models") ~> addHeader(auth) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
        val models = responseAs[List[WorkflowDefJson]]
        models.map(_.id) should be(List(wdId))
      }
    }
  }
}
