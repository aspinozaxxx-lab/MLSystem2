package io.geoalert.mapflow.rest

import java.time.Instant

import scala.concurrent.duration._

import akka.http.scaladsl.model.HttpHeader
import akka.http.scaladsl.model.StatusCodes
import akka.http.scaladsl.testkit.RouteTestTimeout
import akka.http.scaladsl.testkit.ScalatestRouteTest
import akka.testkit.TestDuration
import cats.syntax.option._
import cats.syntax.traverse._
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import doobie.implicits._
import io.circe.generic.auto.exportDecoder
import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.HttpServer
import io.geoalert.mapflow.model.CreateTeamInput
import io.geoalert.mapflow.model.Processing
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.model.TeamMemberRole
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.model.WorkflowSummary
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.service.Services
import io.geoalert.mapflow.service.billing.BillingReport
import io.geoalert.mapflow.service.billing.ProcessingReport
import io.geoalert.mapflow.util.AoiUtil
import io.geoalert.mapflow.util.GeometryUtil
import io.geoalert.mapflow.util.ProcessingUtil
import io.geoalert.mapflow.util.TestAuthenticationUtils
import io.geoalert.mapflow.util.UserUtil

class BillingRoutesSpec extends DbIntegrationTest with ScalatestRouteTest with Services {
  implicit val timeout: RouteTestTimeout = RouteTestTimeout(30.seconds.dilated)

  describe("Billing report resource") {
    /*it("should return user's complete processings") {
      val auth: HttpHeader = TestAuthenticationUtils.authorizationHeaderApiKey()

      val user = UserUtil.createUser(
        "97228df6-5dd9-482e-8e9a-8bd98067b21e",
        areaLimit = 100_000_000L.some,
        aoiAreaLimit = 100_000_000L.some,
      )

      val processing = ProcessingUtil.createProcessing(user)
      ProcessingUtil.completeProcessing(processing)(user)
      createIncompleteProcessing(user)

      val end = Instant.now().plusSeconds(300)
      Get(
        s"/api/v0/users/${user.email}/billing?start_date=2020-02-28T17:00:00Z&end_date=$end"
      ) ~> addHeader(auth) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
        val report = responseAs[BillingReport]

        report.processings should matchPattern {
          case List(
                 ProcessingReport(
                   processing.id,
                   "97228df6-5dd9-482e-8e9a-8bd98067b21e",
                   processing.name,
                   42_000_000,
                   210,
                   _,
                   "API",
                   false,
                 )
               ) =>
        }
      }
    }*/

    /*it("should return team members complete processings") {
      val auth: HttpHeader = TestAuthenticationUtils.authorizationHeaderApiKey()

      val owner = UserUtil.createUser(
        "0a8f9ebb-1d8e-4144-b454-ad3cee552ec9",
        areaLimit = 100_000_000L.some,
        aoiAreaLimit = 100_000_000L.some,
      )
      val member = UserUtil.createUser(
        "b6a716a5-8689-4aef-8efe-895d6e565037",
        areaLimit = 100_000_000L.some,
        aoiAreaLimit = 100_000_000L.some,
      )

      val teamA = teamService
        .createTeam(CreateTeamInput("A-Team"))(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

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
      val processing = ProcessingUtil.createProcessing(member)
      ProcessingUtil.completeProcessing(processing)(member)
      createIncompleteProcessing(member)

      Get(s"/api/v0/users/${owner.email}/billing") ~> addHeader(
        auth
      ) ~!> HttpServer.route ~> check {
        status should be(StatusCodes.OK)
        val report = responseAs[BillingReport]

        report.processings should matchPattern {
          case List(
                 ProcessingReport(
                   processing.id,
                   "b6a716a5-8689-4aef-8efe-895d6e565037",
                   processing.name,
                   42_000_000L,
                   210,
                   _,
                   "API",
                   false,
                 )
               ) =>
        }
      }
    }*/

  }

  def createIncompleteProcessing(user: User): Processing = {
    val processing = ProcessingUtil.createProcessing(user)
    val aois = AoiUtil.createAois(processing, GeometryUtil.createPolygon(42_000_000L))(user)

    aoiService
      .updateAoiStatusAndVrt(aois.map(_.id), Status.Ok, None)
      .transact(xa)
      .unsafeRunSync()

    val wfs = workflowService
      .findWorkflowsWithRequiredActions()
      .transact(xa)
      .unsafeRunSync()
      .filter(wf => aois.map(_.id).contains(wf.aoiId))

    wfs
      .tail
      .traverse(wf =>
        workflowService.updateWorkflowStatus(
          WorkflowSummary(wf),
          Status.Ok,
          Instant.now(),
          None,
          isScheduledUpdate = true,
          List(),
        )
      )
      .rethrowT
      .transact(xa)
      .unsafeRunSync()

    processing
  }
}
