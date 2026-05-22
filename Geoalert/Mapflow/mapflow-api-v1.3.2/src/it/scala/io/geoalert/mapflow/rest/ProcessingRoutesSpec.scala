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
import io.geoalert.mapflow.model.Rating
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.model.UserProject
import io.geoalert.mapflow.model.enums.MemberRole
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.repo.UserProjectsRepo
import io.geoalert.mapflow.rest.json.AoiJson
import io.geoalert.mapflow.rest.json.BlockParametersJson
import io.geoalert.mapflow.rest.json.CreateAndRunProcessingJson
import io.geoalert.mapflow.rest.json.CreateProcessingRatingJson
import io.geoalert.mapflow.rest.json.Decoders
import io.geoalert.mapflow.rest.json.Encoders
import io.geoalert.mapflow.rest.json.ProcessingJson
import io.geoalert.mapflow.rest.json.UpdateProcessingInputJson
import io.geoalert.mapflow.service.Services
import io.geoalert.mapflow.util.AoiUtil
import io.geoalert.mapflow.util.GeometryUtil
import io.geoalert.mapflow.util.ProcessingUtil
import io.geoalert.mapflow.util.ProjectUtil
import io.geoalert.mapflow.util.TestAuthenticationUtils
import io.geoalert.mapflow.util.UserUtil
import io.geoalert.mapflow.util.WorkflowDefUtil

class ProcessingRoutesSpec
    extends DbIntegrationTest
       with ScalatestRouteTest
       with Encoders
       with Decoders
       with Services {
  implicit val timeout: RouteTestTimeout = RouteTestTimeout(5.seconds.dilated)

  describe("Processing resource") {
    describe("Authorized user") {
      it("should be able to create processing using WD name") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)
        WorkflowDefUtil.createWd("Buildings Detection")

        val entity = CreateAndRunProcessingJson(
          name = "My Processing".some,
          description = "description".some,
          projectId = None,
          wdName = "Buildings Detection".some,
          wdId = None,
          geometry = GeometryUtil.fromExtent(0, 0, 0.01, 0.01),
          params = Map("data_provider" -> "mapbox").some,
          meta = Map("source-app" -> "TEST").some,
          none,
        )

        Post("/rest/processings", entity) ~> addHeader(auth) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)

          val processing = responseAs[ProcessingJson]
          processing.name should be("My Processing")
          processing.status should be(Status.InProgress)
          processing.params should be(Map("source_type" -> "xyz", "data_provider" -> "mapbox"))
          processing.meta should be(Map("source-app" -> "TEST"))
          processing.aoiArea should be(1230907)
          processing.aoiCount should be(1)
        }
      }

      it("should be able to create processing in user project") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        val project = ProjectUtil.createProject(UserUtil.regularUser)
        WorkflowDefUtil.createWd("Buildings Detection")

        val entity = CreateAndRunProcessingJson(
          name = "My Processing".some,
          description = "description".some,
          projectId = project.id.some,
          wdName = "Buildings Detection".some,
          wdId = none,
          geometry = GeometryUtil.fromExtent(0, 0, 0.01, 0.01),
          none,
          none,
          Seq(BlockParametersJson("simplification", enabled = true, "Simplification".some)).some,
        )

        Post("/rest/processings", entity) ~> addHeader(auth) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)

          val processing = responseAs[ProcessingJson]

          processing.projectId should be(project.id)
          processing.blocks should matchPattern {
            case Seq(BlockParametersJson("simplification", true, Some("Simplification"))) =>
          }
        }
      }

      it("should be able to retrieve processing") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        val processing = ProcessingUtil.createProcessing(UserUtil.regularUser)

        Get(s"/rest/processings/${processing.id}") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
          val json = responseAs[ProcessingJson]

          json.name should be(processing.name)
        }
      }

      it("should be able to retrieve processing by project id") {
        val user = UserUtil.regularUser
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(user)
        val project = ProjectUtil.createProject(user)
        val processing =
          ProcessingUtil.createProcessing(project.id.some, "Test Processing".some, None)(user)

        Get(s"/rest/projects/${project.id}/processings") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
          val processings = responseAs[List[ProcessingJson]]

          processings.map(_.id) should contain(processing.id)
        }
      }

      it("should be able to retrieve processing by project id if role readOnly") {
        val admin = UserUtil.admin
        val user = UserUtil.regularUser
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(user)
        val project = ProjectUtil.createProject(admin)
        val processing =
          ProcessingUtil.createProcessing(project.id.some, "Test Processing".some, None)(admin)
        projectService
          .shareProject(UserProject(user.id, project.id, MemberRole.Readonly))(admin)
          .transact(xa)
          .rethrowT
          .unsafeRunSync()
        Get(s"/rest/projects/${project.id}/processings") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
          val processings = responseAs[List[ProcessingJson]]

          processings.map(_.id) should contain(processing.id)
        }
      }

      it("should be able to list project processings") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        val project = ProjectUtil.createProject(UserUtil.regularUser)
        val processing = ProcessingUtil.createProcessing(project.id.some)(UserUtil.regularUser)

        Get(s"/rest/projects/${project.id}/processings") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)

          val processings = responseAs[List[ProcessingJson]]

          processings.map(_.id) should contain(processing.id)
        }
      }

      it("should be able to list default project processings") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        val defaultProject = ProjectUtil.defaultProject(UserUtil.regularUser)
        val processing =
          ProcessingUtil.createProcessing(defaultProject.id.some)(UserUtil.regularUser)

        Get(s"/rest/projects/default/processings") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)

          val processings = responseAs[List[ProcessingJson]]

          processings.map(_.id) should contain(processing.id)
        }
      }

      it("should be able to list all processings") {
        val defaultProject = ProjectUtil.defaultProject(UserUtil.regularUser)
        val defaultProjectProcessing =
          ProcessingUtil.createProcessing(defaultProject.id.some)(UserUtil.regularUser)
        val project = ProjectUtil.createProject(UserUtil.regularUser)
        val processing = ProcessingUtil.createProcessing(project.id.some)(UserUtil.regularUser)

        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        Get(s"/rest/processings") ~> addHeader(auth) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)

          val processings = responseAs[List[ProcessingJson]]

          processings.map(_.id) should contain(processing.id)
          processings.map(_.id) should contain(defaultProjectProcessing.id)
        }
      }

      it("should be able to restart processing") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        val project = ProjectUtil.createProject(UserUtil.regularUser)
        val processing = ProcessingUtil.createProcessing(project.id.some)(UserUtil.regularUser)

        Post(s"/rest/processings/${processing.id}/restart") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)

          val result = responseAs[Int]

          result should be(0)
        }
      }

      it("should be able to download processing AOIs") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        val project = ProjectUtil.createProject(UserUtil.regularUser)
        val processing = ProcessingUtil.createProcessing(project.id.some)(UserUtil.regularUser)
        AoiUtil.createAois(processing, GeometryUtil.fromExtent(0, 0, 0.01, 0.01))(
          UserUtil.regularUser
        )

        Get(s"/rest/processings/${processing.id}/aois") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
          val result = responseAs[List[AoiJson]]
          result.map(_.area).sum should be(1230907)
          result.size should be(1)
        }
      }

      it("should be able to archive processing") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        val project = ProjectUtil.createProject(UserUtil.regularUser)
        val processing = ProcessingUtil.createProcessing(project.id.some)(UserUtil.regularUser)

        Delete(s"/rest/processings/${processing.id}") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          Get(s"/rest/processings/${processing.id}") ~> addHeader(
            auth
          ) ~!> HttpServer.route ~> check {
            status should be(StatusCodes.NotFound)
          }
        }
      }

      it("should be able to update processing") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        val project = ProjectUtil.createProject(UserUtil.regularUser)
        val otherProject = ProjectUtil.createProject(UserUtil.regularUser)
        val processing = ProcessingUtil.createProcessing(project.id.some)(UserUtil.regularUser)

        val entity =
          UpdateProcessingInputJson("New Name".some, "New Description".some, otherProject.id.some)

        Put(s"/rest/processings/${processing.id}", entity) ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)

          val processing = responseAs[ProcessingJson]

          processing.name should be("New Name")
          processing.description should be(Some("New Description"))
          processing.projectId should be(otherProject.id)
        }
      }

      it("should not be able to link processing to other user's project") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        val project = ProjectUtil.createProject(UserUtil.regularUser)
        val otherProject = ProjectUtil.createProject(UserUtil.otherUser)
        val processing = ProcessingUtil.createProcessing(project.id.some)(UserUtil.regularUser)

        val entity = UpdateProcessingInputJson(None, None, otherProject.id.some)

        Put(s"/rest/processings/${processing.id}", entity) ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.Forbidden)
        }
      }
    }

    describe("Unauthorized user") {
      it("should not be able to list project processings") {
        val project = ProjectUtil.createProject(UserUtil.regularUser)
        ProcessingUtil.createProcessing(project.id.some)(UserUtil.regularUser)

        Get(s"/rest/projects/${project.id}/processings") ~!> HttpServer.route ~> check {
          status should be(StatusCodes.Unauthorized)
        }
      }

      it("should not be able to retrieve processing") {
        val processing = ProcessingUtil.createProcessing(UserUtil.regularUser)

        Get(s"/rest/processings/${processing.id}") ~!> HttpServer.route ~> check {
          status should be(StatusCodes.Unauthorized)
        }
      }

      it("should not be able to update processing") {
        val project = ProjectUtil.createProject(UserUtil.regularUser)
        val processing = ProcessingUtil.createProcessing(project.id.some)(UserUtil.regularUser)

        val entity =
          UpdateProcessingInputJson("New Name".some, "New Description".some, project.id.some)

        Put(s"/rest/processings/${processing.id}", entity) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.Unauthorized)
        }
      }
    }

    describe("processing rating") {
      it("should be able to rate processing") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        val project = ProjectUtil.createProject(UserUtil.regularUser)
        val processing = ProcessingUtil.createProcessing(project.id.some)(UserUtil.regularUser)

        Put(
          s"/rest/processings/${processing.id}/rate",
          CreateProcessingRatingJson(5, "Awesome job".some),
        ) ~> addHeader(auth) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
        }

        Get(s"/rest/processings/${processing.id}") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
          val json = responseAs[ProcessingJson]

          json.rating should matchPattern {
            case Some(Rating(5, Some("Awesome job"))) =>
          }
        }

        Put(
          s"/rest/processings/${processing.id}/rate",
          CreateProcessingRatingJson(3, none),
        ) ~> addHeader(auth) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
        }

        Get(s"/rest/processings/${processing.id}") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
          val json = responseAs[ProcessingJson]

          json.rating should matchPattern {
            case Some(Rating(3, None)) =>
          }
        }
      }
    }

    describe("Unprivileged user") {
      it("should not be able to retrieve other user's processing") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.otherUser)

        val processing = ProcessingUtil.createProcessing(UserUtil.regularUser)

        Get(s"/rest/processings/${processing.id}") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.NotFound)
        }
      }

      it("should not be able to see other user's  processings") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.otherUser)

        val project = ProjectUtil.createProject(UserUtil.regularUser)
        ProcessingUtil.createProcessing(project.id.some)(UserUtil.regularUser)

        Get(s"/rest/projects/${project.id}/processings") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.NotFound)
        }
      }

      it("should not be able to update other user's processing") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.otherUser)

        val project = ProjectUtil.createProject(UserUtil.regularUser)
        val processing = ProcessingUtil.createProcessing(project.id.some)(UserUtil.regularUser)

        val entity =
          UpdateProcessingInputJson("New Name".some, "New Description".some, project.id.some)

        Put(s"/rest/processings/${processing.id}", entity) ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.Forbidden)
        }
      }
    }
  }
}
