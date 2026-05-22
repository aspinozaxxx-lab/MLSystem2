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
import io.geoalert.mapflow.model.CreateProjectInput
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.rest.json.Decoders
import io.geoalert.mapflow.rest.json.Encoders
import io.geoalert.mapflow.rest.json.ProjectJson
import io.geoalert.mapflow.rest.json.UpdateProjectInputJson
import io.geoalert.mapflow.rest.json.WorkflowDefJson
import io.geoalert.mapflow.service.Services
import io.geoalert.mapflow.util.ProjectUtil
import io.geoalert.mapflow.util.TestAuthenticationUtils
import io.geoalert.mapflow.util.UserUtil
import io.geoalert.mapflow.util.WorkflowDefUtil

class ProjectRoutesSpec
    extends DbIntegrationTest
       with Encoders
       with Decoders
       with ScalatestRouteTest
       with Services {
  implicit val timeout: RouteTestTimeout = RouteTestTimeout(5.seconds.dilated)

  describe("Project resource") {
    describe("Regular User") {
      it("should be able to create project") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        val entity = CreateProjectInput("New project", "description".some, Some(true))

        Post("/rest/projects", entity) ~> addHeader(auth) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)

          val project = responseAs[ProjectJson]

          project.name should be("New project")
          project.description should be("description".some)
          project.isDefault should be(false)
          project.user.processedArea should be(0)
        }
      }

      it("should be able to retrieve project") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)
        val project = ProjectUtil.createProject(CreateProjectInput("Existing Project", None, None))(
          UserUtil.regularUser
        )

        Get(s"/rest/projects/${project.id}") ~> addHeader(auth) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)

          val project = responseAs[ProjectJson]

          project.name should be("Existing Project")
          project.workflowDefs should be(List())
        }
      }

      it("should be able to update project") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)
        val project = ProjectUtil.createProject(CreateProjectInput("Existing Project", None, None))(
          UserUtil.regularUser
        )

        val entity = UpdateProjectInputJson("New Name".some, "New Description".some)

        Put(s"/rest/projects/${project.id}", entity) ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)

          val project = responseAs[ProjectJson]

          project.name should be("New Name")
          project.description should be(Some("New Description"))
          project.workflowDefs should be(List())
        }
      }

      it("should be able to list projects") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        val project = ProjectUtil.createProject(CreateProjectInput("Existing Project", None, None))(
          UserUtil.regularUser
        )

        Get(s"/rest/projects") ~> addHeader(auth) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)

          val projects = responseAs[List[ProjectJson]]

          projects.map(_.id) should contain(project.id)
        }
      }

      it("should be able to retrieve default project") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        Get(s"/rest/projects/default") ~> addHeader(auth) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)

          val project = responseAs[ProjectJson]

          project.name should be("Default")
        }
      }

      it("should be able to delete project") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        val project = ProjectUtil.createProject(CreateProjectInput("Existing Project", None, None))(
          UserUtil.regularUser
        )

        Delete(s"/rest/projects/${project.id}") ~> addHeader(auth) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
        }

        Get(s"/rest/projects/${project.id}") ~> addHeader(auth) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.NotFound)
        }
      }

      it("should not be able to delete default project") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        val defaultProject = ProjectUtil.defaultProject(UserUtil.regularUser)

        Delete(s"/rest/projects/${defaultProject.id}") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.Forbidden)
        }
      }
    }

    describe("Unauthorized user") {
      it("should not be able to create project") {
        val entity = CreateProjectInput("New project", "description".some, Some(true))

        Post("/rest/projects", entity) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.Unauthorized)
        }
      }

      it("should not be able to list projects") {
        Get("/rest/projects") ~!> HttpServer.route ~> check {
          status should be(StatusCodes.Unauthorized)
        }
      }

      it("should not be able to retrieve default project") {
        Get("/rest/projects/default") ~!> HttpServer.route ~> check {
          status should be(StatusCodes.Unauthorized)
        }
      }

      it("should not be able to retrieve project") {
        val project = ProjectUtil.createProject(CreateProjectInput("Existing Project", None, None))(
          UserUtil.regularUser
        )

        Get(s"/rest/projects/${project.id}") ~!> HttpServer.route ~> check {
          status should be(StatusCodes.Unauthorized)
        }
      }

      it("should not be able to delete project") {
        val project = ProjectUtil.createProject(CreateProjectInput("Existing Project", None, None))(
          UserUtil.regularUser
        )

        Delete(s"/rest/projects/${project.id}") ~!> HttpServer.route ~> check {
          status should be(StatusCodes.Unauthorized)
        }
      }

      it("should not be able to update project") {
        val project = ProjectUtil.createProject(CreateProjectInput("Existing Project", None, None))(
          UserUtil.regularUser
        )

        val entity = UpdateProjectInputJson("New Name".some, None)

        Put(s"/rest/projects/${project.id}", entity) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.Unauthorized)
        }
      }
    }

    describe("Administrator") {
      it("should be able to see other user's projects") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        val project = ProjectUtil.createProject(CreateProjectInput("Existing Project", None, None))(
          UserUtil.regularUser
        )

        Get(s"/rest/projects") ~> addHeader(auth) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)

          val projects = responseAs[List[ProjectJson]]

          projects.map(_.id) should contain(project.id)
        }
      }

      it("should be able to update other user's projects") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        val project = ProjectUtil.createProject(CreateProjectInput("Existing Project", None, None))(
          UserUtil.regularUser
        )

        val entity = UpdateProjectInputJson("New Name".some, "New Description".some)

        Put(s"/rest/projects/${project.id}", entity) ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)

          val project = responseAs[ProjectJson]

          project.name should be("New Name")
          project.description should be(Some("New Description"))
          project.workflowDefs should be(List())
        }
      }
    }

    describe("Unprivileged user") {
      it("should not be able to see other user's projects") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.otherUser)

        val project = ProjectUtil.createProject(CreateProjectInput("Existing Project", None, None))(
          UserUtil.regularUser
        )

        Get(s"/rest/projects") ~> addHeader(auth) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)

          val projects = responseAs[List[ProjectJson]]

          projects.map(_.id) should not contain project.id
        }
      }

      it("should not be able to update other user's projects") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.otherUser)

        val project = ProjectUtil.createProject(CreateProjectInput("Existing Project", None, None))(
          UserUtil.regularUser
        )

        val entity = UpdateProjectInputJson("New Name".some, "New Description".some)

        Put(s"/rest/projects/${project.id}", entity) ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.Forbidden)
        }
      }
    }

    describe("Linking WD to a project") {
      it("should be able to link/unlink WD to/from a project") {
        val auth: HttpHeader = TestAuthenticationUtils.authorizationHeader(UserUtil.regularUser)

        val customWdId = WorkflowDefUtil.createWd(isDefault = false)
        workflowDefService
          .linkWorkflowDefToUser(customWdId, UserUtil.regularUser.id)(UserUtil.admin)
          .transact(xa)
          .unsafeRunSync()

        val project = ProjectUtil.createProject(UserUtil.regularUser)
        Get(s"/rest/projects/${project.id}/models") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
          val models = responseAs[List[WorkflowDefJson]]
          models.map(_.id) should be(List())
        }

        Post(s"/rest/projects/${project.id}/models/$customWdId") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
        }

        Get(s"/rest/projects/${project.id}/models") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
          val models = responseAs[List[WorkflowDefJson]]
          models.map(_.id) should be(List(customWdId))
        }

        Delete(s"/rest/projects/${project.id}/models/$customWdId") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
        }

        Get(s"/rest/projects/${project.id}/models") ~> addHeader(
          auth
        ) ~!> HttpServer.route ~> check {
          status should be(StatusCodes.OK)
          val models = responseAs[List[WorkflowDefJson]]
          models.map(_.id) should be(List())
        }
      }
    }
  }
}
