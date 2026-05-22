package io.geoalert.mapflow.service

import scala.concurrent.Await
import scala.concurrent.ExecutionContext
import scala.concurrent.duration._
import scala.util.Failure
import scala.util.Success
import scala.util.Try

import cats.syntax.option._
import doobie.implicits._
import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.exception.Forbidden
import io.geoalert.mapflow.exception.NotFound
import io.geoalert.mapflow.graphql.GraphQLController
import io.geoalert.mapflow.model._
import io.geoalert.mapflow.model.enums.MemberRole
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.repo.ProcessingRepo
import io.geoalert.mapflow.util.ProcessingUtil
import io.geoalert.mapflow.util.ProjectUtil
import io.geoalert.mapflow.util.UserUtil
import io.geoalert.mapflow.util.WorkflowDefUtil

class ProjectServiceSpec extends DbIntegrationTest with Services {
  implicit val ec: ExecutionContext = ExecutionContext.global

  describe("ProjectService") {
    it(s"should preserve access permissions for users") {
      val user1 = UserUtil.createUser("user1@example.com", None, "pwd123456".some)
      val user2 = UserUtil.createUser("user2@example.com", None, "pwd123456".some)
      val prj = ProjectUtil.createProject(user1)

      Try(
        Await.result(
          projectService
            .getProject(prj.id)(user2)
            .rethrowT
            .transact(xa)
            .unsafeToFuture(),
          2.seconds,
        )
      ) should matchPattern {
        case Failure(NotFound(_)) =>
      }

      Try(
        Await.result(
          projectService
            .getProject(prj.id)(user1)
            .rethrowT
            .transact(xa)
            .unsafeToFuture(),
          2.seconds,
        )
      ) should matchPattern {
        case Success(_) =>
      }

      val updInput = UpdateProjectInput(prj.id, Some("x"), None)
      val updTry = Try(
        Await.result(
          projectService.updateProject(updInput)(user2).transact(xa).unsafeToFuture(),
          2.seconds,
        )
      )
      updTry should matchPattern { case Failure(Forbidden(_)) => }
    }

    it(s"should preserve access permissions for admins") {
      val user1 = UserUtil.createUser("user1@example.com", None, "pwd123456".some)
      val prj = ProjectUtil.createProject(user1)

      projectService.getProject(prj.id)(user1).rethrowT.transact(xa).unsafeRunSync()

      val updInput = UpdateProjectInput(prj.id, Some("x"), None)
      projectService.updateProject(updInput)(user1).transact(xa).unsafeRunSync()
    }

    it(s"should access able to change member role") {
      val user = UserUtil.createUser("user1@example.com", None, "pwd123456".some)
      val admin = UserUtil.admin
      val project = ProjectUtil.createProject(admin)
      projectService
        .shareProject(UserProject(user.id, project.id, MemberRole.Contributor))(admin)
        .transact(xa)
        .rethrowT
        .unsafeRunSync()
      projectService
        .unshareProject(project.id, user.id)(admin)
        .transact(xa)
        .rethrowT
        .unsafeRunSync()
      projectService
        .shareProject(UserProject(user.id, project.id, MemberRole.Readonly))(admin)
        .transact(xa)
        .rethrowT
        .unsafeRunSync()
    }
    it(s"should access able to change member role without unshare") {
      val user = UserUtil.createUser("user1@example.com", None, "pwd123456".some)
      val admin = UserUtil.admin
      val project = ProjectUtil.createProject(admin)
      projectService
        .shareProject(UserProject(user.id, project.id, MemberRole.Contributor))(admin)
        .transact(xa)
        .rethrowT
        .unsafeRunSync()
      projectService
        .shareProject(UserProject(user.id, project.id, MemberRole.Readonly))(admin)
        .transact(xa)
        .rethrowT
        .unsafeRunSync()
    }

    it("should archive all processings when archiving project") {
      val project = ProjectUtil.createProject(UserUtil.regularUser)
      val processingId = ProcessingUtil.createProcessing(project.id.some)(UserUtil.regularUser).id

      val result = projectService
        .archiveProject(project.id)(UserUtil.regularUser)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()
      result should be("OK")

      val processing = ProcessingRepo
        .getProcessing(processingId, UserUtil.regularUser.id.some)
        .value
        .transact(xa)
        .unsafeRunSync()
      processing should matchPattern {
        case Left(_: NotFound) =>
      }
    }
  }

  describe("Workflow Definitions in Project") {
    it("project should contain linked WDs and default WDs") {
      val wd = WorkflowDefUtil.createWd(isDefault = false)
      val defaultWd = WorkflowDefUtil.createWd()
      val project = ProjectUtil.createProject(UserUtil.regularUser)

      workflowDefService
        .linkWorkflowDefToProject(wd, project.id)(UserUtil.admin)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      val prj = projectService
        .getProject(project.id)(UserUtil.regularUser)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      prj.workflowDefs.map(_.id).sorted should be(List(defaultWd, wd).sorted)
    }

    it("default project should contain all user WDs and all default WDs") {
      val userWd1 = WorkflowDefUtil.createWd(isDefault = false)
      val userWd2 = WorkflowDefUtil.createWd(isDefault = false)
      val project = ProjectUtil.defaultProject(UserUtil.regularUser)

      workflowDefService
        .linkWorkflowDefToUser(userWd1, UserUtil.regularUser.id)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      workflowDefService
        .linkWorkflowDefToUser(userWd2, UserUtil.regularUser.id)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      val prj = projectService
        .getProject(project.id)(UserUtil.regularUser)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      prj.workflowDefs.map(_.id).toSet should be(Set(userWd1, userWd2))
    }
  }

  // TODO: Move to separate test of GraphQL Controller
  describe("Paging") {
    it("Should return paged results") {
      Range(0, 25).map(i =>
        ProjectUtil.createProject(CreateProjectInput(s"Project $i", None, false.some))(
          UserUtil.admin
        )
      )

      val result = GraphQLController
        .listProjectsPaged(PagedRequest(20.some, 10.some, None))(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      result.count should be(5)
      result.total should be(25)

      val singleResult = GraphQLController
        .listProjectsPaged(PagedRequest(0.some, 10.some, "Project 7".some))(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      singleResult.count should be(1)
      singleResult.total should be(1)
      singleResult.results.map(_.name).head should be("Project 7")
    }
  }
}
