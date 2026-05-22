package io.geoalert.mapflow.service

import java.util.UUID

import scala.io.Source

import cats.syntax.option._
import doobie.implicits._

import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.exception.AccessDenied
import io.geoalert.mapflow.exception.NotFound
import io.geoalert.mapflow.model.CreateWorkflowDefInput
import io.geoalert.mapflow.model.UpdateWorkflowDefInput
import io.geoalert.mapflow.model.WorkflowDef
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.util.ProjectUtil
import io.geoalert.mapflow.util.UserUtil
import io.geoalert.mapflow.util.WorkflowDefUtil

class WorkflowDefServiceSpec extends DbIntegrationTest with Services {
  val yml: String = Source.fromResource("simple.yml").getLines().toList.reduce(_ + "\n" + _)

  describe("WorkflowDefService") {
    it("should create Workflow Definition for a project") {
      val wd = workflowDefService
        .createWorkflowDef(
          CreateWorkflowDefInput(
            None,
            "Some Buildings in some area",
            "Some long description".some,
            None,
            yml.some,
            1.0.some,
            None,
          ),
          None,
        )(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      wd should matchPattern {
        case WorkflowDef(
               _,
               "Some Buildings in some area",
               Some("Some long description"),
               _,
               _,
               _,
               _,
               _,
               false,
               false,
             ) =>
      }
    }

    it("should update Workflow Definition") {
      val existingWd: UUID = WorkflowDefUtil.createWd()

      val wd = workflowDefService
        .updateWorkflowDef(
          UpdateWorkflowDefInput(
            existingWd,
            None,
            "New Name".some,
            "New Description".some,
            None,
            yml.some,
            17.0.some,
            false.some,
          ),
          None,
        )(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      wd should matchPattern {
        case WorkflowDef(_, "New Name", Some("New Description"), _, _, _, _, _, false, false) =>
      }
    }

    it("should update price from YML") {
      val existingWd: UUID = WorkflowDefUtil.createWd()
      val wd = workflowDefService
        .getWorkflowDef(existingWd)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      wd.workflowDefSummary.pricePerSqKm should be(11)

      val updatedWd = workflowDefService
        .updateWorkflowDef(
          UpdateWorkflowDefInput(
            existingWd,
            None,
            None,
            None,
            None,
            yml.some,
            None,
            false.some,
          ),
          None,
        )(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      updatedWd.workflowDefSummary.pricePerSqKm should be(5)
    }

    it("should retrieve Workflow Definition by ID") {
      val id: UUID = WorkflowDefUtil.createWd()

      val wd = workflowDefService
        .getWorkflowDef(id)(UserUtil.regularUser)
        .transact(xa)
        .unsafeRunSync()

      wd should matchPattern {
        case WorkflowDef(_, "Test workflow definition", None, _, _, _, _, _, false, true) =>
      }
    }

    it("should archive Workflow Definition") {
      val id = WorkflowDefUtil.createWd()

      workflowDefService
        .archiveWorkflowDef(id)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      try
        workflowDefService
          .getWorkflowDef(id)(UserUtil.admin)
          .transact(xa)
          .unsafeRunSync()
      catch {
        case _: NotFound => // Expected exception
      }
    }
  }

  describe("Default WD") {
    it("should create Default Workflow Definition with empty Project ID") {
      val wd = workflowDefService
        .createWorkflowDef(
          CreateWorkflowDefInput(None, "Default WD", None, None, yml.some, None, None),
          None,
        )(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      wd should matchPattern {
        case WorkflowDef(_, "Default WD", _, _, _, _, _, _, false, false) =>
      }
    }
  }

  describe("Permission check") {
    it("regular user should not be able to create Workflow Definition") {
      try
        workflowDefService
          .createWorkflowDef(
            CreateWorkflowDefInput(
              None,
              "Some Buildings in some area",
              "Some long description".some,
              None,
              yml.some,
              1.0.some,
              None,
            ),
            None,
          )(UserUtil.admin)
          .transact(xa)
          .unsafeRunSync()
      catch {
        case _: AccessDenied =>
      }
    }

    it("regular user should not be able to update Workflow Definition") {
      val existingWd: UUID = WorkflowDefUtil.createWd()

      try
        workflowDefService
          .updateWorkflowDef(
            UpdateWorkflowDefInput(
              existingWd,
              None,
              "New Name".some,
              "New Description".some,
              None,
              yml.some,
              17.0.some,
              None,
            ),
            None,
          )(UserUtil.admin)
          .transact(xa)
          .unsafeRunSync()
      catch {
        case _: AccessDenied =>
      }
    }

    it("regular user should not be able to archive Workflow Definition") {
      try {
        val id = WorkflowDefUtil.createWd()

        workflowDefService
          .archiveWorkflowDef(id)(UserUtil.admin)
          .transact(xa)
          .unsafeRunSync()
      }
      catch {
        case _: AccessDenied =>
      }
    }
  }

  describe("Workflow Definition Linking") {
    it("should link WD to a user when linking to a project") {
      val wdId = WorkflowDefUtil.createWd(isDefault = false)

      val project = ProjectUtil.createProject(UserUtil.regularUser)
      workflowDefService
        .linkWorkflowDefToProject(wdId, project.id)(UserUtil.admin)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      val wds = workflowDefService
        .listWorkflowDefLinkedToUser(UserUtil.regularUser.id)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      wds.map(_.id) should be(List(wdId))
    }

    it("should unlink WD from all user's projects when WD id unlinked from a user") {
      val wdId = WorkflowDefUtil.createWd(isDefault = false)

      val project = ProjectUtil.createProject(UserUtil.regularUser)
      workflowDefService
        .linkWorkflowDefToUser(wdId, UserUtil.regularUser.id)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      workflowDefService
        .linkWorkflowDefToProject(wdId, project.id)(UserUtil.regularUser)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      workflowDefService
        .unlinkWorkflowDefFromUser(wdId, UserUtil.regularUser.id)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      val userWds = workflowDefService
        .listWorkflowDefLinkedToUser(UserUtil.regularUser.id)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()
        .map(_.id)

      userWds should be(List())

      val projectWds = workflowDefService
        .listWorkflowDefLinkedToProject(project.id)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()
        .map(_.id)

      projectWds should be(List())
    }
  }
}
