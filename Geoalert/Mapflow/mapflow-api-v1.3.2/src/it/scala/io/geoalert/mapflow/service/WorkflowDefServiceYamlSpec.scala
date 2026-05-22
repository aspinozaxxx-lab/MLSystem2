package io.geoalert.mapflow.service

import java.util.UUID

import scala.io.Source

import doobie.implicits._

import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.model.CreateWorkflowDefInput
import io.geoalert.mapflow.model.UpdateWorkflowDefInput
import io.geoalert.mapflow.model.WorkflowDef
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.util.UserUtil
import io.geoalert.mapflow.util.WorkflowDefUtil

class WorkflowDefServiceYamlSpec extends DbIntegrationTest with Services {
  val yml: String = Source.fromResource("simple.yml").getLines().toList.reduce(_ + "\n" + _)

  describe("WorkflowDefService YML processing") {
    it("Should be able to create WorkflowDef") {
      val wd: WorkflowDef = workflowDefService
        .createWorkflowDef(
          CreateWorkflowDefInput(
            None,
            "Cows",
            Some("Detect cows"),
            None,
            Some(yml),
            Some(5.0),
            None,
          ),
          None,
        )(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      wd should matchPattern {
        case WorkflowDef(_, "Cows", Some("Detect cows"), _, _, _, _, _, false, false) =>
      }

      wd.yml.contains("description: Source selection") should be(true)
    }

    it("Should be able to edit WorkflowDef") {
      val existingWd: UUID = WorkflowDefUtil.createWd()

      val id: UUID = workflowDefService
        .updateWorkflowDef(
          UpdateWorkflowDefInput(
            existingWd,
            None,
            Some("Sheeps"),
            Some("Detect sheeps"),
            None,
            Some(yml),
            Some(13.0),
            None,
          ),
          None,
        )(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()
        .id

      val wd = workflowDefService
        .getWorkflowDef(id)(UserUtil.regularUser)
        .transact(xa)
        .unsafeRunSync()
      wd should matchPattern {
        case WorkflowDef(_, "Sheeps", Some("Detect sheeps"), _, _, _, _, _, false, true) =>
      }

    }

    it("Should not be able to create WorkflowDef with incorrect YML") {
      try {
        workflowDefService
          .createWorkflowDef(
            CreateWorkflowDefInput(
              None,
              "Cows",
              Some("Detect cows"),
              None,
              Some("incorrect yaml [1, 2>"),
              Some(5.0),
              None,
            ),
            None,
          )(UserUtil.regularUser)
          .transact(xa)
          .unsafeRunSync()
        fail("ApplicationError expected")
      }
      catch {
        case _: ApplicationError =>
        case _: Exception => fail("ApplicationError expected")
      }
    }
  }
}
