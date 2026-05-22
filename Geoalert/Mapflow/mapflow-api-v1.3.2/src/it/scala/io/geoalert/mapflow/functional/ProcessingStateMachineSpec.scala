package io.geoalert.mapflow.functional

import java.time.LocalDateTime
import java.util.UUID

import cats.syntax.option._
import doobie.implicits._
import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.exception.AreaLimitExceeded
import io.geoalert.mapflow.model.RequiredAction
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.model.WorkflowSummary
import io.geoalert.mapflow.repo.AoiRepo
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.repo.ProcessingRepo
import io.geoalert.mapflow.service.Services
import io.geoalert.mapflow.service.WorkflowUpdater
import io.geoalert.mapflow.service.billing.ProcessingReport
import io.geoalert.mapflow.service.we.model.WorkflowResponse
import io.geoalert.mapflow.util.ProcessingUtil
import io.geoalert.mapflow.util.UserUtil
import org.scalatest.Assertion

/** This test covers end-to-end processing life cycle in terms of billing (BillingService) and work progress (ProjressService)
  */
class ProcessingStateMachineSpec extends DbIntegrationTest with Services {
  describe("Billing and limits") {
    it(s"should not start processing when user limit exceeded") {
      val user =
        UserUtil.createUser(
          "bb855a4b-e109-410c-b00d-8455ba6af790",
          None,
          None,
          80_000_000L.some,
          100_000_000L.some,
        )

      val processing1 = ProcessingUtil.createProcessing(area = 42_000_000L.some)(user)
      runProcessingService
        .runProcessing(processing1.id)(user)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      val processing2 = ProcessingUtil.createProcessing(area = 42_000_000L.some)(user)
      val either = runProcessingService
        .runProcessing(processing2.id)(user)
        .value
        .transact(xa)
        .unsafeRunSync()

      either should matchPattern {
        case Left(AreaLimitExceeded(84_000_000L, 80_000_000L)) =>
      }
    }

    it("should hold processed area") {
      val user =
        UserUtil.createUser(
          "bb855a4b-e109-410c-b00d-8455ba6af790",
          None,
          None,
          300_000_000L.some,
          100_000_000L.some,
        )

      val processing = ProcessingUtil.createProcessing(area = 42_000_000L.some)(user)
      runProcessingService
        .runProcessing(processing.id)(user)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      val workflows = workflowService
        .findWorkflowsWithRequiredActions()
        .transact(xa)
        .unsafeRunSync()
      workflows.foreach { wf =>
        WorkflowUpdater
          .updateProgress(
            WorkflowSummary(wf),
            WorkflowResponse(42, List(), Status.InProgress.repr, LocalDateTime.now()),
          )
          .unsafeRunSync()
      }

      checkProcessedArea(user)(42_000_000L)
      val report = billingReportService
        .generateBillingReport(user.email, None, None)(user)
        .transact(xa)
        .unsafeRunSync()
      report.processings.size should be(0)

      checkProgress(processing.id)(Status.InProgress, 0)
    }

    it("should increase progress percent when workflow complete") {
      val user =
        UserUtil.createUser(
          "bb855a4b-e109-410c-b00d-8455ba6af790",
          None,
          None,
          300_000_000L.some,
          100_000_000L.some,
        )

      val processing = ProcessingUtil.createProcessing(area = 42_000_000L.some)(user)
      runProcessingService
        .runProcessing(processing.id)(user)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      val workflows = workflowService
        .findWorkflowsWithRequiredActions()
        .transact(xa)
        .unsafeRunSync()

      // Complete only the first workflow
      workflows.headOption.foreach { wf =>
        WorkflowUpdater
          .updateProgress(
            WorkflowSummary(wf),
            WorkflowResponse(42, List(), Status.Ok.repr, LocalDateTime.now()),
          )
          .unsafeRunSync()
      }

      checkProcessedArea(user)(42_000_000L)

      checkProgress(processing.id)(Status.InProgress, 25)
    }

    it("should refund processed are if processing failed") {
      val user =
        UserUtil.createUser(
          "bb855a4b-e109-410c-b00d-8455ba6af790",
          None,
          None,
          300_000_000L.some,
          100_000_000L.some,
        )

      val processing = ProcessingUtil.createProcessing(area = 42_000_000L.some)(user)
      runProcessingService
        .runProcessing(processing.id)(user)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      val workflows = workflowService
        .findWorkflowsWithRequiredActions()
        .transact(xa)
        .unsafeRunSync()
      workflows.headOption.foreach { wf =>
        WorkflowUpdater
          .updateProgress(
            WorkflowSummary(wf),
            WorkflowResponse(42, List(), Status.Ok.repr, LocalDateTime.now()),
          )
          .unsafeRunSync()
      }
      workflows.tail.headOption.foreach { wf =>
        WorkflowUpdater
          .updateProgress(
            WorkflowSummary(wf),
            WorkflowResponse(42, List(), Status.Failed.repr, LocalDateTime.now()),
          )
          .unsafeRunSync()
      }

      val report = billingReportService
        .generateBillingReport(user.email, None, None)(user)
        .transact(xa)
        .unsafeRunSync()
      report.processings.size should be(0)

      checkProcessedArea(user)(42_000_000L)

      checkProgress(processing.id)(Status.InProgress, 25)
    }

    it("should hold and debit processed are if failed processing was restarted") {
      val user =
        UserUtil.createUser(
          "bb855a4b-e109-410c-b00d-8455ba6af790",
          None,
          None,
          300_000_000L.some,
          100_000_000L.some,
        )

      val processing = ProcessingUtil.createProcessing(area = 42_000_000L.some)(user)
      runProcessingService
        .runProcessing(processing.id)(user)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      val workflows = workflowService
        .findWorkflowsWithRequiredActions()
        .transact(xa)
        .unsafeRunSync()
      workflows.headOption.foreach { wf =>
        WorkflowUpdater
          .updateProgress(
            WorkflowSummary(wf),
            WorkflowResponse(42, List(), Status.Failed.repr, LocalDateTime.now()),
          )
          .unsafeRunSync()
      }
      workflows.tail.foreach { wf =>
        WorkflowUpdater
          .updateProgress(
            WorkflowSummary(wf),
            WorkflowResponse(42, List(), Status.Ok.repr, LocalDateTime.now()),
          )
          .unsafeRunSync()
      }

      checkProgress(processing.id)(Status.Failed, 74)

      runProcessingService
        .restartProcessing(processing.id)(user)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      workflows.headOption.foreach { wf =>
        WorkflowUpdater
          .updateProgress(
            WorkflowSummary(wf),
            WorkflowResponse(42, List(), Status.Ok.repr, LocalDateTime.now()),
          )
          .unsafeRunSync()
      }

      checkProgress(processing.id)(Status.Ok, 100)

      checkProcessedArea(user)(42_000_000L)

      val report = billingReportService
        .generateBillingReport(user.email, None, None)(user)
        .transact(xa)
        .unsafeRunSync()
      report.processings.size should be(1)
      report.processings.size should be(1)
      report.processings.head should matchPattern {
        case ProcessingReport(
               _,
               "bb855a4b-e109-410c-b00d-8455ba6af790",
               "Test workflow definition",
               42_000_000L,
               210,
               Some(_),
               "API",
               false,
             ) =>
      }
    }

    it("should debit processed area when processing was complete") {
      val user =
        UserUtil.createUser(
          "bb855a4b-e109-410c-b00d-8455ba6af790",
          None,
          None,
          300_000_000L.some,
          100_000_000L.some,
        )

      val processing = ProcessingUtil.createProcessing(area = 42_000_000L.some)(user)
      runProcessingService
        .runProcessing(processing.id)(user)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      val workflows = workflowService
        .findWorkflowsWithRequiredActions()
        .transact(xa)
        .unsafeRunSync()

      workflows.foreach { wf =>
        WorkflowUpdater
          .updateProgress(
            WorkflowSummary(wf),
            WorkflowResponse(42, List(), Status.Ok.repr, LocalDateTime.now()),
          )
          .unsafeRunSync()
      }

      checkProcessedArea(user)(42_000_000L)
      checkProgress(processing.id)(Status.Ok, 100)

      val report = billingReportService
        .generateBillingReport(user.email, None, None)(user)
        .transact(xa)
        .unsafeRunSync()
      report.processings.size should be(1)
      report.processings.head should matchPattern {
        case ProcessingReport(
               _,
               "bb855a4b-e109-410c-b00d-8455ba6af790",
               "Test workflow definition",
               42_000_000L,
               210,
               Some(_),
               "API",
               false,
             ) =>
      }
    }
  }

  describe("Archiving processing") {
    it("should cancel workflows after archiving processing") {
      val processing =
        ProcessingUtil.createProcessing(area = 42_000_000L.some)(UserUtil.regularUser)
      runProcessingService
        .runProcessing(processing.id)(UserUtil.regularUser)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      val workflowsToStart = workflowService
        .findWorkflowsWithRequiredActions()
        .transact(xa)
        .unsafeRunSync()

      workflowEngineService
        .startWorkflows(workflowsToStart)
        .unsafeRunSync()

      workflowsToStart.foreach { wf =>
        WorkflowUpdater
          .updateProgress(
            WorkflowSummary(wf),
            WorkflowResponse(42, List(), Status.InProgress.repr, LocalDateTime.now()),
          )
          .unsafeRunSync()
      }

      processingService
        .archiveProcessing(processing.id)(UserUtil.regularUser)
        .transact(xa)
        .unsafeRunSync()
      val workflowsToCancel = workflowService
        .findWorkflowsWithRequiredActions()
        .transact(xa)
        .unsafeRunSync()

      workflowsToCancel.flatMap(_.requiredAction).distinct should be(List(RequiredAction.cancel))
      workflowsToCancel.size should be(workflowsToStart.size)

      workflowEngineService
        .cancelWorkflows(workflowsToCancel)
        .unsafeRunSync()

      workflowsToCancel.foreach { wf =>
        WorkflowUpdater
          .updateProgress(
            WorkflowSummary(wf),
            WorkflowResponse(42, List(), Status.Cancelled.repr, LocalDateTime.now()),
          )
          .unsafeRunSync()
      }

      // Check what's happened
      checkProgress(processing.id)(Status.Unprocessed, 0)

      val aoiIds = workflowsToStart.map(_.aoiId)
      val aois = AoiRepo
        .getAois(aoiIds.some, none)
        .transact(xa)
        .unsafeRunSync()

      aois.map(aoi => Status(aoi.status)) should be(List(Status.Cancelled))

      val progress = progressService
        .getAoisProgress(aois)
        .transact(xa)
        .unsafeRunSync()

      progress.values.map(_.status) should be(List(Status.Cancelled))
    }
  }

  def checkProcessedArea(user: User)(value: Long): Assertion =
    billingService
      .getUserAccount(user)
      .transact(xa)
      .unsafeRunSync()
      .processedArea should be(value)

  def checkProgress(processingId: UUID)(status: Status, percent: Int): Assertion = {
    val processing = ProcessingRepo
      .getProcessing(processingId, includeArchived = true)
      .rethrowT
      .transact(xa)
      .unsafeRunSync()
    val progress = progressService
      .getProcessingProgress(processing)
      .transact(xa)
      .unsafeRunSync()

    progress.status should be(status)
    progress.percentCompleted should be(percent)
  }
}
