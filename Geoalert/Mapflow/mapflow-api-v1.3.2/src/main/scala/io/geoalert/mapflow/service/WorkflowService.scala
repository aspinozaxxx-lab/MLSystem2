package io.geoalert.mapflow.service

import java.time.Instant
import java.util.UUID

import cats.data.EitherT
import cats.data.NonEmptyList
import cats.syntax.applicative._
import cats.syntax.option._
import cats.syntax.traverse._
import com.typesafe.scalalogging.LazyLogging
import doobie.ConnectionIO
import doobie.implicits._
import io.geoalert.mapflow.Config.defaultAoiStartTime
import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.implicits.GeometryOps._
import io.geoalert.mapflow.model.Aoi
import io.geoalert.mapflow.model.Message
import io.geoalert.mapflow.model.Processing
import io.geoalert.mapflow.model.RequiredAction
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.model.Workflow
import io.geoalert.mapflow.model.WorkflowSummary
import io.geoalert.mapflow.repo.AoiRepo
import io.geoalert.mapflow.repo.ProcessingDto
import io.geoalert.mapflow.repo.ProcessingRepo
import io.geoalert.mapflow.repo.ProjectRepo
import io.geoalert.mapflow.repo.UserRepo
import io.geoalert.mapflow.repo.WorkflowRepo
import io.geoalert.mapflow.service.billing.BillingService
import io.geoalert.mapflow.service.notification.NotificationService
import io.geoalert.mapflow.util.BatchingService

import geotrellis.vector.Geometry
import geotrellis.vector.Projected

/** Workflow Service is responsible for managing Workflow entities of White-Maps.
  *
  * For Workflow Engine interaction WorkflowEngineService
  */
class WorkflowService(
    billingService: BillingService,
    notificationService: NotificationService,
    progressService: ProgressService,
    reviewService: ReviewService,
  ) extends LazyLogging {
  def createWorkflows(aois: Seq[Aoi], processing: Processing): ConnectionIO[Seq[UUID]] = {
    def createForAoi(aoi: Aoi): ConnectionIO[Seq[UUID]] = {
      val partitions = withAreas(
        aoi.area,
        BatchingService.partitionGeometry(
          aoi.geometry,
          BatchingService.estimatePartitionSize(processing),
        ),
      )
      for {
        ids <- partitions.traverse {
          case (p, a) => WorkflowRepo.createWorkflow(aoi.id, processing.workflowDef.id, p, a)
        }
      } yield ids
    }
    aois.flatTraverse(createForAoi)
  }

  // Computes polygon areas in such a way that their sum equals to `totalArea`
  def withAreas(
      totalArea: Long,
      ps: List[Projected[Geometry]],
    ): Seq[(Projected[Geometry], Long)] = {
    val realAreas = ps.map(_.areaInMeters())
    val delta = (totalArea - realAreas.sum) / ps.length
    // If 'a' is very little and delta < 0, leave 'a' as is
    val tail = realAreas.tail.map(a => if (a + delta > 0) a + delta else a)
    logger.debug(s"totalArea: $totalArea, realAreas: $realAreas, delta: $delta, tail: $tail")
    ps zip (totalArea - tail.sum) :: tail
  }

  def cancelWorkflows(ids: List[UUID]): ConnectionIO[Unit] = {
    def cancelWorkflow(id: UUID): ConnectionIO[Unit] =
      WorkflowRepo.updateWorkflow(id, Status.Cancelled, Instant.now(), None)

    NonEmptyList.fromList(ids) match {
      case Some(value) =>
        for {
          _ <- WorkflowRepo.setRequiredAction(value, RequiredAction.cancel.some)
          // Some workflows may be not started yet, so it need to be canceled manually
          _ <- ids.traverse(cancelWorkflow)
        } yield {}
      case None => {}.pure[ConnectionIO]
    }
  }

  def findWorkflowsWithRequiredActions(): ConnectionIO[List[Workflow]] =
    WorkflowRepo.workflowsAreNotNullActions(locked = false)

  def findRunningWorkflowIdsByProcessing(processingIds: List[UUID]): ConnectionIO[List[UUID]] =
    NonEmptyList
      .fromList(processingIds)
      .map(value => WorkflowRepo.findRunningWorkflowIdsByProcessing(value))
      .getOrElse(List[UUID]().pure[ConnectionIO])

  private def handleStatusChange(processing: ProcessingDto): ConnectionIO[Unit] =
    for {
      progress <- progressService.getProcessingProgress(processing)
      _ <-
        if (progress.status == Status.Ok)
          updateBillingData(processing)
        else if (progress.status == Status.Failed || progress.status == Status.Cancelled)
          billingService.discardTransaction(processing)
        else
          {}.pure[ConnectionIO]
    } yield {}

  private def updateBillingData(processing: ProcessingDto): ConnectionIO[Unit] =
    for {
      project <- ProjectRepo.getProject(processing.projectId, includeArchived = true).rethrowT
      user <- UserRepo.getUser(project.userId).rethrowT
      _ <-
        if (user.reviewWorkflowEnabled)
          reviewService.reviewProcessing(processing.id)
        else
          billingService.confirmTransaction(processing)
    } yield {}

  def updateWorkflowStatus(
      wf: WorkflowSummary,
      status: Status,
      statusUpdateDate: Instant,
      externalId: Option[String],
      isScheduledUpdate: Boolean,
      messages: List[Message],
    ): EitherT[ConnectionIO, ApplicationError, Unit] =
    if (
        wf.status == status && wf.externalId == externalId && wf
          .completionDate
          .contains(statusUpdateDate)
    )
      EitherT.rightT[ConnectionIO, ApplicationError] {}
    else
      for {
        aoi <- AoiRepo.getAoi(wf.aoiId, None)
        _ = logger.info(s"[WorkflowService] AOI is found ID: ${aoi.id}")
        processing <- ProcessingRepo.getProcessing(aoi.processingId, includeArchived = true)
        _ = logger.info(s"[WorkflowService] Processing is found ID: ${processing.id}")
        workflows <- EitherT.right[ApplicationError](
          WorkflowRepo.getWorkflowByAoiId(aoi.id)
        )
        _ = logger.info(s"[WorkflowService] Workflows are found by aoiId: ${workflows.map(_.id)}")
        // 1. Update workflow progress in the DB
        _ <- EitherT.right[ApplicationError](
          WorkflowRepo.updateWorkflow(wf.id, status, statusUpdateDate, externalId)
        )
        _ = logger.info(s"[WorkflowService] Workflow is updated in the DB wfId: ${wf.id} status: ${status}")
        _ <- EitherT.right[ApplicationError](
          AoiRepo
            .updateAoiStartTime(List(aoi.id), Instant.now().minusSeconds(defaultAoiStartTime))
            .whenA(workflows.count(_.status == Status.Ok) == 0 && status == Status.Ok)
        )
        // 2. Invalidate cache
        _ = progressService.invalidateCache(
          List(processing.projectId),
          List(processing.id),
          List(aoi.id),
        )
        // 3. Update AOI progress in DB
        progress <- EitherT.right[ApplicationError](progressService.getAoiProgress(aoi))
        _ = logger.info(s"[WorkflowService] AOI progress is found status: ${progress.status}")
        _ <- EitherT.right[ApplicationError](
          AoiRepo.updateAoiProgress(wf.aoiId, aoi.area, progress.details, processing.created.some)
        )
        _ = logger.info(s"[WorkflowService] AOI progress is updated in the DB details: ${progress.details}")
        // 4. Invalidate cache
        _ = progressService.invalidateCache(
          List(processing.projectId),
          List(processing.id),
          List(aoi.id),
        )
        // 3. Update billing
        _ <- EitherT.right[ApplicationError](handleStatusChange(processing))
        _ = logger.info(s"[WorkflowService] Billing data is updated")
        _ <- EitherT.right[ApplicationError](
          notificationService.sendFailedProcessingNotification(
            processing,
            wf.id,
            wf.externalId,
            wf.status,
            status,
            isScheduledUpdate,
            messages,
          )
        )
      } yield ()

  def restartFailedWorkflowsInProcessing(processingId: UUID): ConnectionIO[Int] =
    WorkflowRepo.restartFailedWorkflowsInProcessing(processingId)
}

object WorkflowService {
  def apply(
      billingService: BillingService,
      notificationService: NotificationService,
      progressService: ProgressService,
      reviewService: ReviewService,
    ): WorkflowService =
    new WorkflowService(billingService, notificationService, progressService, reviewService)
}
