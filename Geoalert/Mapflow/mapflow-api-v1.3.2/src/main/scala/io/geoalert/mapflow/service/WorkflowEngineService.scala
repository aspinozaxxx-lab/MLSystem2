package io.geoalert.mapflow.service

import java.net.URL
import java.time.Instant
import java.time.ZoneOffset
import java.time.temporal.ChronoUnit
import java.util.UUID
import java.util.concurrent.Executors

import scala.concurrent.ExecutionContext
import scala.concurrent.duration._

import cats.data.EitherT
import cats.data.NonEmptyList
import cats.effect.IO
import cats.implicits.catsSyntaxApplicativeId
import cats.implicits.toFlatMapOps
import cats.implicits.toFunctorOps
import cats.implicits.toTraverseOps
import cats.syntax.option._
import com.typesafe.scalalogging.LazyLogging
import doobie.ConnectionIO
import doobie.implicits._
import io.circe.syntax._
import io.geoalert.mapflow.AkkaSystem.system
import io.geoalert.mapflow.Config
import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.implicits.CommonOps.SeqOps
import io.geoalert.mapflow.model.BlockConfig
import io.geoalert.mapflow.model.DataProvider
import io.geoalert.mapflow.model.Message
import io.geoalert.mapflow.model.ProcessingParams
import io.geoalert.mapflow.model.RequiredAction
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.model.Status.InProgress
import io.geoalert.mapflow.model.Workflow
import io.geoalert.mapflow.model.WorkflowSummary
import io.geoalert.mapflow.repo.AoiRepo
import io.geoalert.mapflow.repo.BlockParameters
import io.geoalert.mapflow.repo.DataProviderRepo
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.repo.ProcessingRepo
import io.geoalert.mapflow.repo.RasterLayerRepo
import io.geoalert.mapflow.repo.VectorLayerRepo
import io.geoalert.mapflow.repo.WorkflowRepo
import io.geoalert.mapflow.service.notification.NotificationService
import io.geoalert.mapflow.service.we.WorkflowEngine
import io.geoalert.mapflow.service.we.model.RunWorkflowSummary
import io.geoalert.mapflow.util.HttpUtils

class WorkflowEngineService(
    progressService: ProgressService,
    notificationService: NotificationService,
    workflowEngine: WorkflowEngine,
    workflowService: WorkflowService,
  ) extends LazyLogging {
  implicit val ec: ExecutionContext = ExecutionContext.fromExecutor(Executors.newFixedThreadPool(2))

  private val schedulerEc = ExecutionContext.fromExecutor(Executors.newFixedThreadPool(3))

  def scheduleWorkflowSync(): Unit =
    system
      .scheduler
      .scheduleWithFixedDelay(0.seconds, 5.seconds) { () =>
        try {
          logger.info("[WorkflowEngineService] Started workflows synchronization")
          startProcess().unsafeRunSync()
        }
        catch {
          case ex: Exception =>
            logger.error("[WorkflowEngineService] Workflows synchronization failed", ex)
        }
      }(schedulerEc)

  private def startProcess(): IO[Unit] =
    WorkflowRepo
      .workflowsAreNotNullActions(locked = false)
      .transact(xa)
      .flatMap { workflows =>
        logger.info(
          s"[WorkflowEngineService] Found wfIds ${workflows.map(_.id)} workflows to synchronization"
        )
        NonEmptyList
          .fromList(workflows.map(_.id))
          .traverse(WorkflowRepo.setWorkflowsLocked(_, bool = true, Instant.now().some))
          .transact(xa)
          .flatMap { _ =>
            logger.info(s"[WorkflowEngineService] Locked workflows ${workflows.map(_.id)}")
            if (workflows.nonEmpty) syncWorkflows(workflows)
            else {
              logger.info(s"[WorkflowEngineService] No locked workflows not found to sync")
              WorkflowRepo
                .workflowsAreLockedAndNotNullActions(
                  lockedAt = Instant.now().minus(10, ChronoUnit.MINUTES).some
                )
                .flatTap { lockedWfs =>
                  NonEmptyList
                    .fromList(lockedWfs.map(_.id))
                    .traverse(
                      WorkflowRepo.setWorkflowsLocked(_, bool = true, lockedAt = Instant.now().some)
                    )
                }
                .transact(xa)
                .flatMap { lockedWfs =>
                  if (lockedWfs.nonEmpty) syncWorkflows(lockedWfs)
                  else
                    logger
                      .info(
                        s"[WorkflowEngineService] Locked before 10 minutes workflows not found to sync"
                      )
                      .pure[IO]
                }

            }
          }
      }

  def syncWorkflows(workflows: List[Workflow]): IO[Unit] = {
    if (workflows.nonEmpty)
      logger.info(
        s"[WorkflowEngineService] Synchronizing ${workflows.size} workflows with workflow engine"
      )
    val workflowsToStart = workflows.filter(_.requiredAction.contains(RequiredAction.start))
    val workflowsToRestart = workflows.filter(_.requiredAction.contains(RequiredAction.restart))
    val workflowsToCancel = workflows.filter(_.requiredAction.contains(RequiredAction.cancel))

    logger.info(
      s"[WorkflowEngineService] Starting ${workflowsToStart.size} workflows, cancelling ${workflowsToCancel.size} workflows, restarting ${workflowsToRestart.size} workflows"
    )

    for {
      _ <- startWorkflows(workflowsToStart)
      _ <- cancelWorkflows(workflowsToCancel)
      _ <- restartWorkflows(workflowsToRestart)
    } yield logger.info(s"[WorkflowEngineService] Synchronization of workflows completed")
  }

  def startWorkflows(workflows: List[Workflow]): IO[List[Unit]] =
    if (workflows.nonEmpty) {
      logger.info(
        s"[WorkflowEngineService] Starting workflows in Workflow Engine: ${workflows.map(_.id).shortString(5)}"
      )
      workflows.traverse(startWorkflow).map { wfs =>
        logger.info(s"[WorkflowEngineService] Workflows started in Workflow Engine: ${wfs.size}")
        wfs
      }
    }
    else
      IO.pure(List(()))

  def cancelWorkflows(workflows: List[Workflow]): IO[List[Unit]] =
    if (workflows.nonEmpty) {
      logger.info(
        s"[WorkflowEngineService] Cancelling workflows in Workflow Engine: ${workflows.map(_.id).shortString(5)}"
      )
      for {
        _ <- cancelWorkflow(workflows)
        _ = logger.info(
          s"[WorkflowEngineService] Workflows cancelled in Workflow Engine: ${workflows.size}"
        )
      } yield List(())
    }
    else
      IO.pure(List(()))

  def restartWorkflows(workflows: List[Workflow]): IO[List[Unit]] =
    if (workflows.nonEmpty) {
      logger.info(
        s"[WorkflowEngineService] Restarting workflows in Workflow Engine: ${workflows.map(_.id).shortString(5)}"
      )
      workflows.map(restartWorkflow).sequence.map { wfs =>
        logger.info(s"[WorkflowEngineService] Workflows restarted in Workflow Engine: ${wfs.size}")
        wfs
      }
    }
    else
      IO.pure(List(()))

  def startWorkflow(workflow: Workflow): IO[Unit] = {
    def getDataProvider(
        idOpt: Option[UUID]
      ): EitherT[ConnectionIO, ApplicationError, Option[DataProvider]] =
      idOpt match {
        case Some(value) =>
          EitherT.right[ApplicationError](DataProviderRepo.getOneById(value).map(_.map(_.toDomain)))
        case None => EitherT.rightT[ConnectionIO, ApplicationError](None)
      }

    def prepareBlockParameters(
        config: Seq[BlockConfig],
        params: Seq[BlockParameters],
      ): Seq[BlockParameters] =
      for {
        bc <- config
        bp = params.find(_.name == bc.name)
      } yield BlockParameters(
        bc.name,
        bp.map(_.enabled).getOrElse(bc.defaultEnabled),
        bc.displayName.some,
      )
    val eitherT = for {
      workflow <- WorkflowRepo.getWorkflow(workflow.id)
      aoi <- AoiRepo.getAoi(workflow.aoiId, None)
      _ = logger.info(s"[WorkflowEngineService] AOI is found")
      processing <- ProcessingRepo.getProcessing(aoi.processingId, None)
      _ = logger.info(s"[WorkflowEngineService] Processing is found")
      vectorLayer <- VectorLayerRepo.getVectorLayer(processing.vectorLayerId)
      _ = logger.info(s"[WorkflowEngineService] Vector layer is found")
      rasterLayer <- RasterLayerRepo.getRasterLayer(processing.rasterLayerId)
      _ = logger.info(s"[WorkflowEngineService] Raster layer is found")
      dataProvider <- getDataProvider(processing.dataProviderId)
      _ = logger.info(s"[WorkflowEngineService] Data provider is found")
      progress <- EitherT.right[ApplicationError](
        progressService.getProcessingProgress(processing)
      )
      _ = logger.info(s"[WorkflowEngineService] Progress is found")
      areaInProgress = progress
        .details
        .find(_.status == InProgress)
        .map(_.area)
        .getOrElse(0L)
      params = WorkflowEngineService.prepareProcessingParams(
        ProcessingParams(processing.params),
        dataProvider,
      )
      _ = logger.info(s"[WorkflowEngineService] Processing params are prepared")
      blockConfig = workflow.workflowDef.workflowDefSummary.blocks
      _ = logger.info(s"[WorkflowEngineService] Block config is found")
      blockParams = processing.blocks.map(BlockParameters(_)).getOrElse(Seq())
    } yield RunWorkflowSummary(
      workflow,
      vectorLayer,
      rasterLayer,
      processing.id,
      params,
      areaInProgress,
      prepareBlockParameters(blockConfig, blockParams),
    )

    def updateRequiredAction(status: Status): ConnectionIO[Unit] = {
      logger.info(s"[WorkflowEngineService] Updating required action for wfId: ${workflow.id}")
      if (status != Status.Failed || workflow.failedCount >= 4) {
        logger.info(
          s"[WorkflowEngineService] Update wfId: ${workflow.id} required_action to null and locked false"
        )
        WorkflowRepo
          .setRequiredAction(NonEmptyList.one(workflow.id), None)
          .flatMap(_ =>
            WorkflowRepo.setWorkflowsLocked(NonEmptyList.one(workflow.id), bool = false)
          )
      }
      else {
        logger.info(
          s"[WorkflowEngineService] Update wfId: ${workflow.id} failedCount: ${workflow.failedCount + 1}"
        )
        WorkflowRepo.updateWorkflowFailedCount(workflow.id, workflow.failedCount + 1)
      }
    }

    eitherT.rethrowT.transact(xa).flatMap { summary =>
      if (summary.wf.requiredAction.contains(RequiredAction.start)) {
        logger.info(s"[WorkflowEngineService] Request will be sending to the postWorkflow")
        (for {
          response <- workflowEngine.postWorkflow(summary).to[ConnectionIO]
          _ = logger.info(
            s"[WorkflowEngineService] Workflow engine postWorkflow response: $response"
          )
          _ = logger.info(
            s"[WorkflowEngineService] Workflow engine response wfId: ${workflow.id} status: ${response.status}"
          )
          status = Status.fromWeStatus(response.status)
          _ <- workflowService
            .updateWorkflowStatus(
              WorkflowSummary(
                workflow.id,
                workflow.aoiId,
                workflow.externalId,
                workflow.status,
                workflow.completionDate,
              ),
              status,
              response.statusUpdateDate.toInstant(ZoneOffset.UTC),
              response.id.toString.some,
              isScheduledUpdate = false,
              List(
                Message(
                  "mapflowApi.consistencyError",
                  List(),
                  "Unable to create workflow in Workflow Engine",
                )
              ),
            )
            .rethrowT
          _ <- updateRequiredAction(status)
        } yield ()).transact(xa)
      }
      else {}.pure[IO]
    }

  }

  def restartWorkflow(wf: Workflow): IO[Unit] = {
    val eitherT = for {
      workflow <- WorkflowRepo.getWorkflow(wf.id)
      aoi <- AoiRepo.getAoi(workflow.aoiId, None)
      processing <- ProcessingRepo.getProcessing(aoi.processingId, None)
    } yield (processing, workflow)

    eitherT
      .rethrowT
      .flatMap {
        case (processing, workflow) =>
          if (
              workflow.requiredAction.contains(RequiredAction.restart) && workflow
                .externalId
                .nonEmpty
          )
            for {
              externalId <- IO
                .fromOption(workflow.externalId)(
                  new IllegalStateException("Attempting to restart workflow without externalId")
                )
                .to[ConnectionIO]

              status <- workflowEngine.restartWorkflow(externalId).to[ConnectionIO]
              _ <- WorkflowRepo.setRequiredAction(NonEmptyList.one(workflow.id), None)
              _ <- WorkflowRepo.setWorkflowsLocked(NonEmptyList.one(workflow.id), bool = false)
              _ <- workflowService
                .updateWorkflowStatus(
                  WorkflowSummary(workflow),
                  status,
                  Instant.now(),
                  externalId.some,
                  isScheduledUpdate = false,
                  List(),
                )
                .rethrowT
              _ = notificationService.removeRestartingProcessing(processing.id)
            } yield ()
          else
            WorkflowRepo
              .setRequiredAction(NonEmptyList.one(workflow.id), RequiredAction.start.some)
              .flatTap { _ =>
                WorkflowRepo.setWorkflowsLocked(NonEmptyList.one(workflow.id), bool = false)
              }
      }
      .transact(xa)
  }

  def cancelWorkflow(workflows: List[Workflow]): IO[Unit] =
    NonEmptyList
      .fromList(workflows.flatMap(_.externalId))
      .fold(
        NonEmptyList
          .fromList(workflows.map(_.id))
          .traverse(ids =>
            WorkflowRepo
              .setRequiredAction(ids, None)
              .flatTap(_ => WorkflowRepo.setWorkflowsLocked(ids, bool = false))
          )
          .void
      ) { externalIds =>
        for {
          _ <- workflowEngine.cancelWorkflows(externalIds.toList).to[ConnectionIO]
          _ = logger.info(
            s"[WorkflowEngineService] Workflows cancelledIds: ${workflows.map(_.id)}"
          )
          _ <- NonEmptyList
            .fromList(workflows.map(_.id))
            .traverse(ids =>
              WorkflowRepo
                .setRequiredAction(ids, None)
                .flatTap(_ => WorkflowRepo.setWorkflowsLocked(ids, bool = false))
            )
          _ = logger.info(s"[WorkflowEngineService] Required actions are set to Null")
        } yield ()
      }
      .transact(xa)
}

object WorkflowEngineService extends LazyLogging {
  def apply(
      progressService: ProgressService,
      workflowEngine: WorkflowEngine,
      workflowService: WorkflowService,
      notificationService: NotificationService,
    ): WorkflowEngineService =
    new WorkflowEngineService(progressService, notificationService, workflowEngine, workflowService)

  def prepareProcessingParams(
      params: ProcessingParams,
      dataProviderOpt: Option[DataProvider],
    ): Map[String, String] = {
    val url = for {
      url <- params.url
      dataProvider <- dataProviderOpt
      token <- dataProvider.credentialsToken
    } yield "url" -> url.replace("{token}", token)

    // TODO: Move to HeadService
    val headParams = if (dataProviderOpt.exists(_.name.toLowerCase().contains("head_202"))) {
      // params for Head basemaps
      val yearOpt = for {
        urlString <- params.url
        urlYearOpt = HttpUtils.parseQueryParameters(new URL(urlString).getQuery).get("year")
        paramsYearOpt = params.rest.get("year")
        year <- (paramsYearOpt.toList ++ urlYearOpt.toList).headOption
      } yield year
      val url = Config.tileproxyUrl.replace("{year}", yearOpt.getOrElse("2022"))

      Seq(
        "url" -> url,
        "header" -> Map("x-api-key" -> Config.tileproxyApiKey).asJson.noSpacesSortKeys,
        "source_type" -> "tms",
        // Limit connection rate as otherwise HEAD api fails with timeout error
        "connection_limit" -> "4",
      ).toMap
    }
    else if (dataProviderOpt.exists(_.name.toLowerCase().contains("head_imagery")))
      // params for head imagery
      Seq("source_type" -> "head_imagery").toMap
    else
      Map[String, String]()

    params.toMap ++ url ++ headParams
  }
}
