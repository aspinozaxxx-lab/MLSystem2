package io.geoalert.mapflow.service

import java.time.Instant
import java.time.ZoneOffset
import java.util.concurrent.Executors

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor
import scala.concurrent.duration._

import cats.data.EitherT
import cats.effect.IO
import cats.implicits.catsSyntaxApplicativeError
import cats.instances.list._
import cats.syntax.traverse._
import com.typesafe.scalalogging.LazyLogging
import doobie.implicits._
import io.geoalert.mapflow.AkkaSystem.system
import io.geoalert.mapflow.DefaultWeConfig
import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.implicits.CommonOps._
import io.geoalert.mapflow.model.Message
import io.geoalert.mapflow.model.MessageParameter
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.model.WorkflowSummary
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.repo.WorkflowRepo
import io.geoalert.mapflow.service.we.WorkflowEngine
import io.geoalert.mapflow.service.we.model.WorkflowResponse

object WorkflowUpdater extends LazyLogging with Services with DefaultWeConfig {
  val workflowEngine: WorkflowEngine = WorkflowEngine()

  implicit private val ec: ExecutionContextExecutor =
    ExecutionContext.fromExecutor(Executors.newFixedThreadPool(4))

  def scheduleUpdates(): Unit =
    system.scheduler.scheduleWithFixedDelay(0.seconds, workflowUpdateInterval.seconds) { () =>
      try
        updateWorkflowProgresses()
      catch {
        case e: Exception => logger.error(s"[WorkflowUpdater] Error updating workflow progress", e)
      }
    }

  def updateProgress(wf: WorkflowSummary, response: Option[WorkflowResponse]): IO[Unit] = {
    logger.debug(s"[WorkflowUpdater] Updating progress for $wf, $response")
    response match {
      case Some(response) =>
        updateProgress(wf, response).recover {
          case err: Exception =>
            logger.error(s"Error updating workflow progress for $wf", err)
        }
      case None => markWorkflowFailed(wf)
    }
  }

  def markWorkflowFailed(wf: WorkflowSummary): IO[Unit] = {
    logger.warn(s"[WorkflowUpdater] Workflow $wf not found in Workflow Engine")
    workflowService
      .updateWorkflowStatus(
        wf,
        Status.Failed,
        Instant.now(),
        None,
        isScheduledUpdate = true,
        List(
          Message(
            "mapflowApi.consistencyError",
            List(),
            "Workflow cannot be found in Workflow Engine",
          )
        ),
      )
      .rethrowT
      .transact(xa)
  }

  def updateProgress(wf: WorkflowSummary, response: WorkflowResponse): IO[Unit] = {
    val messages = response
      .stages
      .flatMap(_.messages)
      .flatten
      .map(m =>
        Message(m.code, m.parameters.toList.map(p => MessageParameter(p._1, p._2)), m.message)
      )

    val io = for {
      _ <- EitherT.right[ApplicationError](aoiService.updateAoiMessages(wf.aoiId, messages))
      _ <- workflowService.updateWorkflowStatus(
        wf,
        Status.fromWeStatus(response.status),
        response.statusUpdateDate.toInstant(ZoneOffset.UTC),
        None,
        isScheduledUpdate = true,
        messages,
      )
    } yield ()

    io.rethrowT.transact(xa)
  }

  def updateWorkflowProgresses(): Unit = {
    logger.debug("[WorkflowUpdater] Started to update workflow progress")

    def batch(wfs: List[WorkflowSummary]): IO[List[Unit]] =
      if (wfs.isEmpty)
        // Skip sending requests
        IO.pure(List())
      else {
        logger.debug(s"[WorkflowUpdater] Updating wf statuses: ${wfs.map(_.id).shortString(5)}")
        for {
          responses <- workflowEngine.getWorkflows(wfs.flatMap(_.externalId))
          _ = logger.debug(
            s"[WorkflowUpdater] Received responses for wfs: ${responses.values.map(wf => s"wfId: ${wf.id} wfStatus: ${wf.status}").take(5)}"
          )
          wfsAndResponses = wfs.flatMap(w => w.externalId.map(responses.get).map((w, _)))
          res <- wfsAndResponses.traverse { case (w, r) => updateProgress(w, r) }
        } yield res
      }

    (for {
      wfs <- WorkflowRepo.getWorkflowsToUpdate.transact(xa)
      _ = logger.debug(s"[WorkflowUpdater] Updating wf statuses: ${wfs.map(_.id).shortString(5)}")
      _ <- wfs.sliding(weBatchSize, weBatchSize).toList.traverse(batch)
      _ = logger.debug(s"[WorkflowUpdater] Complete updating wf statuses: ${wfs.map(_.id).shortString(5)}")
    } yield {}).unsafeRunSync()
  }
}
