package io.geoalert.mapflow.service.we

import java.time.LocalDateTime
import java.util.concurrent.Executors

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor
import scala.concurrent.Future
import scala.concurrent.duration._
import scala.util.Failure

import akka.http.scaladsl.client.RequestBuilding.Get
import akka.http.scaladsl.client.RequestBuilding.Post
import akka.http.scaladsl.model.ContentType
import akka.http.scaladsl.model.HttpEntity
import akka.http.scaladsl.model.HttpRequest
import akka.http.scaladsl.model.HttpResponse
import akka.http.scaladsl.model.MediaTypes
import akka.http.scaladsl.model.Multipart
import akka.stream.StreamTcpException
import akka.stream.scaladsl.Sink
import akka.stream.scaladsl.Source
import cats.effect.ContextShift
import cats.effect.IO
import cats.instances.future._
import cats.syntax.applicative._
import com.google.common.util.concurrent.ThreadFactoryBuilder
import com.typesafe.scalalogging.LazyLogging
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import io.circe.generic.auto._

import io.geoalert.mapflow.AkkaSystem.system
import io.geoalert.mapflow.DefaultWeConfig
import io.geoalert.mapflow.exception.ExternalSystemError
import io.geoalert.mapflow.implicits.CommonOps._
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.service.we.model.PostWorkflowRequest
import io.geoalert.mapflow.service.we.model.RunWorkflowSummary
import io.geoalert.mapflow.service.we.model.WorkflowDefResponse
import io.geoalert.mapflow.service.we.model.WorkflowFilter
import io.geoalert.mapflow.service.we.model.WorkflowRequest
import io.geoalert.mapflow.service.we.model.WorkflowResponse
import io.geoalert.mapflow.util.HttpUtils

import geotrellis.vector._

object ProductionWorkflowEngine extends WorkflowEngine with LazyLogging with DefaultWeConfig {
  implicit private val executionContext: ExecutionContextExecutor = ExecutionContext.global

  implicit private lazy val cs: ContextShift[IO] = IO.contextShift(
    ExecutionContext.fromExecutor(
      Executors.newFixedThreadPool(
        8,
        new ThreadFactoryBuilder().setNameFormat("we-client-%d").build(),
      )
    )
  )

  private val connectionFlow = HttpUtils.httpConnection(weUrl)

  override def getLatestDefVersion(id: Long): IO[Int] = {
    val future = runRequest(Get(s"/api/v0/definitions/$id"))
      .flatMap(HttpUtils.parseResponse[WorkflowDefResponse](_, "WE"))
      .map(_.latestVersion.version)
    IO.fromFuture(IO(future))
  }

  override def postDef(definition: String): IO[Long] = {
    import Multipart.FormData
    val definitionForm = FormData.BodyPart(
      "definition",
      HttpEntity(ContentType.Binary(MediaTypes.`application/octet-stream`), definition.getBytes),
      Map("filename" -> "definition.yml"),
    )
    val request = FormData(definitionForm).toEntity()

    val future = (for {
      response <- runRequest(Post(s"/api/v0/definitions", request))
      workflowDefResponse <- HttpUtils.parseResponse[WorkflowDefResponse](response, "WE")
    } yield workflowDefResponse.id).recoverWith {
      case e => Future.failed(ExternalSystemError("WE", e))
    }

    IO.fromFuture(IO(future))
  }

  override def postWorkflow(summary: RunWorkflowSummary): IO[WorkflowResponse] = {
    val request = PostWorkflowRequest(summary)

    logger.debug(s"Posting new workflow to WE: $request")

    def future: Future[WorkflowResponse] = for {
      response <- runRequest(Post(s"/api/v0/workflows", request))
      workflowResponse <- HttpUtils.parseResponse[WorkflowResponse](response, "WE")
    } yield workflowResponse

    val handledFuture = future.recoverWith {
      case e: StreamTcpException => Future.failed(e)
      case e =>
        logger.error("Error posting workflow", e)
        // TODO: Use Either[Error, WorkflowResponse] instead of IO
        WorkflowResponse(0, List(), Status.Failed.repr, LocalDateTime.now()).pure[Future]
    }

    IO.fromFuture(IO(handledFuture))
  }

  override def restartWorkflow(externalId: String): IO[Status] = {
    logger.debug(s"Posting restart workflow request to WE for externalId = $externalId")

    def future: Future[Status] = for {
      response <- runRequest(Post(s"/api/v0/workflows/$externalId/restart?failedStagesOnly=true"))
      _ <- HttpUtils.extractResponseBodyAsString(response, "WE")
    } yield Status.InProgress: Status

    val handledFuture = future.recoverWith {
      case e: StreamTcpException => Future.failed(e)
      case e =>
        logger.error("Error restarting workflow", e)
        Status.Failed.pure[Future]
    }

    IO.fromFuture(IO(handledFuture))
  }

  override def cancelWorkflows(externalIds: List[String]): IO[Unit] = {
    logger.debug(
      s"Posting cancel workflows request for externalIds = ${externalIds.shortString(3)}"
    )

    def cancelBatch(ids: List[String]): IO[List[Unit]] = {
      val future = for {
        response <- runRequest(
          Post(s"/api/v0/workflows/batchCancel?workflowIds=${ids.mkString(",")}")
        )
        _ <- HttpUtils.extractResponseBodyAsString(response, "WE")
      } yield List(())

      IO.fromFuture(IO(future))
    }

    externalIds.batchTraverseSeq(100)(cancelBatch).map(_ => ())
  }

  override def getWorkflows(externalIds: List[String]): IO[Map[String, WorkflowResponse]] = {
    logger.debug(s"Fetching workflows ${externalIds.shortString(3)} from WE")

    val request = WorkflowRequest(WorkflowFilter(externalIds), 0)

    val future = for {
      response <- runRequest(Post(s"/api/v0/workflows/page", request))
      workflowResponses <- HttpUtils.parseResponse[List[WorkflowResponse]](response, "WE")
      _ = logger.debug(s"Received response with ${workflowResponses.size} workflows")
    } yield workflowResponses.map(w => w.id.toString -> w).toMap

    IO.fromFuture(IO(future))
  }

  private def runRequest(request: HttpRequest): Future[HttpResponse] = {
    val requestFuture = Source
      .single(request)
      .via(connectionFlow)
      .runWith(Sink.head)

    requestFuture.onComplete {
      case Failure(e) => logger.error(s"Error trying to call WE", e)
      case _ =>
    }

    Future.firstCompletedOf(
      Seq(
        akka
          .pattern
          .after(3.minutes, system.scheduler)(
            Future.failed(new RuntimeException("WorkflowEngine request timeout"))
          ),
        requestFuture,
      )
    )
  }
}
