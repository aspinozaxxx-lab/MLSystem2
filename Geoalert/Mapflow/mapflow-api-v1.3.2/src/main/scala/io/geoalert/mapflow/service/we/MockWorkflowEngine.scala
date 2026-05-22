package io.geoalert.mapflow.service.we

import java.time.LocalDateTime

import scala.util.Random

import cats.effect.IO
import cats.syntax.applicative._

import io.geoalert.mapflow.Config._
import io.geoalert.mapflow.model._
import io.geoalert.mapflow.service.we.model._

object MockWorkflowEngine extends WorkflowEngine {
  override def getLatestDefVersion(id: Long): IO[Int] = 0.pure[IO]

  override def postDef(definition: String): IO[Long] = Random.nextLong().pure[IO]

  override def postWorkflow(summary: RunWorkflowSummary): IO[WorkflowResponse] =
    WorkflowResponse(Random.nextLong(), List(), randomStatus.repr, LocalDateTime.now()).pure[IO]

  override def restartWorkflow(externalId: String): IO[Status] =
    Status.InProgress.pure[IO]

  override def cancelWorkflows(externalIds: List[String]): IO[Unit] =
    ().pure[IO]

  override def getWorkflows(externalIds: List[String]): IO[Map[String, WorkflowResponse]] =
    IO.pure(
      externalIds
        .map(s => (s, WorkflowResponse(s.toLong, List(), randomStatus.repr, LocalDateTime.now())))
        .toMap
    )

  def randomStatus: Status = Random.nextInt(100) match {
    case n if n < mockWeFailedPercent => Status.Failed
    case n if n < mockWeFailedPercent + mockWeInProgressPercent => Status.InProgress
    case _ => Status.Ok
  }
}
