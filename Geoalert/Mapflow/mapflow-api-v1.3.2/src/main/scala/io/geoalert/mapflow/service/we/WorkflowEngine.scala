package io.geoalert.mapflow.service.we

import cats.effect.IO

import io.geoalert.mapflow.TestEnvConfig
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.service.we.model._

trait WorkflowEngine {
  def getLatestDefVersion(id: Long): IO[Int]

  def postDef(definition: String): IO[Long]

  def postWorkflow(summary: RunWorkflowSummary): IO[WorkflowResponse]

  def restartWorkflow(externalId: String): IO[Status]

  def cancelWorkflows(externalIds: List[String]): IO[Unit]

  def getWorkflows(externalIds: List[String]): IO[Map[String, WorkflowResponse]]
}

object WorkflowEngine extends TestEnvConfig {
  private lazy val instance = if (testEnv) MockWorkflowEngine else ProductionWorkflowEngine

  def apply(): WorkflowEngine = instance
}
