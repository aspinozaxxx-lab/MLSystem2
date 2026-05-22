package io.geoalert.mapflow

import scala.concurrent.Await
import scala.concurrent.duration._

import com.typesafe.scalalogging.LazyLogging
import io.geoalert.mapflow.AkkaSystem._
import io.geoalert.mapflow.repo.Migration
import io.geoalert.mapflow.service._

object Application extends App with LazyLogging with Services with DefaultDbConfig {
  scala.sys.addShutdownHook(() -> shutdown())

  Migration()

  HttpServer()

  WorkflowUpdater.scheduleUpdates()
  reviewService.scheduleUpdates()
  workflowEngineService.scheduleWorkflowSync()

  val version: String = getClass.getPackage.getImplementationVersion

  logger.info(s"Server started version $version")

  def shutdown(): Unit = {
    system.terminate()
    Await.result(system.whenTerminated, 30.seconds)
  }
}
