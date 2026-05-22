package io.geoalert.mapflow.util

import java.time.Instant
import java.util.UUID

import cats.syntax.option._
import cats.syntax.traverse._
import doobie.implicits._

import io.geoalert.mapflow.model.CreateAoisFromGeometryInput
import io.geoalert.mapflow.model.CreateProcessingInput
import io.geoalert.mapflow.model.Processing
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.model.WorkflowSummary
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.service.Services

object ProcessingUtil extends Services {
  def createProcessing(user: User): Processing = {
    val project = ProjectUtil.defaultProject(user)
    createProcessing(project.id.some)(user)
  }

  def createProcessing(
      projectId: Option[UUID] = None,
      name: Option[String] = None,
      area: Option[Long] = None,
    )(
      user: User
    ): Processing = {
    val wdId = WorkflowDefUtil.createWd()
    val project = projectId.getOrElse(ProjectUtil.defaultProject(user).id)

    val processing = processingService
      .createProcessing(
        CreateProcessingInput(project, None, None, wdId, None, name, None, cost = 24)
      )(user)
      .rethrowT
      .transact(xa)
      .unsafeRunSync()

    area.foreach { area =>
      val geom = GeometryUtil.createPolygon(area)

      val aoiInput = CreateAoisFromGeometryInput(processing.id, geom)

      aoiService
        .createAoisFromGeometry(aoiInput)(user)
        .transact(xa)
        .unsafeRunSync()
    }

    processingService
      .getProcessing(processing.id)(user)
      .rethrowT
      .transact(xa)
      .unsafeRunSync()
  }

  def completeProcessing(processing: Processing)(user: User): Processing = {
    val aois = AoiUtil.createAois(processing, GeometryUtil.createPolygon(42_000_000L))(user)

    aoiService
      .updateAoiStatusAndVrt(aois.map(_.id), Status.Ok, None)
      .transact(xa)
      .unsafeRunSync()

    val wfs = workflowService
      .findWorkflowsWithRequiredActions()
      .transact(xa)
      .unsafeRunSync()
      .filter(wf => aois.map(_.id).contains(wf.aoiId))

    wfs
      .traverse(wf =>
        workflowService.updateWorkflowStatus(
          WorkflowSummary(wf),
          Status.Ok,
          Instant.now(),
          None,
          isScheduledUpdate = true,
          List(),
        )
      )
      .rethrowT
      .transact(xa)
      .unsafeRunSync()

    processing
  }
}
