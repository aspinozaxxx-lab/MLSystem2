package io.geoalert.mapflow.util

import java.util.UUID

import doobie.implicits._
import io.geoalert.mapflow.model._
import io.geoalert.mapflow.repo._
import Db.xa
import io.geoalert.mapflow.service.Services

import geotrellis.vector._

object AoiUtil extends Services {
  def createAois(processing: Processing, geometry: Projected[Geometry])(user: User): List[Aoi] = {

    val io = for {
      _ <- aoiService.createAois(processing, geometry)(user)
      aois <- aoiService.getProcessingAois(processing.id)(user)
      _ <- workflowService.createWorkflows(aois, processing)
    } yield aois

    io.transact(xa).unsafeRunSync()
  }
  def createWorkflows(
      processing: Processing,
      geometry: Projected[Geometry],
    )(
      user: User
    ): Seq[UUID] = {

    val io = for {
      _ <- aoiService.createAois(processing, geometry)(user)
      aois <- aoiService.getProcessingAois(processing.id)(user)
      ids <- workflowService.createWorkflows(aois, processing)
    } yield ids

    io.transact(xa).unsafeRunSync()
  }
}
