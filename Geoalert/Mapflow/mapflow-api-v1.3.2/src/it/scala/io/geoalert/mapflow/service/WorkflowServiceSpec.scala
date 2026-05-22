package io.geoalert.mapflow.service

import java.time.Instant
import java.time.temporal.ChronoUnit

import scala.concurrent.Await
import scala.concurrent.ExecutionContext
import scala.concurrent.duration._

import akka.actor.ActorSystem
import akka.http.scaladsl.model.HttpEntity
import akka.stream.scaladsl.Sink
import cats.data.NonEmptyList
import cats.implicits.catsSyntaxOptionId
import doobie.implicits._
import io.geoalert.mapflow.AkkaSystem
import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.model.AoiFilter
import io.geoalert.mapflow.model.RequiredAction
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.repo.WorkflowRepo
import io.geoalert.mapflow.util.AoiUtil
import io.geoalert.mapflow.util.GeometryUtil
import io.geoalert.mapflow.util.ProcessingUtil
import io.geoalert.mapflow.util.UserUtil
import org.scalatest.FunSpecLike
import org.scalatest.Matchers

import geotrellis.proj4.LatLng

class WorkflowServiceSpec extends Matchers with FunSpecLike with Services {
  implicit val ec: ExecutionContext = ExecutionContext.global

  implicit val system: ActorSystem = AkkaSystem.system

  describe("WorkflowService") {
    it("should update workflow locked") {
      val processing = ProcessingUtil.createProcessing(UserUtil.regularUser)
      val workflowIds = AoiUtil.createWorkflows(
        processing,
        GeometryUtil.fromExtent(83, 54, 83.01, 54.01).withSRID(LatLng.epsgCode.get),
      )(UserUtil.regularUser)

      val sd = workflowIds.head
      WorkflowRepo
        .setWorkflowsLocked(NonEmptyList.one(sd), bool = true, Instant.now().some)
        .transact(xa)
        .unsafeRunSync()
      val wd = WorkflowRepo.getWorkflow(sd).transact(xa).rethrowT.unsafeRunSync()
      wd.locked should be(true)
      wd.lockedAt.isDefined should be(true)
      wd.requiredAction should contain(RequiredAction.start)
    }

    it("should be non empty workflow locked before 10 minutes") {
      val processing = ProcessingUtil.createProcessing(UserUtil.regularUser)
      val workflowIds = AoiUtil.createWorkflows(
        processing,
        GeometryUtil.fromExtent(83, 54, 83.01, 54.01).withSRID(LatLng.epsgCode.get),
      )(UserUtil.regularUser)

      val wdId = workflowIds.head
      WorkflowRepo
        .setWorkflowsLocked(NonEmptyList.one(wdId), bool = true, Instant.now().some)
        .transact(xa)
        .unsafeRunSync()
      val wds = WorkflowRepo
        .workflowsAreLockedAndNotNullActions(Instant.now().some)
        .transact(xa)
        .unsafeRunSync()
      wds.exists(_.id == wdId) should be(true)
    }
  }
}
