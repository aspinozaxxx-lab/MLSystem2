package io.geoalert.mapflow.service

import cats.instances.list._
import cats.syntax.traverse._
import doobie.ConnectionIO
import doobie.implicits._

import io.geoalert.mapflow.Config._
import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.model.AoiStats
import io.geoalert.mapflow.model.ProgressDetail
import io.geoalert.mapflow.model.Status._
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.util.GeometryUtil._
import io.geoalert.mapflow.util.ProcessingUtil._
import io.geoalert.mapflow.util.UserUtil

import geotrellis.vector._

class RunProcessingConcurrentSpec extends DbIntegrationTest with Services {
  val requestsNum = 100

  private def testRunImport(geoms: List[Projected[Geometry]]): Unit = {
    val processing = createProcessing(UserUtil.admin)

    def addGeom(geom: Projected[Geometry]): ConnectionIO[AoiStats] =
      aoiService.createAois(processing, geom)(UserUtil.admin)

    val f = for {
      _ <- geoms.traverse(addGeom)
      _ <- runProcessingService.runProcessing(processing)(UserUtil.admin).rethrowT
      aois <- aoiService.getProcessingAois(processing.id)(UserUtil.admin)
      project <- projectService.getProject(processing.projectId)(UserUtil.admin).rethrowT
      processing <- processingService.getProcessing(processing.id)(UserUtil.admin).rethrowT
    } yield (aois, processing, project)

    val (aois, prc, project) = f
      .transact(xa)
      .unsafeRunSync()

    val area = aois.map(_.area).sum
    val count = aois.size

    project.area should be(area)
    project.aoiCount should be(count)

    prc.area should be(area)
    prc.aoiCount should be(count)

    prc.progress.status should be(InProgress)
    prc.progress.details should be(ProgressDetail(InProgress, count, area, None) :: Nil)

    project.progress.status should be(InProgress)
    project.progress.details should be(ProgressDetail(InProgress, 1, area, None) :: Nil)

    aois.map(_.progress.status).distinct should be(InProgress :: Nil)

    for (aoi <- aois)
      aoi.progress.details should matchPattern {
        case ProgressDetail(InProgress, _, aoi.area, None) :: Nil =>
      }
  }

  describe("RunProcessingService") {
    it(s"should concurrently run aois without collisions") {
      val geom = for {
        i <- 0 to requestsNum * 3
        xy0 = defaultPartitionSize * 2 * i
      } yield fromExtent(
        xy0,
        xy0,
        xy0 + defaultPartitionSize * 1.5,
        xy0 + defaultPartitionSize * 1.5,
      )

      testRunImport(geom.toList)
    }
  }
}
