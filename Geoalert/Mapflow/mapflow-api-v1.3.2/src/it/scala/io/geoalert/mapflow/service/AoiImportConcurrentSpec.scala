package io.geoalert.mapflow.service

import cats.instances.list._
import cats.syntax.option._
import cats.syntax.traverse._
import doobie.ConnectionIO
import doobie.implicits._

import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.implicits.GeometryOps._
import io.geoalert.mapflow.model._
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.repo._
import io.geoalert.mapflow.util.GeometryUtil
import io.geoalert.mapflow.util.ProcessingUtil
import io.geoalert.mapflow.util.UserUtil

import geotrellis.vector._

class AoiImportConcurrentSpec extends DbIntegrationTest with Services {
  val requestsNum = 250

  describe("AoiService") {
    it(s"should concurrently insert single polygons without collisions") {
      val processing = ProcessingUtil.createProcessing(UserUtil.regularUser)

      val newGeom = GeometryUtil.fromExtent(1, 1, 1.001, 1.001)

      def addGeom(geom: Projected[Geometry]): ConnectionIO[AoiStats] =
        aoiService.createAoisFromGeometry(CreateAoisFromGeometryInput(processing.id, geom))(
          UserUtil.regularUser
        )

      val io = for {
        _ <- (1 to requestsNum).toList.traverse(_ => addGeom(newGeom))
        aoiFilter = AoiFilter(processingIds = List(processing.id).some, geometry = newGeom.some)
        aois <- AoiRepo.getAoisWithFilter(aoiFilter, None)
        prj <- projectService.getProject(processing.projectId)(UserUtil.regularUser).rethrowT
        prcs <- processingService.getProcessings(None, List(processing.projectId).some)(
          UserUtil.regularUser
        )
      } yield (aois, prcs.head, prj)

      val (aois, prc, project) = io.transact(xa).unsafeRunSync()

      aois.map(_.geometry) should (contain theSameElementsAs Seq(newGeom))

      val expectedArea = newGeom.areaInMeters()

      prc.area should be(expectedArea)
      prc.aoiCount should be(1)
      prc.progress.details.map(_.area).sum should be(expectedArea)

      project.area should be(expectedArea)
      project.aoiCount should be(1)
      project.progress.details.map(_.area).sum should be(expectedArea)
    }
  }
}
