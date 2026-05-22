package io.geoalert.mapflow.service

import scala.concurrent.ExecutionContext
import scala.io.Source

import cats.syntax.option._
import doobie.implicits._

import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.implicits.GeometryOps._
import io.geoalert.mapflow.model._
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.repo._
import io.geoalert.mapflow.util.AoiUtil
import io.geoalert.mapflow.util.GeometryUtil
import io.geoalert.mapflow.util.ProcessingUtil
import io.geoalert.mapflow.util.ProjectUtil
import io.geoalert.mapflow.util.UserUtil

import geotrellis.proj4.LatLng
import geotrellis.vector._

class AoiImportSpec extends DbIntegrationTest with Services {
  implicit val ec: ExecutionContext = ExecutionContext.global

  private def testImport(
      initGeoms: List[Geometry],
      newGeoms: List[Geometry],
      expectedGeoms: List[Geometry],
    ) = {
    val projectId = ProjectUtil.createProject(UserUtil.regularUser).id
    val processing = ProcessingUtil.createProcessing(projectId.some)(UserUtil.regularUser)

    AoiUtil.createAois(processing, GeometryCollection(initGeoms).withSRID(LatLng.epsgCode.get))(
      UserUtil.admin
    )
    AoiUtil.createAois(processing, GeometryCollection(newGeoms).withSRID(LatLng.epsgCode.get))(
      UserUtil.admin
    )

    val aoiFilter = AoiFilter(processingIds = List(processing.id).some, geometry = None)

    val io = for {
      aois <- AoiRepo.getAoisWithFilter(aoiFilter, None)
      project <- projectService.getProject(projectId)(UserUtil.regularUser).rethrowT
      processing <- processingService.getProcessing(processing.id)(UserUtil.regularUser).rethrowT
    } yield (aois, processing, project)

    val (aois, prc, prj) = io.transact(xa).unsafeRunSync()

    aois.map(_.geometry) should (contain theSameElementsAs expectedGeoms.map(
      _.withSRID(LatLng.epsgCode.get)
    ))

    val expectedArea = expectedGeoms.map(_.withSRID(LatLng.epsgCode.get).areaInMeters()).sum

    prc.area should be(expectedArea)
    prc.aoiCount should be(expectedGeoms.size)
    prc.progress.details.map(_.area).sum should be(expectedArea)

    prj.area should be(expectedArea)
    prj.aoiCount should be(expectedGeoms.size)
    prj.progress.details.map(_.area).sum should be(expectedArea)
  }

  describe("AoiService") {
    it(s"should insert a single polygon into an empty processing") {
      testImport(
        List(),
        List(GeometryUtil.fromExtent(1, 1, 2, 2)),
        List(GeometryUtil.fromExtent(1, 1, 2, 2)),
      )
    }
  }

  describe("merging geometries for AOI") {
    it("should merge nearby geometries to multipolygon") {
      val json = Source.fromResource("union.geojson").getLines().toList.reduce(_ + "\n" + _)
      val geoms = GeometryUtil.parse(json).withSRID(4326)

      val processing = ProcessingUtil.createProcessing(UserUtil.regularUser)
      val aois = aoiService
        .createAois(processing, geoms)(UserUtil.regularUser)
        .transact(xa)
        .unsafeRunSync()

      aois.count should be(4)
    }
  }
}
