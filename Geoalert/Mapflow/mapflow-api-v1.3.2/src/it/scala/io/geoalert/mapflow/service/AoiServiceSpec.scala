package io.geoalert.mapflow.service

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor
import scala.concurrent.duration._
import scala.io.Source

import cats.syntax.option._
import doobie.implicits._
import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.exception.AccumulativeAoiImportError
import io.geoalert.mapflow.exception.TooLargeProcessing
import io.geoalert.mapflow.model._
import io.geoalert.mapflow.model.enums.MemberRole
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.util.GeometryUtil
import io.geoalert.mapflow.util.ProcessingUtil
import io.geoalert.mapflow.util.UserUtil

import geotrellis.vector.Geometry
import geotrellis.vector.Projected

class AoiServiceSpec extends DbIntegrationTest with Services {
  implicit val ec: ExecutionContextExecutor = ExecutionContext.global

  private def loadGeometry(file: String): Projected[Geometry] = {
    val json = Source.fromResource(file).getLines().toList.reduce(_ + "\n" + _)
    GeometryUtil.parse(json)
  }

  describe("AoiService") {
    it(s"should correctly return aois without offset or limit") {
      val geom = loadGeometry("five_geometries.geojson")
      val processing = ProcessingUtil.createProcessing(UserUtil.regularUser)

      aoiService
        .createAois(processing, geom)(UserUtil.regularUser)
        .transact(xa)
        .unsafeRunSync()

      val filter = AoiFilter(processingIds = List(processing.id).some, geometry = None)

      val aois = aoiService
        .getAois(filter, None, None, None)(UserUtil.regularUser)
        .transact(xa)
        .unsafeRunSync()

      aois.aois should have size 5
      aois.hasMore should be(false)
    }

    it(s"should correctly return when shared project") {
      val geom = loadGeometry("five_geometries.geojson")
      val user1 = UserUtil.regularUser
      val user2 = UserUtil.regularUser
      val processing = ProcessingUtil.createProcessing(user2)
      projectService
        .shareProject(
          UserProject(user1.id, processing.projectId, MemberRole.Contributor)
        )(user2)
        .transact(xa)
        .rethrowT
        .unsafeRunSync()
      aoiService
        .createAois(processing, geom)(user2)
        .transact(xa)
        .unsafeRunSync()

      val filter = AoiFilter(processingIds = List(processing.id).some, geometry = None)

      val aois = aoiService
        .getAois(filter, None, None, None)(user2)
        .transact(xa)
        .unsafeRunSync()

      aois.aois should have size 5
      aois.hasMore should be(false)
    }

    it(s"should correctly return aois with offset and no limit") {
      val geom = loadGeometry("five_geometries.geojson")
      val processing = ProcessingUtil.createProcessing(UserUtil.regularUser)

      aoiService
        .createAois(processing, geom)(UserUtil.regularUser)
        .transact(xa)
        .unsafeRunSync()

      val filter = AoiFilter(processingIds = List(processing.id).some, geometry = None)

      val aois = aoiService
        .getAois(filter, None, 1.some, None)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      aois.aois should have size 4
      aois.hasMore should be(false)
    }

    it(s"should correctly return aois with offset and limit") {
      val geom = loadGeometry("five_geometries.geojson")
      val processing = ProcessingUtil.createProcessing(UserUtil.regularUser)

      aoiService
        .createAois(processing, geom)(UserUtil.regularUser)
        .transact(xa)
        .unsafeRunSync()

      val filter = AoiFilter(processingIds = List(processing.id).some, geometry = None)

      val aois = aoiService
        .getAois(filter, None, 1.some, 1.some)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      aois.aois should have size 1
      aois.hasMore should be(true)

      val aoisFull = aoiService
        .getAois(filter, None, 1.some, 10.some)(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      aoisFull.aois should have size 4
      aoisFull.hasMore should be(false)
    }

    it("should fail to create aoi larger then aoiAreaLimit") {
      val user = UserUtil.createUser("bb855a4b-e109-410c-b00d-8455ba6af790", None, "password".some, None, 100L.some)
      val prc = ProcessingUtil.createProcessing(user)
      val geom = GeometryUtil.createPolygon(42_000L)
      val aoiInput = CreateAoisFromGeometryInput(prc.id, geom)
      val aoi = aoiService
        .createAoisFromGeometry(aoiInput)(user)
        .transact(xa)
        .attempt
        .unsafeRunTimed(10.seconds)
      aoi should matchPattern {
        case Some(Left(TooLargeProcessing(42_000L, 100L))) =>
      }
    }

    it("Should fail while trying create too large AOI") {
      val user =
        UserUtil.createUser("bb855a4b-e109-410c-b00d-8455ba6af790", None, "password".some, None, 10_000_000L.some)
      val prc = ProcessingUtil.createProcessing(user)
      val area = GeometryUtil.createPolygon(50_000_000L)
      val aoiInput = CreateAoisFromGeometryInput(prc.id, area)
      val aoi = aoiService
        .createAoisFromGeometry(aoiInput)(user)
        .transact(xa)
        .attempt
        .unsafeRunTimed(10.seconds)

      aoi should matchPattern {
        case Some(Left(TooLargeProcessing(50_000_000L, 10_000_000L))) =>
      }
    }

    it("Should not fail while crossing 180 longitude") {
      val user =
        UserUtil.createUser("bb855a4b-e109-410c-b00d-8455ba6af790", None, "password".some, None, 100000000L.some)
      val prc = ProcessingUtil.createProcessing(user)
      val area = GeometryUtil.fromExtent(179.99, -0.01, 180.01, 0.01)
      val aoiInput = CreateAoisFromGeometryInput(prc.id, area)
      val aoi = aoiService
        .createAoisFromGeometry(aoiInput)(user)
        .transact(xa)
        .attempt
        .unsafeRunTimed(10.seconds)

      aoi should matchPattern {
        case Some(Right(AoiStats(1, 2461814, _))) =>
      }
    }

    it("Should not able to create AOI based on geometry in incorrect projection") {
      val geom = loadGeometry("invalid_projection.geojson")
      val processing = ProcessingUtil.createProcessing(UserUtil.regularUser)

      val aoiInput = CreateAoisFromGeometryInput(processing.id, geom)
      try
        aoiService
          .createAoisFromGeometry(aoiInput)(UserUtil.regularUser)
          .transact(xa)
          .unsafeRunSync()
      catch {
        case _: AccumulativeAoiImportError =>
      }
    }

    it("should throw BadRequest exception if geometry is in wrong projection") {
      val geom = loadGeometry("invalid_projection.geojson")
      val processing = ProcessingUtil.createProcessing(UserUtil.regularUser)

      try
        aoiService
          .createAois(processing, geom)(UserUtil.regularUser)
          .transact(xa)
          .unsafeRunSync()
      catch {
        case _: AccumulativeAoiImportError =>
      }
    }
  }
}
