package io.geoalert.mapflow.util

import scala.io.Source

import org.scalatest.FunSpec
import org.scalatest.Matchers

import io.geoalert.mapflow.implicits.GeometryOps._

import geotrellis.vector._

class BartchingServiceSpec extends FunSpec with Matchers {
  describe("BatchMerging") {
    it("should process single geometry") {
      val geom = GeometryUtil.fromExtent(83, 54, 83.1, 54.1)

      val result = BatchingService.mergeGeometriesToBatches(geom.toPolygons.map(_.geom), 0.05)

      result.size should be(1)
    }

    it("should merge multiple simple features into one large geometry collection") {
      val json = Source.fromResource("union.geojson").getLines().toList.reduce(_ + "\n" + _)

      val geom = GeometryUtil.parse(json)

      val result = BatchingService.mergeGeometriesToBatches(geom.geom.toPolygons, 1.00)

      result.size should be(1)
      result.head.bounds.extent should be(geom.geom.extent)
    }

    it("should merge multiple simple features into multiple MultiPolygons") {
      val json = Source.fromResource("union.geojson").getLines().toList.reduce(_ + "\n" + _)

      val geom = GeometryUtil.parse(json)

      val result = BatchingService.mergeGeometriesToBatches(geom.geom.toPolygons, 0.05)

      result.size should be(4)
      result.map(_.bounds).reduce(_.combine(_)).extent should be(geom.geom.extent)
      Math.abs(result.map(_.asGeometry.getArea).sum - geom.geom.getArea) < 1e-5 should be(true)
    }
  }

  describe("Workflow Service") {
    it("Should not throw java.lang.IllegalArgumentException: Precision 34 inadequate...") {
      val geometry = Polygon(
        (80.30229996656722, 51.62392360028478),
        (80.25726753696135, 51.59763157547681),
        (80.29614168559549, 51.56450586597285),
        (80.33559317307072, 51.567137671532585),
        (80.35118132178043, 51.61651557068735),
        (80.32904999953823, 51.62464044219622),
        (80.30229996656722, 51.62392360028478),
      )

      val result = BatchingService.partitionGeometry(Projected(geometry, 4326), 4000e-5)

      val p1 = Polygon(
        (80.28857213190105, 51.570956086635334),
        (80.26085669865991, 51.594573154084536),
        (80.28857213190105, 51.594573154084536),
        (80.28857213190105, 51.570956086635334),
      )

      val p2 = Polygon(
        (80.25726753696135, 51.59763157547681),
        (80.28857213190105, 51.615908652551305),
        (80.28857213190105, 51.594573154084536),
        (80.26085669865991, 51.594573154084536),
        (80.25726753696135, 51.59763157547681),
      )

      val p3 = Polygon(
        (80.29614168559549, 51.56450586597285),
        (80.28857213190105, 51.570956086635334),
        (80.28857213190105, 51.594573154084536),
        (80.31987672684073, 51.594573154084536),
        (80.31987672684073, 51.56608922866732),
        (80.29614168559549, 51.56450586597285),
      )
      val p4 = Polygon(
        (80.28857213190105, 51.615908652551305),
        (80.30229996656722, 51.62392360028478),
        (80.31987672684073, 51.624394618711),
        (80.31987672684073, 51.594573154084536),
        (80.28857213190105, 51.594573154084536),
        (80.28857213190105, 51.615908652551305),
      )

      val p5 = Polygon(
        (80.34425430262996, 51.594573154084536),
        (80.33559317307072, 51.567137671532585),
        (80.31987672684073, 51.56608922866732),
        (80.31987672684073, 51.594573154084536),
        (80.34425430262996, 51.594573154084536),
      )

      val p6 = Polygon(
        (80.35118132178043, 51.61651557068735),
        (80.34425430262996, 51.594573154084536),
        (80.31987672684073, 51.594573154084536),
        (80.31987672684073, 51.624394618711),
        (80.32904999953823, 51.62464044219622),
        (80.35118132178043, 51.61651557068735),
      )

      result should be(
        List(
          Projected(p1, 4326),
          Projected(p2, 4326),
          Projected(p3, 4326),
          Projected(p4, 4326),
          Projected(p5, 4326),
          Projected(p6, 4326),
        )
      )
    }
  }
}
