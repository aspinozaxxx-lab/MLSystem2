package io.geoalert.mapflow.util

import geotrellis.proj4.LatLng
import geotrellis.vector.{Extent, Projected}
import org.scalatest.{FunSpec, Matchers}

class Gml3WriterSpec extends FunSpec with Matchers {
  describe("Conversion Geometry to GML 3 string") {
    it("should convert polygon to GML") {
      val geom = Extent(0, 0, 10, 10).toPolygon()

      val gml = Gml3Writer().write(Projected(geom, LatLng.epsgCode.get))
      gml should be(
        """
          |<Polygon xmlns="http://www.opengis.net/gml">
          |<exterior xmlns="http://www.opengis.net/gml">
          |<LinearRing xmlns="http://www.opengis.net/gml">
          |<posList xmlns="http://www.opengis.net/gml" srsDimension="2">0.0 0.0 0.0 10.0 10.0 10.0 10.0 0.0 0.0 0.0</posList>
          |</LinearRing>
          |</exterior>
          |</Polygon>
          |""".stripMargin.replace("\n", ""))
    }
  }

}
