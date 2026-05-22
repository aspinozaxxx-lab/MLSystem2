package io.geoalert.mapflow.util

import java.io.StringWriter
import java.io.Writer

import org.locationtech.jts.geom.Geometry
import org.locationtech.jts.geom.LinearRing

import geotrellis.proj4.CRS
import geotrellis.proj4.LatLng
import geotrellis.vector._

class Gml3Writer(writer: Writer) {
  def write(projected: Projected[Geometry]): String = {
    val geometry =
      projected.reproject(CRS.fromEpsgCode(projected.srid), LatLng)(LatLng.epsgCode.get).geom
    writeGeometry(geometry)
    writer.toString
  }

  def asString: String = writer.toString

  private def writeGeometry(geometry: Geometry): Unit = geometry match {
    case g: Polygon => writePolygon(g)
    case _ => throw new UnsupportedOperationException(s"Unexpected geometry type $geometry")
  }

  private def writePolygon(polygon: Polygon): Unit = {
    writer.write("<Polygon xmlns=\"http://www.opengis.net/gml\">")

    writer.write("<exterior xmlns=\"http://www.opengis.net/gml\">")
    writeLineRing(polygon.getExteriorRing)
    writer.write("</exterior>")

    if (polygon.getNumInteriorRing > 0)
      for {
        i <- Range(0, polygon.getNumInteriorRing)
        ring = polygon.getInteriorRingN(i)
      } yield writeLineRing(ring)

    writer.write("</Polygon>")
  }

  private def writeLineRing(ring: LinearRing): Unit = {
    writer.write("<LinearRing xmlns=\"http://www.opengis.net/gml\">")
    writer.write("<posList xmlns=\"http://www.opengis.net/gml\" srsDimension=\"2\">")
    val coordStr =
      ring.getCoordinates.map(coord => "" + coord.getX + " " + coord.getY).reduce(_ + " " + _)
    writer.write(coordStr)
    writer.write("</posList>")
    writer.write("</LinearRing>")
  }
}

object Gml3Writer {
  def apply(): Gml3Writer = new Gml3Writer(new StringWriter())
}
