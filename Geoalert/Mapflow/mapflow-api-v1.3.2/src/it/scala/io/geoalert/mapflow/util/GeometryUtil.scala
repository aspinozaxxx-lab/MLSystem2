package io.geoalert.mapflow.util

import geotrellis.proj4.CRS
import geotrellis.proj4.LatLng
import geotrellis.vector.Extent
import geotrellis.vector.Geometry
import geotrellis.vector.GeometryCollection
import geotrellis.vector.ProjectGeometry
import geotrellis.vector.Projected
import geotrellis.vector.io.json.GeoJson
import geotrellis.vector.io.json.JsonFeatureCollection
import geotrellis.vector.reproject.Reproject

object GeometryUtil {
  def fromExtent(
      xmin: Double,
      ymin: Double,
      xmax: Double,
      ymax: Double,
    ): Projected[Geometry] =
    Extent(xmin, ymin, xmax, ymax).toPolygon().withSRID(4326)

  def parse(geojson: String): Projected[Geometry] = {
    val c = GeoJson
      .parse[JsonFeatureCollection](geojson)
      .getAllGeometries()
      .toList

    GeometryCollection(c).withSRID(LatLng.epsgCode.get)
  }

  def createPolygon(area: Long = 9000): Projected[Geometry] = {
    val side = Math.sqrt(area.toDouble)

    val bias = 10000.0
    // Reuse the same projection we're using for area calculation
    val srcSrc = CRS.fromString(s"+proj=cea +ellps=WGS84 +lon_0=0")
    Reproject(Extent(bias, bias, bias + side, bias + side).toPolygon(), srcSrc, LatLng)
      .withSRID(LatLng.epsgCode.get)
  }
}
