package io.geoalert.mapflow.implicits

import org.locationtech.jts.simplify.DouglasPeuckerSimplifier

import io.geoalert.mapflow.util.Gml3Writer

import geotrellis.proj4.CRS
import geotrellis.proj4.LatLng
import geotrellis.proj4.WebMercator
import geotrellis.vector._

object GeometryOps {
  implicit class GeometryOps(val value: Geometry) extends AnyVal {
    def toPolygons: List[Polygon] = value match {
      case m: MultiPolygon => m.polygons.toList
      case p: Polygon => List(p)
      case gc: GeometryCollection => gc.geometries.flatMap(_.toPolygons).toList
      case _ => List()
    }
  }

  implicit class GeometryCollectionOps(val value: List[Geometry]) extends AnyVal {
    def toPolygons: List[Polygon] = value.flatMap(_.toPolygons)
  }

  implicit class ProjectedGeometryOps(val value: Projected[Geometry]) extends AnyVal {
    def bufferProjected(distance: Double): Projected[Geometry] =
      value.buffer(distance).withSRID(value.srid)

    def buffer0: Projected[Geometry] = value.bufferProjected(0)

    def buffer0AndSimplify(tolerance: Double): Projected[Geometry] = {
      val a = value.buffer(0)
      val b = DouglasPeuckerSimplifier.simplify(a, tolerance)
      b.withSRID(value.srid)
    }

    def toPolygons: List[Projected[Polygon]] = value.geom.toPolygons.map(_.withSRID(value.srid))

    def toGml3: String = Gml3Writer().write(value)

    def intersection(other: Projected[Geometry]): Projected[Geometry] = {
      val otherReprojected =
        if (value.srid == other.srid)
          other.geom.reproject(CRS.fromEpsgCode(other.srid), CRS.fromEpsgCode(value.srid))
        else other.geom
      value.geom.intersection(otherReprojected).withSRID(value.srid)
    }

    def intersectionSafer(other: Projected[Geometry]): Projected[Geometry] =
      try
        intersection(other)
      catch {
        case _: Throwable =>
          value.bufferProjected(0.0000001).intersection(other.bufferProjected(0.0000001))
      }

    def difference(other: Projected[Geometry]): Projected[Geometry] = {
      val otherReprojected =
        if (value.srid == other.srid)
          other.geom.reproject(CRS.fromEpsgCode(other.srid), CRS.fromEpsgCode(value.srid))
        else other.geom
      value.geom.difference(otherReprojected).withSRID(value.srid)
    }

    def differenceSafer(other: Projected[Geometry]): Projected[Geometry] =
      try
        difference(other)
      catch {
        case _: Throwable =>
          value.bufferProjected(0.0000001).difference(other.bufferProjected(0.0000001))
      }

    def union(other: Projected[Geometry]): Projected[Geometry] = {
      val otherReprojected =
        if (value.srid == other.srid)
          other.geom.reproject(CRS.fromEpsgCode(other.srid), CRS.fromEpsgCode(value.srid))
        else other.geom
      value.geom.union(otherReprojected).withSRID(value.srid)
    }

    def unionSafer(other: Projected[Geometry]): Projected[Geometry] =
      try
        union(other)
      catch {
        case _: Throwable =>
          value.bufferProjected(0.0000001).union(other.bufferProjected(0.0000001))
      }

    def reprojectTo(targetCrs: CRS): Projected[Geometry] = (value.srid, targetCrs.epsgCode) match {
      case (_, None) => sys.error(s"Cannot reproject to $targetCrs")
      case (3857, Some(trgSrid)) => value.reproject(WebMercator, targetCrs)(trgSrid)
      case (4326, Some(trgSrid)) => value.reproject(LatLng, targetCrs)(trgSrid)
      case (srid, _) => sys.error(s"Unexpected source SRID: $srid")
    }

    def areaInMeters(): Long = {
      val g = value.reprojectTo(LatLng)
      // Move projection center to the center of a feature
      val centroid = g.getCentroid
      val targetCrs = CRS.fromString(s"+proj=cea +ellps=WGS84 +lon_0=${centroid.getX}")
      val projected = g.reproject(LatLng, targetCrs)(0)
      projected.getArea.round
    }
  }

  implicit class ProjectedGeometryCollectionOps(val value: List[Projected[Geometry]])
      extends AnyVal {
    def toPolygons: List[Projected[Polygon]] = value.flatMap(_.toPolygons)
  }
}
