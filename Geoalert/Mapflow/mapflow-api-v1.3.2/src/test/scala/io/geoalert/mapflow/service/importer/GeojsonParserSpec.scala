package io.geoalert.mapflow.service.importer

import java.io.ByteArrayInputStream

import cats.instances.list._
import cats.syntax.applicative._
import cats.syntax.option._
import geotrellis.proj4.{LatLng, WebMercator}
import io.geoalert.mapflow.implicits.GeometryOps._
import geotrellis.vector._
import org.scalatest._

class GeojsonParserSpec extends FunSpec with Matchers {
  val coordLists = List(
    "[[[1.1, 2.1], [2.1, 2.1], [33.45, 5.0], [1.1, 2.1]]]",
    "[[[1, 1], [2, 2], [3, 3], [1, 1]]]"
  )

  val ps = for {
    c <- coordLists
    g <- List(s"""{"type": "Polygon", "coordinates": $c}""", s"""{"coordinates": $c, "type": "Polygon"}""")
  } yield g

  val ms = (for {
    c1 <- coordLists
    c2 <- coordLists if c1 != c2
    g <- List(
      s"""{"type": "MultiPolygon", "coordinates": [$c1, $c2]}""",
      s"""{"coordinates": [$c1, $c2], "type": "MultiPolygon"}""",
      s"""{"type": "MultiPolygon", "coordinates": [$c1]}""",
      s"""{"coordinates": [$c1], "type": "MultiPolygon"}"""
    )
  } yield g).distinct

  val gcs = {
    val gsLists = ps.take(3).replicateA(2) ++ List(ms.take(1)) ++
      List(List(s"""{"type": "GeometryCollection", "geometries": [${ps.head}]}"""))
    for {
      gs <- gsLists
      gc <- List(
        s"""{"type": "GeometryCollection", "geometries": [${gs.mkString(", ")}]}""",
        s"""{"geometries": [${gs.mkString(", ")}], "type": "GeometryCollection"}"""
      )
    } yield gc
  }

  val geoms = ps ++ ms ++ gcs

  val features = for {
    g <- geoms
    f <- List(s"""{"type": "Feature", "geometry": $g}""", s"""{"geometry": $g, "type": "Feature"}""")
  } yield f

  lazy val featureTriplets =
    (features.filter(_.contains(""""Polygon"""")).take(4) ++
      features.filter(_.contains(""""MultiPolygon"""")).take(4) ++
      features.filter(_.contains(""""GeometryCollection"""")).take(4)).combinations(3)

  def featureCollection(features: List[String], crs: Option[String] = None) =
    """{"type": "FeatureCollection", """ +
       s"${crs.map(s => s""""crs": "$s", """).getOrElse("")}" +
       s""""features": [ ${features.mkString(", ")}]}"""

  def parse(geojson: String) =
    GeojsonParser.polygonStream(new ByteArrayInputStream(geojson.getBytes())).toList

  describe("GeojsonParser") {
    it(s"should parse each geometry correctly") {
      for (f <- features) {
        val geojson = featureCollection(List(f))
        parse(geojson) should be(List(f.parseGeoJson[Geometry]()).toPolygons)
      }
    }

    it(s"should parse all triplets of geometries correctly") {
      for (fs <- featureTriplets) {
        val geojson = featureCollection(fs)
        parse(geojson) should be(fs.map(_.parseGeoJson[Geometry]()).toPolygons)
      }
    }

    it(s"should read CRS and parse each geometry correctly") {
      for (f <- features) {
        val geojson = featureCollection(List(f), "EPSG:3857".some)
        parse(geojson) should be(List(f.parseGeoJson[Geometry]()).toPolygons.map(_.reproject(WebMercator, LatLng)))
      }
    }
  }
}
