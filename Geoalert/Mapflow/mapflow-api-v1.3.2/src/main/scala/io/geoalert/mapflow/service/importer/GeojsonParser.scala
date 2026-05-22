package io.geoalert.mapflow.service.importer

import scala.util.Try

import cats.syntax.either._
import cats.syntax.option._
import com.fasterxml.jackson.core.JsonFactory
import com.fasterxml.jackson.core.JsonToken._
import com.fasterxml.jackson.core.TreeNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.typesafe.scalalogging.LazyLogging

import geotrellis.proj4.CRS
import geotrellis.proj4.LatLng
import geotrellis.proj4.WebMercator
import geotrellis.vector._

object GeojsonParser extends LazyLogging {
  def polygonStream(geojson: java.io.InputStream): LazyList[Polygon] = {
    val jsonFactory = new JsonFactory()
    val parser = jsonFactory.createParser(geojson)
    parser.setCodec(new ObjectMapper())

    def nextFieldName(): Option[String] = parser.nextToken() match {
      case null => None
      case FIELD_NAME => parser.currentName().some
      case _ => nextFieldName()
    }

    def toFieldCond(cond: String => Boolean): Option[String] = nextFieldName() match {
      case None => None
      case Some(name) if cond(name) => Some(name)
      case _ => toFieldCond(cond)
    }

    def toField(name: String): Option[Unit] = toFieldCond(_ == name).map(_ => ())

    def coordinatesToPolygon(coordinates: String, crs: CRS): Polygon =
      s"""{"type": "Polygon", "coordinates": $coordinates}"""
        .parseGeoJson[Polygon]()
        .reproject(crs, LatLng)

    // Expects current position to be on the first coordinate number: ...[2.0, ...
    def readCoordinates(): String = {
      val buf = new StringBuilder("[[[" + parser.getText)
      var n = 3
      var last = VALUE_NUMBER_FLOAT
      while (n > 0) {
        val t = parser.nextToken()
        if (t == END_ARRAY) {
          n -= 1
          buf.append(']')
        }
        else if (t.isNumeric) {
          if (last.isNumeric) buf.append(',')
          buf.append(parser.getText)
        }
        else if (t == START_ARRAY) {
          n += 1
          if (last == END_ARRAY) buf.append(',')
          buf.append('[')
        }
        else sys.error(s"Unexpected token: 
        last = t
      }
      buf.toString()
    }

    def readCoordinatesList(): LazyList[String] = {
      def isStartArray = parser.nextToken() == START_ARRAY
      if (isStartArray && isStartArray && isStartArray)
        if (isStartArray) { // MultiPolygon
          def nextCoordinates(): LazyList[String] = parser.nextToken() match {
            case t if t.isNumeric => LazyList.cons(readCoordinates(), nextCoordinates())
            case START_ARRAY =>
              if (isStartArray && isStartArray) nextCoordinates()
              else sys.error(s"Unexpected token")
            case END_ARRAY =>
              while (parser.nextToken() != END_OBJECT) {}
              LazyList.empty
            case t => sys.error(s"Unexpected token: 
          }
          nextCoordinates()
        }
        else { // Polygon
          val coords = readCoordinates()
          while (parser.nextToken() != END_OBJECT) {}
          LazyList.cons(coords, LazyList.empty)
        }
      else sys.error("Coordinates list expected")
    }

    def readGeometryStartingWithCoordinates(crs: CRS): LazyList[Polygon] =
      readCoordinatesList().map(coordinatesToPolygon(_, crs))

    def readGeometryCollection(crs: CRS): LazyList[Polygon] =
      if (parser.nextToken() == START_ARRAY) {
        def nextGeometry(crs: CRS): LazyList[Polygon] =
          parser.nextToken() match {
            case START_OBJECT => readGeometry(crs) #::: nextGeometry(crs)
            case END_ARRAY =>
              while (parser.nextToken() != END_OBJECT) {}
              LazyList.empty
            case t => sys.error(s"Unexpected token: 
          }
        nextGeometry(crs)
      }
      else sys.error("Unexpected token")

    // Expects current position to be at '{'
    def readGeometry(crs: CRS): LazyList[Polygon] = parser.currentToken() match {
      case START_OBJECT =>
        def readNextGeom(crs: CRS): LazyList[Polygon] =
          toFieldCond(n => n == "coordinates" || n == "geometries" || n == "type") match {
            case Some("coordinates") => readGeometryStartingWithCoordinates(crs)
            case Some("geometries") => readGeometryCollection(crs)
            case Some("type") =>
              parser.nextTextValue() match {
                case "FeatureCollection" | "Feature" | "Polygon" | "MultiPolygon" |
                     "GeometryCollection" =>
                  readNextGeom(crs)
                case _ => sys.error(s"Unsupported geometry type: ${parser.getText()}")
              }
            case _ => LazyList.empty
          }
        readNextGeom(crs)
      case _ => LazyList.empty
    }

    def stream(crs: CRS): LazyList[Polygon] = toField("geometry") match {
      case Some(_) =>
        parser.nextToken()
        readGeometry(crs) #::: stream(crs)
      case None => LazyList.empty
    }

    def readCrs() = {
      def parseCrs(crs: String) =
        if (crs == "urn:ogc:def:crs:OGC:1.3:CRS84") LatLng.some
        else if (crs contains "3857") WebMercator.some
        else
          Try(CRS fromString crs)
            .toOption
            .orElse(Try(CRS fromWKT crs).toOption.flatten)
            .orElse(Try(CRS fromName crs).toOption)

      parser.nextToken() match {
        case VALUE_STRING =>
          val crs = parser.getText
          Either.fromOption(parseCrs(crs), new RuntimeException(s"Couldn't parse CRS: $crs"))
        case START_OBJECT =>
          val tree = parser.readValueAsTree[TreeNode]()
          val opt = Try(tree.get("properties").get("name").toString.replace("\"", ""))
            .toOption
            .flatMap(parseCrs)
          Either.fromOption(opt, new RuntimeException(s"Couldn't parse CRS: $tree"))
        case _ =>
          Left(
            new RuntimeException(s"Couldn't parse CRS, unexpected token: }")
          )
      }
    }

    toFieldCond(n => n == "crs" || n == "features") match {
      case Some("crs") => stream(readCrs().toTry.get)
      case Some("features") =>
        logger.info("CRS not specified, will use default LatLng.")
        stream(LatLng)
      case _ => LazyList.empty
    }
  }
}
