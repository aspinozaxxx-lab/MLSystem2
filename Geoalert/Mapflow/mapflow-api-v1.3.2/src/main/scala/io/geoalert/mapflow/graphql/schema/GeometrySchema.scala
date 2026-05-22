package io.geoalert.mapflow.graphql.schema

import scala.util.Try

import cats.syntax.either._
import io.circe.parser.decode
import sangria.schema.Argument
import sangria.schema.IntType
import sangria.schema.OptionInputType
import sangria.schema.ScalarAlias
import sangria.schema.StringType
import sangria.validation.ValueCoercionViolation

import geotrellis.vector.Geometry
import geotrellis.vector.Projected
import geotrellis.vector._

trait GeometrySchema {
  case class GeometryCoercionViolation(e: Throwable)
      extends ValueCoercionViolation(s"Couldn't parse geojson: $e")
  implicit val GeometryType: ScalarAlias[Projected[Geometry], String] =
    ScalarAlias[Projected[Geometry], String](
      StringType,
      _.geom.toGeoJson(),
      s =>
        Try(s.parseGeoJson[Geometry]().withSRID(4326))
          .toEither
          .leftMap(e => GeometryCoercionViolation(e)),
    )
  val GeometryArg: Argument[Option[Projected[Geometry]]] =
    Argument("geometry", OptionInputType(GeometryType))

  case class BboxCoercionViolation(msg: String)
      extends ValueCoercionViolation(s"Couldn't parse bbox: $msg")
  implicit val BboxType: ScalarAlias[Extent, String] = ScalarAlias[Extent, String](
    StringType,
    extent => s"[${extent.xmin}, ${extent.ymin}, ${extent.xmax}, ${extent.ymax}]",
    s =>
      decode[List[Double]](s)
        .leftMap(e => BboxCoercionViolation(e.toString))
        .flatMap {
          case xmin :: ymin :: xmax :: ymax :: Nil =>
            Try(Extent(xmin, ymin, xmax, ymax))
              .toEither
              .leftMap(e => BboxCoercionViolation(e.toString))
          case _ => BboxCoercionViolation(s"Wrong bbox input: $s").asLeft
        },
  )
  val BboxArg: Argument[Extent] = Argument("bbox", BboxType)

  val XResArg: Argument[Int] = Argument("xRes", IntType)
  val YResArg: Argument[Int] = Argument("yRes", IntType)
}
