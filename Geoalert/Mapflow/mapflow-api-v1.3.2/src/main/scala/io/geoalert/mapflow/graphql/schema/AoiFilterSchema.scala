package io.geoalert.mapflow.graphql.schema

import java.util.UUID

import io.circe.generic.auto._
import sangria.macros.derive.deriveInputObjectType
import sangria.macros.derive.deriveObjectType
import sangria.marshalling.CoercedScalaResultMarshaller
import sangria.marshalling.FromInput
import sangria.marshalling.ResultMarshaller
import sangria.marshalling.circe._
import sangria.schema.Argument
import sangria.schema.InputObjectType
import sangria.schema.ObjectType

import io.geoalert.mapflow.graphql.GraphQLContext
import io.geoalert.mapflow.model._

import geotrellis.vector.Geometry
import geotrellis.vector.Projected

trait AoiFilterSchema extends CommonSchema with GeometrySchema {
  implicit val AoiFilterFromInput: FromInput[AoiFilter] = new FromInput[AoiFilter] {
    override val marshaller: ResultMarshaller = CoercedScalaResultMarshaller.default
    override def fromResult(node: marshaller.Node): AoiFilter = {
      val fields = node.asInstanceOf[Map[String, Any]]
      AoiFilter(
        fields.get("ids").flatMap(_.asInstanceOf[Option[Vector[UUID]]].map(_.toList)),
        fields.get("processingIds").flatMap(_.asInstanceOf[Option[Vector[UUID]]].map(_.toList)),
        fields.get("statuses").flatMap(_.asInstanceOf[Option[Vector[Status]]].map(_.toList)),
        fields.get("geometry").flatMap(_.asInstanceOf[Option[Projected[Geometry]]]),
      )
    }
  }

  implicit val AoiFilterType: ObjectType[GraphQLContext, AoiFilter] =
    deriveObjectType[GraphQLContext, AoiFilter]()

  implicit val AoiFilterInputType: InputObjectType[AoiFilter] = deriveInputObjectType[AoiFilter]()

  val AoiFilterArg: Argument[AoiFilter] = Argument("filter", AoiFilterInputType)
}
