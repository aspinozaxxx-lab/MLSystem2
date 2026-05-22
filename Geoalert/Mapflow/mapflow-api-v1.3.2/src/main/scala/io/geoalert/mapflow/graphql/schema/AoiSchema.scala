package io.geoalert.mapflow.graphql.schema

import java.util.UUID

import io.circe.generic.auto._
import sangria.macros.derive.ExcludeFields
import sangria.macros.derive.deriveInputObjectType
import sangria.macros.derive.deriveObjectType
import sangria.marshalling.CoercedScalaResultMarshaller
import sangria.marshalling.FromInput
import sangria.marshalling.ResultMarshaller
import sangria.marshalling.circe._
import sangria.schema._

import io.geoalert.mapflow.graphql.GraphQLContext
import io.geoalert.mapflow.model._

import geotrellis.vector.Geometry
import geotrellis.vector.Projected

trait AoiSchema extends CommonSchema with GeometrySchema with ProcessingSchema {
  implicit val AoiType: ObjectType[GraphQLContext, Aoi] = deriveObjectType[GraphQLContext, Aoi](
    ExcludeFields("processingId")
  )

  implicit val AoiListType: ObjectType[GraphQLContext, AoiList] =
    deriveObjectType[GraphQLContext, AoiList]()

  implicit val AoiStatsType: ObjectType[GraphQLContext, AoiStats] =
    deriveObjectType[GraphQLContext, AoiStats]()

  implicit val AoiSortFieldEnum: EnumType[AoiSortField] = EnumType[AoiSortField](
    "AoiSortField",
    None,
    List(
      EnumValue(AoiSortField.Area.repr, value = AoiSortField.Area),
      EnumValue(AoiSortField.PercentCompleted.repr, value = AoiSortField.PercentCompleted),
      EnumValue(AoiSortField.Status.repr, value = AoiSortField.Status),
    ),
  )

  implicit val AoiSortEntryInputFromInput: FromInput[AoiSortEntry] = new FromInput[AoiSortEntry] { // TODO WTF?! WHY DO I NEED TO DO THIS???
    override val marshaller: ResultMarshaller = CoercedScalaResultMarshaller.default
    override def fromResult(node: marshaller.Node): AoiSortEntry = {
      val fields = node.asInstanceOf[Map[String, Any]]
      AoiSortEntry(
        fields("field").asInstanceOf[AoiSortField],
        fields.get("desc").flatMap(_.asInstanceOf[Option[Boolean]]),
      )
    }
  }

  implicit val AoiSortEntryInputType: InputObjectType[AoiSortEntry] =
    deriveInputObjectType[AoiSortEntry]()

  val AoiSortArg: Argument[Option[Seq[AoiSortEntry]]] =
    Argument("sort", OptionInputType(ListInputType(AoiSortEntryInputType)))

  implicit val CreateAoisFromGeometryInputFromInput: FromInput[CreateAoisFromGeometryInput] =
    new FromInput[CreateAoisFromGeometryInput] {
      // TODO can we reduce this kind of boilerplate definitions?
      override val marshaller: ResultMarshaller = CoercedScalaResultMarshaller.default
      override def fromResult(node: marshaller.Node): CreateAoisFromGeometryInput = {
        val fields = node.asInstanceOf[Map[String, Any]]
        CreateAoisFromGeometryInput(
          fields("processingId").asInstanceOf[UUID],
          fields("geometry").asInstanceOf[Projected[Geometry]],
        )
      }
    }

  val CreateAoisFromGeometryInputType: InputObjectType[CreateAoisFromGeometryInput] =
    deriveInputObjectType[CreateAoisFromGeometryInput]()

  val CreateAoisFromGeometryArg: Argument[CreateAoisFromGeometryInput] =
    Argument("data", CreateAoisFromGeometryInputType)

  implicit val CreateAoisFromFileInputFromInput: FromInput[CreateAoisFromFileInput] =
    new FromInput[CreateAoisFromFileInput] {
      override val marshaller: ResultMarshaller = CoercedScalaResultMarshaller.default
      override def fromResult(node: marshaller.Node): CreateAoisFromFileInput = {
        val fields = node.asInstanceOf[Map[String, Any]]
        CreateAoisFromFileInput(
          fields("processingId").asInstanceOf[UUID],
          fields("file").asInstanceOf[Upload],
        )
      }
    }

  val CreateAoisFromFileInputType: InputObjectType[CreateAoisFromFileInput] =
    deriveInputObjectType[CreateAoisFromFileInput]()

  val CreateAoisFromFileArg: Argument[CreateAoisFromFileInput] =
    Argument("data", CreateAoisFromFileInputType)

  val AoiIdArg: Argument[UUID] = Argument("aoiId", UuidIdType)
}
