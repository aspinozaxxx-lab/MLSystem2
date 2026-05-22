package io.geoalert.mapflow.graphql.schema

import java.util.UUID

import io.circe.generic.auto._
import sangria.macros.derive.AddFields
import sangria.macros.derive.ExcludeFields
import sangria.macros.derive.deriveInputObjectType
import sangria.macros.derive.deriveObjectType
import sangria.marshalling.CoercedScalaResultMarshaller
import sangria.marshalling.FromInput
import sangria.marshalling.ResultMarshaller
import sangria.marshalling.circe._
import sangria.schema.Argument
import sangria.schema.BooleanType
import sangria.schema.Field
import sangria.schema.FloatType
import sangria.schema.InputObjectType
import sangria.schema.ListType
import sangria.schema.ObjectType
import sangria.schema.OptionInputType

import io.geoalert.mapflow.graphql.GraphQLContext
import io.geoalert.mapflow.model._
import io.geoalert.mapflow.repo.BlockParameters

trait WorkflowDefSchema extends CommonSchema with UploadSchema {
  implicit val BlockParametersType: ObjectType[GraphQLContext, BlockParameters] =
    deriveObjectType[GraphQLContext, BlockParameters]()
  implicit val BlockConfigType: ObjectType[GraphQLContext, BlockConfig] =
    deriveObjectType[GraphQLContext, BlockConfig](
      ExcludeFields("price")
    )
  implicit val WorkflowDefType: ObjectType[GraphQLContext, WorkflowDef] =
    deriveObjectType[GraphQLContext, WorkflowDef](
      ExcludeFields("weName", "weId"),
      AddFields(
        Field(
          "blocks",
          ListType(BlockConfigType),
          resolve = ctx => ctx.value.workflowDefSummary.blocks,
        ),
      ),
    )

  val CreateWorkflowDefInputType: InputObjectType[CreateWorkflowDefInput] =
    deriveInputObjectType[CreateWorkflowDefInput]()

  val UpdateWorkflowDefInputType: InputObjectType[UpdateWorkflowDefInput] =
    deriveInputObjectType[UpdateWorkflowDefInput]()

  implicit val createWorkflowDefInputFromInput: FromInput[CreateWorkflowDefInput] =
    new FromInput[CreateWorkflowDefInput] {
      override val marshaller: ResultMarshaller =
        CoercedScalaResultMarshaller.default
      override def fromResult(node: marshaller.Node): CreateWorkflowDefInput = {
        val fields = node.asInstanceOf[Map[String, Any]]
        CreateWorkflowDefInput(
          fields.get("projectId").flatMap(_.asInstanceOf[Option[UUID]]),
          fields("name").asInstanceOf[String],
          fields.get("description").flatMap(_.asInstanceOf[Option[String]]),
          fields.get("file").flatMap(_.asInstanceOf[Option[Upload]]),
          fields.get("ymlString").flatMap(_.asInstanceOf[Option[String]]),
          fields.get("pricePerSqKm").flatMap(_.asInstanceOf[Option[Double]]),
          fields.get("isDefault").flatMap(_.asInstanceOf[Option[Boolean]]),
        )
      }
    }

  implicit val updateWorkflowDefInputFromInput: FromInput[UpdateWorkflowDefInput] =
    new FromInput[UpdateWorkflowDefInput] {
      override val marshaller: ResultMarshaller =
        CoercedScalaResultMarshaller.default
      override def fromResult(node: marshaller.Node): UpdateWorkflowDefInput = {
        val fields = node.asInstanceOf[Map[String, Any]]
        UpdateWorkflowDefInput(
          fields("id").asInstanceOf[UUID],
          fields.get("projectId").flatMap(_.asInstanceOf[Option[UUID]]),
          fields.get("name").flatMap(_.asInstanceOf[Option[String]]),
          fields.get("description").flatMap(_.asInstanceOf[Option[String]]),
          fields.get("file").flatMap(_.asInstanceOf[Option[Upload]]),
          fields.get("ymlString").flatMap(_.asInstanceOf[Option[String]]),
          fields.get("pricePerSqKm").flatMap(_.asInstanceOf[Option[Double]]),
          fields.get("isDefault").flatMap(_.asInstanceOf[Option[Boolean]]),
        )
      }
    }

  val CreateWorkflowDefArg: Argument[CreateWorkflowDefInput] =
    Argument("data", CreateWorkflowDefInputType)
  val UpdateWorkflowDefArg: Argument[UpdateWorkflowDefInput] =
    Argument("data", UpdateWorkflowDefInputType)

  val IsDefaultArg: Argument[Option[Boolean]] = Argument("isDefault", OptionInputType(BooleanType))

  val WorkflowDefIdArg: Argument[UUID] = Argument("workflowDefId", UuidIdType)
}
