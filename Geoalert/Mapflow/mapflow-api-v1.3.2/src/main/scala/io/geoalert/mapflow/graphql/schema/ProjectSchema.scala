package io.geoalert.mapflow.graphql.schema

import java.util.UUID

import io.circe.generic.auto._
import io.geoalert.mapflow.graphql.GraphQLContext
import io.geoalert.mapflow.model._
import io.geoalert.mapflow.service.PagedResponse
import sangria.macros.derive.deriveInputObjectType
import sangria.macros.derive.deriveObjectType
import sangria.marshalling.circe._
import sangria.schema.Argument
import sangria.schema.InputObjectType
import sangria.schema.ListInputType
import sangria.schema.ObjectType
import sangria.schema.OptionInputType
import schema._

trait ProjectSchema {
  implicit val ProjectType: ObjectType[GraphQLContext, Project] =
    deriveObjectType[GraphQLContext, Project]()

  implicit val ProjectBriefType: ObjectType[GraphQLContext, ProjectBrief] =
    deriveObjectType[GraphQLContext, ProjectBrief]()

  implicit val ProjectsPagedResponseType: ObjectType[GraphQLContext, PagedResponse[ProjectBrief]] =
    pagedResponseType[ProjectBrief]("PagedProjectBrief")
  val CreateProjectInputType: InputObjectType[CreateProjectInput] =
    deriveInputObjectType[CreateProjectInput]()
  val UpdateProjectInputType: InputObjectType[UpdateProjectInput] =
    deriveInputObjectType[UpdateProjectInput]()
  val ShareProjectInputType: InputObjectType[UserProject] =
    deriveInputObjectType[UserProject]()

  val CreateProjectArg: Argument[CreateProjectInput] = Argument("data", CreateProjectInputType)
  val UpdateProjectArg: Argument[UpdateProjectInput] = Argument("data", UpdateProjectInputType)
  val ShareProjectArg: Argument[UserProject] = Argument("data", ShareProjectInputType)
  val ProjectIdsArg: Argument[Option[Seq[UUID]]] =
    Argument("projectIds", OptionInputType(ListInputType(UuidIdType)))
  val ProjectIdArg: Argument[UUID] = Argument("projectId", UuidIdType)
}
