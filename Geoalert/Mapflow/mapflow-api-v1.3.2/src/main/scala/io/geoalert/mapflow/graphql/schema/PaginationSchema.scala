package io.geoalert.mapflow.graphql.schema

import io.circe.generic.auto._
import sangria.macros.derive.ObjectTypeName
import sangria.macros.derive.deriveInputObjectType
import sangria.macros.derive.deriveObjectType
import sangria.marshalling.circe._
import sangria.schema.Argument
import sangria.schema.InputObjectType
import sangria.schema.ObjectType

import io.geoalert.mapflow.graphql.GraphQLContext
import io.geoalert.mapflow.service.PagedRequest
import io.geoalert.mapflow.service.PagedResponse

trait PaginationSchema extends ProjectSchema {
  implicit val PagedRequestType: ObjectType[GraphQLContext, PagedRequest] =
    deriveObjectType[GraphQLContext, PagedRequest]()

  implicit val PagedRequestInputType: InputObjectType[PagedRequest] =
    deriveInputObjectType[PagedRequest]()

  def pagedResponseType[A: Lambda[M => ObjectType[GraphQLContext, M]]](
      name: String
    ): ObjectType[GraphQLContext, PagedResponse[A]] =
    deriveObjectType[GraphQLContext, PagedResponse[A]](ObjectTypeName(name))

  val PagedRequestArg: Argument[PagedRequest] = Argument("filter", PagedRequestInputType)
}
