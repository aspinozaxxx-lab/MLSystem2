package io.geoalert.mapflow.graphql.schema

import sangria.macros.derive.deriveObjectType
import sangria.schema.Argument
import sangria.schema.ListInputType
import sangria.schema.ObjectType
import sangria.schema.OptionInputType

import io.geoalert.mapflow.graphql.GraphQLContext
import io.geoalert.mapflow.model._

trait ProgressSchema extends CommonSchema {
  implicit val ProgressDetailType: ObjectType[GraphQLContext, ProgressDetail] =
    deriveObjectType[GraphQLContext, ProgressDetail]()

  implicit val ProgressType: ObjectType[GraphQLContext, Progress] =
    deriveObjectType[GraphQLContext, Progress]()

  val StatusesArg: Argument[Option[Seq[Status]]] =
    Argument("statuses", OptionInputType(ListInputType(StatusEnum)))
}
