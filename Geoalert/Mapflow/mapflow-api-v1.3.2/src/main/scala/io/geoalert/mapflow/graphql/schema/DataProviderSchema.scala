package io.geoalert.mapflow.graphql.schema

import java.util.UUID

import io.circe.generic.auto._
import io.geoalert.mapflow.graphql.GraphQLContext
import io.geoalert.mapflow.model.CreateDataProviderInput
import io.geoalert.mapflow.model.DataProvider
import io.geoalert.mapflow.model.UpdateDataProviderInput
import sangria.macros.derive.ExcludeFields
import sangria.macros.derive.deriveInputObjectType
import sangria.macros.derive.deriveObjectType
import sangria.marshalling.circe._
import sangria.schema.Argument
import sangria.schema.InputObjectType
import sangria.schema.ObjectType

trait DataProviderSchema extends CommonSchema {
  implicit val DataProviderType: ObjectType[GraphQLContext, DataProvider] =
    deriveObjectType[GraphQLContext, DataProvider](
      ExcludeFields("pricePerMp")
    )
  implicit val CreateDataProviderInputType: InputObjectType[CreateDataProviderInput] =
    deriveInputObjectType[CreateDataProviderInput]()
  implicit val UpdateDataProviderInputType: InputObjectType[UpdateDataProviderInput] =
    deriveInputObjectType[UpdateDataProviderInput]()

  val CreateDataProviderArg: Argument[CreateDataProviderInput] =
    Argument("data", CreateDataProviderInputType)
  val UpdateDataProviderArg: Argument[UpdateDataProviderInput] =
    Argument("data", UpdateDataProviderInputType)

  val DataProviderIdArg: Argument[UUID] = Argument("dataProviderId", UuidIdType)
}
