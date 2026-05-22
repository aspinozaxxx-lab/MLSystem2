package io.geoalert.mapflow.graphql.schema

import sangria.macros.derive.IncludeMethods
import sangria.macros.derive.deriveObjectType
import sangria.schema.ObjectType
import schema._

import io.geoalert.mapflow.graphql.GraphQLContext
import io.geoalert.mapflow.model.VectorLayer

trait VectorLayerSchema {
  implicit val VectorLayerType: ObjectType[GraphQLContext, VectorLayer] =
    deriveObjectType[GraphQLContext, VectorLayer](
      IncludeMethods("tileJsonUrl", "tileUrl")
    )
}
