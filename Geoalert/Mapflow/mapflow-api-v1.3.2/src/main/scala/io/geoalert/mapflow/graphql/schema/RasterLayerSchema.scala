package io.geoalert.mapflow.graphql.schema

import sangria.macros.derive.IncludeMethods
import sangria.macros.derive.deriveObjectType
import sangria.schema.ObjectType
import schema._

import io.geoalert.mapflow.graphql.GraphQLContext
import io.geoalert.mapflow.model.RasterLayer

trait RasterLayerSchema {
  implicit val RasterLayerType: ObjectType[GraphQLContext, RasterLayer] =
    deriveObjectType[GraphQLContext, RasterLayer](
      IncludeMethods("tileJsonUrl", "tileUrl")
    )
}
