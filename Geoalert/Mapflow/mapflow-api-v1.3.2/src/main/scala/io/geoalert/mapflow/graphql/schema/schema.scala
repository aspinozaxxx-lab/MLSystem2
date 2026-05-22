package io.geoalert.mapflow.graphql.schema

object schema
    extends CommonSchema
       with GeometrySchema
       with VectorLayerSchema
       with RasterLayerSchema
       with WorkflowDefSchema
       with AoiFilterSchema
       with AoiSchema
       with ProcessingSchema
       with ProjectSchema
       with ProgressSchema
       with UserSchema
       with UploadSchema
       with BillingSchema
       with PaginationSchema
       with TeamSchema {}
