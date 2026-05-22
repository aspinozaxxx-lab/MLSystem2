package io.geoalert.mapflow.graphql.schema

import java.util.UUID

import io.circe.generic.auto._
import io.geoalert.mapflow.graphql.GraphQLContext
import io.geoalert.mapflow.graphql.args.processing.ProcessingFilters
import io.geoalert.mapflow.model.BlockParametersInput
import io.geoalert.mapflow.model.CreateProcessingInput
import io.geoalert.mapflow.model.Message
import io.geoalert.mapflow.model.MessageParameter
import io.geoalert.mapflow.model.Processing
import io.geoalert.mapflow.model.Rating
import io.geoalert.mapflow.model.ReviewStatus
import io.geoalert.mapflow.model.UpdateProcessingInput
import io.geoalert.mapflow.model.enums.ProcessingSortOrder
import io.geoalert.mapflow.model.enums.SortBy
import io.geoalert.mapflow.repo.ProcessingReviewDto
import io.geoalert.mapflow.service.PagedResponse
import sangria.macros.derive.ExcludeFields
import sangria.macros.derive.ExcludeInputFields
import sangria.macros.derive.TransformValueNames
import sangria.macros.derive.deriveEnumType
import sangria.macros.derive.deriveInputObjectType
import sangria.macros.derive.deriveObjectType
import sangria.marshalling.circe._
import sangria.schema.Argument
import sangria.schema.EnumType
import sangria.schema.EnumValue
import sangria.schema.InputObjectType
import sangria.schema.ObjectType
import sangria.util.StringUtil

trait ProcessingSchema
    extends CommonSchema
       with VectorLayerSchema
       with RasterLayerSchema
       with WorkflowDefSchema
       with GeometrySchema
       with UserSchema
       with ProgressSchema
       with DataProviderSchema
       with PaginationSchema {
  implicit val RatingType: ObjectType[GraphQLContext, Rating] =
    deriveObjectType[GraphQLContext, Rating]()

  implicit val ReviewDtoType: ObjectType[GraphQLContext, ProcessingReviewDto] =
    deriveObjectType[GraphQLContext, ProcessingReviewDto]()
  implicit val ReviewStatusEnum: EnumType[ReviewStatus] = EnumType(
    "ReviewStatus",
    None,
    List(
      EnumValue(ReviewStatus.InReview.repr, value = ReviewStatus.InReview),
      EnumValue(ReviewStatus.Accepted.repr, value = ReviewStatus.Accepted),
      EnumValue(ReviewStatus.NotAccepted.repr, value = ReviewStatus.NotAccepted),
      EnumValue(ReviewStatus.Refunded.repr, value = ReviewStatus.Refunded),
    ),
  )

  implicit val ProcessingSortOrderEnum: EnumType[ProcessingSortOrder] =
    deriveEnumType[ProcessingSortOrder](
      TransformValueNames(StringUtil.camelCaseToUnderscore(_).toUpperCase)
    )

  implicit val SortByEnum: EnumType[SortBy] =
    deriveEnumType[SortBy](
      TransformValueNames(StringUtil.escapeString(_).toUpperCase)
    )

  implicit val ProcessingType: ObjectType[GraphQLContext, Processing] =
    deriveObjectType[GraphQLContext, Processing](
      ExcludeFields("params", "cost")
    )

  implicit val MessageType: ObjectType[GraphQLContext, Message] =
    deriveObjectType[GraphQLContext, Message]()

  implicit val MessageParameterType: ObjectType[GraphQLContext, MessageParameter] =
    deriveObjectType[GraphQLContext, MessageParameter]()

  implicit val BlockParametersInputType: InputObjectType[BlockParametersInput] =
    deriveInputObjectType[BlockParametersInput]()

  implicit val ProcessingPagedResponseType: ObjectType[GraphQLContext, PagedResponse[Processing]] =
    pagedResponseType[Processing]("PagedProcessing")
  val CreateProcessingInputType: InputObjectType[CreateProcessingInput] =
    deriveInputObjectType[CreateProcessingInput](
      ExcludeInputFields("params", "source")
    )
  val UpdateProcessingInputType: InputObjectType[UpdateProcessingInput] =
    deriveInputObjectType[UpdateProcessingInput]()
  implicit val ProcessingFiltersInputType: InputObjectType[ProcessingFilters] =
    deriveInputObjectType[ProcessingFilters]()

  val ProcessingFiltersArg: Argument[ProcessingFilters] =
    Argument("filters", ProcessingFiltersInputType)

  val CreateProcessingArg: Argument[CreateProcessingInput] =
    Argument("data", CreateProcessingInputType)
  val UpdateProcessingArg: Argument[UpdateProcessingInput] =
    Argument("data", UpdateProcessingInputType)
  val ProcessingIdArg: Argument[UUID] = Argument("processingId", UuidIdType)
}
