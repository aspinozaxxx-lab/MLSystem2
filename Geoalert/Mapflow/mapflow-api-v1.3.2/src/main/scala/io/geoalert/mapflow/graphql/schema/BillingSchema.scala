package io.geoalert.mapflow.graphql.schema

import java.time.Instant

import sangria.macros.derive.deriveObjectType
import sangria.schema.Argument
import sangria.schema.ObjectType
import sangria.schema.OptionInputType
import sangria.schema.StringType
import schema._

import io.geoalert.mapflow.graphql.GraphQLContext
import io.geoalert.mapflow.service.billing.BillingReport
import io.geoalert.mapflow.service.billing.ProcessingReport

trait BillingSchema {
  implicit val ProcessingReportType: ObjectType[GraphQLContext, ProcessingReport] =
    deriveObjectType[GraphQLContext, ProcessingReport]()

  val EmailArg: Argument[String] = Argument("email", StringType)

  val StartDateArg: Argument[Option[Instant]] = Argument("start_date", OptionInputType(InstantType))
  val EndDateArg: Argument[Option[Instant]] = Argument("end_date", OptionInputType(InstantType))

  implicit val BillingReportType: ObjectType[GraphQLContext, BillingReport] =
    deriveObjectType[GraphQLContext, BillingReport]()
}
