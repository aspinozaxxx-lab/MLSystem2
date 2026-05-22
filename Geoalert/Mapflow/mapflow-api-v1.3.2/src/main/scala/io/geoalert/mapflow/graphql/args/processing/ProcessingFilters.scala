package io.geoalert.mapflow.graphql.args.processing

import java.time.Instant

import io.geoalert.mapflow.model.enums.ProcessingSortOrder
import io.geoalert.mapflow.model.enums.SortBy
import io.geoalert.mapflow.model.Status
import sangria.macros.derive.GraphQLDeprecated

case class ProcessingFilters(
    @GraphQLDeprecated("Use the terms field instead")
    emails: Option[Seq[String]],
    statuses: Option[Seq[Status]],
    terms: Option[String],
    dateFrom: Option[Instant],
    dateTo: Option[Instant],
    limit: Option[Int],
    offset: Option[Int],
    sortOrder: Option[ProcessingSortOrder],
    sortBy: Option[SortBy],
  )
