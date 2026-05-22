package io.geoalert.mapflow.model

import io.geoalert.mapflow.model.DataSource.DataSource
import io.geoalert.mapflow.model.SourceType.SourceType

case class BlockConfig(
    name: String,
    displayName: String,
    optional: Boolean,
    price: Double,
    defaultEnabled: Boolean,
  )

case class WorkflowDefSummary(
    name: String,
    sourceType: Option[SourceType],
    source: Option[DataSource],
    pricePerSqKm: Double,
    zoom: Option[Int],
    url: Option[String],
    userInputBucket: Option[String],
    partitionSize: Option[Double],
    blocks: Seq[BlockConfig],
  )
