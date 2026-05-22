package io.geoalert.mapflow.model

case class PngLinkInput(
    tileRow: Int,
    tileColumn: Int,
    tileMatrix: Int,
    connectIdType: String,
    cqlFilter: Option[String],
  )
