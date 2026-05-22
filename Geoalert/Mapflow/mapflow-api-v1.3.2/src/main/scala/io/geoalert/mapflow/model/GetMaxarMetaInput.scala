package io.geoalert.mapflow.model

case class GetMaxarMetaInput(
    url: String,
    connectId: String,
    body: Option[String],
  )
