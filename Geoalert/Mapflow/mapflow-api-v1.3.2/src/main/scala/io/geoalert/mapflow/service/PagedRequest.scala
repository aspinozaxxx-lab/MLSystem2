package io.geoalert.mapflow.service

case class PagedRequest(
    offset: Option[Int],
    limit: Option[Int],
    filter: Option[String],
  )
