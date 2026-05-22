package io.geoalert.mapflow.service

case class PagedResponse[T](
    results: List[T],
    total: Int,
    count: Int,
  )
