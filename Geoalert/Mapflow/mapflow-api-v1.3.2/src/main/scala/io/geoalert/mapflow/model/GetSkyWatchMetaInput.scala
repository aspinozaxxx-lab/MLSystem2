package io.geoalert.mapflow.model

case class GetSkyWatchMetaInput(
    location: Location,
    resolution: String,
    coverage: Int,
    start_date: String,
    end_date: String,
    order_by: List[String],
  )

case class Location(coordinates: List[List[List[Double]]], `type`: String)
