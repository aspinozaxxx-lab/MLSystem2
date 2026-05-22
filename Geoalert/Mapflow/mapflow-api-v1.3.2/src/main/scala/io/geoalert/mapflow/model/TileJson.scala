package io.geoalert.mapflow.model

case class TileJson(
    bounds: Seq[Double],
    center: Seq[Double],
    name: String,
    minzoom: Int,
    maxzoom: Int,
    tiles: Seq[String],
    vector_layers: Seq[Map[String, String]],
  )
