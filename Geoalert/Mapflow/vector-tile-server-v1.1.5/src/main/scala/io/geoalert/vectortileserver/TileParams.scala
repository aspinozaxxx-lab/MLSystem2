package io.geoalert.vectortileserver

case class TileParams(minAreaFactor: Option[Double],
                      simplifyFactor: Option[Double],
                      simplifyMaxZoom: Option[Int],
                      minZoom: Option[Int],
                      maxFeatures: Option[Int],
                      asPoints: Option[Boolean]) {
  def toUrlParams(): String =
    List(
      minAreaFactor.map(v => s"min_area_factor=$v"),
      simplifyFactor.map(v => s"simplify_factor=$v"),
      simplifyMaxZoom.map(v => s"simplify_max_zoom=$v"),
      minZoom.map(v => s"min_zoom=$v"),
      maxFeatures.map(v => s"max_features=$v"),
      asPoints.map(v => s"points=$v")
    ).flatten.mkString("?", "&", "")
}

object TileParams {
  def apply(map: Map[String, String]): TileParams =
    TileParams(
      map.get("min_area_factor").map(_.toDouble),
      map.get("simplify_factor").map(_.toDouble),
      map.get("simplify_max_zoom").map(_.toInt),
      map.get("min_zoom").map(_.toInt),
      map.get("max_features").map(_.toInt),
      map.get("points").map(_.toBoolean)
    )
}
