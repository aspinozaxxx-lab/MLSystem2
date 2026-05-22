package io.geoalert.mapflow.model

case class ProcessingParams(
    url: Option[String],
    rasterLogin: Option[String],
    rasterPassword: ],
    zoom: Option[String],
    sourceType: Option[String],
    dataProvider: Option[String],
    rest: Map[String, String],
  ) {
  val isCredentialsSpecified: Boolean =
    rasterLogin.exists(_.nonEmpty) && rasterPassword.exists(_.nonEmpty)

  def toMap: Map[String, String] = Seq(
    url.map("url" -> _),
    rasterLogin.map("raster_login" -> _),
    rasterPassword.map("raster_password" -> _),
    zoom.map("zoom" -> _),
    sourceType.map("source_type" -> _),
    dataProvider.map("data_provider" -> _),
  ).flatten.toMap ++ rest
}

object ProcessingParams {
  def apply(params: Map[String, String]): ProcessingParams = new ProcessingParams(
    params.get("url"),
    params.get("raster_login"),
    params.get("raster_password"),
    params.get("zoom"),
    params.get("source_type"),
    params.get("data_provider"),
    // NB: Partition size cannot be adjusted by the end user
    params - "url" - "raster_login" - "raster_password" - "zoom" - "source_type" - "partition_size",
  )
}
