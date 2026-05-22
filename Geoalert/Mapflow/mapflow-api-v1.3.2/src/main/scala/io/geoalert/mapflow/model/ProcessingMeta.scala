package io.geoalert.mapflow.model

import cats.MonadThrow
import cats.implicits.toTraverseOps
import io.circe.Decoder
import io.circe.Encoder
import io.circe.HCursor
import io.circe.Json
import io.circe.generic.semiauto.deriveEncoder
import io.geoalert.mapflow.syntax.all.circeSyntaxDecoderOps
import io.geoalert.mapflow.syntax.all.circeSyntaxJsonDecoderOps

case class ProcessingMeta(
    source: Option[String], // Contains 'maxar'
    maxarProduct: Option[String], // Contains: securewatch, vivid, basemaps
    sourceApp: Option[String], // Contains qgis
    rest: Map[String, String],
  ) {
  def toMap: Map[String, String] = Seq(
    source.map("source" -> _),
    maxarProduct.map("maxar_product" -> _),
    sourceApp.map("source-app" -> _),
  ).flatten.toMap ++ rest
}

object ProcessingMeta {
  def parseJson[F[_]: MonadThrow](json: Option[Json]): F[Option[ProcessingMeta]] =
    if (json.exists(_.isObject))
      json.traverse(_.decodeAsF[F, ProcessingMeta])
    else json.flatMap(_.asString).traverse(_.decodeAsF[F, ProcessingMeta])

  def apply(meta: Map[String, String]): ProcessingMeta = new ProcessingMeta(
    meta.get("source"),
    meta.get("maxar_product"),
    meta.get("source-app"),
    meta - "source" - "maxar_product" - "source-app",
  )
  implicit val encoder: Encoder[ProcessingMeta] = deriveEncoder[ProcessingMeta]

  implicit val decoder: Decoder[ProcessingMeta] = (c: HCursor) =>
    for {
      source <- c.downField("source").as[Option[String]]
      maxarProduct <- c.downField("maxarProduct").as[Option[String]]
      sourceApp <- c.downField("sourceApp").as[Option[String]]
      restMap <- c.downField("rest").as[Option[Map[String, String]]]
      restFields <- (c.keys.getOrElse(Set.empty).toSet -- Set(
        "source",
        "maxarProduct",
        "sourceApp",
        "rest",
      ))
        .toList
        .traverse(key => c.downField(key).as[String].map(key -> _))
    } yield ProcessingMeta(
      source,
      maxarProduct,
      sourceApp,
      restMap.getOrElse(Map.empty) ++ restFields.toMap,
    )
}
