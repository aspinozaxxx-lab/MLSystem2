package io.geoalert.vectortileserver

import akka.http.scaladsl.marshallers.sprayjson.SprayJsonSupport
import spray.json.{DefaultJsonProtocol, RootJsonFormat}

case class TileJson(bounds: Seq[Double],
                    center: Seq[Double],
                    name: String,
                    minzoom: Int,
                    maxzoom: Int,
                    tiles: Seq[String],
                    vector_layers: Seq[Map[String, String]])

trait TileJsonSupport extends SprayJsonSupport with DefaultJsonProtocol {
  implicit val tileJsonFormat: RootJsonFormat[TileJson] = jsonFormat7(TileJson)
}