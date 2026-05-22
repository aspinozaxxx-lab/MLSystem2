package io.geoalert.rastertileserver.rest.model

import akka.http.scaladsl.marshallers.sprayjson.SprayJsonSupport
import spray.json.{DefaultJsonProtocol, RootJsonFormat}

case class TileJson(tilejson: String,
                    bounds: Seq[Double],
                    center: Seq[Double],
                    name: String,
                    minzoom: Int,
                    maxzoom: Int,
                    tiles: Seq[String])

trait TileJsonSupport extends SprayJsonSupport with DefaultJsonProtocol {
  implicit val tileJsonFormat: RootJsonFormat[TileJson] = jsonFormat7(TileJson)
}
