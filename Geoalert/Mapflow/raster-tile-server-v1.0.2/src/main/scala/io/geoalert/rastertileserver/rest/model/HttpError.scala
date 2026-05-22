package io.geoalert.rastertileserver.rest.model

import akka.http.scaladsl.marshallers.sprayjson.SprayJsonSupport
import spray.json.{DefaultJsonProtocol, RootJsonFormat}

case class HttpError(code: String, message: String)

trait HttpErrorSupport extends SprayJsonSupport with DefaultJsonProtocol {
  implicit val httpErrorFormat: RootJsonFormat[HttpError] = jsonFormat2(HttpError)
}

