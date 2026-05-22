package io.geoalert.mapflow.rest

import akka.http.scaladsl.model.ContentTypes
import akka.http.scaladsl.model.HttpEntity
import akka.http.scaladsl.model.headers.HttpOriginRange
import akka.http.scaladsl.model.headers.`Access-Control-Allow-Origin`
import akka.http.scaladsl.model.headers.`Content-Type`
import akka.http.scaladsl.server.Directives
import akka.http.scaladsl.server.Route

import io.geoalert.mapflow.Config.keycloakOIDCClientId
import io.geoalert.mapflow.Config.keycloakRealm
import io.geoalert.mapflow.Config.keycloakUrl

object ConfigurationResource extends Directives with Authorization with RestImplicits {
  val config: String =
    s"""
      |{
      |  "realm": "$keycloakRealm",
      |  "auth-server-url": "$keycloakUrl",
      |  "ssl-required": "external",
      |  "resource": "$keycloakOIDCClientId",
      |  "public-client": true,
      |  "confidential-port": 0
      |}
      |""".stripMargin

  def getConfiguration: Route = (path("config" / "keycloak.json") & get) {
    respondWithHeaders(
      Seq(
        `Content-Type`(ContentTypes.`application/json`),
        `Access-Control-Allow-Origin`(HttpOriginRange.*),
      )
    ) {
      complete(HttpEntity(ContentTypes.`application/json`, config))
    }
  }

  val routes: Route = concat(
    getConfiguration
  )
}
