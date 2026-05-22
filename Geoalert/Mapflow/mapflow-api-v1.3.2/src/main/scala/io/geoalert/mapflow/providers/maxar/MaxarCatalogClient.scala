package io.geoalert.mapflow.providers.maxar

import akka.http.scaladsl.model.HttpResponse
import cats.effect.IO

import io.geoalert.mapflow.TestEnvConfig

trait MaxarCatalogClient {
  def getMaxarMetaOld(
      url: String,
      body: Option[String],
      maxarLogin: String,
      maxarPassword: ,
    ): IO[HttpResponse]

  def searchMeta(
      username: String,
      password: ,
      token: ,
      input: MaxarCatalogRequest,
    ): IO[List[MaxarFeature]]

  def getDiscoveryApiMetadata(legacyId: String): IO[Option[DiscoveryApiMetadata]]
}

object MaxarCatalogClient extends TestEnvConfig {
  lazy val instance: MaxarCatalogClient =
    if (testEnv) new MockMaxarMetaClient() else new ProductionMaxarCatalogClient()
  def apply(): MaxarCatalogClient = instance
}
