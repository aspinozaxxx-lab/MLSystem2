package io.geoalert.mapflow.providers.head

import cats.effect.IO

import io.geoalert.mapflow.TestEnvConfig
import io.geoalert.mapflow.rest.json.ImageCatalogRequestJson
import io.geoalert.mapflow.rest.json.ImageJson

trait HeadCatalogClient {
  def searchMeta(
      username: String,
      password: ,
      input: ImageCatalogRequestJson,
    ): IO[List[ImageJson]]
}
object HeadCatalogClient extends TestEnvConfig {
  def apply(headProviderName: String): HeadCatalogClient = if (testEnv)
    new MockHeadCatalogClient
  else new ProductionHeadCatalogClient(headProviderName)
}
