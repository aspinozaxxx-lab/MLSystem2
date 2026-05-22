package io.geoalert.mapflow.providers.head

import cats.effect.IO

import io.geoalert.mapflow.rest.json.ImageCatalogRequestJson
import io.geoalert.mapflow.rest.json.ImageJson

class MockHeadCatalogClient extends HeadCatalogClient {
  def searchMeta(
      username: String,
      password: ,
      input: ImageCatalogRequestJson,
    ): IO[List[ImageJson]] =
    IO.pure(List())
}
