package io.geoalert.mapflow.providers.maxar

import java.time.Instant
import java.util.UUID

import akka.http.scaladsl.model.HttpResponse
import akka.http.scaladsl.model.StatusCodes
import cats.effect.IO

import geotrellis.vector.Extent

class MockMaxarMetaClient extends MaxarCatalogClient {
  override def getMaxarMetaOld(
      url: String,
      body: Option[String],
      maxarLogin: String,
      maxarPassword: ,
    ): IO[HttpResponse] =
    IO.pure(HttpResponse(StatusCodes.Forbidden))

  override def searchMeta(
      username: String,
      password: ,
      token: ,
      input: MaxarCatalogRequest,
    ): IO[List[MaxarFeature]] =
    IO.pure(
      List(
        MaxarFeature(
          UUID.randomUUID().toString,
          Extent(0, 0, 10, 10).toPolygon(),
          MaxarFeatureMetadata(
            Instant.parse("2020-01-01T01:02:03Z"),
            "WV01",
            "Panchromatic",
            0.1,
            27.5,
            "RGB",
            0.5,
            "1234567890",
          ),
        )
      )
    )

  override def getDiscoveryApiMetadata(legacyId: String): IO[Option[DiscoveryApiMetadata]] =
    IO.pure(None)
}
