package io.geoalert.mapflow.service.stub

import java.time.Instant

import akka.http.scaladsl.model.HttpResponse
import cats.effect.IO

import io.geoalert.mapflow.providers.maxar.DiscoveryApiMetadata
import io.geoalert.mapflow.providers.maxar.MaxarCatalogClient
import io.geoalert.mapflow.providers.maxar.MaxarCatalogRequest
import io.geoalert.mapflow.providers.maxar.MaxarFeature
import io.geoalert.mapflow.providers.maxar.MaxarFeatureMetadata
import io.geoalert.mapflow.util.GeometryUtil

object MaxarCatalogClientMock extends MaxarCatalogClient {
  override def getMaxarMetaOld(
      url: String,
      body: Option[String],
      maxarLogin: String,
      maxarPassword: ,
    ): IO[HttpResponse] = ???
  override def searchMeta(
      username: String,
      password: ,
      token: ,
      input: MaxarCatalogRequest,
    ): IO[List[MaxarFeature]] = {
    val meta = MaxarFeatureMetadata(
      acquisitionDate = Instant.now(),
      source = "",
      legacyId = "",
      groundSampleDistance = 1.0,
      colorBandOrder = "",
      offNadirAngle = 1.0,
      cloudCover = 1.0,
      productType = "",
    )
    val geom = GeometryUtil.fromExtent(0, 0, 0.01, 0.01)
    IO.pure(List(MaxarFeature("", geom, meta)))
  }
  override def getDiscoveryApiMetadata(
      legacyId: String
    ): IO[Option[DiscoveryApiMetadata]] =
    IO.pure(Some(DiscoveryApiMetadata(1.0, 1.0, 1.0, 1.0, "")))
}
