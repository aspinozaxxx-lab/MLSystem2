package io.geoalert.mapflow.rest.controller

import akka.http.scaladsl.model.HttpResponse
import akka.http.scaladsl.model.StatusCodes
import cats.effect.IO

import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.exception.AccessDenied
import io.geoalert.mapflow.model.PngLinkInput
import io.geoalert.mapflow.providers.maxar.MaxarCatalogClient
import io.geoalert.mapflow.providers.maxar.MaxarTilesProxy
import io.geoalert.mapflow.service.MaxarService
import io.geoalert.mapflow.service.Services
import io.geoalert.mapflow.util.UserStub

class MaxarControllerSpec extends DbIntegrationTest with UserStub with Services {
  val service: MaxarService = new MaxarService(
    MaxarCatalogClient(),
    new MaxarTilesProxy() {
      override def proxySingleTile(
          url: String,
          username: String,
          password: ,
        ): IO[HttpResponse] =
        url match {
          case "https://securewatch.maxar.com/earthservice/wmtsaccess?SERVICE=WMTS&VERSION=1.0.0&STYLE=&REQUEST=GetTile&LAYER=DigitalGlobe:ImageryTileService&FORMAT=image/png&TileRow=10757&TileCol=17456&TileMatrixSet=EPSG:3857&TileMatrix=EPSG:3857:15&CONNECTID=39a1c1fc-ef1f-4e6f-8aa1-0ebe6522296d&CQL_FILTER=feature_id=%27062925f14ada50ab5e3e577375afbc7e%27" =>
            IO.pure(HttpResponse())
          case "https://securewatch.maxar.com/earthservice/wmtsaccess?SERVICE=WMTS&VERSION=1.0.0&STYLE=&REQUEST=GetTile&LAYER=DigitalGlobe:ImageryTileService&FORMAT=image/png&TileRow=10757&TileCol=17456&TileMatrixSet=EPSG:3857&TileMatrix=EPSG:3857:10&CONNECTID=39a1c1fc-ef1f-4e6f-8aa1-0ebe6522296d&CQL_FILTER=feature_id=%27062925f14ada50ab5e3e577375afbc7e%27" =>
            IO.pure(HttpResponse())
          case "https://securewatch.maxar.com/earthservice/wmtsaccess?SERVICE=WMTS&VERSION=1.0.0&STYLE=&REQUEST=GetTile&LAYER=DigitalGlobe:ImageryTileService&FORMAT=image/png&TileRow=10757&TileCol=17456&TileMatrixSet=EPSG:3857&TileMatrix=EPSG:3857:15&CONNECTID=SOME_CONNECT_ID&CQL_FILTER=feature_id=%27062925f14ada50ab5e3e577375afbc7e%27" =>
            IO.pure(HttpResponse())
          case "https://securewatch.maxar.com/earthservice/wmtsaccess?SERVICE=WMTS&VERSION=1.0.0&STYLE=&REQUEST=GetTile&LAYER=DigitalGlobe:ImageryTileService&FORMAT=image/png&TileRow=10757&TileCol=17456&TileMatrixSet=EPSG:3857&TileMatrix=EPSG:3857:10&CONNECTID=SOME_CONNECT_ID&CQL_FILTER=feature_id=%27062925f14ada50ab5e3e577375afbc7e%27" =>
            IO.pure(HttpResponse())
          case _ => fail(s"Unexpected url: $url")
        }
    },
    dataProviderService,
  ) {
    override val zoomConstraint: Int = 12
  }

  describe("Maxar controller") {
    it("Should not limit zoom for administrators") {
      val response = service
        .proxySingleTile(
          PngLinkInput(
            10757,
            17456,
            15,
            "securewatch",
            Some("feature_id=%27062925f14ada50ab5e3e577375afbc7e%27"),
          )
        )(admin)
        .unsafeRunSync()

      response.status should be(StatusCodes.OK)
    }

    it("Should restrict zoom for premium users") {
      assertThrows[AccessDenied](
        service
          .proxySingleTile(
            PngLinkInput(
              10757,
              17456,
              15,
              "securewatch",
              Some("feature_id=%27062925f14ada50ab5e3e577375afbc7e%27"),
            )
          )(premiumUser)
          .unsafeRunSync()
      )
    }

    it("Should not restrict zoom less then 12 for premium users") {
      val response = service
        .proxySingleTile(
          PngLinkInput(
            10757,
            17456,
            10,
            "securewatch",
            Some("feature_id=%27062925f14ada50ab5e3e577375afbc7e%27"),
          )
        )(premiumUser)
        .unsafeRunSync()

      response.status should be(StatusCodes.OK)
    }

    it("Should restrict zoom for regular users") {
      assertThrows[AccessDenied](
        service.proxySingleTile(
          PngLinkInput(
            10757,
            17456,
            15,
            "securewatch",
            Some("feature_id=%27062925f14ada50ab5e3e577375afbc7e%27"),
          )
        )(regularUser)
      )
    }

    it("Should not restrict zoom less then 12 for regular users") {
      val response = service
        .proxySingleTile(
          PngLinkInput(
            10757,
            17456,
            10,
            "securewatch",
            Some("feature_id=%27062925f14ada50ab5e3e577375afbc7e%27"),
          )
        )(regularUser)
        .unsafeRunSync()

      response.status should be(StatusCodes.OK)
    }

    it("should parse maxar url") {
      val id = service.parseMaxarUrl(
        "https://securewatch.digitalglobe.com/earthservice/wmtsaccess?SERVICE=WMTS&VERSION=1.0.0&STYLE=&REQUEST=GetTile&LAYER=DigitalGlobe:ImageryTileService&FORMAT=image/jpeg&TileRow={y}&TileCol={x}&TileMatrixSet=EPSG:3857&TileMatrix=EPSG:3857:{z}&CQL_FILTER=feature_id='056daf9ff49edf65e98dd7f9ad6a98e3'&CONNECTID=b05c51d9-7428-48c1-9c69-d8f57618d9d4"
      )
      id should be(Some("056daf9ff49edf65e98dd7f9ad6a98e3"))
    }
  }
}
