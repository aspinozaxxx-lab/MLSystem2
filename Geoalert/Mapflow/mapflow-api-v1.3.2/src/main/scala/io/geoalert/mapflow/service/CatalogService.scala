package io.geoalert.mapflow.service

import cats.syntax.option._
import doobie.ConnectionIO

import io.geoalert.mapflow.exception.NotFound
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.providers.maxar.MaxarCatalogRequest
import io.geoalert.mapflow.rest.json.ImageCatalogRequestJson
import io.geoalert.mapflow.rest.json.ImageCatalogResponseJson
import io.geoalert.mapflow.rest.json.ImageJson

class CatalogService(maxarService: MaxarService, headService: HeadService) {
  def searchMeta(
      input: ImageCatalogRequestJson
    )(
      user: User
    ): ConnectionIO[ImageCatalogResponseJson] = {
    val request = MaxarCatalogRequest(
      input.aoi.some,
      input.acquisitionDateFrom,
      input.acquisitionDateTo,
      input.minResolution,
      input.maxResolution,
      input.maxCloudCover,
      input.minOffNadirAngle,
      input.maxOffNadirAngle,
      None,
    )

    for {
//      maxarImages <- maxarService.searchMeta(request)(user)
      headImages <- headService.searchMeta(input)
      // TODO: Other catalogs goes here
    } yield ImageCatalogResponseJson(headImages)
  }

  def fetchMeta(imageId: String)(user: User): ConnectionIO[ImageJson] = {
    val request = MaxarCatalogRequest(
      None,
      None,
      None,
      None,
      None,
      None,
      None,
      None,
      None,
      imageId.some,
    )

    for {
      // TODO: get image metadata from cache
      maxarImages <- maxarService.searchMeta(request)(user)
      // TODO: Other catalogs goes here
    } yield maxarImages.headOption.getOrElse(throw NotFound(imageId))
  }
}

object CatalogService {
  def apply(maxarService: MaxarService, headService: HeadService): CatalogService =
    new CatalogService(maxarService, headService)
}
