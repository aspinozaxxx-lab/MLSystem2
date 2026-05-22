package io.geoalert.mapflow.service

import doobie.free.connection.ConnectionIO

import io.geoalert.mapflow.providers.head.HeadCatalogClient
import io.geoalert.mapflow.rest.json.ImageCatalogRequestJson
import io.geoalert.mapflow.rest.json.ImageJson

class HeadService(dataProviderService: DataProviderService) {
  private val headProviderName = "head_imagery"
  private val client: HeadCatalogClient = HeadCatalogClient(headProviderName)
  def searchMeta(input: ImageCatalogRequestJson): ConnectionIO[List[ImageJson]] =
    for {
      dps <- dataProviderService.listDataProvidersByName(headProviderName)
      dp = dps.head
      images <- client
        .searchMeta(dp.credentialsUsername.get, dp.credentialsPassword.get, input)
        .to[ConnectionIO]
    } yield images
}

object HeadService {
  def apply(dataProviderService: DataProviderService): HeadService = new HeadService(
    dataProviderService
  )
}
