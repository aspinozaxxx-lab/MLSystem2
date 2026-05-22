package io.geoalert.mapflow.rest

import akka.http.scaladsl.server.Directives
import akka.http.scaladsl.server.Route
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import io.circe.generic.auto.exportDecoder

import io.geoalert.mapflow.model.GetMaxarMetaInput
import io.geoalert.mapflow.model.GetSkyWatchMetaInput
import io.geoalert.mapflow.model.PngLinkInput
import io.geoalert.mapflow.rest.controller.SkyWatchController
import io.geoalert.mapflow.service.Services

object MetaResource extends Directives with Authorization with RestImplicits with Services {
  private val skyWatchController: SkyWatchController = SkyWatchController()

  def searchDefaultMeta: Route =
    (path("meta") & post & authorized & entity(as[GetMaxarMetaInput])) { (user, input) =>
      toComplete(maxarService.requestImageryFromCatalogOld(input)(user))
    }

  def searchMaxarMeta: Route =
    (path("meta" / "maxar") & post & authorized & entity(as[GetMaxarMetaInput])) { (user, input) =>
      toComplete(maxarService.requestImageryFromCatalogOld(input)(user))
    }

  def searchSkywatchMeta: Route =
    (path("meta" / "skywatch" / "id") & post & authorized & entity(as[GetSkyWatchMetaInput])) {
      (user, input) =>
        complete(skyWatchController.getSkyWatchAnswerId(input)(user))
    }

  def searchSkywatchMetaPage: Route = (path(
    "meta" / "skywatch" / "page"
  ) & get & authorized & parameters(Symbol("id").as[String], Symbol("cursor").?)) {
    (user, id, cursor) =>
      complete(skyWatchController.getSkyWatchMetaPage(id, cursor)(user))
  }

  def maxarProxy: Route = (path("png") & get & authorized & parameters(
    Symbol("TileRow").as[Int],
    Symbol("TileCol").as[Int],
    Symbol("TileMatrix").as[Int],
    Symbol("CONNECTID").as[String],
    Symbol("CQL_FILTER").?,
  )) { (user, tileRow, tileColumn, tileMatrix, connectIdType, cqlFilter) =>
    val input = PngLinkInput(tileRow, tileColumn, tileMatrix, connectIdType, cqlFilter)
    toComplete(maxarService.proxySingleTile(input)(user))
  }

  val routes: Route = concat(
    searchDefaultMeta,
    searchSkywatchMeta,
    searchSkywatchMetaPage,
    searchMaxarMeta,
    maxarProxy,
  )
}
