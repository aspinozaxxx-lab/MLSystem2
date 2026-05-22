package io.geoalert.mapflow.service

import java.util.UUID

import scala.collection.immutable.ArraySeq

import cats.data.EitherT
import com.typesafe.scalalogging.LazyLogging
import doobie.ConnectionIO
import doobie.implicits._
import sangria.schema.Args

import io.geoalert.mapflow.Config._
import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.graphql.schema.schema._
import io.geoalert.mapflow.model.Permission.ViewAnyProject
import io.geoalert.mapflow.model.TileJson
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.repo.AoiRepo

import geotrellis.layer.ZoomedLayoutScheme
import geotrellis.proj4.LatLng
import geotrellis.proj4.WebMercator
import geotrellis.raster.GridBounds
import geotrellis.vector.Extent

class LayerService extends LazyLogging {
  private val layoutScheme = ZoomedLayoutScheme(WebMercator, 256)

  def getGeojsonLayer(args: Args)(user: User): ConnectionIO[String] = {
    val (processingId, bbox, xRes, yRes) =
      (args arg ProcessingIdArg, args arg BboxArg, args arg XResArg, args arg YResArg)

    val io = for {
      processingId <- EitherT(
        Validations.processingExists(processingId, user.userFilter(ViewAnyProject))
      )
      geojson <- EitherT.right[ApplicationError](
        AoiRepo.getGeojsonLayer(processingId, bbox, xRes, yRes)
      )
    } yield geojson

    io.rethrowT
  }

  def getMvtLayer(
      processingId: UUID,
      x: Int,
      y: Int,
      z: Int,
    ): ConnectionIO[Array[Byte]] = {
    val bbox = layoutScheme
      .levelForZoom(z)
      .layout
      .layoutForBounds(GridBounds(x, y, x, y))
      .extent
      .reproject(WebMercator, LatLng)

    val io = for {
      processingId <- EitherT(Validations.processingExists(processingId, None))
      mvt <- EitherT.right[ApplicationError](AoiRepo.getMvtLayer(processingId, bbox))
    } yield mvt

    io.rethrowT
  }

  def getTileJson(processingId: UUID): ConnectionIO[TileJson] = {
    def buildTileJson(extent: Extent) = {
      val tilesUrls =
        Array[String](s"$externalUrl/tiles/aoi/{z}/{x}/{y}.vector.pbf?processingId=$processingId")

      val vectorLayer = Map[String, String](
        "id" -> "aoi",
        "description" -> "",
        "minzoom" -> "0",
        "maxzoom" -> "15",
      )

      TileJson(
        bounds = Seq(extent.xmin, extent.ymin, extent.xmax, extent.ymax),
        center = Seq(extent.center.getX, extent.center.getY),
        name = "aoi",
        minzoom = 0,
        maxzoom = 15,
        tiles = ArraySeq.unsafeWrapArray(tilesUrls),
        vector_layers = Seq(vectorLayer),
      )
    }

    val io = for {
      processingId <- EitherT(Validations.processingExists(processingId, None))
      summary <- EitherT.right[ApplicationError](
        AoiRepo.getAoiSummariesByProcessings(List(processingId))
      )
    } yield buildTileJson(summary(processingId).bbox)

    io.rethrowT
  }
}

object LayerService {
  def apply(): LayerService = new LayerService()
}
