package io.geoalert.vectortileserver

import cats.effect.{IO, LiftIO}
import doobie.ConnectionIO
import geotrellis.proj4.{LatLng, WebMercator}
import geotrellis.raster.GridBounds
import org.locationtech.jts.io.WKBReader
import com.typesafe.scalalogging.LazyLogging
import doobie.implicits._
import doobie.postgres.implicits._
import doobie.util.fragment.Fragment.const
import geotrellis.layer.ZoomedLayoutScheme
import geotrellis.vector._

import java.util.UUID

object LayerService extends Config with Db with LazyLogging{
  private val layoutScheme = ZoomedLayoutScheme(WebMercator, 256)

  def getTile(layerId: UUID,
              x: Int, y: Int, z: Int,
              params: TileParams): ConnectionIO[Array[Byte]] = {

    if (z < params.minZoom.getOrElse(defaultMinZoom)) {
      LiftIO[ConnectionIO].liftIO(IO.pure(Array[Byte]()))
    } else {
      val tileExtent = layoutScheme.levelForZoom(z)
        .layout
        .layoutForBounds(GridBounds(x, y, x, y))
        .extent
        .reproject(WebMercator, LatLng)
        .toPolygon()
      val frAreaGeom = geomFromText(tileExtent.toText)

      val (frGeom, frArea) =
        if (params.asPoints.getOrElse(false))
          (fr"st_closestpoint(geometry, st_centroid(geometry))", fr"")
        else if (z > params.simplifyMaxZoom.getOrElse(defaultSimplifyMaxZoom))
          (fr"geometry", fr"")
        else {
          val simplify = params.simplifyFactor.getOrElse(defaultSimplifyFactor) * Math.pow(2, 10.0 - z)
          val area = params.minAreaFactor.getOrElse(defaultMinAreaFactor) * Math.pow(4, 10.0 - z)
          (fr"ST_SimplifyPreserveTopology(geometry, $simplify)",
            fr"and ( st_area(f.geometry) > $area or st_area(f.geometry) = 0 )")
        }

      val sql = fr"select ST_AsMVT(frows, 'vector_layer', 4096, 'geometry') from" ++
        fr"(select f.id, f.attributes, st_asmvtgeom(" ++
        frGeom ++ fr"," ++ frAreaGeom ++ fr", 4096, 256, true) as geometry" ++
        const(s"from $dbSchema.feature f ") ++
        fr"where f.layer_id = $layerId and" ++
        fr"st_intersects(f.geometry," ++ frAreaGeom ++ fr") = true" ++
        frArea ++ fr"limit ${params.maxFeatures.getOrElse(defaultMaxFeatures)}) as frows"
      logger.debug(sql.toString())
      for {
        bin <- sql.query[Option[Array[Byte]]].to[List]
        maybeBin = bin.headOption.flatten.filter(_.nonEmpty)
      } yield maybeBin.getOrElse(Array[Byte]())
    }
  }

  def getTileJson(layerId: UUID,
                  params: TileParams): ConnectionIO[Option[TileJson]] = {
    val tilesUrls = List[String](
      s"$externalUrl/api/layers/$layerId/tiles/{z}/{x}/{y}.vector.pbf${params.toUrlParams()}"
    )

    val sql = const(s"""select st_asbinary(extent) from $dbSchema.""") ++  sql"""layer where id = $layerId"""
    logger.debug(sql.toString())
    val vectorLayer = Seq(
      Some("id" -> "vector_layer"),
      Some("description" -> ""),
      params.minZoom.map(z => "minzoom" -> String.valueOf(z)),
      Some("maxzoom" -> "25")
    ).flatten.toMap

    for {
      bin <- sql.query[Option[Array[Byte]]].to[List]
      maybeBin = bin.headOption.flatten.filter(_.nonEmpty)
      maybeExt = maybeBin.map(b => new WKBReader().read(b).extent)
    } yield maybeExt map { e =>
      TileJson(
        bounds = Seq(e.xmin, e.ymin, e.xmax, e.ymax),
        center = Seq(e.center.x, e.center.y),
        name = layerId.toString,
        minzoom = params.minZoom.getOrElse(defaultMinZoom),
        maxzoom = 25,
        tiles = tilesUrls,
        vector_layers = Seq(vectorLayer)
      )
    }
  }

  def countLayers() : ConnectionIO[Int] = {
    val sql = const(s"SELECT count(*) FROM $dbSchema.layer")
    logger.debug(sql.toString())
    sql.query[Int].unique
  }

  private def geomFromText(geom: String) =
    fr"st_setsrid(st_geomfromtext($geom), 4326)"
}
