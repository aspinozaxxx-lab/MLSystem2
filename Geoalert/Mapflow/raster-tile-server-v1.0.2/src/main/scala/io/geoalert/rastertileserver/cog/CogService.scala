package io.geoalert.rastertileserver.cog

import com.github.blemale.scaffeine.{Cache, Scaffeine}
import geotrellis.layer.{SpatialKey, ZoomedLayoutScheme}
import geotrellis.proj4.{LatLng, WebMercator}
import geotrellis.raster.gdal.config.GDALOptionsConfig
import geotrellis.raster.{CellSize, MultibandTile, Raster, RasterSource}
import geotrellis.vector.io.json.JsonFeatureCollection
import geotrellis.vector.{Geometry, Projected, ProjectedExtent}
import io.geoalert.rastertileserver.Config.{cplVsilCurlCacheSize, externalUrl, gdalThreadPoolSize, maxSourcesPerTile, minioAccessKey, minioHost, minioPort, minioSecretKey, useGdalLibrary, vsiCacheSize}
import io.geoalert.rastertileserver.exception.NotFound
import io.geoalert.rastertileserver.rest.model.TileJson
import io.geoalert.rastertileserver.s3.{AttributeStoreCache, MinioS3Client}
import io.minio.errors.ErrorResponseException
import software.amazon.awssdk.services.s3.model.NoSuchKeyException

import java.net.URI
import java.util.concurrent.Executors
import scala.collection.parallel.CollectionConverters._
import scala.concurrent.{ExecutionContext, ExecutionContextExecutorService, Future, blocking}
import scala.concurrent.duration._

class CogService extends geotrellis.vector.io.json.Implicits {
  private val layoutScheme = ZoomedLayoutScheme(WebMercator)

  val maskGeometryCache: Cache[String, List[Geometry]] = Scaffeine()
    .expireAfterWrite(10.minutes)
    .maximumSize(1000)
    .build()

  val rasterSourceCache: ThreadLocal[Cache[URI, RasterSource]] = ThreadLocal.withInitial(() => Scaffeine()
    .expireAfterWrite(10.minutes)
    .maximumSize(1000)
    .build())

  if (useGdalLibrary) {
    val minioUri = if (minioPort == ""){minioHost} else {s"$minioHost:$minioPort"}
    GDALOptionsConfig.registerOption("AWS_HTTPS", "NO")
    GDALOptionsConfig.registerOption("AWS_VIRTUAL_HOSTING", "FALSE")
    GDALOptionsConfig.registerOption("AWS_S3_ENDPOINT", minioUri)
    GDALOptionsConfig.registerOption("AWS_ACCESS_KEY_ID", minioAccessKey)
    GDALOptionsConfig.registerOption("AWS_SECRET_ACCESS_KEY", minioSecretKey)
    GDALOptionsConfig.registerOption("VSI_CACHE_SIZE", vsiCacheSize)
    GDALOptionsConfig.registerOption("CPL_VSIL_CURL_CACHE_SIZE", cplVsilCurlCacheSize)
    GDALOptionsConfig.setRegistryOptions
  }

  implicit val blockingExecutionContext: ExecutionContextExecutorService =
    ExecutionContext.fromExecutorService(Executors.newFixedThreadPool(gdalThreadPoolSize))

  def getTile(x: Int, y: Int, zoom: Int, uri: URI, maskUri: Option[String]): Future[Option[Array[Byte]]] =
    Future(getTileSync(x, y, zoom, uri, maskUri))(blockingExecutionContext)

  def getTileSync(x: Int, y: Int, zoom: Int, uri: URI, maskUri: Option[String]): Option[Array[Byte]] = {
    val layout = layoutScheme.levelForZoom(zoom).layout
    val keyExtent = layout.mapTransform(SpatialKey(x, y))
    val gridExtent = layout.createAlignedGridExtent(keyExtent)

    def getGeometryFromS3(uri: String): List[Geometry] = {
      try {
        val geoJsonString: String = MinioS3Client.getObject(uri)

        val fc = geoJsonString.parseGeoJson[JsonFeatureCollection]()
        fc.getAllGeometries().toList.map(geom => {
          Projected(geom, LatLng.epsgCode.get)
            .reproject(LatLng, WebMercator)(WebMercator.epsgCode.get)
        })
      } catch {
        case _@(_: NoSuchKeyException | _: ErrorResponseException) =>
          throw NotFound(s"Resource not found: $uri")
      }
    }

    def getTileFromSingleFile(uri: URI): Option[Raster[MultibandTile]] =
      rasterSourceCache.get().get(uri, uri => CogRasterSource(uri))
        .resampleToRegion(gridExtent)
        .read(keyExtent)

    def getTileFromMultipleFiles(uri: URI): Option[Raster[MultibandTile]] =
      AttributeStoreCache.get(uri)
        .query(uri.toString, ProjectedExtent(keyExtent, layoutScheme.crs))
        .to(LazyList)
        .take(maxSourcesPerTile)
        .par
        .map(_.uri)
        .flatMap(getTileFromSingleFile)
        .reduceOption(_ merge _)

    try {
      val tileOpt = if (uri.getPath.endsWith(".tif")) {
        getTileFromSingleFile(uri)
      } else {
        getTileFromMultipleFiles(uri)
      }

      val geometryOpt = maskUri.map(maskGeometryCache.get(_, getGeometryFromS3))

      tileOpt.map(tile => {
        geometryOpt match {
          case Some(geoms) =>
            val t = tile.mask(geoms)
            t
          case None => tile
        }
      }).map(_.tile.renderPng())
    } catch {
      case _: NoSuchKeyException =>
        throw NotFound(s"Resource not found: $uri")
    }
  }

  def getTileJson(uri: URI): TileJson = blocking {
    def getMaxZoom(rs: RasterSource) = {
      val cellSize = CellSize(rs.extent, rs.cols.toInt, rs.rows.toInt)
      ZoomedLayoutScheme(rs.crs).levelFor(rs.extent, cellSize).zoom
    }

    try {
      val (extent, maxZoom) = if (uri.getPath.endsWith(".tif")) {
        val source = CogRasterSource(uri)

        val extentLatLng = source.extent.reproject(source.crs, LatLng)
        (extentLatLng, getMaxZoom(source))
      } else {
        val attributeStore = AttributeStoreCache.get(uri)
        val index = attributeStore.query(uri.toString)
        if (index.isEmpty) {
          throw NotFound(s"Resource $uri not found")
        }
        val extentLatLng = index
          .map(md => md.extent.reproject(md.crs, LatLng))
          .reduce(_ combine _)
        val rasterSourceUri = index.head.uri
        (extentLatLng, getMaxZoom(CogRasterSource(rasterSourceUri)))
      }

      val tilesUrls = Seq(s"$externalUrl/api/v0/cogs/tiles/{z}/{x}/{y}.png?uri=$uri")

      TileJson(
        tilejson = "2.2.0", // see https://github.com/mapbox/tilejson-spec
        bounds = Seq(extent.xmin, extent.ymin, extent.xmax, extent.ymax),
        center = Seq(extent.center.getX, extent.center.getY),
        name = uri.toString,
        minzoom = 0,
        maxzoom = maxZoom,
        tiles = tilesUrls
      )
    } catch {
      case _: NoSuchKeyException =>
        throw NotFound(s"Resource not found: $uri")
    }
  }

  def getBounds(uri: URI): String = blocking {
    val extent = if (uri.getPath.endsWith(".tif")) {
      CogRasterSource(uri).extent
    } else {
      val index = AttributeStoreCache.get(uri)
        .query(uri.toString)

      if (index.isEmpty) {
        throw NotFound(s"Resource $uri not found")
      }

      index.map(md => md.extent)
        .reduce(_ combine _)
    }

    "{\"bounds\"" + s": [${extent.xmin}, ${extent.ymin}, ${extent.xmax}, ${extent.ymax}]}"
  }
}

object CogService {
  def apply(): CogService = new CogService()
}