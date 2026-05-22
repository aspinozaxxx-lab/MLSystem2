package io.geoalert.rastertileserver.cog

import java.net.URI
import java.net.URLDecoder
import java.nio.charset.StandardCharsets

import com.typesafe.scalalogging.LazyLogging
import geotrellis.proj4.WebMercator
import geotrellis.raster.RasterSource
import geotrellis.raster.gdal.GDALRasterSource
import geotrellis.raster.gdal.config.GDALOptionsConfig
import geotrellis.raster.geotiff.GeoTiffRasterSource

import io.geoalert.rastertileserver.Config
import io.geoalert.rastertileserver.exception.BadRequest

object CogRasterSource extends LazyLogging {
  def apply(uri: URI): RasterSource = {
    val uriString = uri.toASCIIString

    try {
      val rasterSource = if (Config.useGdalLibrary) {
        val path = URLDecoder
          .decode(uriString, StandardCharsets.UTF_8)
          .replace("s3://", "/vsis3/")
        GDALRasterSource(path)
      }
      else
        GeoTiffRasterSource(uriString)

      rasterSource.crs

      if (rasterSource.crs != WebMercator)
        throw BadRequest(s"GeoTIFF projection EPSG:3857 is required, but ${rasterSource.crs} found")

      rasterSource
    }
    catch {
      case e: Exception =>
        logger.error(s"Error reading raster from $uriString", e)
        GDALOptionsConfig.setRegistryOptions
        logger.info(s"Set GDAL options configuration")
        throw e
    }

  }
}
