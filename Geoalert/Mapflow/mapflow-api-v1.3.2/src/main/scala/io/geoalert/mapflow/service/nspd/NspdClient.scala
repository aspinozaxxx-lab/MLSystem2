package io.geoalert.mapflow.service.nspd

import cats.effect.IO

import io.geoalert.mapflow.Config.testEnv
import io.geoalert.mapflow.service.nspd.response.Datasource
import io.geoalert.mapflow.service.nspd.response.Layer
import io.geoalert.mapflow.service.nspd.response.MapConfig

trait NspdClient {
  def getMapfile(url: String): IO[MapConfig]
}

object NspdClient {
  def make: NspdClient =
    if (testEnv)
      new NoOpNspdClient
    else
      new LiveNspdClient

  private class NoOpNspdClient extends NspdClient {
    override def getMapfile(url: String): IO[MapConfig] =
      IO(
        MapConfig(
          Layer(
            name = "Test",
            srs = "+init=epsg:3857",
            styleName = "style",
            datasource = Datasource(`type` = "gdal", file = "/vrt/example/uri"),
          )
        )
      )
  }
}
