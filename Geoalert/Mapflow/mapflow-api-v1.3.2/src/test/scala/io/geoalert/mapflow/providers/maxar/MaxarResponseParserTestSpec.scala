package io.geoalert.mapflow.providers.maxar

import geotrellis.proj4.LatLng
import geotrellis.vector.Projected
import org.apache.commons.codec.Charsets
import io.geoalert.mapflow.implicits.GeometryOps._
import org.scalatest.{FunSpec, Matchers}

import java.time.Instant

class MaxarResponseParserTestSpec extends FunSpec with Matchers {
  describe("Maxar Catalog API response parser") {
    it("should parse actual Maxar response") {
      val json = new String(getClass.getResourceAsStream("maxar_meta_response.json").readAllBytes(), Charsets.UTF_8)

      val result = SearchMetaResponseParser.parseSearchMetaResponse(json)

      result.size should be(23)
      val img = result.find(_.id == "5ef5e102c195dc8d8e44191e01647721").get.toImageJson
      img.id should be("5ef5e102c195dc8d8e44191e01647721")
      Projected(img.footprint, LatLng.epsgCode.get).areaInMeters() should be(274_463_559)
      img.sensor should be("GE01")
      img.productType should be("Pan Sharpened Natural Color")
      img.offNadirAngle should be(24.135431)
      img.acquisitionDate should be(Instant.parse("2022-09-10T15:42:12Z"))
      img.cloudCover should be(0.0012740794)
      img.colorBandOrder should be("RGB")
      img.pixelResolution should be(0.48)
    }
  }
}
