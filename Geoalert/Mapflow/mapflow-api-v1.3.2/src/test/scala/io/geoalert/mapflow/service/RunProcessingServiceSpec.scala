package io.geoalert.mapflow.service

import cats.implicits.catsSyntaxOptionId
import io.geoalert.mapflow.model.{BlockConfig, DataProvider, DataSource, ProcessingParams, SourceType, WorkflowDefSummary}
import io.geoalert.mapflow.service.RunProcessingService
import org.scalatest.{FunSpec, Matchers}

import java.util.UUID

class RunProcessingServiceSpec extends FunSpec with Matchers with Services {
  describe("RunProcessingService") {
    it("Should not replace url in geotiff provider params") {
      val params = ProcessingParams(Map("url" -> "s3://test/test.tiff", "source_type" -> "local", "data_provider" -> "GTIFF"))
      val provider = DataProvider(
        UUID.fromString("c433e83a-8da4-4a44-8257-291dbd0c2da3"),
        "GTIFF",
        "GeoTIFF",
        Some(""),
        None,
        0.0,
        None,
        None,
        None,
        isDefault = true,
        None)
      val actual = RunProcessingService.overrideDataProviderUrl(params, provider.some)
      val expected = ProcessingParams(Map("url" -> "s3://test/test.tiff", "source_type" -> "local", "data_provider" -> "GTIFF"))

      actual shouldBe expected
    }

    it("Should  replace url with providers") {
      val params = ProcessingParams(Map("url" -> "some.wrong.url", "source_type" -> "xyz", "data_provider" -> "MyProvider"))
      val provider = DataProvider(
        UUID.fromString("c433e83a-8da4-4a44-8257-291dbd0c2da3"),
        "MyProvider",
        "MyProvider",
        Some("http://proper.link/{x}/{y}/{z}"),
        None,
        0.0,
        None,
        None,
        None,
        isDefault = true,
        None)
      val actual = RunProcessingService.overrideDataProviderUrl(params, provider.some)
      val expected = ProcessingParams(Map("url" -> "http://proper.link/{x}/{y}/{z}", "source_type" -> "xyz", "data_provider" -> "MyProvider"))

      actual shouldBe expected
    }
  }
}
