package io.geoalert.mapflow.service

import java.util.UUID

import cats.syntax.option._
import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.model.DataProvider
import io.geoalert.mapflow.model.ProcessingParams

class WorkflowEngineServiceSpec extends DbIntegrationTest {
  describe("utility methods") {
    it("should set token in the URL") {
      val params =
        ProcessingParams(Map("url" -> "http://example.com/{z}/{x}/{y}.png?token=}"))
      val provider = DataProvider(
        UUID.randomUUID(),
        "mapbox",
        "",
        None,
        None,
        0,
        None,
        None,
        "1234567890987654321".some,
        isDefault = true,
        None,
      )
      val map = WorkflowEngineService.prepareProcessingParams(params, provider.some)

      map should be(Map("url" -> "http://example.com/{z}/{x}/{y}.png?token=
    }

    it("should ignore params without url ") {
      val params = ProcessingParams(Map("source" -> "local"))
      val provider = DataProvider(
        UUID.randomUUID(),
        "mapbox",
        "",
        None,
        None,
        0,
        None,
        None,
        "1234567890987654321".some,
        isDefault = true,
        None,
      )
      val map = WorkflowEngineService.prepareProcessingParams(params, provider.some)

      map should be(Map("source" -> "local"))
    }

    it("should ignore url without token placeholder ") {
      val params = ProcessingParams(Map("url" -> "s3://minio/test.tif"))
      val provider = DataProvider(
        UUID.randomUUID(),
        "mapbox",
        "",
        None,
        None,
        0,
        None,
        None,
        "1234567890987654321".some,
        isDefault = true,
        None,
      )
      val map = WorkflowEngineService.prepareProcessingParams(params, provider.some)

      map should be(Map("url" -> "s3://minio/test.tif"))
    }

    it("should not rely on Data Provider") {
      val params =
        ProcessingParams(Map("url" -> "http://example.com/{z}/{x}/{y}.png?token=}"))
      val map = WorkflowEngineService.prepareProcessingParams(params, None)

      map should be(Map("url" -> "http://example.com/{z}/{x}/{y}.png?token=}"))
    }
    it("should append HEAD parameters") {
      val params = ProcessingParams(Map())
      val map = WorkflowEngineService.prepareProcessingParams(
        params,
        DataProvider(
          UUID.randomUUID(),
          "HEAD_2021",
          "HEAD (2021)",
          none,
          none,
          0,
          "user".some,
          "pass".some,
          none,
          isDefault = false,
          None,
        ).some,
      )

      map should be(
        Map(
          "url" -> "https://app.mapflow.ai/tiles/satimagery/{z}/{x}/{y}.png?year=2022",
          "header" -> "{\"x-api-key\":\"7c57e21577ebea7e212970ee7ad9dab3408f9418\"}",
          "source_type" -> "tms",
          "connection_limit" -> "4",
        )
      )
    }
  }
}
