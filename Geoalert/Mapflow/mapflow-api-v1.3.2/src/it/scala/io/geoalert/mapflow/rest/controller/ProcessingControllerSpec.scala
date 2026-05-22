package io.geoalert.mapflow.rest.controller

import java.time.Instant
import java.util.UUID
import scala.util.Failure
import scala.util.Try
import cats.syntax.option._
import doobie.implicits._
import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.exception.BadRequest
import io.geoalert.mapflow.exception.Forbidden
import io.geoalert.mapflow.model.DataProvider
import io.geoalert.mapflow.model.ProcessingMeta
import io.geoalert.mapflow.model.ProcessingParams
import io.geoalert.mapflow.model.WorkflowDefSummary
import io.geoalert.mapflow.providers.maxar.MaxarTilesProxy
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.rest.json.CreateAndRunProcessingJson
import io.geoalert.mapflow.service.MaxarService
import io.geoalert.mapflow.service.RunProcessingService
import io.geoalert.mapflow.service.Services
import io.geoalert.mapflow.service.stub.MaxarCatalogClientMock
import io.geoalert.mapflow.util.GeometryUtil
import io.geoalert.mapflow.util.UserStub
import io.geoalert.mapflow.util.UserUtil
import io.geoalert.mapflow.util.WorkflowDefUtil
import org.scalatest.OptionValues

class ProcessingControllerSpec extends DbIntegrationTest with UserStub with Services with OptionValues {
  val maxarServiceMock: MaxarService =
    MaxarService(MaxarCatalogClientMock, MaxarTilesProxy(), dataProviderService)

  val runProcessingServiceWithMockMaxar: RunProcessingService = RunProcessingService(
    aoiService,
    billingService,
    processingService,
    progressService,
    projectService,
    userService,
    workflowService,
    nspdClient,
    dataProviderService,
    maxarServiceMock,
    workflowDefService,
    costCalculatorService,
  )

  private val wd: WorkflowDefSummary =
    WorkflowDefSummary("Test WD", None, None, 0, None, None, None, None, Seq())

  describe("createAndRun test") {
    it("should return same params") {
      val params =
        ProcessingParams(Map("raster_login" -> "test-login", "raster_password" -> "test-password"))
      val meta = ProcessingMeta(Map("source" -> "other"))

      val actual = costCalculatorService
        .useMaxarCredentialsIfNeeded(params, meta, wd)(regularUser)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()
        .toMap
      val expected = Map("raster_login" -> "test-login", "raster_password" -> "test-password")

      assertResult(expected)(actual)
    }

    it("should return same params without changing without source in meta") {
      val params = ProcessingParams(
        Map("raster_login" -> "default-login", "raster_password" -> "default-password")
      )
      val meta = ProcessingMeta(Map())

      val actual = costCalculatorService
        .useMaxarCredentialsIfNeeded(params, meta, wd)(regularUser)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()
        .toMap

      val expected = Map("raster_login" -> "default-login", "raster_password" -> "default-password")

      assertResult(expected)(actual)
    }

    it("should throw bad request (empty login)") {
      val params = ProcessingParams(
        Map("raster_login" -> "", "raster_password" -> "password", "testKey" -> "testValue")
      )
      val meta = ProcessingMeta(Map("source" -> "maxar"))

      assertThrows[BadRequest](
        costCalculatorService
          .useMaxarCredentialsIfNeeded(params, meta, wd)(regularUser)
          .rethrowT
          .transact(xa)
          .unsafeRunSync()
      )
    }

    it("should throw bad request (empty password)") {
      val params = ProcessingParams(
        Map("raster_login" -> "login", "raster_password" -> "", "testKey" -> "testValue")
      )
      val meta = ProcessingMeta(Map("source" -> "maxar"))

      assertThrows[BadRequest](
        costCalculatorService
          .useMaxarCredentialsIfNeeded(params, meta, wd)(regularUser)
          .rethrowT
          .transact(xa)
          .unsafeRunSync()
      )
    }

    it("should work with empty meta") {
      val params = ProcessingParams(
        Map("raster_login" -> "login", "raster_password" -> "password", "testKey" -> "testValue")
      )
      val meta = ProcessingMeta(Map())

      val actual = costCalculatorService
        .useMaxarCredentialsIfNeeded(params, meta, wd)(regularUser)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()
        .toMap

      val expected =
        Map("raster_login" -> "login", "raster_password" -> "password", "testKey" -> "testValue")
      assertResult(expected)(actual)
    }

    it("should work with empty params") {
      val params = ProcessingParams(Map())
      val meta = ProcessingMeta(Map())

      val actual = costCalculatorService
        .useMaxarCredentialsIfNeeded(params, meta, wd)(regularUser)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()
        .toMap

      val expected = Map()
      assertResult(expected)(actual)
    }

    it("should use user credentials from url") {
      val connectId = "39a1c1fc-ef1f-4e6f-8aa1-0ebe6522296d"
      val dataProvider = DataProvider(
        id = UUID.fromString(connectId),
        name = "securewatch",
        displayName = "test",
        urlTemplate = None,
        pricePerMp = 1L,
        previewUrl = None,
        credentialsUsername = None,
        credentialsPassword = ,
        credentialsToken = ,
        isDefault = true,
        mapfileUri = None,
      )
      val url =
        s"""https://securewatch.digitalglobe.com/earthservice/wmtsaccess?SERVICE=WMTS&VERSION=1.0.0&STYLE=&REQUEST=GetTile&LAYER=DigitalGlobe:ImageryTileService&FORMAT=image/png&TileRow=10757&TileCol=17456&TileMatrixSet=EPSG:3857&TileMatrix=EPSG:3857:15&CONNECTID=$connectId&CQL_FILTER=feature_id='testId'"""
      val params = ProcessingParams(
        Map("url" -> url, "raster_login" -> "test-user", "raster_password" -> "user-pass")
      )
      val actual = runProcessingServiceWithMockMaxar
        .loadImageMetadataIfNeeded(
          params,
          Some(dataProvider),
          wd.copy(userInputBucket = Some("test")),
        )(regularUser)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()
        .toMap
      assertResult(Some("1.0"))(actual.get("sun_elevation"))
    }

    it("should not let expired user to run processings") {
      val user =
        UserUtil.createUser(
          "97228df6-5dd9-482e-8e9a-8bd98067b21e",
          activeUntil = Instant.now().minusSeconds(3600).some,
        )
      val wdId = WorkflowDefUtil.createWd()
      val geom = GeometryUtil.createPolygon()

      Try(
        runProcessingService
          .createAndRun(
            CreateAndRunProcessingJson(none, none, none, none, wdId.some, geom, none, none, none)
          )(user)
          .rethrowT
          .transact(xa)
          .unsafeRunSync()
      ) should matchPattern {
        case Failure(Forbidden(_)) =>
      }
    }

    it("should allow non-expired user to run processings") {
      val user =
        UserUtil.createUser(
          "97228df6-5dd9-482e-8e9a-8bd98067b21e",
          activeUntil = Instant.now().plusSeconds(3600).some,
        )
      val wdId = WorkflowDefUtil.createWd()
      val geom = GeometryUtil.createPolygon(42_000_000L)

      val processingId = runProcessingService
        .createAndRun(
          CreateAndRunProcessingJson(none, none, none, none, wdId.some, geom, none, none, none)
        )(user)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()
        .id

      // This check may be simplified after the cost was added to ProcessingJson
      val processing = processingService
        .getProcessing(processingId)(UserUtil.admin)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      processing.cost should be(210.some)
    }


    it("should add GTIFF provider to local data request") {
      val user =
        UserUtil.createUser(
          "97228df6-5dd9-482e-8e9a-8bd98067b21e",
          activeUntil = Instant.now().plusSeconds(3600).some,
        )
      val wdId = WorkflowDefUtil.createWd()
      val geom = GeometryUtil.createPolygon(42_000_000L)
      val params = Map(
        "url" -> "s3://data/test.tiff",
        "source_type" -> "local")

      val processingId = runProcessingService
        .createAndRun(
          CreateAndRunProcessingJson(none, none, none, none, wdId.some, geom, params.some, none, none)
        )(user)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()
        .id

      val processing = processingService
        .getProcessing(processingId)(UserUtil.admin)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      processing.dataProvider.value.name should be("GTIFF")
      processing.params.toMap should be(Map("data_provider" -> "GTIFF",
                                            "url" -> "s3://data/test.tiff",
                                            "source_type" -> "local"))
     }
  }
}
