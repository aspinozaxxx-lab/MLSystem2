package io.geoalert.mapflow

import java.net.URL
import java.util.UUID
import scala.concurrent.duration._
import scala.util.Try
import io.geoalert.mapflow.model.BillingType

trait DefaultConfig
    extends DefaultDbConfig
       with DefaultWeConfig
       with DefaultExternalSystemConfig
       with DefaultAoiLayerConfig
       with MiscConfig
       with TestEnvConfig
       with DefaultTelegramConfig
       with LimitsConfig
       with KeycloakConfig
       with DefaultProcessingReviewConfig
       with DefaultAvanpostConfig

trait MiscConfig {

  /** The port of the HTTP server */
  val port: Int = sys.env.getOrElse("PORT", "8080").toInt

  /** URL of this system, accessible from the web */
  val externalUrl: String = sys.env.getOrElse("ORIGIN", s"http://localhost:$port")

  /** S3 bucket to store COGs */
  val rastersBucket = sys.env.getOrElse("RASTERS_BUCKET", "ml-rasters")

  /** AOIs smaller than this are ignored during import */
  val zeroArea = 1e-10 // ~1 sq meter.

  /** Used to simplify AOIs during import. */
  val aoiSmpl = 1e-5 // ~1 meter.

  val defaultPaidProviders: Set[String] =
    sys.env.getOrElse("DEFAULT_PAID_PROVIDERS", s"securewatch,vivid").split("\\s*,\\s*").toSet
}

trait KeycloakConfig {
  /** JWT key used for coding/decoding auth tokens */
  val jwtKey: String = sys.env.getOrElse("JWT_KEY", "secretKey")

  val keycloakUrl: String = sys.env.getOrElse("KEYCLOAK_URL", "https://auth.mapflow.ai/auth")
  val keycloakRealm: String = sys.env.getOrElse("KEYCLOAK_REALM", "mapflow-duty")
  val keycloakOIDCClientId: String = sys.env.getOrElse("KEYCLOAK_OIDC_CLIENT", "whitemaps")

  val keycloakManagementClientId: String =
    sys.env.getOrElse("KEYCLOAK_MANAGEMENT_CLIENT", "front-view-users")
  val keycloakManagementClientSecret:  =
    sys.env.getOrElse("KEYCLOAK_MANAGEMENT_CLIENT_SECRET", "tOpaYQfxKPnxjaTeutkBlp3EIZHvyJNo")
}

trait LimitsConfig {
  /** Max partition size in degrees (both dimensions) */
  val defaultPartitionSize: Double = sys
    .env
    .get("DEFAULT_PARTITION_SIZE")
    .map(_.toDouble)
    .getOrElse(0.05) // ~5000 meters

  /** Partition size for source type sentinel_l2a */
  val sentinelPartitionSize: Double = sys
    .env
    .get("SENTINEL_PARTITION_SIZE")
    .map(_.toDouble)
    .getOrElse(2000000e-5)

  /** Max amount AOIs allowed to be fetched via API. Used as a guard against excessive load. */
  val maxAoiFetch = 1000

  /** Max AOIs for one CreateAndRun request * */
  val maxAoisPerProcessing: Int = sys
    .env
    .get("MAX_AOIS_PER_PROCESSING")
    .map(_.toInt)
    .getOrElse(10)

  /** Limits for users */
  val defaultAreaLimit: Long =
    sys.env.getOrElse("DEFAULT_AREA_LIMIT", "50000000").toLong
  val defaultAoiAreaLimit: Long =
    sys.env.getOrElse("DEFAULT_AOI_AREA_LIMIT", "50000000").toLong
  val defaultMemoryLimit: Long =
    sys.env.getOrElse("DEFAULT_MEMORY_LIMIT", "1000000000").toLong

  val defaultBillingType: BillingType =
    BillingType.fromString(sys.env.getOrElse("DEFAULT_BILLING_TYPE", "NONE"))
}

trait TestEnvConfig {

  /** Enables test HTTP routes */
  val testEnv: Boolean = sys.env.getOrElse("TEST_ENV", "false").toBoolean

  /** Migrations to populate DB with test data */
  val testData: Option[String] = sys.env.get("TEST_DATA")

  /** Determines the probability of receiving a `FAILED` status in response from WE (mocked WE only) */
  val mockWeFailedPercent: Int =
    sys.env.getOrElse("MOCK_WE_FAILED_PERCENT", "0").toInt

  /** Determines the probability of receiving an `IN_PROGRESS` status in response from WE (mocked WE only) */
  val mockWeInProgressPercent: Int =
    sys.env.getOrElse("MOCK_WE_IN_PROGRESS_PERCENT", "50").toInt
}

trait DefaultDbConfig {
  val dbPort: Int = sys.env.getOrElse("DB_PORT", "5432").toInt
  val dbHost: String = sys.env.getOrElse("DB_HOST", "localhost")
  val dbName: String = sys.env.getOrElse("DB_NAME", "mapflow")
  val dbUser: String = sys.env.getOrElse("DB_USER", "postgres")
  val dbPassword:  = sys.env.getOrElse("DB_PASSWORD", "1234Qq")
  val dbSchema: String = sys.env.getOrElse("DB_SCHEMA", "mapflow")
  val dbUrl: String =
    s"jdbc:postgresql://$dbHost:$dbPort/$dbName"

  val dbUrlMigration: String =
    s"jdbc:postgresql://$dbHost:$dbPort/$dbName"

  /** Max length of IN clause in SQL queries (e.g., max amount of ids, when querying by ids) */
  val maxInFrLen = 20000

  val connectionPoolSize: Int = sys.env.getOrElse("CONNECTION_POOL_SIZE", "50").toInt
  val leakDetectionThresholdMs: Long =
    sys.env.getOrElse("CONNECTION_LEAK_THRESHOLD_MS", "30000").toLong
}

trait DefaultWeConfig {
  lazy val weUrl = new URL(
    sys.env.getOrElse("WORKFLOW_ENGINE_URL", "http://localhost:8060")
  )

  lazy val weBatchSize: Int = sys.env.getOrElse("WORKFLOW_ENGINE_BATCH_SIZE", "100").toInt

  /** A string that identifies this app instance, used to track workflow owners */
  val systemId: String = sys.env.getOrElse("SYSTEM_ID", "mapflow")

  /** Minimum workflow priority value */
  val minPriority: Int = sys.env.getOrElse("MIN_PRIORITY", "1").toInt

  /** Maximum workflow priority value */
  val maxPriority: Int = sys.env.getOrElse("MAX_PRIORITY", "10").toInt

  /** WE polling interval */
  val workflowUpdateInterval: Int =
    sys.env.getOrElse("PROGRESS_UPDATE_INTERVAL", "5").toInt
}

trait DefaultExternalSystemConfig {
  lazy val httpProxy: Option[URL] = sys.env.get("HTTP_PROXY").map(new URL(_))

  lazy val vectorProcessorUrl = new URL(
    sys.env.getOrElse("VECTOR_PROCESSOR_URL", "http://localhost:8700")
  )

  lazy val vectorTileServerUrl: String =
    sys.env.getOrElse("VECTOR_TILE_SERVER_URL", "http://localhost:8600")

  lazy val rasterTileServerUrl: String =
    sys.env.getOrElse("RASTER_TILE_SERVER_URL", "http://localhost:8500")

  val s3Url: String =
    sys.env.getOrElse("MINIO_URL", "http://localhost:9000")

  val s3AccessKey:  =
    sys.env.getOrElse("MINIO_ACCESS_KEY", "CDX9VOT08Z44JC2D3TPV")

  val s3SecretKey:  = sys
    .env
    .getOrElse(
      "MINIO_SECRET_KEY",
      "VRdU292pgyplJlFXHrHs1+I9G3060Os+sTz2DeGe",
    )

  val discoveryApiKey: String = sys.env.getOrElse("MAXAR_DISCOVERY_API_KEY", "")

  val zoomConstraint: Int = sys.env.getOrElse("ZOOM_CONSTRAINT", "12").toInt

  /** Current latest version of plugin, semver string(starting from "1.7.0"). */
  val wmApiVersion: String = sys.env.getOrElse("WM_API_VERSION", "1")

  /** SkyWatch Data Provider */
  val skyWatchApiKey: String = sys.env.getOrElse("SKYWATCH_API_KEY", "")

  val skyWatchUrl: String =
    sys.env.getOrElse("SKYWATCH_URL", "https://api.skywatch.co/earthcache/archive/search")

  val wmAdministratorEmail: String =
    sys.env.getOrElse("WM_ADMINISTRATOR_EMAIL", "admin@geoalert.io")

  val apiKey: String =
    sys.env.getOrElse("API_KEY", "PjP4KG&y7^mDB%g*PUQxu5sr%^Hj2i6Ef9355Z3ldc&z70nachRLEV!RUxA5TVP@")

  // We use single tileproxy instance because HEAD supports only one session
  val tileproxyUrl: String = sys
    .env
    .getOrElse(
      "TILEPROXY_URL",
      "https://app.mapflow.ai/tiles/satimagery/{z}/{x}/{y}.png?year={year}",
    )
  val tileproxyApiKey: String =
    sys.env.getOrElse("TILEPROXY_API_KEY", "7c57e21577ebea7e212970ee7ad9dab3408f9418")

  val billingEngineUrl: String =
    sys.env.getOrElse("BILLING_ENGINE_URL", "http://billing-engine:8080/")
  val billingEngineApiKey: String = sys
    .env
    .getOrElse(
      "BILLING_ENGINE_API_KEY",
      "NO8jscmHu0k@Do^IRLMuOzI&S#h6ajwjLS36^2ZFo!ujAE1JwpTCkN!v5bY2T5!S",
    )
  val billingDisabled: Boolean = sys.env.getOrElse("BILLING_DISABLED", "true").toBoolean
}

trait DefaultAoiLayerConfig {
  val geojsonLayerSmplFactor = 1500 // extent.width / 1500
  val geojsonLayerSmplMax = 30e-3 // ~ 3000 meters

  /** Determines when to convert a polygon to a point */
  val pointThreshold = 6 // in pixels

  val mvtLayerSmplFactor: Int = 4096 / 2 // extent.width / (4096 / 2)
  val mvtLayerSmplMax = 5e-2 // ~ 5000 meters

  val minPartitionSize = 0.01 // in degrees
  val maxPartitionSize = 2 // in degrees
  val defaultAoiStartTime: Int = sys.env.getOrElse("AOI_START_TIME", "300").toInt
}
trait DefaultAvanpostConfig {
  val avanpostDisabled: Boolean = sys.env.getOrElse("AVANPOST_DISABLED", "false").toBoolean

  val avanpostUrl: String = sys.env.getOrElse("AVANPOST_URL", "http://localhost/")
  val avanpostActorId: String = sys.env.getOrElse("AVANPOST_ACTOR_ID", "")
  val avanpostUserGroupIds: List[UUID] =
    sys
      .env
      .getOrElse("USER_GROUP_IDS", "4134b782-2aa4-4e7a-8d35-314ff04e9bde")
      .split(",")
      .toList
      .flatMap(str => Try(UUID.fromString(str)).toOption)
  val avanpostAdminGroupIds: List[UUID] =
    sys
      .env
      .getOrElse("ADMIN_GROUP_IDS", "c700172a-e8cf-4c8e-87ec-4f91928dddc0")
      .split(",")
      .toList
      .flatMap(str => Try(UUID.fromString(str)).toOption)
}

trait DefaultProcessingReviewConfig {
  val autoConfirmProcessingsInterval: Duration =
    sys.env.getOrElse("REVIEW_AUTO_CONFIRM_INTERVAL_HOURS", "1440").toInt.hours
  val autoConfirmCheckInterval: FiniteDuration = 5.minutes
}

trait DefaultTelegramConfig {
  val environment: String = sys.env.getOrElse("ENVIRONMENT", "staging")

  val telegramApiUrl = new URL("https://api.telegram.org")

  /** Telegram bot API Token */
  val telegramToken:  = sys.env.getOrElse("TELEGRAM_TOKEN", "")

  /** Chat ID for notifications */
  val telegramChatId: String = sys.env.getOrElse("TELEGRAM_CHAT_ID", "")

  /** Enable Telegram notifications about failed processings */
  val enableTelegramNotificationAboutFailedProcessings: Boolean = sys
    .env
    .getOrElse("ENABLE_FAILED_PROCESSINGS_NOTIFICATIONS", "true")
    .toBoolean
}
