package io.geoalert.mapflow.service

import java.net.URL
import java.util.concurrent.Executors

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor
import scala.concurrent.Future
import scala.concurrent.duration._
import scala.util.matching.Regex

import akka.actor.ActorSystem
import akka.http.scaladsl.model.HttpResponse
import akka.http.scaladsl.model.StatusCodes
import akka.http.scaladsl.unmarshalling.Unmarshal
import cats.data.EitherT
import cats.effect.ContextShift
import cats.effect.IO
import cats.implicits._
import com.github.blemale.scaffeine.AsyncCache
import com.github.blemale.scaffeine.Scaffeine
import com.google.common.util.concurrent.ThreadFactoryBuilder
import com.typesafe.scalalogging.LazyLogging
import doobie.ConnectionIO
import doobie.implicits._

import io.geoalert.mapflow.AkkaSystem
import io.geoalert.mapflow.DefaultExternalSystemConfig
import io.geoalert.mapflow.exception.AccessDenied
import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.exception.BadRequest
import io.geoalert.mapflow.exception.ExternalSystemError
import io.geoalert.mapflow.model.DataProvider
import io.geoalert.mapflow.model.GetMaxarMetaInput
import io.geoalert.mapflow.model.Permission.NoZoomRestrictionsForMaxar
import io.geoalert.mapflow.model.PngLinkInput
import io.geoalert.mapflow.model.ProcessingMeta
import io.geoalert.mapflow.model.ProcessingParams
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.model.UserCredentials
import io.geoalert.mapflow.model.WorkflowDefSummary
import io.geoalert.mapflow.providers.maxar.DiscoveryApiMetadata
import io.geoalert.mapflow.providers.maxar.MaxarCatalogClient
import io.geoalert.mapflow.providers.maxar.MaxarCatalogRequest
import io.geoalert.mapflow.providers.maxar.MaxarSatelliteOrbitHeight
import io.geoalert.mapflow.providers.maxar.MaxarTilesProxy
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.rest.json.ImageJson
import io.geoalert.mapflow.util.HttpUtils

class MaxarService(
    maxarCatalogClient: MaxarCatalogClient,
    maxarTilesProxy: MaxarTilesProxy,
    dataProviderService: DataProviderService,
  ) extends LazyLogging
       with DefaultExternalSystemConfig {
  implicit val system: ActorSystem = AkkaSystem.system
  implicit private val executionContext: ExecutionContextExecutor = ExecutionContext.global

  implicit private lazy val cs: ContextShift[IO] = IO.contextShift(
    ExecutionContext.fromExecutor(
      Executors.newFixedThreadPool(
        8,
        new ThreadFactoryBuilder().setNameFormat("maxar-client-%d").build(),
      )
    )
  )

  val cache: AsyncCache[String, List[DataProvider]] = Scaffeine()
    .expireAfterWrite(10.minutes)
    .maximumSize(1000)
    .buildAsync[String, List[DataProvider]]()

  private val IMAGE_ID_REGEX: Regex = "feature_id='(\\w+?)'".r

  def parseMaxarUrl(urlString: String): Option[String] = {
    val url = new URL(urlString)
    val query = HttpUtils.parseQueryParameters(url.getQuery)
    for {
      filter <- query.get("CQL_FILTER")
      reMatch <- IMAGE_ID_REGEX.findFirstMatchIn(filter)
      imageId = reMatch.group(1)
    } yield imageId
  }

  def getDefaultDataProviders(maxarProduct: String): Future[List[DataProvider]] =
    cache.getFuture(maxarProduct, loadDefaultDataProviders)

  private def loadDefaultDataProviders(maxarProduct: String): Future[List[DataProvider]] =
    dataProviderService.listDataProvidersByName(maxarProduct).transact(xa).unsafeToFuture()

  def cheapestProviderForAnImage(
      dataProviders: List[DataProvider],
      imageId: String,
    ): IO[Option[DataProvider]] =
    for {
      responses <- dataProviders.sortBy(_.pricePerMp).traverse(checkImageInProvider(imageId))
    } yield responses.find(_._1).map(_._2)

  def addImageMetadataToParams(
      params: ProcessingParams,
      maybeDataProvider: Option[DataProvider],
      maybeUserCred: Option[UserCredentials],
      wd: WorkflowDefSummary,
    ): EitherT[ConnectionIO, ApplicationError, ProcessingParams] = {
    logger.debug(
      s"Adding image metadata to Processing params $params, ${maybeDataProvider
          .fold("--")(_.displayName)}, $wd"
    )
    val urlOpt = (params.url ++ wd.url).headOption

    val maxarUsername = maybeDataProvider
      .flatMap(_.credentialsUsername)
      .orElse(maybeUserCred.flatMap(_.username))
      .getOrElse(throw new IllegalStateException("Data provider must contain credentialsUsername"))
    val maxarPassword = 
      .flatMap(_.credentialsPassword)
      .orElse(maybeUserCred.flatMap(_.password))
      .getOrElse(throw new IllegalStateException("Data provider must contain credentialsPassword"))
    val connectId = maybeDataProvider
      .flatMap(_.credentialsToken)
      .orElse(maybeUserCred.flatMap(_.token))
      .getOrElse(throw new IllegalStateException("Data provider must contain credentialsToken"))

    def requestLegacyId(imageId: String): EitherT[IO, ApplicationError, String] =
      for {
        features <- EitherT.right[ApplicationError](
          maxarCatalogClient.searchMeta(
            maxarUsername,
            maxarPassword,
            connectId,
            MaxarCatalogRequest(featureId = imageId.some),
          )
        )
        legacyIdOpt = features.map(_.metadata.legacyId).headOption
        legacyId <- EitherT.fromOption[IO](
          legacyIdOpt,
          ExternalSystemError("Unable to perform Maxar Catalog request"): ApplicationError,
        )
      } yield legacyId

    def requestMetadata(legacyId: String): EitherT[IO, ApplicationError, DiscoveryApiMetadata] =
      for {
        metadataOpt <- EitherT.right[ApplicationError](
          maxarCatalogClient.getDiscoveryApiMetadata(legacyId)
        )
        metadata <- EitherT.fromOption[IO](
          metadataOpt,
          ExternalSystemError("Unable to perform Maxar Discovery API  request"): ApplicationError,
        )
      } yield metadata

    val io = for {
      url <- EitherT
        .fromOption[IO](urlOpt, BadRequest("'url' parameter must be defined"): ApplicationError)
      imageId <- EitherT.fromOption[IO](
        parseMaxarUrl(url),
        BadRequest("cannot extract feature_id from URL"): ApplicationError,
      )
      legacyId <- requestLegacyId(imageId)
      metadata <- requestMetadata(legacyId)
      _ = logger.debug(s"Received Maxar Image meta: $metadata")
    } yield ProcessingParams(
      params.toMap ++ Map(
        "sat_elevation" -> offNadirAngleToSatelliteElevation(
          metadata.offNadirAvg,
          MaxarSatelliteOrbitHeight.orbitMeanHeightMeters(metadata.vehicleName),
        ).toString,
        "sun_azimuth" -> metadata.sunAzimuthAvg.toString,
        "sat_azimuth" -> ((180 + metadata.targetAzimuthAvg) % 360).toString,
        "sun_elevation" -> metadata.sunElevationAvg.toString,
      )
    )

    EitherT(io.value.to[ConnectionIO])
  }

  def offNadirAngleToSatelliteElevation(
      offNadirAngle: Double,
      orbitHeightMeters: Double,
    ): Double = {
    val earthRadiusMeters = 6_371_000

    val phi = Math.sin(Math.toRadians(offNadirAngle))
    Math.toDegrees(Math.acos(phi * (earthRadiusMeters + orbitHeightMeters) / earthRadiusMeters))
  }

  def addMaxarCredentialsToParams(
      params: ProcessingParams,
      meta: ProcessingMeta,
      wd: WorkflowDefSummary,
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, ProcessingParams] = {
    val urlOpt = (params.url ++ wd.url).headOption
    logger.debug(s"Adding maxar creds to url ${urlOpt}")
    // 1. Parse Maxar URL to find Connection ID and Image ID
    // 2. Check available data providers in order to find suitable provider
    val io = for {
      url <- EitherT
        .fromOption[IO](urlOpt, BadRequest("'url' parameter must be defined"): ApplicationError)
      imageId <- EitherT.fromOption[IO](
        parseMaxarUrl(url),
        BadRequest("cannot extract feature_id from URL"): ApplicationError,
      )
      maxarProduct <- EitherT.fromOption[IO](
        meta.maxarProduct,
        BadRequest("'maxar_product' parameter must be defined"): ApplicationError,
      )
      dataProviders = user.availableDataProviders.filter(_.name.equalsIgnoreCase(maxarProduct))
      cheapestProviderOpt <- EitherT.right(cheapestProviderForAnImage(dataProviders, imageId))
      cheapestProvider <- EitherT.fromOption[IO](
        cheapestProviderOpt,
        BadRequest(
          "No suitable data provider found for the processing",
          "MAXAR_PROVIDERS_UNAVAILABLE",
        ): ApplicationError,
      )
      _ = logger.info(
        s"Chosen provider: ${cheapestProvider.id}, ${cheapestProvider.name}, ${cheapestProvider.credentialsToken}"
      )
    } yield ProcessingParams(
      params.toMap ++ Seq(
        cheapestProvider.credentialsUsername.map("raster_login" -> _),
        cheapestProvider.credentialsPassword.map("raster_password" -> _),
        cheapestProvider.credentialsToken.map(connectId => "url" -> s"$url&CONNECTID=$connectId"),
      ).flatten.toMap
    )

    EitherT(io.value.to[ConnectionIO])
  }

  def checkImageInProvider(
      imageId: String
    )(
      dataProvider: DataProvider
    ): IO[(Boolean, DataProvider)] = {
    val maxarUsername = dataProvider
      .credentialsUsername
      .getOrElse(throw new IllegalStateException("Data provider must contain credentialsUsername"))
    val maxarPassword = 
      .credentialsPassword
      .getOrElse(throw new IllegalStateException("Data provider must contain credentialsPassword"))
    val connectId = dataProvider
      .credentialsToken
      .getOrElse(throw new IllegalStateException("Data provider must contain credentialsToken"))

    val request = MaxarCatalogRequest(featureId = imageId.some)

    maxarCatalogClient
      .searchMeta(maxarUsername, maxarPassword, connectId, request)
      .map(list => (list.nonEmpty, dataProvider))
  }

  def selectProviderForMetaSearch(
      maxarProduct: String,
      userDataProviders: List[DataProvider],
    ): ConnectionIO[Option[DataProvider]] = {
    val userProviders = userDataProviders
      .filter(_.name.equalsIgnoreCase(maxarProduct))

    val providersIo: ConnectionIO[List[DataProvider]] =
      if (userProviders.isEmpty)
        dataProviderService.listDataProvidersByName(maxarProduct)
      else
        IO.pure(userProviders).to[ConnectionIO]

    for {
      providers <- providersIo
    } yield providers.sortBy(_.pricePerMp).reverse.headOption
  }

  def requestImageryFromCatalogOld(
      input: GetMaxarMetaInput
    )(
      user: User
    ): ConnectionIO[HttpResponse] = {
    logger.info(
      s"Requesting imagery from Maxar catalog for ${user.email}, request connectId=${input.connectId}, URL: ${input.url}"
    )

    val maxarProduct = input.connectId

    for {
      bestProvider <- selectProviderForMetaSearch(maxarProduct, user.availableDataProviders)
      response <- bestProvider match {
        case None =>
          logger.debug(s"Data Provider '$maxarProduct' is not available for '${user.email}'")
          val res = HttpResponse(
            StatusCodes.Forbidden,
            Nil,
            s"Data Provider '$maxarProduct' is not available for '${user.email}'",
          )
          IO.pure(res).to[ConnectionIO]
        case Some(dataProvider) =>
          logger.info(
            s"Chosen provider: ${dataProvider.id}, ${dataProvider.name}, ${dataProvider.credentialsToken}"
          )
          retrieveCatalogForDataProvider(dataProvider, input).to[ConnectionIO]
      }
    } yield response
  }

  def searchMeta(request: MaxarCatalogRequest)(user: User): ConnectionIO[List[ImageJson]] =
    for {
      bestProvider <- selectProviderForMetaSearch("securewatch", user.availableDataProviders)
      images <- bestProvider match {
        case None =>
          IO.pure(List()).to[ConnectionIO]
        case Some(dp) =>
          val connectId = dp
            .credentialsToken
            .getOrElse(
              throw new IllegalStateException(
                s"credentialsToken not defined for data provider ${dp.id}"
              )
            )
          val maxarLogin = dp
            .credentialsUsername
            .getOrElse(
              throw new IllegalStateException(
                s"credentialsUsername not defined for data provider ${dp.id}"
              )
            )
          val maxarPassword = 
            .credentialsPassword
            .getOrElse(
              throw new IllegalStateException(
                s"credentialsPassword not defined for data provider ${dp.id}"
              )
            )
          maxarCatalogClient
            .searchMeta(maxarLogin, maxarPassword, connectId, request)
            .to[ConnectionIO]
      }
    } yield images.map(_.toImageJson)

  def proxySingleTile(input: PngLinkInput)(user: User): IO[HttpResponse] = {
    val maxarProduct = input.connectIdType

    val zoom = getCorrectZoomUsingConstraints(input.tileMatrix)(user)
    val x = input.tileColumn
    val y = input.tileRow

    val userDefaultProductOpt = user
      .availableDataProviders
      .sortBy(_.pricePerMp)
      .findLast(_.name.equalsIgnoreCase(maxarProduct))

    for {
      dataProviders <- IO.fromFuture(IO(getDefaultDataProviders(maxarProduct)))
      dataProviderOpt = (userDefaultProductOpt.toList ++ dataProviders
        .sortBy(_.pricePerMp)
        .findLast(_.name.equalsIgnoreCase(maxarProduct))
        .toList).headOption
      response <- dataProviderOpt match {
        case None =>
          IO.pure(
            HttpResponse(
              StatusCodes.Forbidden,
              Nil,
              s"User has no access to $maxarProduct data provider",
            )
          )
        case Some(dataProvider) => proxySingleTile(zoom, x, y, input.cqlFilter, dataProvider)
      }
    } yield response
  }

  private def proxySingleTile(
      zoom: Int,
      x: Int,
      y: Int,
      cqlFilter: Option[String],
      dataProvider: DataProvider,
    ): IO[HttpResponse] = {
    val url =
      s"""
         |https://securewatch.maxar.com/earthservice/wmtsaccess?
         |SERVICE=WMTS&
         |VERSION=1.0.0&
         |STYLE=&
         |REQUEST=GetTile&
         |LAYER=DigitalGlobe:ImageryTileService&
         |FORMAT=image/png&
         |TileRow=$y&
         |TileCol=$x&
         |TileMatrixSet=EPSG:3857&
         |TileMatrix=EPSG:3857:$zoom&
         |CONNECTID=${dataProvider.credentialsToken.getOrElse("")}
         |&CQL_FILTER=${cqlFilter.getOrElse("")}
         |"""
        .stripMargin
        .replace("\n", "")

    maxarTilesProxy.proxySingleTile(
      url,
      dataProvider.credentialsUsername.getOrElse(""),
      dataProvider.credentialsPassword.getOrElse(""),
    )
  }

  private def getCorrectZoomUsingConstraints(zoom: Int)(user: User): Int =
    if (user.role.hasPermission(NoZoomRestrictionsForMaxar))
      zoom
    else if (zoom > zoomConstraint)
      throw AccessDenied(s"Access to zoom level > $zoomConstraint is denied")
    else
      zoom

  private def retrieveCatalogForDataProvider(
      dataProvider: DataProvider,
      input: GetMaxarMetaInput,
    ): IO[HttpResponse] = {
    val urlWithConnectId = s"${input.url}&CONNECTID=${dataProvider.credentialsToken.get}"

    logger.debug(
      s"Sending request ${input.body} to $urlWithConnectId authorized by ${dataProvider.credentialsUsername.getOrElse("")}"
    )

    maxarCatalogClient
      .getMaxarMetaOld(
        urlWithConnectId,
        input.body,
        dataProvider.credentialsUsername.getOrElse(""),
        dataProvider.credentialsPassword.getOrElse(""),
      )
      .map { response =>
        if (response.status != StatusCodes.OK) {
          val responseString = Unmarshal(response.entity).to[String]
          logger.error(s"Maxar responded ${response.status} $responseString")
          HttpResponse(StatusCodes.BadGateway)
        }
        else
          response
      }
  }
}

object MaxarService {
  def apply(
      maxarCatalogClient: MaxarCatalogClient,
      maxarTilesProxy: MaxarTilesProxy,
      dataProviderService: DataProviderService,
    ): MaxarService = new MaxarService(maxarCatalogClient, maxarTilesProxy, dataProviderService)
}
