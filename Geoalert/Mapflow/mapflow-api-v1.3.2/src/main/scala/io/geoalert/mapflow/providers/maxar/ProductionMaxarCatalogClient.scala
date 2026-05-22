package io.geoalert.mapflow.providers.maxar

import java.util.concurrent.Executors

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor
import scala.concurrent.Future

import akka.http.scaladsl.Http
import akka.http.scaladsl.marshallers.xml.ScalaXmlSupport
import akka.http.scaladsl.model.FormData
import akka.http.scaladsl.model.HttpMethods
import akka.http.scaladsl.model.HttpRequest
import akka.http.scaladsl.model.HttpResponse
import akka.http.scaladsl.model.StatusCodes
import akka.http.scaladsl.model.headers.Authorization
import akka.http.scaladsl.model.headers.BasicHttpCredentials
import akka.http.scaladsl.model.headers.RawHeader
import cats.effect.ContextShift
import cats.effect.IO
import com.google.common.util.concurrent.ThreadFactoryBuilder
import com.typesafe.scalalogging.LazyLogging

import io.geoalert.mapflow.AkkaSystem.system
import io.geoalert.mapflow.Config.discoveryApiKey
import io.geoalert.mapflow.exception.ExternalSystemError
import io.geoalert.mapflow.util.HttpUtils

class ProductionMaxarCatalogClient
    extends LazyLogging
       with MaxarCatalogClient
       with ScalaXmlSupport {
  implicit private val executionContext: ExecutionContextExecutor = ExecutionContext.global

  implicit private lazy val cs: ContextShift[IO] = IO.contextShift(
    ExecutionContext.fromExecutor(
      Executors.newFixedThreadPool(
        8,
        new ThreadFactoryBuilder().setNameFormat("maxar-catalog-client-%d").build(),
      )
    )
  )

  override def getMaxarMetaOld(
      url: String,
      body: Option[String],
      maxarLogin: String,
      maxarPassword: ,
    ): IO[HttpResponse] = {
    val auth = Authorization(BasicHttpCredentials(maxarLogin, maxarPassword))
    val request = body match {
      case None => HttpRequest(HttpMethods.GET, url, List(auth))
      case Some(value) => HttpRequest(HttpMethods.POST, url, Seq(auth), value)
    }

    val future = Http().singleRequest(request, settings = HttpUtils.proxySettings())
    IO.fromFuture(IO(future))
  }

  override def searchMeta(
      maxarLogin: String,
      maxarPassword: ,
      connectId: String,
      input: MaxarCatalogRequest,
    ): IO[List[MaxarFeature]] = {
    val url = s"https://securewatch.digitalglobe.com/catalogservice/wfsaccess?connectid=$connectId"

    val body = CatalogRequestBuilder.buildRequest(input)

    logger.debug(s"Maxar Catalog request: $body")
    val auth = Authorization(BasicHttpCredentials(maxarLogin, maxarPassword))
    val request = HttpRequest(HttpMethods.POST, url, Seq(auth), body)

    def apiIntersectionFilter(feature: MaxarFeature): Boolean = {
      val opt = for {
        aoi <- input.aoi
        minPercent <- input.minAoiIntersectionPercent
        percent = (feature.geometry.intersection(aoi).getArea / aoi.getArea) * 100
      } yield percent >= minPercent

      opt.getOrElse(true)
    }
    val future = for {
      response <- Http().singleRequest(request, settings = HttpUtils.proxySettings())
      features <- parseSearchMetaResponse(response)
    } yield features.filter(apiIntersectionFilter)

    IO.fromFuture(IO(future))
  }

  override def getDiscoveryApiMetadata(legacyId: String): IO[Option[DiscoveryApiMetadata]] = {
    val url = "https://api.discover.digitalglobe.com/v1/services/ImageServer/query"

    val body = FormData(
      "outFields" -> "*",
      "where" -> s"image_identifier in ('$legacyId')",
      "returnCountOnly" -> "false",
      "f" -> "json",
    ).toEntity

    val request = HttpRequest(
      HttpMethods.POST,
      url,
      Seq(
        RawHeader("x-api-key", discoveryApiKey)
      ),
      body,
    )

    logger.debug(s"Maxar Discovery API request: $body")

    val future = Http()
      .singleRequest(request, settings = HttpUtils.proxySettings())
      .flatMap(parseDiscoveryApiResponse)
    IO.fromFuture(IO(future))
  }

  private def parseDiscoveryApiResponse(
      response: HttpResponse
    ): Future[Option[DiscoveryApiMetadata]] =
    if (response.status == StatusCodes.OK)
      HttpUtils
        .extractResponseBodyAsString(response, "maxar")
        .map(DiscoveryApiResponseParser.parseResponse)

    // Future.failed(new RuntimeException)
    else
      HttpUtils.extractResponseBodyAsString(response, "maxar").map { str =>
        logger.error(s"Failed to get metadata from Maxar Discovery Service. Server responded: $str")
        throw ExternalSystemError("Maxar request failed")
      }

  private def parseSearchMetaResponse(response: HttpResponse): Future[List[MaxarFeature]] =
    if (response.status == StatusCodes.OK)
      HttpUtils
        .extractResponseBodyAsString(response, "maxar")
        .map(SearchMetaResponseParser.parseSearchMetaResponse)
    else
      HttpUtils.extractResponseBodyAsString(response, "maxar").map { str =>
        logger.error(s"Failed to search maxar catalog. Server responded: $str")
        throw ExternalSystemError("Maxar request failed")
      }
}
