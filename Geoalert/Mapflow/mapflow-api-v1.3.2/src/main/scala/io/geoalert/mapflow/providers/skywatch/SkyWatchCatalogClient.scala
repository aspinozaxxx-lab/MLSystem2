package io.geoalert.mapflow.providers.skywatch

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor
import scala.concurrent.Future

import akka.http.scaladsl.Http
import akka.http.scaladsl.model.ContentTypes
import akka.http.scaladsl.model.HttpEntity
import akka.http.scaladsl.model.HttpMethods
import akka.http.scaladsl.model.HttpRequest
import akka.http.scaladsl.model.HttpResponse
import akka.http.scaladsl.model.headers.RawHeader
import com.typesafe.scalalogging.LazyLogging
import io.circe.syntax.EncoderOps

import io.geoalert.mapflow.AkkaSystem.system
import io.geoalert.mapflow.DefaultExternalSystemConfig
import io.geoalert.mapflow.model.GetSkyWatchMetaInput
import io.geoalert.mapflow.rest.MetaResource.getSkyWatchMetaInputEncoder
import io.geoalert.mapflow.util.HttpUtils

class SkyWatchCatalogClient extends LazyLogging with DefaultExternalSystemConfig {
  implicit private val executionContext: ExecutionContextExecutor = ExecutionContext.global

  def getSkyWatchMetaAnswerId(input: GetSkyWatchMetaInput): Future[String] = {
    val request = HttpRequest(
      method = HttpMethods.POST,
      uri = skyWatchUrl,
      entity = HttpEntity(
        ContentTypes.`application/json`,
        input.asJson.toString(),
      ),
    ).withHeaders(RawHeader("x-api-key", skyWatchApiKey))

    for {
      response <- runRequest(request)
      responseBody <- HttpUtils.extractResponseBodyAsString(
        response,
        "SkyWatch",
      )
    } yield responseBody
  }

  def getSkyWatchMetaPage(
      answerId: String,
      cursor: Option[String],
    ): Future[HttpResponse] = {
    val uri: String = cursor match {
      case Some(c) => s"$skyWatchUrl/$answerId/search_results?cursor=$c"
      case None => s"$skyWatchUrl/$answerId/search_results"
    }

    val request = HttpRequest(
      method = HttpMethods.GET,
      uri = uri,
    ).withHeaders(RawHeader("x-api-key", skyWatchApiKey))

    runRequest(request)
  }

  private def runRequest(httpRequest: HttpRequest): Future[HttpResponse] =
    Http().singleRequest(httpRequest, settings = HttpUtils.proxySettings())
}
