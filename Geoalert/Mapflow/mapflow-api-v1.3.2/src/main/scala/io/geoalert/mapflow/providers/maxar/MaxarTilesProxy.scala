package io.geoalert.mapflow.providers.maxar

import java.util.concurrent.Executors

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor

import akka.http.scaladsl.Http
import akka.http.scaladsl.model.ContentType
import akka.http.scaladsl.model.HttpEntity
import akka.http.scaladsl.model.HttpMethods
import akka.http.scaladsl.model.HttpRequest
import akka.http.scaladsl.model.HttpResponse
import akka.http.scaladsl.model.MediaTypes
import akka.http.scaladsl.model.StatusCodes
import akka.http.scaladsl.model.headers.Authorization
import akka.http.scaladsl.model.headers.BasicHttpCredentials
import cats.effect.ContextShift
import cats.effect.IO
import com.google.common.util.concurrent.ThreadFactoryBuilder
import com.typesafe.scalalogging.LazyLogging

import io.geoalert.mapflow.AkkaSystem.system
import io.geoalert.mapflow.util.HttpUtils

class MaxarTilesProxy extends LazyLogging {
  implicit private val executionContext: ExecutionContextExecutor = ExecutionContext.global

  implicit private lazy val cs: ContextShift[IO] = IO.contextShift(
    ExecutionContext.fromExecutor(
      Executors.newFixedThreadPool(
        128,
        new ThreadFactoryBuilder().setNameFormat("maxar-tiles-proxy-%d").build(),
      )
    )
  )

  def proxySingleTile(
      url: String,
      credentialsUsername: String,
      credentialsPassword: ,
    ): IO[HttpResponse] = {
    logger.debug(s"Downloading single maxar tile $url using maxar account $credentialsUsername")
    val authorization = Authorization(
      BasicHttpCredentials(credentialsUsername, credentialsPassword)
    )

    val request = HttpRequest(HttpMethods.GET, url, List(authorization))

    val future = Http().singleRequest(request, settings = HttpUtils.proxySettings()).map {
      case okResponse @ HttpResponse(StatusCodes.OK, _, _, _) =>
        HttpResponse(
          entity = HttpEntity(
            ContentType(MediaTypes.`image/png`),
            okResponse.entity.dataBytes,
          )
        )

      case notOkResponse => notOkResponse
    }

    IO.fromFuture(IO(future))
  }
}

object MaxarTilesProxy {
  def apply(): MaxarTilesProxy = new MaxarTilesProxy()
}
