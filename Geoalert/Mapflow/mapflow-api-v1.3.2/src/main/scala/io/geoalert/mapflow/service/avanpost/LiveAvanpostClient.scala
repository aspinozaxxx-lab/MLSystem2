package io.geoalert.mapflow.service.avanpost

import java.net.URL
import java.util.UUID

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor
import scala.concurrent.Future
import scala.concurrent.duration.DurationInt
import scala.util.Failure

import akka.http.scaladsl.client.RequestBuilding.Get
import akka.http.scaladsl.model.HttpRequest
import akka.http.scaladsl.model.HttpResponse
import akka.http.scaladsl.model.headers.RawHeader
import akka.stream.scaladsl.Sink
import akka.stream.scaladsl.Source
import com.typesafe.scalalogging.LazyLogging
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import io.geoalert.mapflow.AkkaSystem.system
import io.geoalert.mapflow.DefaultAvanpostConfig
import io.geoalert.mapflow.service.avanpost.responses.UserInfo
import io.geoalert.mapflow.util.HttpUtils

class LiveAvanpostClient extends AvanpostClient with DefaultAvanpostConfig with LazyLogging {
  implicit private val executionContext: ExecutionContextExecutor = ExecutionContext.global
  private val connectionFlow = HttpUtils.httpConnection(new URL(avanpostUrl))
  override def userInfo(token: , id: UUID): Future[UserInfo] = {
    val url = avanpostUrl.replace("{}", id.toString)
    logger.info(s"""
           Request to avanpost:
            UserId: $id
            Url: $url
            ActorId: $avanpostActorId
            Headers: ${List(
              RawHeader("X-Actor-ID", avanpostActorId),
            )}
           """)
    for {
      response <- runRequest(
        Get(url).withHeaders(
          RawHeader("X-Actor-ID", avanpostActorId),
        )
      )
      userInfo <- HttpUtils.parseResponse[UserInfo](response, "BE")
      _ = logger.info(s"""Avanpost user groups: $userInfo""")
    } yield userInfo
  }

  private def runRequest(request: HttpRequest): Future[HttpResponse] = {
    val requestFuture = Source
      .single(request)
      .via(connectionFlow)
      .runWith(Sink.head)

    requestFuture.onComplete {
      case Failure(e) => logger.error(s"Error trying to call Avanpost api", e)
      case _ =>
    }

    Future.firstCompletedOf(
      Seq(
        akka
          .pattern
          .after(1.minute, system.scheduler)(
            Future.failed(new RuntimeException("Avanpost request timeout"))
          ),
        requestFuture,
      )
    )
  }
}
