package io.geoalert.mapflow.service.nspd

import java.net.URL
import java.util.concurrent.Executors

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor
import scala.concurrent.Future
import scala.concurrent.duration.DurationInt
import scala.util.Failure
import scala.xml.NodeSeq

import akka.http.scaladsl.client.RequestBuilding.Get
import akka.http.scaladsl.marshallers.xml.ScalaXmlSupport._
import akka.http.scaladsl.model.HttpRequest
import akka.http.scaladsl.model.HttpResponse
import akka.http.scaladsl.model.headers.RawHeader
import akka.stream.scaladsl.Sink
import akka.stream.scaladsl.Source
import cats.effect.ContextShift
import cats.effect.IO
import cats.implicits.catsSyntaxApplicativeId
import cats.implicits.catsSyntaxMonadError
import com.google.common.util.concurrent.ThreadFactoryBuilder
import com.typesafe.scalalogging.LazyLogging
import io.geoalert.mapflow.AkkaSystem.system
import io.geoalert.mapflow.DefaultAvanpostConfig
import io.geoalert.mapflow.exception.MapfileFetchError
import io.geoalert.mapflow.exception.MapfileParseError
import io.geoalert.mapflow.service.nspd.response.Datasource
import io.geoalert.mapflow.service.nspd.response.Layer
import io.geoalert.mapflow.service.nspd.response.MapConfig
import io.geoalert.mapflow.util.HttpUtils

class LiveNspdClient extends NspdClient with DefaultAvanpostConfig with LazyLogging {
  implicit private val executionContext: ExecutionContextExecutor = ExecutionContext.global
  implicit private lazy val cs: ContextShift[IO] = IO.contextShift(
    ExecutionContext.fromExecutor(
      Executors.newFixedThreadPool(
        4,
        new ThreadFactoryBuilder().setNameFormat("nspd-client-%d").build(),
      )
    )
  )
  override def getMapfile(url: String): IO[MapConfig] = {
    logger.info(s"""
           Request to Nspd:
            Url: $url
            ActorId: $avanpostActorId
            Headers: ${RawHeader("X-Actor-ID", avanpostActorId)}
           """)
    val requestFuture = for {
      response <- runRequest(
        Get(url).withHeaders(
          RawHeader("X-Actor-ID", avanpostActorId)
        ),
        url,
      ).adaptError {
        case error =>
          MapfileFetchError(error, "Error occurred while get xml file")
      }
      xml <- HttpUtils
        .parseResponse[NodeSeq](response, "NSPD_CLIENT")
        .adaptError {
          case error =>
            logger.error(s"Error occurred while parse response: $response", error)
            MapfileFetchError(error, "Error occurred while get xml file")
        }
      mapConfig <- xml
        .headOption
        .fold(Future.failed[MapConfig](MapfileParseError(None, "Unprocessable entity")))(
          parseMapConfig(_).pure[Future]
        )
    } yield mapConfig
    IO.fromFuture(IO(requestFuture))
  }
  private def parseMapConfig(xml: scala.xml.Node): MapConfig = {
    val layerNode = (xml \ "Layer").head
    val layer = Layer(
      layerNode \@ "name",
      layerNode \@ "srs",
      (layerNode \ "StyleName").text,
      parseDatasource((layerNode \ "Datasource").head),
    )

    MapConfig(layer)
  }

  private def parseDatasource(xml: scala.xml.Node): Datasource =
    Datasource(
      (xml \ "Parameter").find(p => (p \@ "name") == "type").map(_.text).getOrElse(""),
      (xml \ "Parameter").find(p => (p \@ "name") == "file").map(_.text).getOrElse(""),
    )
  private def runRequest(request: HttpRequest, url: String): Future[HttpResponse] = {
    val requestFuture = Source
      .single(request)
      .via(HttpUtils.httpConnection(new URL(url)))
      .runWith(Sink.head)

    requestFuture.onComplete {
      case Failure(e) => logger.error(s"Error trying to call Nspd api", e)
      case _ =>
    }

    Future.firstCompletedOf(
      Seq(
        akka
          .pattern
          .after(1.minute, system.scheduler)(
            Future.failed(new RuntimeException("Nspd request timeout"))
          ),
        requestFuture,
      )
    )
  }
}
