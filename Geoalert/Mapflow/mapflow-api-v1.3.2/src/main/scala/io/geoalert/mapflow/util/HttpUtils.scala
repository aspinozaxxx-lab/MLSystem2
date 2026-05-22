package io.geoalert.mapflow.util

import java.net.InetSocketAddress
import java.net.URL

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor
import scala.concurrent.Future

import akka.http.scaladsl.ClientTransport
import akka.http.scaladsl.Http
import akka.http.scaladsl.model.HttpRequest
import akka.http.scaladsl.model.HttpResponse
import akka.http.scaladsl.settings.ClientConnectionSettings
import akka.http.scaladsl.settings.ConnectionPoolSettings
import akka.http.scaladsl.unmarshalling.Unmarshal
import akka.http.scaladsl.unmarshalling.Unmarshaller
import akka.stream.scaladsl.Flow
import akka.util.ByteString
import com.typesafe.scalalogging.LazyLogging

import io.geoalert.mapflow.AkkaSystem._
import io.geoalert.mapflow.Config
import io.geoalert.mapflow.exception.ExternalSystemError

object HttpUtils extends LazyLogging {
  implicit private val ec: ExecutionContextExecutor = ExecutionContext.global

  def proxySettings(): ConnectionPoolSettings =
    Config
      .httpProxy
      .map { p =>
        val httpsProxyTransport =
          ClientTransport.httpsProxy(InetSocketAddress.createUnresolved(p.getHost, p.getPort))

        ConnectionPoolSettings(system)
          .withConnectionSettings(
            ClientConnectionSettings(system)
              .withTransport(httpsProxyTransport)
          )
      }
      .getOrElse(ConnectionPoolSettings.default)

  def httpConnection(url: URL): Flow[HttpRequest, HttpResponse, Future[Http.OutgoingConnection]] = {
    val https = url.getProtocol.equalsIgnoreCase("https")
    val host = url.getHost
    val port = url.getPort match {
      case -1 if https => 443
      case -1 if !https => 80
      case p => p
    }

    if (https) Http().outgoingConnectionHttps(host = host, port = port)
    else Http().outgoingConnection(host = host, port = port)
  }

  def parseResponse[A](
      r: HttpResponse,
      systemName: String,
    )(implicit
      um: Unmarshaller[HttpResponse, A]
    ): Future[A] = {
    logger.debug(s"Received response from $systemName (status ${r.status.intValue})")
    if (r.status.isSuccess()) Unmarshal(r).to[A]
    else
      r.entity.dataBytes.runFold(ByteString.empty)(_ ++ _).flatMap { body =>
        Future.failed(ExternalSystemError(systemName, r.status.intValue(), body.utf8String))
      }
  }

  def parseResponseAsString(r: HttpResponse, systemName: String): Future[String] = {
    logger.debug(s"Received response from $systemName (status ${r.status.intValue})")
    if (r.status.isSuccess()) r.entity.dataBytes.runFold(ByteString.empty)(_ ++ _).map(_.utf8String)
    else
      r.entity.dataBytes.runFold(ByteString.empty)(_ ++ _).flatMap { body =>
        Future.failed(ExternalSystemError(systemName, r.status.intValue(), body.utf8String))
      }
  }

  def extractResponseBodyAsString(r: HttpResponse, systemName: String): Future[String] = {
    logger.debug(s"Received response from $systemName (status ${r.status.intValue})")
    if (r.status.isSuccess()) r.entity.dataBytes.runFold(ByteString.empty)(_ ++ _).map(_.utf8String)
    else
      r.entity.dataBytes.runFold(ByteString.empty)(_ ++ _).flatMap { body =>
        Future.failed(ExternalSystemError(systemName, r.status.intValue(), body.utf8String))
      }
  }

  def parseQueryParameters(query: String): Map[String, String] =
    if (query == null)
      Map()
    else
      query
        .split("&")
        .collect {
          case s"$key=$value" => key -> value
        }
        .toMap

  def parseUriParameters(uri: String): Map[String, String] = {
    val parts = uri split "\\?"
    if (parts.length > 1) {
      val query = parts(1)
      query
        .split("&")
        .collect {
          case s"$key=$value" => key -> value
        }
        .toMap
    }
    else Map.empty
  }
}
