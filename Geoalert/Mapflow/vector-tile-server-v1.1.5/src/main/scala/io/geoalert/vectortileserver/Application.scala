package io.geoalert.vectortileserver

import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.model.HttpMethods
import akka.http.scaladsl.server.{Directives, Route}
import ch.megard.akka.http.cors.scaladsl.CorsDirectives.{cors, corsRejectionHandler}
import ch.megard.akka.http.cors.scaladsl.settings.CorsSettings
import com.typesafe.scalalogging.LazyLogging
import io.geoalert.vectortileserver.rest.RestRoute

import scala.concurrent.ExecutionContextExecutor
import scala.util.{Failure, Success}
import scala.concurrent.duration._

object Application extends App with Directives with LazyLogging {
  val corsSettings: CorsSettings = CorsSettings.defaultSettings.withAllowedMethods(
    List(HttpMethods.GET, HttpMethods.HEAD, HttpMethods.POST, HttpMethods.DELETE, HttpMethods.OPTIONS)
  )

  implicit val system: ActorSystem = ActorSystem("vector-tile-server")
  implicit val executionContext: ExecutionContextExecutor = system.dispatcher

  val routes: Route = Route.seal {
    handleRejections(corsRejectionHandler) {
      cors(corsSettings) {
        withRequestTimeout(60.minutes) {
          RestRoute.routes
        }
      }
    }
  }

  val bindingFuture = Http().newServerAt("0.0.0.0", 8080).bindFlow(routes)

  bindingFuture.onComplete {
    case Success(serverBinding) => logger.info(s"Vector Tile Server is listening to ${serverBinding.localAddress}")
    case Failure(error) => logger.error(s"Error: ${error.getMessage}")
  }
}
