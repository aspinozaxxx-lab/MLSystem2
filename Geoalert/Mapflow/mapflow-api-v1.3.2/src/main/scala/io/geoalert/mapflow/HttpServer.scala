package io.geoalert.mapflow

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.util.Failure
import scala.util.Success

import akka.http.scaladsl.Http
import akka.http.scaladsl.model._
import akka.http.scaladsl.server.Directives
import akka.http.scaladsl.server.ExceptionHandler
import akka.http.scaladsl.server.Route
import ch.megard.akka.http.cors.scaladsl.CorsDirectives.cors
import ch.megard.akka.http.cors.scaladsl.settings.CorsSettings
import com.typesafe.scalalogging.LazyLogging
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import io.circe.generic.auto._

import io.geoalert.mapflow.AkkaSystem._
import io.geoalert.mapflow.exception._
import io.geoalert.mapflow.graphql.GraphQLRoute
import io.geoalert.mapflow.rest.RestRoute
import io.geoalert.mapflow.rest.json.HttpError

object HttpServer extends LazyLogging with MiscConfig with Directives {
  val corsSettings: CorsSettings = CorsSettings
    .defaultSettings
    .withAllowedMethods(
      List(
        HttpMethods.GET,
        HttpMethods.HEAD,
        HttpMethods.POST,
        HttpMethods.PUT,
        HttpMethods.DELETE,
        HttpMethods.OPTIONS,
      )
    )

  implicit def exceptionHandler: ExceptionHandler = ExceptionHandler {
    case e =>
      extractRequest { request =>
        e match {
          case err: UserError =>
            logger.debug(s"Bad request ${request.method} ${request.uri}:", err)
          case _: AuthenticationError =>
          case err: ApplicationError =>
            logger.warn(s"Error processing request to ${request.method} ${request.uri}", err)
          case _ =>
            logger.error(s"Unexpected error ${request.method} ${request.uri}", e)
        }

        e match {
          case e: AuthenticationError =>
            val body = HttpError(e.code, e.getMessage)
            deleteCookie("token") & complete(e.statusCode -> body)
          case e: ApplicationError =>
            val body = HttpError(e.code, e.getMessage)
            complete(e.statusCode -> body)
          case _ =>
            val err = InternalServerError("Unexpected server error")
            val body = HttpError(err.code, err.getMessage)
            complete(err.statusCode -> body)
        }
      }
  }

  val route: Route = Route.seal {
    cors(corsSettings) {
      withRequestTimeout(60.minutes) {
        concat(
          GraphQLRoute.routes,
          RestRoute.publicApiRoutes,
          RestRoute.internalApiRoutes,
        )
      }
    }
  }

  def apply(): Unit = {
    val bindingFuture = Http().newServerAt("0.0.0.0", port).bindFlow(route)

    bindingFuture.onComplete {
      case Success(serverBinding) =>
        println(s"Http server is listening to ${serverBinding.localAddress}")
      case Failure(error) => println(s"Error: ${error.getMessage}")
    }
  }
}
