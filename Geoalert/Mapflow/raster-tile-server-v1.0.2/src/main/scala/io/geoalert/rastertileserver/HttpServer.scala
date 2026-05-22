package io.geoalert.rastertileserver

import akka.http.scaladsl.Http
import akka.http.scaladsl.model.HttpMethods
import akka.http.scaladsl.server.{Directives, ExceptionHandler, Route}
import ch.megard.akka.http.cors.scaladsl.CorsDirectives.{cors, corsRejectionHandler}
import ch.megard.akka.http.cors.scaladsl.settings.CorsSettings
import com.typesafe.scalalogging.LazyLogging
import io.geoalert.rastertileserver.AkkaSystem.system
import io.geoalert.rastertileserver.Config.{port, minioHost, minioPort, minioEndpoint}
import io.geoalert.rastertileserver.exception.{ApplicationError, InternalServerError, NotFound, UserError}
import io.geoalert.rastertileserver.rest.RestRoute
import io.geoalert.rastertileserver.rest.model.{HttpError, HttpErrorSupport}

import scala.concurrent.ExecutionContext
import scala.util.{Failure, Success}
import scala.concurrent.duration._


object HttpServer extends LazyLogging with Directives with HttpErrorSupport {
  private val corsSettings: CorsSettings = CorsSettings.defaultSettings.withAllowedMethods(
    List(HttpMethods.GET, HttpMethods.HEAD, HttpMethods.POST, HttpMethods.DELETE, HttpMethods.OPTIONS)
  )

  implicit val executionContext:ExecutionContext = system.dispatcher

  val route: Route = Route.seal {
    handleRejections(corsRejectionHandler) {
      cors(corsSettings) {
        withRequestTimeout(1.minute) {
          RestRoute.routes
        }
      }
    }
  }

  implicit def exceptionHandler: ExceptionHandler = ExceptionHandler { case e =>
    (cors(corsSettings) & extractUri) { uri =>
      val appErr: ApplicationError = e match {
        case appErr: NotFound =>
          appErr
        case err: UserError =>
          logger.warn(s"Bad request $uri:", e)
          err
        case appErr: ApplicationError =>
          logger.error(s"Error processing request to $uri:", e)
          appErr
        case _ =>
          logger.error(s"Error processing request to $uri:", e)
          InternalServerError(e.toString): ApplicationError
      }
      val body = HttpError(appErr.code, appErr.getMessage)
      complete(appErr.statusCode -> body)
    }
  }

  def apply(): Unit = {
    val bindingFuture = Http()
      .newServerAt("0.0.0.0", port)
      .bindFlow(route)

    bindingFuture.onComplete {
      case Success(serverBinding) => logger.info(s"Raster Tile Server is listening to ${serverBinding.localAddress}. " +
        s"minioEndpoint: $minioEndpoint, minioHost: $minioHost, minioPort: $minioPort")
      case Failure(error) => logger.error(error.toString)
    }
  }
}
