package io.geoalert.rastertileserver.rest

import akka.http.scaladsl.model.{ContentType, HttpEntity, MediaTypes, StatusCodes}
import akka.http.scaladsl.server.Directives._
import akka.http.scaladsl.server.Route
import com.typesafe.scalalogging.LazyLogging
import io.geoalert.rastertileserver.cog.CogService
import io.geoalert.rastertileserver.rest.model.TileJsonSupport
import io.prometheus.client.{Counter, Histogram}

import java.net.URI
import scala.concurrent.ExecutionContext
import scala.util.{Failure, Success}

object CogResource extends TileJsonSupport with LazyLogging {
  val cogService: CogService = CogService()

  val tileLatency: Histogram = Histogram.build()
    .name("tile_latency_seconds")
    .help("single tile request latency in seconds")
    .register()

  val tilesCount: Counter = Counter.build()
    .name("tile_requests")
    .help("Number of tiles requests")
    .register()

  val errorsCount: Counter = Counter.build()
    .name("tile_request_errors")
    .help("Number of errors in tiles requests")
    .register()

  implicit val ec: ExecutionContext = ExecutionContext.global

  val singleTile: Route = (path("cogs" / "tiles" / IntNumber / IntNumber / IntNumber ~ ".png") & get &
    parameters(
      Symbol("uri").as[String],
      Symbol("mask_uri").as[String].optional,
    )) { (z, x, y, uri, maskUri) =>

    val timer = tileLatency.startTimer()
    val future = cogService.getTile(x, y, z, new URI(uri), maskUri)

    onComplete(future)  {
      case Success(Some(data)) =>
        timer.observeDuration()
        complete(HttpEntity(ContentType(MediaTypes.`image/png`), data))
      case Success(None) =>
        timer.observeDuration()
        complete(StatusCodes.NoContent)
      case Failure(ex) =>
        logger.error("Failed to load tile", ex)
        errorsCount.inc()
        timer.observeDuration()
        complete(StatusCodes.InternalServerError)
    }
  }

  val tilesJson: Route = (path("cogs" / "tiles.json") & get & parameter("uri")) { uri =>
    complete(cogService.getTileJson(new URI(uri)))
  }

  val bounds: Route = (path("cogs" / "bounds") & get & parameter("uri")) { uri =>
    complete(HttpEntity(cogService.getBounds(new URI(uri))))
  }

  val routes: Route = concat(singleTile, tilesJson, bounds)
}
