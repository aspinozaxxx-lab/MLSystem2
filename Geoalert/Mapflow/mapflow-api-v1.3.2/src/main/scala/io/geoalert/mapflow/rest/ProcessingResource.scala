package io.geoalert.mapflow.rest

import scala.util.Failure
import scala.util.Success
import java.util.UUID

import _root_.io.geoalert.mapflow.exception.ApplicationError
import _root_.io.geoalert.mapflow.graphql.args.processing.ProcessingFilters
import _root_.io.geoalert.mapflow.model.Processing
import _root_.io.geoalert.mapflow.model.UpdateProcessingInput
import akka.http.scaladsl.model.ContentTypes
import akka.http.scaladsl.model.HttpEntity
import akka.http.scaladsl.model.headers.RawHeader
import akka.http.scaladsl.server.Directives
import akka.http.scaladsl.server.Route
import akka.stream.scaladsl.Source
import cats.implicits.toBifunctorOps
import cats.implicits.toTraverseOps
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import doobie.implicits._
import io.circe.generic.auto.exportDecoder
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.rest.json.CreateAndRunProcessingJson
import io.geoalert.mapflow.rest.json.CreateProcessingRatingJson
import io.geoalert.mapflow.rest.json.ProcessingJson
import io.geoalert.mapflow.rest.json.UpdateProcessingInputJson
import io.geoalert.mapflow.service.Services

import geotrellis.vector._

object ProcessingResource extends Directives with Authorization with RestImplicits with Services {
  private val downloadHeaders =
    List(RawHeader("Content-Disposition", "attachment; filename=features.geojson"))

  def retrieveProjectProcessings: Route =
    (path("projects" / JavaUUID / "processings") & get & authorized) { (projectId, user) =>
      toComplete(projectService.getProjectProcessings(projectId)(user))
    }

  def retrieveDefaultProjectProcessings: Route =
    (path("projects" / "default" / "processings") & get & authorized) { user =>
      toComplete(projectService.getDefaultPrjProcessings(user))
    }

  def retrieveProcessing: Route = (path("processings" / JavaUUID) & get & authorized) {
    (id, user) =>
      toComplete(
        processingService
          .getProcessing(id)(user)
          .leftWiden[ApplicationError]
          .semiflatMap(ProcessingJson(_))
      )
  }

  def downloadProcessingAsCsv: Route =
    (path("processings" / "stats") & post & authorized & parameters(Symbol("type").?) & entity(
      as[ProcessingFilters]
    )) {
      (
          _,
          responseType,
          filter,
      ) =>
        responseType.map(_.toLowerCase) match {
          case Some("json") =>
            toComplete(
              processingService.getProcessings(filter).map(_.results.map(_.toProcessingDetails))
            )
          case _ =>
            val headers = Source.single(Processing.CsvHeaders)
            val processingSource = processingService
              .getProcessings(filter)
              .map(processings => Source(processings.results.map(_.toCSVField)))
            completeAsCsv(processingSource.map(source => headers.concat(source)))
        }
    }

  def listProcessings: Route = (path("processings") & get & authorized) { user =>
    toComplete(
      processingService
        .getProcessings()(user)
        .flatMap(_.traverse(ProcessingJson(_)))
    )
  }

  def createAndRunProcessing: Route =
    (path("processings") & post & authorized & entity(as[CreateAndRunProcessingJson])) {
      (user, input) =>
        toComplete(runProcessingService.createAndRun(input)(user))
    }

  def updateProcessing: Route =
    (path("processings" / JavaUUID) & put & authorized & entity(as[UpdateProcessingInputJson])) {
      (
          id,
          user,
          input,
      ) =>
        toComplete(
          processingService
            .updateProcessing(
              UpdateProcessingInput(
                id,
                name = input.name,
                description = input.description,
                projectId = input.projectId,
              )
            )(user)
            .flatMap(ProcessingJson(_))
        )
    }

  def archiveProcessing: Route = (path("processings" / JavaUUID) & delete & authorized) {
    (id, user) =>
      toComplete(processingService.archiveProcessing(id)(user))
  }

  def cancelProcessing: Route = (path("processings" / JavaUUID / "cancel") & post & authorized) {
    (id, user) =>
      toComplete(processingService.cancelProcessing(id)(user))
  }

  def restartProcessing: Route = (path("processings" / JavaUUID / "restart") & post & authorized) {
    (id, user) =>
      toComplete(runProcessingService.restartProcessing(id)(user))
  }

  val createProcessingRate: Route =
    (path("processings" / JavaUUID / "rate") & put & authorized & entity(
      as[CreateProcessingRatingJson]
    )) {
      (
          id,
          user,
          rating,
      ) =>
        toComplete(processingService.rateProcessing(id, rating.rating, rating.feedback)(user))
    }

  // Deprecated. Will be removed in API 2
  def listProcessingAois: Route = (path("processings" / JavaUUID / "aois") & get & authorized) {
    (id, user) =>
      toComplete(projectService.getProcessingAois(id)(user))
  }

  def downloadResults: Route =
    path("processings" / JavaUUID / "result") {
      id =>
        (get & authorized) { user =>
          parameters(Symbol("aoiId").as[UUID].?) { aoiIdOpt =>
            respondWithHeaders(downloadHeaders) {
              onComplete(
                resultService.downloadResult(id, aoiIdOpt).transact(xa).unsafeToFuture()
              ) {
                case Success(v) => complete(HttpEntity.Chunked(ContentTypes.`application/octet-stream`, v))
                case Failure(ex) => failWith(ex)
              }
            }
          }
        }
    }

  val routes: Route = concat(
    retrieveProjectProcessings,
    retrieveDefaultProjectProcessings,
    retrieveProcessing,
    listProcessings,
    createAndRunProcessing,
    archiveProcessing,
    restartProcessing,
    listProcessingAois,
    downloadResults,
    updateProcessing,
    downloadProcessingAsCsv,
    createProcessingRate,
  )
}
