package io.geoalert.mapflow.rest

import akka.http.scaladsl.server.Directives
import akka.http.scaladsl.server.Route
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import io.circe.generic.auto.exportDecoder

import io.geoalert.mapflow.rest.json.Decoders
import io.geoalert.mapflow.rest.json.ProcessingReviewDetailsJson
import io.geoalert.mapflow.rest.json.ProcessingReviewInputJson
import io.geoalert.mapflow.service.Services

object ProcessingReviewResource
    extends Directives
       with Authorization
       with RestImplicits
       with Decoders
       with Services {
  val acceptProcessing: Route =
    (path("processings" / JavaUUID / "acceptation") & put & authorized) { (processingId, user) =>
      toComplete(reviewService.acceptProcessing(processingId)(user))
    }

  val rejectProcessing: Route =
    (path("processings" / JavaUUID / "rejection") & put & authorized & entity(
      as[ProcessingReviewInputJson]
    )) { (processingId, user, input) =>
      toComplete(reviewService.rejectProcessing(processingId, input.comment, input.features)(user))
    }

  val getReview: Route = (path("processings" / JavaUUID / "review") & get & authorized) {
    (processingId, user) =>
      toComplete(
        reviewService.getProcessingReview(processingId)(user).map(ProcessingReviewDetailsJson(_))
      )
  }

  val getReviewFeatures: Route =
    (path("processings" / JavaUUID / "review_features") & get & authorized) {
      (processingId, user) =>
        toComplete(reviewService.getProcessingReviewFeatures(processingId)(user))
    }

  val routes: Route = concat(
    acceptProcessing,
    rejectProcessing,
    getReview,
    getReviewFeatures,
  )
}
