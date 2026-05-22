package io.geoalert.mapflow.service

import java.time.Instant
import java.time.temporal.ChronoUnit
import java.util.UUID
import java.util.concurrent.Executors

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor
import scala.concurrent.duration._

import cats.data.EitherT
import cats.syntax.bifunctor._
import cats.syntax.option._
import cats.syntax.traverse._
import com.typesafe.scalalogging.LazyLogging
import doobie.free.connection.ConnectionIO
import doobie.implicits._

import io.geoalert.mapflow.AkkaSystem.system
import io.geoalert.mapflow.DefaultProcessingReviewConfig
import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.exception.InternalServerError
import io.geoalert.mapflow.exception.NotFound
import io.geoalert.mapflow.model.Permission
import io.geoalert.mapflow.model.ReviewStatus
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.repo.ProcessingRepo
import io.geoalert.mapflow.repo.ProcessingReviewDto
import io.geoalert.mapflow.repo.ProcessingReviewRepo
import io.geoalert.mapflow.service.billing.BillingService

import geotrellis.vector.io.json.JsonFeatureCollection

class ReviewService(billingService: BillingService)
    extends DefaultProcessingReviewConfig
       with LazyLogging {
  implicit private val ec: ExecutionContextExecutor =
    ExecutionContext.fromExecutor(Executors.newFixedThreadPool(2))

  def acceptProcessing(
      processingId: UUID
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, String] = {
    logger.info(
      s"Processing Review: ${user.email} accepted results for processing $processingId"
    )
    for {
      processing <- ProcessingRepo.getProcessing(
        processingId,
        user.userFilter(Permission.ConfirmProcessingReview),
      )
      _ <- EitherT.right[ApplicationError](
        billingService.confirmTransaction(processing)
      )
      _ <- EitherT.right[ApplicationError](
        ProcessingReviewRepo.update(processingId, ReviewStatus.Accepted)
      )
    } yield "OK"
  }

  def acceptProcessingUnsafe(
      processingId: UUID
    ): EitherT[ConnectionIO, ApplicationError, Unit] =
    for {
      processing <- ProcessingRepo
        .getProcessing(processingId, none, includeArchived = true)
        .leftWiden[ApplicationError]
      _ <- EitherT.right[ApplicationError](
        billingService.confirmTransaction(processing)
      )
      _ <- EitherT.right[ApplicationError](
        ProcessingReviewRepo.update(processingId, ReviewStatus.Accepted)
      )
    } yield {}

  def rejectProcessing(
      processingId: UUID,
      comment: String,
      features: Option[JsonFeatureCollection],
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, Unit] = {
    logger.info(
      s"Processing Review: ${user.email} rejected results for processing $processingId"
    )
    for {
      processing <- ProcessingRepo.getProcessing(
        processingId,
        user.userFilter(Permission.ConfirmProcessingReview),
      )
      _ <- EitherT.right[ApplicationError](
        ProcessingReviewRepo.update(
          processing.id,
          ReviewStatus.NotAccepted,
          comment.some,
          features,
        )
      )
    } yield {}
  }

  def returnToReview(
      processingId: UUID
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, String] = {
    Validations.checkPermission(user, Permission.ConfirmProcessingReview)
    for {
      currentStatus <- EitherT.right[ApplicationError](
        ProcessingReviewRepo.get(processingId).map(_.map(_.reviewStatus))
      )
      _ = logger.info(
        s"Processing Review: ${user.email} returned processing $processingId to IN_REVIEW status"
      )
      result <-
        if (currentStatus.contains(ReviewStatus.NotAccepted))
          EitherT.right[ApplicationError](
            ProcessingReviewRepo
              .update(processingId, ReviewStatus.InReview)
              .map(_ => "OK")
          )
        else {
          logger.error(
            s"Processing Review: ${user.email} failed to return processing $processingId to IN_REVIEW. Status: $ReviewStatus"
          )
          EitherT.leftT[ConnectionIO, String](
            InternalServerError(
              "Processing return to review was failed"
            ): ApplicationError
          )
        }
    } yield result

  }

  def reviewProcessing(processingId: UUID): ConnectionIO[Unit] = {
    logger.info(
      s"Processing Review: completed processing $processingId moved to IN_REVIEW status"
    )
    ProcessingReviewRepo.create(processingId).map(_ -> {})
  }

  def getProcessingReview(
      processingId: UUID
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, ProcessingReviewDto] =
    for {
      processing <- ProcessingRepo.getProcessing(
        processingId,
        user.userFilter(Permission.ViewAnyProject),
      )
      review <- EitherT.fromOptionF(
        ProcessingReviewRepo.get(processing.id),
        NotFound(
          s"Cannot find review status for processing $processingId"
        ): ApplicationError,
      )
    } yield review

  def getProcessingReviewFeatures(
      processingId: UUID
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, JsonFeatureCollection] =
    for {
      processing <- ProcessingRepo.getProcessing(
        processingId,
        user.userFilter(Permission.ViewAnyProject),
      )
      features <- EitherT.fromOptionF(
        ProcessingReviewRepo.getProcessingReviewFeatures(processing.id),
        NotFound(
          s"Cannot find review features for processing $processingId"
        ): ApplicationError,
      )
    } yield features

  def refund(
      processingId: UUID
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, String] = {
    Validations.checkPermission(user, Permission.ConfirmProcessingReview)
    logger.info(
      s"Processing Review: ${user.email} confirmed refund for processing $processingId"
    )
    for {
      processing <- ProcessingRepo
        .getProcessing(processingId, none, includeArchived = true)
        .leftWiden[ApplicationError]
      _ <- EitherT.right[ApplicationError](
        billingService.discardTransaction(processing)
      )
      _ <- EitherT.right[ApplicationError](
        ProcessingReviewRepo.update(processingId, ReviewStatus.Refunded)
      )
    } yield "OK"
  }

  // TODO: Download geometry

  def autoAcceptProcessings(): Unit = {
    val io = for {
      processingIds <- ProcessingReviewRepo.listProcessingsInReviewBefore(
        Instant
          .now()
          .minus(autoConfirmProcessingsInterval.toSeconds, ChronoUnit.SECONDS)
      )
      _ = if (processingIds.nonEmpty)
        logger.info(
          s"Following processings were in review for $autoConfirmProcessingsInterval and will be automatically accepted: ${processingIds
              .mkString(",")}"
        )
      _ <- processingIds.traverse(id => acceptProcessingUnsafe(id).rethrowT)
    } yield {}

    io.transact(xa).unsafeRunSync()
  }

  def scheduleUpdates(): Unit =
    system
      .scheduler
      .scheduleWithFixedDelay(
        0.seconds,
        autoConfirmCheckInterval,
      ) { () =>
        try
          autoAcceptProcessings()
        catch {
          case e: Exception =>
            logger.error(s"Error while updating review status", e)
        }
      }
}
