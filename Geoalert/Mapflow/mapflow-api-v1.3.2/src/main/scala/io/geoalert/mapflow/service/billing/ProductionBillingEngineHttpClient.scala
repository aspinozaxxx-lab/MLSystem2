package io.geoalert.mapflow.service.billing

import java.net.URL
import java.util.UUID
import java.util.concurrent.Executors

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor
import scala.concurrent.Future
import scala.concurrent.duration._
import scala.util.Failure

import akka.http.scaladsl.client.RequestBuilding.Delete
import akka.http.scaladsl.client.RequestBuilding.Get
import akka.http.scaladsl.client.RequestBuilding.Post
import akka.http.scaladsl.client.RequestBuilding.Put
import akka.http.scaladsl.model.HttpRequest
import akka.http.scaladsl.model.HttpResponse
import akka.http.scaladsl.model.StatusCodes
import akka.http.scaladsl.model.headers.RawHeader
import akka.stream.scaladsl.Sink
import akka.stream.scaladsl.Source
import cats.effect.ContextShift
import cats.effect.IO
import cats.syntax.option._
import com.google.common.util.concurrent.ThreadFactoryBuilder
import com.typesafe.scalalogging.LazyLogging
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import io.circe.generic.auto._

import io.geoalert.mapflow.AkkaSystem.system
import io.geoalert.mapflow.DefaultExternalSystemConfig
import io.geoalert.mapflow.util.HttpUtils

case class UserBalanceJson(
    id: Option[UUID],
    email: String,
    usedCredits: Long,
    remainingCredits: Long,
  )

case class CreditTransactionJson(
    processingId: UUID,
    credits: Long,
    area: Long,
    system: String,
    comment: Option[String],
  )

object ProductionBillingEngineHttpClient
    extends BillingEngineHttpClient
       with DefaultExternalSystemConfig
       with LazyLogging {
  implicit private val executionContext: ExecutionContextExecutor = ExecutionContext.global

  private val authHeader = List(RawHeader("x-api-key", billingEngineApiKey))

  implicit private lazy val cs: ContextShift[IO] = IO.contextShift(
    ExecutionContext.fromExecutor(
      Executors.newFixedThreadPool(
        8,
        new ThreadFactoryBuilder().setNameFormat("billing-engine-%d").build(),
      )
    )
  )

  private val connectionFlow = HttpUtils.httpConnection(new URL(billingEngineUrl))

  logger.info(s"Billing Engine connection is configured at $billingEngineUrl")

  override def getUserBalance(email: String): IO[UserBalanceJson] = {
    val url = s"$billingEngineUrl/api/v0/user/$email/balance"
    logger.debug(s"Get user balance from Billing Engine: GET $url")
    val future = for {
      response <- runRequest(
        Get(url)
          .withHeaders(authHeader)
      )
      balance <-
        if (response.status == StatusCodes.NotFound) {
          logger.warn(
            s"Billing Engine returned 404 NOT FOUND for $email. Either user not found in BE or BE is unavailable at $url"
          )
          Future.successful(UserBalanceJson(none, email, 0, 0))
        }
        else
          HttpUtils.parseResponse[UserBalanceJson](response, "BE")
    } yield balance

    IO.fromFuture(IO(future))
  }

  override def credit(
      email: String,
      processingId: UUID,
      area: Long,
      credits: Long,
    ): IO[UUID] = {
    val url = s"$billingEngineUrl/api/v0/user/$email/credit"
    val payload = CreditTransactionJson(
      processingId,
      credits,
      area,
      "WM",
      "Charging user account for a processing".some,
    )
    logger.debug(s"Create CREDIT transaction in Billing Engine: POST $url, $payload")

    val future = for {
      response <- runRequest(
        Post(url, payload)
          .withHeaders(authHeader)
      )
      transactionId <- HttpUtils.parseResponse[UUID](response, "BE")
    } yield transactionId

    IO.fromFuture(IO(future))
  }

  override def confirmTransaction(processingId: UUID): IO[Unit] = {
    val url = s"$billingEngineUrl/api/v0/processing/$processingId/transaction/confirmation"

    logger.debug(s"Confirm CREDIT transactions in Billing Engine: PUT $url")
    val future = for {
      response <- runRequest(
        Put(url)
          .withHeaders(authHeader)
      )
      txIds <- HttpUtils.parseResponse[List[UUID]](response, "BE")
      _ = logger.info(
        s"Confirmed transactions ${txIds.mkString(", ")} for processing $processingId"
      )
    } yield {}

    IO.fromFuture(IO(future))
  }

  override def discardTransaction(processingId: UUID): IO[Unit] = {
    val url = s"$billingEngineUrl/api/v0/processing/$processingId/transaction"
    logger.debug(s"Discard CREDIT transactions in Billing Engine: DELETE $url")
    val future = for {
      response <- runRequest(
        Delete(url)
          .withHeaders(authHeader)
      )
      txIds <- HttpUtils.parseResponse[List[UUID]](response, "BE")
      _ = logger.info(
        s"Discarded transactions ${txIds.mkString(", ")} for processing $processingId"
      )
    } yield {}

    IO.fromFuture(IO(future))
  }

  private def runRequest(request: HttpRequest): Future[HttpResponse] = {
    val requestFuture = Source
      .single(request)
      .via(connectionFlow)
      .runWith(Sink.head)

    requestFuture.onComplete {
      case Failure(e) => logger.error(s"Error trying to call Billing Engine", e)
      case _ =>
    }

    Future.firstCompletedOf(
      Seq(
        akka
          .pattern
          .after(1.minute, system.scheduler)(
            Future.failed(new RuntimeException("Billing Engine request timeout"))
          ),
        requestFuture,
      )
    )
  }
}
