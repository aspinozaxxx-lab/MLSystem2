package io.geoalert.mapflow.service.billing

import java.util.UUID

import cats.effect.IO
import cats.syntax.option._

object MockBillingEngineHttpClient extends BillingEngineHttpClient {
  override def getUserBalance(email: String): IO[UserBalanceJson] =
    IO.pure(UserBalanceJson(none, email, 0L, 0L))

  override def credit(
      email: String,
      processingId: UUID,
      area: Long,
      credits: Long,
    ): IO[UUID] =
    IO.pure(UUID.randomUUID())

  override def confirmTransaction(processingId: UUID): IO[Unit] =
    IO.pure {}

  override def discardTransaction(processingId: UUID): IO[Unit] =
    IO.pure {}
}
