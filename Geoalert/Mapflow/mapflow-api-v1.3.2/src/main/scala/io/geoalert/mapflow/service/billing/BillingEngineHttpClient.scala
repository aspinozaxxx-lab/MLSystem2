package io.geoalert.mapflow.service.billing

import java.util.UUID
import cats.effect.IO
import io.geoalert.mapflow.Config.billingDisabled
import io.geoalert.mapflow.TestEnvConfig

trait BillingEngineHttpClient extends TestEnvConfig {
  def getUserBalance(email: String): IO[UserBalanceJson]

  def credit(
      email: String,
      processingId: UUID,
      area: Long,
      credits: Long,
    ): IO[UUID]

  def confirmTransaction(processingId: UUID): IO[Unit]

  def discardTransaction(processingId: UUID): IO[Unit]
}

object BillingEngineHttpClient extends TestEnvConfig {
  private lazy val instance =
    if (testEnv || billingDisabled) MockBillingEngineHttpClient else ProductionBillingEngineHttpClient

  def apply(): BillingEngineHttpClient = instance
}
