package io.geoalert.mapflow

object Config extends DefaultConfig {
  override val testData: Option[String] = None

  override val testEnv = true

  override val maxInFrLen = 4

  override val enableTelegramNotificationAboutFailedProcessings = false
}
