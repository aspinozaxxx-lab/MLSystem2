package io.geoalert.mapflow.service.keycloak

import cats.effect.IO

import io.geoalert.mapflow.TestEnvConfig

trait KeycloakClient {
  def getKeycloakUser(email: String): IO[Option[String]]

  def createKeycloakUser(email: String): IO[String]

  def disableKeycloakUser(email: String): IO[String]
}

object KeycloakClient extends TestEnvConfig {
  private lazy val instance = if (testEnv) MockKeycloakClient else ProductionKeycloakClient

  def apply(): KeycloakClient = instance
}
