package io.geoalert.mapflow.service.keycloak

import java.util.UUID

import cats.effect.IO

object MockKeycloakClient extends KeycloakClient {
  override def getKeycloakUser(email: String): IO[Option[String]] = IO.pure(None)

  override def createKeycloakUser(email: String): IO[String] = IO.pure(UUID.randomUUID().toString)

  override def disableKeycloakUser(email: String): IO[String] = IO.pure("OK")
}
