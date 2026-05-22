package io.geoalert.mapflow.service.keycloak

import cats.effect.IO
import com.typesafe.scalalogging.LazyLogging

class KeycloakService extends LazyLogging {
  val client: KeycloakClient = KeycloakClient()

  def createUser(email: String): IO[Boolean] = {
    logger.info(s"Creating new user in Keycloak $email")
    for {
      kcUser <- client.getKeycloakUser(email)
      userCreated <-
        if (kcUser.isDefined) {
          logger.info(s"User already exists in Keycloak $email")
          IO.pure(false)
        }
        else
          for {
            id <- client.createKeycloakUser(email)
            _ = logger.info(s"User successfully created in Keycloak $email $id")
          } yield true
    } yield userCreated
  }

  def disableUser(email: String): IO[String] = {
    logger.info(s"Disable Keycloak user $email")
    for {
      res <- client.disableKeycloakUser(email)
    } yield res

  }
}

object KeycloakService {
  def apply(): KeycloakService = new KeycloakService
}
