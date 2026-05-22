package io.geoalert.mapflow.service.keycloak

import java.net.URL
import java.util.concurrent.Executors

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor
import scala.concurrent.Future
import scala.concurrent.duration._
import scala.util.Failure

import akka.http.scaladsl.client.RequestBuilding.Get
import akka.http.scaladsl.client.RequestBuilding.Post
import akka.http.scaladsl.client.RequestBuilding.Put
import akka.http.scaladsl.model.FormData
import akka.http.scaladsl.model.HttpRequest
import akka.http.scaladsl.model.HttpResponse
import akka.http.scaladsl.model.headers.Authorization
import akka.http.scaladsl.model.headers.OAuth2BearerToken
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
import io.geoalert.mapflow.KeycloakConfig
import io.geoalert.mapflow.util.HttpUtils

case class OidcTokenResponse(access_token: 

//https://www.keycloak.org/docs-api/15.0/rest-api/index.html#_userrepresentation
case class UserRepresentation(
    id: Option[String],
    email: Option[String],
    username: Option[String],
    enabled: Option[Boolean],
    emailVerified: Option[Boolean],
    requiredActions: Option[List[String]],
  )

object ProductionKeycloakClient extends KeycloakClient with LazyLogging with KeycloakConfig {
  implicit private val executionContext: ExecutionContextExecutor = ExecutionContext.global

  implicit private lazy val cs: ContextShift[IO] = IO.contextShift(
    ExecutionContext.fromExecutor(
      Executors.newFixedThreadPool(
        4,
        new ThreadFactoryBuilder().setNameFormat("keycloak-client-%d").build(),
      )
    )
  )

  private val connectionFlow = HttpUtils.httpConnection(new URL(keycloakUrl))
  override def getKeycloakUser(email: String): IO[Option[String]] =
    for {
      token <- authenticateClient()
      users <- findUsers(token, email)
      ids = users.flatMap(_.id)
    } yield ids.headOption

  override def createKeycloakUser(email: String): IO[String] =
    for {
      token <- authenticateClient()
      _ <- createNewUser(token, email)
      users <- findUsers(token, email)
      id = users.flatMap(_.id).head
      _ <- resetPassword(token, id)
    } yield id

  override def disableKeycloakUser(email: String): IO[String] =
    for {
      token <- authenticateClient()
      users <- findUsers(token, email)
      user = users.head
      _ <- disableUser(token, user.id.get)
    } yield "OK"

  private def resetPassword(token: , id: String): IO[Unit] = {
    val future = for {
      _ <- runRequest(
        Put(
          s"/auth/admin/realms/$keycloakRealm/users/$id/execute-actions-email",
          List("UPDATE_PASSWORD"),
        )
          .withHeaders(Authorization(OAuth2BearerToken(token)))
      )
    } yield {}

    IO.fromFuture(IO(future))

  }

  private def createNewUser(token: , email: String): IO[Unit] = {
    val user = UserRepresentation(
      None,
      email.some,
      email.some,
      true.some,
      false.some,
      List("VERIFY_EMAIL", "UPDATE_PASSWORD").some,
    )
    val future = for {
      _ <- runRequest(
        Post(s"/auth/admin/realms/$keycloakRealm/users?username=$email&exact=true", user)
          .withHeaders(Authorization(OAuth2BearerToken(token)))
      )
    } yield {}

    IO.fromFuture(IO(future))
  }

  private def findUsers(token: , email: String): IO[List[UserRepresentation]] = {
    val future = for {
      response <- runRequest(
        Get(s"/auth/admin/realms/$keycloakRealm/users?username=$email&exact=true")
          .withHeaders(Authorization(OAuth2BearerToken(token)))
      )
      res <- HttpUtils.parseResponse[List[UserRepresentation]](response, "KC")
    } yield res

    IO.fromFuture(IO(future))
  }

  private def disableUser(token: , id: String): IO[List[UserRepresentation]] = {
    val user = UserRepresentation(None, None, None, false.some, None, None)
    val future = for {
      response <- runRequest(
        Get(s"/auth/admin/realms/$keycloakRealm/users/$id", user)
          .withHeaders(Authorization(OAuth2BearerToken(token)))
      )
      res <- HttpUtils.parseResponse[List[UserRepresentation]](response, "KC")
    } yield res

    IO.fromFuture(IO(future))
  }

  private def authenticateClient(): IO[String] = {
    val req = FormData(
      Map(
        "grant_type" -> "client_credentials",
        "client_id" -> keycloakManagementClientId,
        "client_secret" -> keycloakManagementClientSecret,
      )
    ).toEntity

    val url = s"/auth/realms/$keycloakRealm/protocol/openid-connect/token"
    logger.debug(s"Requesting KC token: 

    val future = for {
      response <- runRequest(Post(url).withEntity(req))
      res <- HttpUtils.parseResponse[OidcTokenResponse](response, "KC")
    } yield res.access_token

    IO.fromFuture(IO(future))
  }

  private def runRequest(request: HttpRequest): Future[HttpResponse] = {
    val requestFuture = Source
      .single(request)
      .via(connectionFlow)
      .runWith(Sink.head)

    requestFuture.onComplete {
      case Failure(e) => logger.error(s"Error trying to call Keycloak", e)
      case _ =>
    }

    Future.firstCompletedOf(
      Seq(
        akka
          .pattern
          .after(3.minutes, system.scheduler)(
            Future.failed(new RuntimeException("Keycloak request timeout"))
          ),
        requestFuture,
      )
    )
  }
}
