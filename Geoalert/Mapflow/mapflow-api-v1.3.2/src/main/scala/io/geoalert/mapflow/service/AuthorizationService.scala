package io.geoalert.mapflow.service

import java.time.Instant
import java.util.UUID

import scala.concurrent.ExecutionContext
import scala.concurrent.Future

import cats.syntax.either._
import cats.syntax.option._
import com.typesafe.scalalogging.LazyLogging
import doobie.implicits._
import io.circe.generic.extras.Configuration
import io.circe.generic.extras.ConfiguredJsonCodec
import io.circe.parser.decode
import io.circe.syntax._
import io.geoalert.mapflow.DefaultConfig
import io.geoalert.mapflow.exception._
import io.geoalert.mapflow.model._
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.service.AuthorizationService.AvanpostUser
import pdi.jwt.JwtAlgorithm
import pdi.jwt.JwtCirce
import pdi.jwt.JwtClaim
import pdi.jwt.JwtOptions
class AuthorizationService(userService: UserService) extends LazyLogging with DefaultConfig {
  implicit val ec: ExecutionContext = ExecutionContext.global

  val jwtTtlSeconds: Long = 7 * 24 * 60 * 60L

  val jwtOptions: JwtOptions = JwtOptions(signature = false)

  def decodeToken(token:  Either[AuthenticationError, AvanpostUser] =
    JwtCirce
      .decodeRaw(token, jwtOptions)
      .toEither
      .flatMap { claim =>
        logger.info(s"Decoded claim form token: 
        decode[AvanpostUser](claim)
      }
      .leftMap { error =>
        BadToken(error.getMessage)
      }

  def encodeToken(user: User): String = {
    val now = Instant.now()
    val payload = AvanpostUser(UUID.fromString(user.email), user.name, user.preferredUsername)

    val claim = JwtClaim(
      content = payload.asJson.noSpaces,
      expiration = now.plusSeconds(jwtTtlSeconds).getEpochSecond.some,
      issuedAt = now.getEpochSecond.some,
      issuer = "WM".some,
      subject = user.id.toString.some,
      notBefore = now.getEpochSecond.some,
    )

    JwtCirce.encode(claim, jwtKey, JwtAlgorithm.HS256)
  }

  /** Get user from WM database or create new user
    */
  def synchronize(
      email: String,
      role: Role,
      name: Option[String],
      preferredUsername: Option[String],
      avantpostUserId: Option[UUID],
    ): Future[User] =
    userService
      .synchronizeUser(email, role, name, preferredUsername, avantpostUserId)
      .transact(xa)
      .unsafeToFuture()

  def authenticate(login: String, pass: String): Future[Option[User]] =
    userService
      .authorize(login, pass)
      .toOption
      .transact(xa)
      .value
      .unsafeToFuture()
}

object AuthorizationService {
  @ConfiguredJsonCodec
  case class AvanpostUser(
      uid: UUID,
      name: Option[String],
      preferredUsername: Option[String],
    )

  object AvanpostUser {
    implicit val config: Configuration = Configuration.default.withSnakeCaseMemberNames
  }
  def apply(userService: UserService): AuthorizationService = new AuthorizationService(userService)
}
