package io.geoalert.mapflow.rest

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor
import scala.concurrent.Future
import scala.concurrent.duration._
import scala.util.Failure
import scala.util.Success
import akka.http.scaladsl.server.Directive1
import akka.http.scaladsl.server.Directives
import cats.data.EitherT
import cats.implicits.catsSyntaxOptionId
import com.github.blemale.scaffeine.AsyncCache
import com.github.blemale.scaffeine.Scaffeine
import com.typesafe.scalalogging.LazyLogging
import io.geoalert.mapflow.Config.avanpostAdminGroupIds
import io.geoalert.mapflow.Config.avanpostUserGroupIds
import io.geoalert.mapflow.exception.AccessDenied
import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.exception.AuthenticationError
import io.geoalert.mapflow.exception.InternalServerError
import io.geoalert.mapflow.exception.NoToken
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.service.Services
import io.geoalert.mapflow.service.avanpost.responses.UserInfo

trait Authorization extends Directives with Services with LazyLogging {
  implicit private val executionContext: ExecutionContextExecutor = ExecutionContext.global

  val cache: AsyncCache[String, Option[User]] = Scaffeine()
    .expireAfterWrite(10.minutes)
    .maximumSize(1000)
    .buildAsync[String, Option[User]]()

  private def getToken(auth: Option[String]): Option[String] =
    auth.map("(?i)^\\s*bearer\\s*".r.replaceFirstIn(_, "").trim)

  private def getRoleByGroup(userInfo: UserInfo): Either[ApplicationError, Role] = {
    val userGroupIds = userInfo.userGroups.map(_.id)
    val hasAdminRole = userGroupIds.intersect(avanpostAdminGroupIds).nonEmpty
    val hasUserRole = userGroupIds.intersect(avanpostUserGroupIds).nonEmpty
    logger.info(s"""UserGroupIds: $userGroupIds
         Expected avanpost admin groupIds: $avanpostAdminGroupIds
         Expected avanpost user groupIds: $avanpostUserGroupIds
         hasAdminRole: $hasAdminRole
         hasUserRole: $hasUserRole
         """)
    Option
      .when[Role](hasAdminRole)(Role.Admin)
      .orElse(Option.when(hasUserRole)(Role.User))
      .toRight(AccessDenied("User group not matched"))
  }

  private def authorizeJwt(header: String): Future[Either[ApplicationError, User]] = {
    val authTask = for {
      token <- EitherT.fromOption[Future](getToken(Some(header)), NoToken())
      avantpostUser <- EitherT.fromEither[Future](authorizationService.decodeToken(token))
      _ = logger.info(s"Parsed avanpost user from token: 
      role <- EitherT(avanpostClient.userInfo(token, avantpostUser.uid).map(getRoleByGroup))
      user <- EitherT.liftF[Future, ApplicationError, User](
        authorizationService.synchronize(
          avantpostUser.uid.toString,
          role,
          avantpostUser.name,
          avantpostUser.preferredUsername,
          avantpostUser.uid.some,
        )
      )
    } yield user
    authTask.value
  }

  def authorized: Directive1[User] =
    optionallyAuthorized.flatMap {
      case Some(user) => provide(user)
      case None => failWith(new AuthenticationError())
    }

  private def optionallyAuthorized: Directive1[Option[User]] = {
    // User can either be authenticated by AVANPOST_TOKEN cookie OR "Authorization: basic" header
    val decl = optionalCookie("AVANPOST_TOKEN").map(_.map(_.value)) &
      optionalHeaderValueByName("Authorization")

    decl.tflatMap { cookieHeader =>
      val future = cookieHeader match {
        case (Some(value: String), _) => authorizeJwt(value)
        case (None, Some(value: String)) if value.toLowerCase().startsWith("bearer") =>
          authorizeJwt(value)
        case _ => Future.successful(Left(NoToken()))
      }

      onComplete(future).flatMap {
        case Success(Right(user)) => provide(Some(user))
        case Success(Left(error)) => failWith(error)
        case Failure(exception: AuthenticationError) =>
          failWith(exception)
        case Failure(exception: AccessDenied) =>
          logger.debug("Access Denied", exception)
          failWith(exception)
        case Failure(exception: Exception) =>
          logger.error("Unexpected authorization error", exception)
          failWith(InternalServerError("Unexpected authorization error"))
        case _ => failWith(new IllegalStateException("Unexpected authentication state"))
      }
    }
  }
}
