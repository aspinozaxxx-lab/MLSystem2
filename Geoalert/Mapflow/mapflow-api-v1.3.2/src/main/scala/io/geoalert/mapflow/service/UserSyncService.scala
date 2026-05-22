package io.geoalert.mapflow.service

import java.time.Instant
import cats.effect.IO
import cats.syntax.option._
import com.typesafe.scalalogging.LazyLogging
import doobie.ConnectionIO
import io.geoalert.mapflow.Config.defaultAoiAreaLimit
import io.geoalert.mapflow.Config.defaultAreaLimit
import io.geoalert.mapflow.Config.defaultBillingType
import io.geoalert.mapflow.Config.defaultMemoryLimit
import io.geoalert.mapflow.model.BillingType
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.repo.UserDto
import io.geoalert.mapflow.repo.UserRepo

import java.util.UUID

/** Synchronize keycloak user with WM database
  */
class UserSyncService extends LazyLogging {
  /** Check if user with specified email exists in WM DB and create the user if necessary
    * @param email user email
    * @param role user role
    * @return
    */
  def synchronizeUser(
      email: String,
      role: Role,
      areaLimit: Option[Long] = None,
      billingType: Option[BillingType] = None,
      activeUntil: Option[Instant] = None,
      name: Option[String] = None,
      preferredUsername: Option[String] = None,
      avantpostUserId: Option[UUID] = None,
    ): ConnectionIO[UserDto] =
    for {
      maybe <- UserRepo.getUsersWithFilter(None, List(email).some, None).map(_.headOption)
      user <- maybe match {
        case None =>
          logger.info(s"Creating new user: $email")
          UserRepo.createUser(
            email,
            None,
            role,
            areaLimit.getOrElse(defaultAreaLimit),
            defaultAoiAreaLimit,
            defaultBillingType,
            defaultMemoryLimit,
            activeUntil,
            reviewWorkflowEnabled = false,
            name,
            preferredUsername,
            avantpostUserId,
          )
        case Some(user)
             if user.role != role.intVal || areaLimit.isDefined || billingType.isDefined || activeUntil.isDefined ||
               name.isDefined || preferredUsername.isDefined || avantpostUserId.isDefined =>
          logger.info(s"Synchronizing user: $email")
          for {
            _ <- UserRepo.updateUser(
              maybe.get.id,
              None,
              None,
              None,
              billingType,
              None,
              None,
              role.some,
              activeUntil,
              None,
              name,
              preferredUsername,
              avantpostUserId,
            )
            user <- UserRepo.getUser(maybe.get.id).rethrowT
          } yield user
        case Some(user) => IO.pure(user).to[ConnectionIO]
      }

    } yield user
}
