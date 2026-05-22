package io.geoalert.mapflow.util

import java.time.Instant

import scala.concurrent.ExecutionContext

import cats.syntax.option._
import doobie.implicits._
import io.geoalert.mapflow.Config.defaultAoiAreaLimit
import io.geoalert.mapflow.Config.defaultAreaLimit
import io.geoalert.mapflow.model.BillingType
import io.geoalert.mapflow.model.CreateUserInput
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.service.Services
import io.geoalert.mapflow.service.avanpost.AvanpostClient.AvanpostTestAdminId

object UserUtil extends Services {
  implicit val ec: ExecutionContext = ExecutionContext.global

  def regularUser: User =
    userService
      .synchronizeUser("bb855a4b-e109-410c-b00d-8455ba6af790", Role.User, None, None, None)
      .transact(xa)
      .unsafeRunSync()

  def otherUser: User =
    userService
      .synchronizeUser("f5bdd727-cc7e-40d1-b0f2-ced1e1db57f4", Role.User, None, None, None)
      .transact(xa)
      .unsafeRunSync()

  def admin: User = //
    userService
      .synchronizeUser(AvanpostTestAdminId.toString, Role.Admin, None, None, None)
      .transact(xa)
      .unsafeRunSync()

  def createUser(
      email: String,
      role: Option[Role] = None,
      password: ] = None,
      areaLimit: Option[Long] = defaultAreaLimit.some,
      aoiAreaLimit: Option[Long] = defaultAoiAreaLimit.some,
      activeUntil: Option[Instant] = None,
      reviewWorkflowEnabled: Option[Boolean] = None,
    ): User = {
    val input = CreateUserInput(
      email,
      role,
      password,
      areaLimit,
      aoiAreaLimit,
      BillingType.Area,
      None,
      activeUntil,
      reviewWorkflowEnabled = reviewWorkflowEnabled,
    )
    userService
      .createUser(input)(admin)
      .transact(xa)
      .unsafeRunSync()
  }

  def getUserByEmail(email: String): User =
    userService
      .getUserByEmail(email)(admin)
      .transact(xa)
      .unsafeRunSync()
}
