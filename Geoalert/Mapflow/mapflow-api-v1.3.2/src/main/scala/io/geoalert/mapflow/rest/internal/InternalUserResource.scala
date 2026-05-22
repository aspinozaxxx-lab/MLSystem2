package io.geoalert.mapflow.rest.internal

import akka.http.scaladsl.server.Directives
import akka.http.scaladsl.server.Route
import cats.syntax.traverse._
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import io.circe.generic.auto.exportDecoder

import io.geoalert.mapflow.Config.defaultBillingType
import io.geoalert.mapflow.Config.defaultPaidProviders
import io.geoalert.mapflow.model.CreateUserInput
import io.geoalert.mapflow.model.Permission
import io.geoalert.mapflow.model.UpdateUserInput
import io.geoalert.mapflow.rest.Authorization
import io.geoalert.mapflow.rest.RestImplicits
import io.geoalert.mapflow.rest.json.CreateUserInputJson
import io.geoalert.mapflow.rest.json.UpdateUserInputJson
import io.geoalert.mapflow.rest.json.UserJson
import io.geoalert.mapflow.service.Services
import io.geoalert.mapflow.service.Validations

object InternalUserResource extends Directives with Authorization with RestImplicits with Services {

  val createUser: Route = (path("users") & post & authorized & entity(as[CreateUserInputJson])) {
    (user, input) =>
      val result = for {
        user <- userService.createUser(
          CreateUserInput(
            input.email,
            None,
            input.password,
            input.areaLimit,
            input.aoiAreaLimit,
            defaultBillingType,
            input.memoryLimit,
            None,
            None,
          )
        )(user)
      } yield UserJson(user, 0)

      toComplete(result)
  }

  val listUsers: Route = (path("users") & get & authorized & parameters(
    Symbol("offset").as[Int].optional,
    Symbol("limit").as[Int].optional,
  )) { (user, offsetOpt, limitOpt) =>
    toComplete(userService.listUsers(offsetOpt, limitOpt)(user))
  }
  val updateUser: Route =
    (path("users" / Segment) & put & authorized & entity(as[UpdateUserInputJson])) {
      (email, actor, input) =>
        val result = for {
          user <- userService.updateUser(
            UpdateUserInput(
              email,
              None,
              input.password,
              input.areaLimit,
              input.aoiAreaLimit,
              None,
              input.memoryLimit,
              None,
              None,
              None,
            )
          )(actor)
        } yield UserJson(user, 0)

        toComplete(result)
    }

  val getUserStatus: Route = (path("users" / Segment / "status") & get & authorized) {
    (email, actor) =>
      val result = for {
        user <- userService.getUserByEmail(email)(actor)
        userStatus <- userService.getUserStatus(user)
      } yield userStatus

      toComplete(result)
  }

  val archiveUser: Route = (path("users" / Segment) & delete & authorized) { (email, actor) =>
    val result = for {
      user <- userService.getUserByEmail(email)(actor)
      _ <- userService.deleteUser(user.id)(actor)
    } yield "OK"

    toComplete(result)
  }

  val getUser: Route = (path("users" / Segment) & get & authorized) { (email, actor) =>
    val result = for {
      user <- userService.getUserByEmail(email)(actor)
      account <- billingService.getUserAccount(user)
    } yield UserJson(user, account.processedArea)

    toComplete(result)
  }

  def linkPaidDataProviders: Route =
    (path("users" / Segment / "data_providers") & put & authorized) { (email, actor) =>
      logger.info(s"Enabling paid data providers for $email by ${actor.email}")
      val result = for {
        _ <- Validations.checkPermission(actor, Permission.ManageDataProviders)
        user <- userService.getUserByEmail(email)(actor)
        allProviders <- dataProviderService.listDataProviders()(actor)
        providersToLink = allProviders
          .filter(dp => defaultPaidProviders.contains(dp.name))
          .map(_.id)
        _ <- providersToLink.traverse(providerId =>
          dataProviderService.linkDataProvider(user.id, providerId)(actor)
        )
      } yield "OK"

      toComplete(result)
    }

  val routes: Route = concat(
    createUser,
    updateUser,
    getUser,
    getUserStatus,
    listUsers,
    archiveUser,
    linkPaidDataProviders,
  )
}
