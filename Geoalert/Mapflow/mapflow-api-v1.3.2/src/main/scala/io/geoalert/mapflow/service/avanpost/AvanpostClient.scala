package io.geoalert.mapflow.service.avanpost

import java.util.UUID

import scala.concurrent.Future

import io.geoalert.mapflow.Application.authorizationService
import io.geoalert.mapflow.Config.avanpostAdminGroupIds
import io.geoalert.mapflow.Config.avanpostDisabled
import io.geoalert.mapflow.Config.avanpostUserGroupIds
import io.geoalert.mapflow.Config.testEnv
import io.geoalert.mapflow.service.avanpost.responses.UserGroups
import io.geoalert.mapflow.service.avanpost.responses.UserInfo

trait AvanpostClient {
  def userInfo(token: , id: UUID): Future[UserInfo]
}

object AvanpostClient {
  val AvanpostTestAdminId = UUID.fromString("14a852b4-7ac2-4d0d-9551-2e689cf1c157")
  val AvanpostTestAdminId2 = UUID.fromString("196760e9-ff62-41e2-9be3-ba6bc3948b24")
  def make: AvanpostClient =
    if (avanpostDisabled || testEnv)
      new NoOpAvanpostClient
    else
      new LiveAvanpostClient

  private class NoOpAvanpostClient extends AvanpostClient {
    override def userInfo(token: , id: UUID): Future[UserInfo] =
      authorizationService
        .decodeToken(token)
        .fold(
          error => Future.failed(error),
          _ =>
            Future.successful(
              Option
                .when(List(AvanpostTestAdminId2, AvanpostTestAdminId).contains(id))(
                  UserInfo(userGroups = avanpostAdminGroupIds.map(UserGroups(_, "test_admin")))
                )
                .getOrElse(
                  UserInfo(userGroups = avanpostUserGroupIds.map(UserGroups(_, "test_user")))
                )
            ),
        )
  }
}
