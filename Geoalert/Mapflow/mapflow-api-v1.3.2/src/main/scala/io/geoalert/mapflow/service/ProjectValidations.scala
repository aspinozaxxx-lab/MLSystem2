package io.geoalert.mapflow.service

import java.util.UUID

import doobie.ConnectionIO
import doobie.implicits._
import doobie.postgres.implicits._
import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.exception.Forbidden
import io.geoalert.mapflow.exception.NotFound
import io.geoalert.mapflow.model.Project
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.model.enums.MemberRole
import io.geoalert.mapflow.repo.ProjectRepo
import io.geoalert.mapflow.repo.UserProjectsRepo

object ProjectValidations {
  def projectExist(id: UUID)(user: User): ConnectionIO[Either[ApplicationError, UUID]] =
    UserProjectsRepo
      .getUserAvailableProjects(user.id)
      .map(_.filterNot(_.role == MemberRole.Readonly))
      .map { availableProjects =>
        Either.cond(
          user.role == Role.Admin || availableProjects.exists(_.projectId == id),
          id,
          Forbidden("No access")
        )
      }

  def projectNotDefault(id: UUID): ConnectionIO[Either[ApplicationError, UUID]] = for {
    exists <- ProjectRepo.existsByIdWhere(id, Some(fr"is_default = false"))
  } yield Either.cond(exists, id, Forbidden("Default project cannot be deleted"))
}
