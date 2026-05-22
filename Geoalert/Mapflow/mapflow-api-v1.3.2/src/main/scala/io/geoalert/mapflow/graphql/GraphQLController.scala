package io.geoalert.mapflow.graphql

import java.util.UUID

import cats.data.EitherT
import cats.syntax.option._
import com.typesafe.scalalogging.LazyLogging
import doobie.ConnectionIO

import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.model.ProjectBrief
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.model.UserBrief
import io.geoalert.mapflow.repo.ProjectRepo
import io.geoalert.mapflow.repo.WorkflowDefRepo
import io.geoalert.mapflow.service.PagedRequest
import io.geoalert.mapflow.service.PagedResponse
import io.geoalert.mapflow.service.Services
import io.geoalert.mapflow.service.Validations

object GraphQLController extends Services with LazyLogging {
  def listWorkflowDefUsers(
      workflowDefId: UUID
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, List[User]] =
    for {
      _ <- Validations.checkWorkflowDefPermission(workflowDefId, user)
      userIds <- EitherT.right[ApplicationError](WorkflowDefRepo.listLinkedUsers(workflowDefId))
      users <- EitherT.right[ApplicationError](userService.getUsers(userIds.some, None, None)(user))
    } yield users

  def listProjectsPaged(
      request: PagedRequest
    )(
      user: User
    ): ConnectionIO[PagedResponse[ProjectBrief]] =
    for {
      projectIds <- ProjectRepo.filterProjects(
        Option.unless(user.role.isAdmin)(user.id),
        filter = request.filter,
      ).map(_.distinct)
      _ = logger.debug(s"Project search by filter: ${request.filter}, projectIds: $projectIds")
      pagedProjectIds = projectIds.slice(
        request.offset.getOrElse(0),
        request.offset.getOrElse(0) + request.limit.getOrElse(25),
      )
      _ = logger.info(s"pagedProjectIds: $pagedProjectIds")
      projects <- projectService.getProjects(pagedProjectIds)(user)
      _ = logger.info(s"projects: $projects")
      userIds = projects.map(_.userId)
      _ = logger.info(s"userIds: $userIds")
      users <- userService.getUsers(userIds.some)(user)
      _ = logger.info(s"users: $users")
      usersMap = users.map(u => u.id -> u).toMap
      _ = logger.info(s"usersMap: $usersMap")
      briefs = projects.map { project =>
        val userId = project.userId
        val user = usersMap(userId)
        ProjectBrief(
          project.id,
          project.name,
          project.description,
          project.created,
          project.updated,
          UserBrief(
            userId,
            user.email,
            user.name,
            user.preferredUsername,
            user.avantpostUserId
          ),
          project.progress
        )
      }
    } yield PagedResponse(briefs, projectIds.size, projects.size)
}
