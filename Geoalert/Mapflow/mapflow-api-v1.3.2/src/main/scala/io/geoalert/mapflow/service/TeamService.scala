package io.geoalert.mapflow.service

import java.time.Instant
import java.util.UUID

import cats.data.EitherT
import cats.syntax.bifunctor._
import cats.syntax.option._
import cats.syntax.traverse._
import com.typesafe.scalalogging.LazyLogging
import doobie.ConnectionIO
import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.exception.BadRequest
import io.geoalert.mapflow.exception.InternalServerError
import io.geoalert.mapflow.exception.NotFound
import io.geoalert.mapflow.model.BillingType
import io.geoalert.mapflow.model.CreateTeamInput
import io.geoalert.mapflow.model.Permission
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.model.Team
import io.geoalert.mapflow.model.TeamMember
import io.geoalert.mapflow.model.TeamMemberRole
import io.geoalert.mapflow.model.TeamMemberRole.TeamMemberRole
import io.geoalert.mapflow.model.TeamWithMembers
import io.geoalert.mapflow.model.UpdateTeamInput
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.repo.TeamRepo
import io.geoalert.mapflow.repo.UserDto
import io.geoalert.mapflow.repo.UserProjectsRepo
import io.geoalert.mapflow.repo.UserRepo

class TeamService(userSyncService: UserSyncService)
    extends LazyLogging {
  def createTeam(input: CreateTeamInput)(user: User): ConnectionIO[Team] = {
    val either = for {
      _ <- EitherT(Validations.checkPermission(user, Permission.ManageTeams))
      id <- EitherT.right[ApplicationError](TeamRepo.create(input))
      team <- EitherT.right[ApplicationError](TeamRepo.getOneById(id))
    } yield team.getOrElse(throw NotFound[Team](id): ApplicationError)

    either.rethrowT
  }

  def updateTeam(input: UpdateTeamInput)(user: User): ConnectionIO[Team] = {
    val id = input.id

    val either = for {
      _ <- EitherT(Validations.checkPermission(user, Permission.ManageTeams))
      _ <- EitherT.right[ApplicationError](TeamRepo.update(input))
      team <- EitherT.right[ApplicationError](TeamRepo.getOneById(id))
    } yield team.getOrElse(throw NotFound[Team](id): ApplicationError)

    either.rethrowT
  }

  def archiveTeam(id: UUID)(user: User): ConnectionIO[String] = {
    val either = for {
      _ <- EitherT(Validations.checkPermission(user, Permission.ManageTeams))
      _ <- EitherT.right[ApplicationError](TeamRepo.archive(id))
    } yield "OK"

    either.rethrowT
  }

  def getTeam(id: UUID)(user: User): EitherT[ConnectionIO, ApplicationError, TeamWithMembers] =
    for {
      _ <- EitherT(Validations.canManageTeam(id, user))
      team <- EitherT.right[ApplicationError](TeamRepo.getOneById(id))
      members <- EitherT.right[ApplicationError](TeamRepo.listMembers(id))
    } yield TeamWithMembers(team.getOrElse(throw NotFound[Team](id): ApplicationError), members)

  def listManagedTeamMembers(user: User): ConnectionIO[Seq[UUID]] =
    for {
      memberships <- TeamRepo.listTeamMemberships(user.id)
      ownedTeams = memberships.filter(_.role == TeamMemberRole.OWNER).map(_.teamId)
      members <- ownedTeams.flatTraverse(TeamRepo.listMembers)
    } yield members.filter(_.userId != user.id).map(_.userId)

  def listTeams(
      ids: Option[Seq[UUID]],
      userIds: Option[Seq[UUID]],
    )(
      user: User
    ): ConnectionIO[List[Team]] = {
    val either = for {
      _ <- EitherT(Validations.checkPermission(user, Permission.ManageTeams))
      teams <- EitherT.right[ApplicationError](TeamRepo.listTeams(ids, userIds))
    } yield teams

    either.rethrowT
  }

  def linkUserToTeam(
      teamId: UUID,
      email: String,
      role: TeamMemberRole,
      activeUntil: Option[Instant],
      areaLimit: Option[Long],
      creditsLimit: Option[Long],
      failToLinkExistingUser: Boolean,
    )(
      user: User
    ): ConnectionIO[String] = {
    def getTeamBillingType(
        owners: List[TeamMember]
      ): EitherT[ConnectionIO, ApplicationError, Option[BillingType]] = {
      val ownerId = owners.headOption.map(_.userId)
      ownerId match {
        case Some(value) =>
          for {
            user <- UserRepo.getUser(value).leftWiden[ApplicationError]
          } yield user.billingType.some
        case None => EitherT.rightT[ConnectionIO, ApplicationError](None: Option[BillingType])
      }
    }

    val either = for {
      _ <- EitherT(Validations.canManageTeam(teamId, user))
      members <- EitherT.right[ApplicationError](TeamRepo.listMembers(teamId))
      owners = members.filter(_.role == TeamMemberRole.OWNER)
      _ <- EitherT.cond[ConnectionIO](
        role != TeamMemberRole.OWNER || owners.isEmpty || owners.contains(teamId),
        (),
        BadRequest("Only one owner is allowed for a team"): ApplicationError,
      )
      billingType <- getTeamBillingType(owners)
      user <- EitherT.right(
        userSyncService.synchronizeUser(email, Role.User, areaLimit, billingType, activeUntil)
      )
      teams <- EitherT.right[ApplicationError](
        TeamRepo.listTeams(ids = None, userIds = List(user.id).some)
      )
      teamIds = teams.map(_.id)
      _ <- EitherT.cond[ConnectionIO](
        teamIds.isEmpty || teamIds.contains(teamId),
        "",
        BadRequest("Only one team is allowed for a user"): ApplicationError,
      )
      _ <- EitherT.right[ApplicationError](
        TeamRepo.linkUserToTeam(user.id, teamId, role, creditsLimit)
      )
    } yield "OK"

    either.rethrowT
  }

  def updateUserInTeam(
      teamId: UUID,
      email: String,
      role: TeamMemberRole,
      activeUntil: Option[Instant],
      areaLimit: Option[Long],
      creditsLimit: Option[Long],
    )(
      actor: User
    ): ConnectionIO[String] = {
    val either = for {
      _ <- EitherT(Validations.canManageTeam(teamId, actor))
      user <- EitherT.fromOptionF(
        UserRepo.getByEmail(email),
        NotFound(s"User not found by email $email"): ApplicationError,
      )
      members <- EitherT.right[ApplicationError](TeamRepo.listMembers(teamId))
      member = members.find(_.userId == user.id)
      _ <- EitherT.cond[ConnectionIO](
        member.isDefined,
        {},
        BadRequest(s"User $email does not belong to a team $teamId"),
      )
      _ <- EitherT.cond[ConnectionIO](
        member.get.role == TeamMemberRole.MEMBER,
        {},
        BadRequest(s"Team owner can not be edited"),
      )
      _ <- EitherT.right[ApplicationError](
        TeamRepo.updateUserInTeam(user.id, teamId, role, creditsLimit)
      )
      _ <- EitherT.right[ApplicationError](
        UserRepo.updateUser(
          user.id,
          None,
          areaLimit,
          None,
          None,
          None,
          None,
          None,
          activeUntil,
          None,
        )
      )
    } yield "OK"

    either.rethrowT
  }

  def unlinkUserFromTeam(
      id: UUID,
      email: String,
    )(
      user: User
    ): ConnectionIO[String] = {
    val either = for {
      _ <- EitherT(Validations.canManageTeam(id, user))
      userOpt <- EitherT.right[ApplicationError](UserRepo.getByEmail(email))
      user <- EitherT
        .fromOption[ConnectionIO](userOpt, NotFound(s"User $email not found"): ApplicationError)
      members <- EitherT.right[ApplicationError](TeamRepo.listMembers(id))
      memberIds = members.map(_.userId).filterNot(_ == user.id)
      teamProjectIds <- EitherT.right[ApplicationError](
        UserProjectsRepo.getUsersProjectIds(memberIds)
      )
      _ <- EitherT.right[ApplicationError](
        UserProjectsRepo.unshareProjects(user.id, teamProjectIds)
      )
      role = members.filter(_.email == email).map(_.role).headOption
      _ <- EitherT.cond[ConnectionIO](
        role != TeamMemberRole.OWNER.some,
        {},
        BadRequest("Deleting team owners is not allowed"),
      )
      _ <- EitherT.right[ApplicationError](TeamRepo.unlinkUserFromTeam(user.id, id))
      // Reset user areaLimit
      _ <- EitherT.right[ApplicationError](
        UserRepo.updateUser(user.id, None, 0L.some, None, None, None, None, None, None, None)
      )
    } yield "OK"

    either.rethrowT
  }

  def getTeamOwnerForUser(
      userId: UUID
    ): EitherT[ConnectionIO, ApplicationError, Option[UserDto]] = {
    def getTeamOwner(teamId: UUID): EitherT[ConnectionIO, ApplicationError, (Team, UserDto)] =
      for {
        team <- EitherT.fromOptionF(TeamRepo.getOneById(teamId), NotFound[Team](teamId))
        members <- EitherT.right[ApplicationError](TeamRepo.listMembers(teamId))
        ownerId <- EitherT.fromOption[ConnectionIO](
          members
            .filter(_.role == TeamMemberRole.OWNER)
            .map(_.userId)
            .headOption,
          InternalServerError(s"Owner not found for a team $teamId"): ApplicationError,
        )
        owner <- UserRepo.getUser(ownerId).leftWiden[ApplicationError]
      } yield (team, owner)

    for {
      memberships <- EitherT.right[ApplicationError](TeamRepo.listTeamMemberships(userId))
      teamIds = memberships.map(_.teamId)
      teamOwners <- teamIds.traverse(getTeamOwner)
      _ = logger.debug(
        s"User belongs to ${teamOwners.size} teams: " + teamOwners
          .map {
            case (team, owner) => s"${team.name} (${owner.areaLimit})"
          }
          .mkString(", ")
      )
    } yield teamOwners.map(_._2).headOption
  }
}

object TeamService {
  def apply(userSyncService: UserSyncService): TeamService =
    new TeamService(userSyncService)
}
