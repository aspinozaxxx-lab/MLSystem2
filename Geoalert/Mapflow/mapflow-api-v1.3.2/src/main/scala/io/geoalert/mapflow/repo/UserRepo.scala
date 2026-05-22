package io.geoalert.mapflow.repo

import java.time.Instant
import java.util.UUID
import cats.data.EitherT
import cats.data.NonEmptyList
import cats.data.OptionT
import cats.implicits.toTraverseOps
import cats.syntax.applicative._
import cats.syntax.option._
import doobie._
import doobie.implicits._
import doobie.implicits.legacy.instant._
import doobie.postgres.implicits._
import doobie.util.fragment.Fragment.const
import doobie.util.fragments.in
import doobie.util.fragments.whereAndOpt
import io.geoalert.mapflow.Config
import io.geoalert.mapflow.exception.NotFound
import io.geoalert.mapflow.implicits.CommonOps._
import io.geoalert.mapflow.implicits.Postgres._
import io.geoalert.mapflow.model.BillingType
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.repo.util.RepoConstants.UserRepoConstants._
import io.geoalert.mapflow.repo.util.UserRepoWhereClauseMaker._

case class UserDto(
    id: UUID,
    email: String,
    password: ],
    role: Int,
    areaLimit: Long,
    aoiAreaLimit: Long,
    billingType: BillingType,
    created: Instant,
    updated: Instant,
    memoryLimit: Long,
    maxAoisPerProcessing: Int,
    activeUntil: Option[Instant],
    reviewWorkflowEnabled: Boolean,
    name: Option[String] = None,
    preferredUsername: Option[String] = None,
    avantpostUserId: Option[UUID] = None,
  )

object UserRepo extends GenericRepo[UserDto](appUserTable, appUserTableColumns) {
  def createUser(
      email: String,
      pwdHash: Option[String],
      role: Role,
      areaLimit: Long,
      aoiAreaLimit: Long,
      billingType: BillingType,
      memoryLimit: Long,
      activeUntil: Option[Instant],
      reviewWorkflowEnabled: Boolean,
      name: Option[String] = None,
      preferredUsername: Option[String] = None,
      avantpostUserId: Option[UUID] = None,
    ): ConnectionIO[UserDto] = {
    val columns = Seq(
      Some(emailColumn -> email),
      pwdHash.map(passwordColumn -> _),
      Some(roleColumn -> role.intVal),
      Some(areaLimitColumn -> areaLimit),
      Some(aoiAreaLimitColumn -> aoiAreaLimit),
      Some(billingTypeColumn -> billingType.repr),
      Some(createdColumn -> Instant.now()),
      Some(updatedColumn -> Instant.now()),
      Some(memoryLimitColumn -> memoryLimit),
      Some(maxAoisPerProcessingColumn -> Config.maxAoisPerProcessing),
      activeUntil.map(activeUntilColumn -> _),
      Some(reviewWorkflowEnabledColumn -> reviewWorkflowEnabled),
      name.map(nameColumn -> _),
      preferredUsername.map(preferredUsernameColumn -> _),
      avantpostUserId.map(avantpostUserIdColumn -> _),
    ).flatten.toMap

    for {
      id <- create(columns)
      userOrErr <- getUser(id).value
    } yield userOrErr.toTry.get
  }

  def updateUser(
      id: UUID,
      pwdHash: Option[String],
      areaLimit: Option[Long],
      aoiAreaLimit: Option[Long],
      billingType: Option[BillingType],
      memoryLimit: Option[Long],
      maxAoisPerProcessing: Option[Int],
      role: Option[Role],
      activeUntil: Option[Instant],
      reviewWorkflowEnabled: Option[Boolean],
      name: Option[String] = None,
      preferredUsername: Option[String] = None,
      avantpostUserId: Option[UUID] = None,
    ): ConnectionIO[Unit] = {
    val fields = Seq(
      pwdHash.map(passwordColumn -> _),
      areaLimit.map(areaLimitColumn -> _),
      aoiAreaLimit.map(aoiAreaLimitColumn -> _),
      billingType.map(billingTypeColumn -> _.repr),
      role.map(roleColumn -> _.intVal),
      memoryLimit.map(memoryLimitColumn -> _),
      maxAoisPerProcessing.map(maxAoisPerProcessingColumn -> _),
      Some(updatedColumn -> Instant.now()),
      activeUntil.map(activeUntilColumn -> _),
      reviewWorkflowEnabled.map(reviewWorkflowEnabledColumn -> _),
      name.map(nameColumn -> _),
      preferredUsername.map(preferredUsernameColumn -> _),
      avantpostUserId.map(avantpostUserIdColumn -> _),
    ).flatten.toMap

    updateById(fields, id)
  }

  def getUsers(ids: Seq[UUID]): ConnectionIO[List[UserDto]] =
    getAllByIds(ids)
  def getAdmins: ConnectionIO[List[UUID]] =
    const(s"SELECT id FROM $dbSchema.$table WHERE role = ${Role.Admin.intVal}")
      .query[UUID]
      .to[List]
  def getByDpId(id: UUID): ConnectionIO[List[UserDto]] = {
    val sql = const(
      "SELECT " +
        columns.map("u." + _).mkString(", ") +
        s" FROM $dbSchema.$table u INNER JOIN $dbSchema.app_user_data_provider audp ON audp.user_id = u.id "
    ) ++ whereAndOpt(fr"audp.data_provider_id = $id".some)
    sql.query[UserDto].to[List]
  }
  def getByProjectId(ids: List[UUID]): ConnectionIO[Map[UUID, UserDto]] =
    NonEmptyList
      .fromList(ids)
      .traverse { projectIds =>
        val sql = const(
          "SELECT p.id, " +
            columns.map("u." + _).mkString(", ") +
            s" FROM $dbSchema.$table u INNER JOIN $dbSchema.project p ON p.user_id = u.id "
        ) ++ whereAndOpt(in(fr"p.id", projectIds).some)
        sql.query[(UUID, UserDto)].to[List]
      }
      .map(_.toList.flatten.toMap)

  def getUser(id: UUID): EitherT[ConnectionIO, NotFound, UserDto] =
    getUsers(List(id)).headOrNotFound(id)

  def getByEmail(email: String): ConnectionIO[Option[UserDto]] = (for {
    dto <- OptionT(getOneWhere(fr"email=$email".some))
    user <- getUser(dto.id).toOption
  } yield user).value

  def getUsersWithFilter(
      ids: Option[List[UUID]],
      emails: Option[List[String]],
      roles: Option[List[Role]],
    ): ConnectionIO[List[UserDto]] =
    for {
      idList <- getUserIds(ids)
      dtos <- getDTOs(Option(idList), emails, roles)
    } yield dtos

  private[repo] def getUserIds(ids: Option[List[UUID]]): ConnectionIO[List[UUID]] =
    ids match {
      case Some(list) if list.isEmpty =>
        List[UUID]().pure[ConnectionIO]
      case _ =>
        getAllIdsWhere(false, whereClause("id", ids))
    }

  private[repo] def getDTOs(
      ids: Option[List[UUID]],
      emails: Option[List[String]],
      roles: Option[List[Role]],
    ): ConnectionIO[List[UserDto]] =
    (ids, emails, roles) match {
      case (Some(Nil), _, _) | (_, Some(Nil), _) | (_, _, Some(Nil)) =>
        List[UserDto]().pure[ConnectionIO]
      case _ =>
        getAllWhere(
          forUpdate = false,
          whereClause(idColumn, ids),
          whereClause(emailColumn, emails),
          whereClause(roleColumn, roles),
        )()
    }
}
