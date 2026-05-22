package io.geoalert.mapflow.repo

import java.util.UUID

import cats.data.NonEmptyList
import cats.implicits.catsSyntaxOptionId
import doobie._
import doobie.implicits._
import doobie.postgres.implicits._
import doobie.util.fragment.Fragment.const
import doobie.util.fragments.in
import doobie.util.fragments.whereAndOpt
import io.geoalert.mapflow.implicits.Postgres._
import io.geoalert.mapflow.model.UserProject
import io.geoalert.mapflow.model.enums.MemberRole

object UserProjectsRepo
    extends GenericRepo[UserProject](
      table = "user_projects",
      columns = Seq(
        "user_id",
        "project_id",
        "role",
      ),
      idField = "project_id",
    ) {
  def create(userProjects: List[UserProject]): ConnectionIO[Int] = {
    val sql =
      s"INSERT INTO $dbSchema.$table (${columns.mkString(",")}) VALUES (?,?,?) ON CONFLICT (user_id, project_id) DO UPDATE SET role = EXCLUDED.role"

    Update[UserProject](sql).updateMany[List](userProjects)
  }

  def unshareProject(userId: UUID, projectId: UUID): ConnectionIO[Int] = {
    val sql = const(s"DELETE FROM $dbSchema.$table") ++ whereAndOpt(
      fr"user_id=$userId".some,
      fr"project_id=$projectId".some,
    )
    sql.update.run
  }

  def unshareProjects(userId: UUID, projectIds: List[UUID]): ConnectionIO[Int] = {
    val sql = const(s"DELETE FROM $dbSchema.$table") ++ whereAndOpt(
      fr"user_id=$userId".some,
      NonEmptyList.fromList(projectIds).map(in(const("project_id"), _)),
    )
    sql.update.run
  }

  def getUserProjectIds(userId: UUID): ConnectionIO[List[UUID]] =
    getAllIdsWhere(forUpdate = false, fr"user_id = $userId".some)

  def getUsersProjectIds(userIds: List[UUID]): ConnectionIO[List[UUID]] =
    getAllIdsWhere(
      forUpdate = false,
      NonEmptyList.fromList(userIds).map(in(const("user_id"), _)),
    )

  def getUserAvailableProjects(userId: UUID): ConnectionIO[List[UserProject]] =
    getAllWhere(forUpdate = false, fr"user_id = $userId".some)()

  def getUserProjects(userId: UUID): ConnectionIO[List[UserProject]] =
    getAllWhere(forUpdate = false, fr"user_id = $userId".some)().map(
      _.filter(_.role == MemberRole.Owner)
    )
}
