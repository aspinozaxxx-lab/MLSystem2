package io.geoalert.mapflow.repo

import java.time.Instant
import java.util.UUID

import cats.data.NonEmptySeq
import cats.syntax.option._
import doobie.Fragments._
import doobie._
import doobie.implicits._
import doobie.implicits.legacy.instant._
import doobie.postgres.implicits._
import doobie.util.fragment.Fragment.const
import io.geoalert.mapflow.implicits.Postgres._
import io.geoalert.mapflow.model.CreateTeamInput
import io.geoalert.mapflow.model.Team
import io.geoalert.mapflow.model.TeamMember
import io.geoalert.mapflow.model.TeamMemberRole
import io.geoalert.mapflow.model.TeamMemberRole.TeamMemberRole
import io.geoalert.mapflow.model.TeamMembership
import io.geoalert.mapflow.model.UpdateTeamInput

object TeamRepo
    extends GenericRepo[Team](
      "team",
      Seq(
        "id",
        "name",
        "created",
        "updated",
        "archived",
      ),
    ) {
  def create(input: CreateTeamInput): ConnectionIO[UUID] = {
    val columns = Seq(
      ("name" -> input.name).some,
      ("created" -> Instant.now()).some,
      ("updated" -> Instant.now()).some,
    ).flatten.toMap

    create(columns)
  }

  def update(input: UpdateTeamInput): ConnectionIO[Unit] = {
    val columns = Seq(
      input.name.map("name" -> _),
      ("updated" -> Instant.now()).some,
    ).flatten.toMap

    updateById(columns, input.id)
  }

  def listTeams(ids: Option[Seq[UUID]], userIds: Option[Seq[UUID]]): ConnectionIO[List[Team]] = {
    val sql = const("SELECT " + columns.map("t." + _).mkString(", ")) ++
      const(s"FROM $dbSchema.team t ") ++
      const(s"LEFT JOIN $dbSchema.team_member tm ON tm.team_id = t.id ") ++
      fr"WHERE " ++
      andOpt(
        fr"t.archived = FALSE".some,
        ids.flatMap(NonEmptySeq.fromSeq).map(in(fr"t.id", _)),
        userIds.flatMap(NonEmptySeq.fromSeq).map(in(fr"tm.user_id", _)),
      ) ++
      fr"group by t.id"

    sql.query[Team].to[List]
  }

  def linkUserToTeam(
      userId: UUID,
      teamId: UUID,
      role: TeamMemberRole,
      creditsLimit: Option[Long],
    ): ConnectionIO[Int] = {
    val sql =
      const(s"INSERT INTO $dbSchema.team_member (team_id, user_id, role, credits_limit) ") ++
        fr"VALUES ($teamId, $userId, $role, $creditsLimit) ON CONFLICT DO NOTHING"
    sql.update.run
  }

  def updateUserInTeam(
      userId: UUID,
      teamId: UUID,
      role: TeamMemberRole,
      creditsLimit: Option[Long],
    ): ConnectionIO[Int] = {
    val sql = const(
      s"UPDATE $dbSchema.team_member"
    ) ++ fr"SET role = $role, credits_limit = $creditsLimit " ++
      fr"WHERE team_id = $teamId AND user_id = $userId"
    sql.update.run
  }

  def unlinkUserFromTeam(userId: UUID, teamId: UUID): ConnectionIO[Int] = {
    val sql = const(s"DELETE FROM $dbSchema.team_member") ++ fr"WHERE team_id=$teamId AND user_id=$userId"
    sql.update.run
  }

  def listMembers(id: UUID): ConnectionIO[List[TeamMember]] = {
    val sql =
      const(s"SELECT u.id, u.email, tm.role, u.active_until, tm.credits_limit FROM $dbSchema.team_member tm ") ++
        const(s"JOIN $dbSchema.app_user u ON tm.user_id = u.id ") ++
        const(s"JOIN $dbSchema.team t ON tm.team_id = t.id ") ++
        fr"WHERE t.archived = FALSE AND t.id = $id"

    sql.query[TeamMember].to[List]
  }

  def teamOwner(id: UUID): ConnectionIO[Option[TeamMember]] = {
    val sql =
      const(s"SELECT u.id, u.email, tm.role, u.active_until, tm.credits_limit FROM $dbSchema.team_member tm ") ++
        const(s"JOIN $dbSchema.app_user u ON tm.user_id = u.id ") ++
        const(s"JOIN $dbSchema.team t ON tm.team_id = t.id ") ++
        fr"WHERE t.archived = FALSE AND tm.role = ${TeamMemberRole.OWNER}" ++
        const(s"AND tm.team_id IN (SELECT team_id FROM $dbSchema.team_member") ++ fr"WHERE user_id = $id LIMIT 1)"

    sql.query[TeamMember].option
  }

  def listTeamMemberships(userId: UUID): ConnectionIO[List[TeamMembership]] = {
    val sql =
      const(s"SELECT tm.team_id, t.name, tm.role, u.active_until, tm.credits_limit FROM $dbSchema.team_member tm ") ++
        const(s"JOIN $dbSchema.team t ON t.id = tm.team_id ") ++
        const(s"JOIN $dbSchema.app_user u ON tm.user_id = u.id ") ++
        fr"WHERE tm.user_id = $userId " ++
        fr" AND t.archived = FALSE"

    sql.query[TeamMembership].to[List]
  }
}
