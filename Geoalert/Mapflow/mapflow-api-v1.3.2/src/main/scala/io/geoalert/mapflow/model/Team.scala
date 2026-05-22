package io.geoalert.mapflow.model

import java.time.Instant
import java.util.UUID

import io.geoalert.mapflow.model.TeamMemberRole.TeamMemberRole

/** Multiple users can belong to a team. All users within the team shares the
  * same limit
  */
case class Team(
    id: UUID,
    name: String,
    created: Instant,
    updated: Instant,
    archived: Boolean,
  )

case class TeamWithMembers(team: Team, members: List[TeamMember])

case class TeamMembership(
    teamId: UUID,
    name: String,
    role: TeamMemberRole,
    activeUntil: Option[Instant],
    creditsLimit: Option[Long],
  )

object TeamMemberRole extends Enumeration {
  type TeamMemberRole = Value

  val OWNER, MEMBER = Value

  def fromString(name: String): Option[TeamMemberRole] =
    values.find(_.toString.equals(name))
}

case class TeamMember(
    userId: UUID,
    email: String,
    role: TeamMemberRole.TeamMemberRole,
    activeUntil: Option[Instant],
    creditsLimit: Option[Long],
  )

case class CreateTeamInput(name: String)

case class UpdateTeamInput(id: UUID, name: Option[String])
