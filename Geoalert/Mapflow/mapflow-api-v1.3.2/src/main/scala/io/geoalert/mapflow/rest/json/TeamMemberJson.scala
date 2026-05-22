package io.geoalert.mapflow.rest.json

import java.time.Instant
import java.util.UUID

import io.geoalert.mapflow.model.TeamMember
import io.geoalert.mapflow.model.TeamMemberRole

case class TeamMemberJson(
    userId: UUID,
    email: String,
    role: TeamMemberRole.TeamMemberRole,
    activeUntil: Option[Instant],
    processedArea: Long,
    remainingArea: Long,
    areaLimit: Long,
    creditsUsed: Long,
    creditsLeft: Long,
    creditsLimit: Long,
  )

object TeamMemberJson {
  def apply(
      tm: TeamMember,
      processedArea: Long,
      remainingArea: Long,
      areaLimit: Long,
      creditsUsed: Long,
      creditsLeft: Long,
      creditsLimit: Long,
    ): TeamMemberJson = new TeamMemberJson(
    tm.userId,
    tm.email,
    tm.role,
    tm.activeUntil,
    processedArea,
    remainingArea,
    areaLimit,
    creditsUsed,
    creditsLeft,
    creditsLimit,
  )
}
