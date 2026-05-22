package io.geoalert.mapflow.rest.json

import java.time.Instant

import io.geoalert.mapflow.model.TeamMemberRole.TeamMemberRole

case class MembershipJson(
    role: Option[TeamMemberRole],
    activeUntil: Option[Instant],
    creditsLimit: Option[Long],
    areaLimit: Option[Long],
  )
