package io.geoalert.mapflow.rest

import akka.http.scaladsl.server.Directives
import akka.http.scaladsl.server.Route
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import io.circe.generic.auto.exportDecoder
import io.geoalert.mapflow.model.TeamMemberRole
import io.geoalert.mapflow.rest.json.Decoders
import io.geoalert.mapflow.rest.json.MembershipJson
import io.geoalert.mapflow.service.Services

object TeamResource
    extends Directives
       with Authorization
       with RestImplicits
       with Decoders
       with Services {
  val listTeamMembers: Route = (path("team" / JavaUUID / "member") & get & authorized) {
    (teamId, user) =>
      toComplete(billingService.listTeamMembers(teamId)(user))
  }

  val addTeamMember: Route = (path("team" / JavaUUID / "member" / Segment) & post & authorized &
    entity(as[MembershipJson])) { (teamId, email, user, membership) =>
    toComplete(
      teamService.linkUserToTeam(
        teamId,
        email,
        membership.role.getOrElse(TeamMemberRole.MEMBER),
        membership.activeUntil,
        membership.areaLimit,
        membership.creditsLimit,
        failToLinkExistingUser = true,
      )(user)
    )
  }

  val updateTeamMember: Route = (path("team" / JavaUUID / "member" / Segment) & put & authorized &
    entity(as[MembershipJson])) { (teamId, email, user, membership) =>
    toComplete(
      teamService.updateUserInTeam(
        teamId,
        email,
        membership.role.getOrElse(TeamMemberRole.MEMBER),
        membership.activeUntil,
        membership.areaLimit,
        membership.creditsLimit,
      )(user)
    )
  }

  val removeTeamMember: Route =
    (path("team" / JavaUUID / "member" / Segment) & delete & authorized) { (teamId, email, user) =>
      toComplete(teamService.unlinkUserFromTeam(teamId, email)(user))
    }

  val routes: Route = concat(
    listTeamMembers,
    addTeamMember,
    updateTeamMember,
    removeTeamMember,
  )
}
