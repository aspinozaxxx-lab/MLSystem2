package io.geoalert.mapflow.functional

import cats.syntax.option._
import doobie.implicits._

import io.geoalert.mapflow.DbIntegrationTest
import io.geoalert.mapflow.exception.AreaLimitExceeded
import io.geoalert.mapflow.model.CreateTeamInput
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.model.TeamMemberRole
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.service.Services
import io.geoalert.mapflow.service.billing.UserAccount
import io.geoalert.mapflow.util.ProcessingUtil
import io.geoalert.mapflow.util.UserUtil

/** Test billing for team account
  */
class TeamAccountSpec extends DbIntegrationTest with Services {
  describe("Team Accounts") {
    it("should fail if team balance exceeded") {
      val teamOwner = UserUtil.createUser(
        "0a8f9ebb-1d8e-4144-b454-ad3cee552ec9",
        Role.User.some,
        None,
        42_000_000L.some,
        100_000_000L.some,
      )
      val teamMember = UserUtil.createUser(
        "b6a716a5-8689-4aef-8efe-895d6e565037",
        Role.User.some,
        None,
        42_000_000L.some,
        100_000_000L.some,
      )

      setupTeam(teamOwner, teamMember)

      val ownerProcessing = ProcessingUtil.createProcessing(area = 40_000_000L.some)(teamOwner)

      // Should OK
      billingService
        .hold(ownerProcessing)(teamOwner)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      val ownerAccount = billingService
        .getUserAccount(teamOwner)
        .transact(xa)
        .unsafeRunSync()

      ownerAccount should matchPattern {
        case UserAccount(40_000_000L, 2_000_000L, 42_000_000L, 0, 0) =>
      }

      val memberAccount = billingService
        .getUserAccount(teamMember)
        .transact(xa)
        .unsafeRunSync()

      memberAccount should matchPattern {
        case UserAccount(0L, 2_000_000L, 42_000_000L, 0, 0) =>
      }

      // Should fail as insufficient area
      val memberProcessing = ProcessingUtil.createProcessing(area = 40_000_000L.some)(teamMember)
      val either = billingService
        .hold(memberProcessing)(teamMember)
        .value
        .transact(xa)
        .unsafeRunSync()

      either should matchPattern {
        case Left(AreaLimitExceeded(80_000_000L, 42_000_000L)) =>
      }
    }

    it("should debit team balance as well as user balance") {
      val teamOwner = UserUtil.createUser(
        "0a8f9ebb-1d8e-4144-b454-ad3cee552ec9",
        Role.User.some,
        None,
        42_000_000L.some,
        100_000_000L.some,
      )
      val teamMember = UserUtil.createUser(
        "b6a716a5-8689-4aef-8efe-895d6e565037",
        Role.User.some,
        None,
        42_000_000L.some,
        100_000_000L.some,
      )

      setupTeam(teamOwner, teamMember)

      val ownerProcessing = ProcessingUtil.createProcessing(area = 1_000_000L.some)(teamOwner)

      // Should OK
      billingService
        .hold(ownerProcessing)(teamOwner)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      // Should fail as insufficient area
      val memberProcessing = ProcessingUtil.createProcessing(area = 1_000_000L.some)(teamMember)
      billingService
        .hold(memberProcessing)(teamMember)
        .rethrowT
        .transact(xa)
        .unsafeRunSync()

      val ownerBalance = billingService
        .getUserAccount(teamOwner)
        .transact(xa)
        .unsafeRunSync()

      val memberBalance = billingService
        .getUserAccount(teamMember)
        .transact(xa)
        .unsafeRunSync()

      ownerBalance.processedArea should be(2_000_000L)
      memberBalance.processedArea should be(1_000_000L)
    }

    def setupTeam(teamOwner: User, teamMember: User) = {
      val team = teamService
        .createTeam(CreateTeamInput(""))(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      teamService
        .linkUserToTeam(
          team.id,
          teamOwner.email,
          TeamMemberRole.OWNER,
          None,
          None,
          None,
          failToLinkExistingUser = false,
        )(UserUtil.admin)
        .transact(xa)
        .unsafeRunSync()

      teamService
        .linkUserToTeam(
          team.id,
          teamMember.email,
          TeamMemberRole.MEMBER,
          None,
          None,
          None,
          failToLinkExistingUser = false,
        )(teamOwner)
        .transact(xa)
        .unsafeRunSync()
    }
  }
}
