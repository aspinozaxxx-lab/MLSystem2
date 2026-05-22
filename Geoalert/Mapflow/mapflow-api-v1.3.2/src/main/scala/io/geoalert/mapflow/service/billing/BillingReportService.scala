package io.geoalert.mapflow.service.billing

import java.time.Instant

import cats.syntax.applicative._
import doobie.ConnectionIO

import io.geoalert.mapflow.model.ProcessingMeta
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.repo.BillingRepo
import io.geoalert.mapflow.service.TeamService
import io.geoalert.mapflow.service.UserService

class BillingReportService(teamService: TeamService, userService: UserService) {
  def generateBillingReport(
      email: String,
      startDate: Option[Instant],
      endDate: Option[Instant],
    )(
      actor: User
    ): ConnectionIO[BillingReport] =
    for {
      user <-
        if (actor.email == email)
          actor.pure[ConnectionIO]
        else
          userService.getUserByEmail(email)(actor)
      managedUsers <- teamService.listManagedTeamMembers(user)
//      processings <- listCompleteProcessings(managedUsers :+ user.id, startDate, endDate)
      records <- BillingRepo.listCompleteAois(startDate, endDate, managedUsers :+ user.id)
      defaultStartDate = records
        .map(_.completionDate.getOrElse(Instant.now()))
        .sorted
        .headOption
        .getOrElse(Instant.now())
    } yield BillingReport(
      startDate.getOrElse(defaultStartDate),
      endDate.getOrElse(Instant.now()),
      records
        .map(p =>
          ProcessingReport(
            p.processingId,
            p.email,
            p.name,
            p.area,
            p.cost,
            p.completionDate,
            ProcessingMeta(p.meta).sourceApp.getOrElse("API"),
            p.archived,
          )
        ),
    )
}

object BillingReportService {
  def apply(teamService: TeamService, userService: UserService): BillingReportService =
    new BillingReportService(teamService, userService)
}
