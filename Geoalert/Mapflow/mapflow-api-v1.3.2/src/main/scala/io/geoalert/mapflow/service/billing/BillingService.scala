package io.geoalert.mapflow.service.billing

import java.util.UUID

import cats.data.EitherT
import cats.syntax.applicative._
import cats.syntax.bifunctor._
import cats.syntax.option._
import cats.syntax.traverse._
import com.typesafe.scalalogging.LazyLogging
import doobie.ConnectionIO
import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.exception.AreaLimitExceeded
import io.geoalert.mapflow.model.BillingType
import io.geoalert.mapflow.model.Permission.UnlimitedProcessing
import io.geoalert.mapflow.model.Processing
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.repo.ProcessedAreaRepo
import io.geoalert.mapflow.repo.ProcessingDto
import io.geoalert.mapflow.repo.ProjectRepo
import io.geoalert.mapflow.repo.UserDto
import io.geoalert.mapflow.repo.UserRepo
import io.geoalert.mapflow.rest.json.TeamMemberJson
import io.geoalert.mapflow.service.TeamService

case class UserAccount(
    processedArea: Long,
    remainingArea: Long,
    areaLimit: Long,
    creditsLeft: Long,
    creditsUsed: Long,
  )

class BillingService(teamService: TeamService) extends LazyLogging {
  val billingEngineHttpClient: BillingEngineHttpClient = BillingEngineHttpClient()

  def hold(processing: Processing)(user: User): EitherT[ConnectionIO, ApplicationError, Unit] = {
    val area = processing.area
    val cost = processing.cost.getOrElse(0L)
    logger.info(s"Holding $cost credits and $area sq.km. for user ${user.email}")

    def holdCredits(user: UserDto): EitherT[ConnectionIO, ApplicationError, Unit] =
      if (user.billingType != BillingType.Credits)
        EitherT.rightT[ConnectionIO, ApplicationError] {}
      else {
        logger.info(
          f"Holding user ${user.email} for the processing ${processing.id} in the amount of $cost credits"
        )
        EitherT.right[ApplicationError](
          billingEngineHttpClient
            .credit(user.email, processing.id, area, cost)
            .map { _ => }
            .to[ConnectionIO]
        )
      }

    def holdArea(user: UserDto): EitherT[ConnectionIO, ApplicationError, Unit] =
      if (user.billingType != BillingType.Area)
        EitherT.rightT[ConnectionIO, ApplicationError] {}
      else {
        logger.info(
          f"Holding user ${user.email} for the processing ${processing.id} in the amount of $area sq.km."
        )
        for {
          _ <- checkAreaLimitReached(area, user)
          _ <- EitherT.right[ApplicationError](
            ProcessedAreaRepo.holdProcessedArea(processing.id, user.id, area)
          )
        } yield {}
      }

    for {
      userDto <- UserRepo.getUser(user.id).leftWiden[ApplicationError]
      teamOwner <- teamService.getTeamOwnerForUser(user.id)
      accounts = (userDto :: teamOwner.toList).distinct
      _ <- accounts.traverse(holdCredits)
      _ <- accounts.traverse(holdArea)
    } yield {}
  }

  def confirmTransaction(processing: ProcessingDto): ConnectionIO[Unit] = {
    logger.info(s"Debit user account for processing ${processing.id}")

    for {
      project <- ProjectRepo.getProject(processing.projectId).rethrowT
      owner <- UserRepo.getUser(project.userId).rethrowT
      _ <-
        if (owner.billingType == BillingType.Area)
          ProcessedAreaRepo.debitProcessedArea(processing.id)
        else
          billingEngineHttpClient.confirmTransaction(processing.id).to[ConnectionIO]
    } yield {}
  }

  def discardTransaction(processing: ProcessingDto): ConnectionIO[Unit] = {
    logger.info(s"Refund user account for processing ${processing.id}")

    for {
      project <- ProjectRepo.getProject(processing.projectId).rethrowT
      owner <- UserRepo.getUser(project.userId).rethrowT
      _ <-
        if (owner.billingType == BillingType.Area)
          ProcessedAreaRepo.deleteProcessedArea(processing.id)
        else
          billingEngineHttpClient.discardTransaction(processing.id).to[ConnectionIO]
    } yield {}
  }

  private def getUserAccount(dto: UserDto, processedArea: Long): ConnectionIO[UserAccount] = {
    def calculateTeamRemainingArea(teamOwner: UserDto): ConnectionIO[Long] =
      for {
        ownerProcessedArea <- ProcessedAreaRepo.getProcessedAreaByUserId(teamOwner.id)
      } yield teamOwner.areaLimit - ownerProcessedArea

    for {
      teamOwner <- teamService.getTeamOwnerForUser(dto.id).rethrowT
      ownerAccount = teamOwner.filterNot(_.id == dto.id)
      teamRemainingArea <- ownerAccount.traverse(calculateTeamRemainingArea)
      userRemainingArea = dto.areaLimit - processedArea
      remainingArea = Math.max(0, (userRemainingArea :: teamRemainingArea.toList).min)
      userBalance <-
        if (dto.billingType == BillingType.Credits)
          billingEngineHttpClient.getUserBalance(dto.email).to[ConnectionIO]
        else
          UserBalanceJson(dto.id.some, dto.email, 0, 0).pure[ConnectionIO]
      teamOwnerBalance <- ownerAccount.traverse { owner =>
        if (dto.billingType == BillingType.Credits)
          billingEngineHttpClient.getUserBalance(owner.email).to[ConnectionIO]
        else
          UserBalanceJson(dto.id.some, dto.email, 0, 0).pure[ConnectionIO]
      }
      creditsLeft = Math.max(
        0,
        (userBalance.remainingCredits :: teamOwnerBalance.map(_.remainingCredits).toList).min,
      )
    } yield UserAccount(
      processedArea,
      remainingArea,
      dto.areaLimit,
      creditsLeft,
      userBalance.usedCredits,
    )
  }

  def getUserAccount(user: User): ConnectionIO[UserAccount] =
    for {
      userDto <- UserRepo.getUser(user.id).rethrowT
      account <- getUserAccount(userDto)
    } yield account

  def getUserAccount(user: UserDto): ConnectionIO[UserAccount] =
    for {
      processedArea <- ProcessedAreaRepo.getProcessedAreaByUserId(user.id)
      account <- getUserAccount(user, processedArea)
    } yield account

  def getUserAccounts(userIds: Seq[UUID]): ConnectionIO[Map[UUID, UserAccount]] =
    for {
      users <- UserRepo.getUsers(userIds)
      processedAreas <- ProcessedAreaRepo.getProcessedAreaByUserIds(userIds)
      userAccounts <- users
        .traverse(dto =>
          getUserAccount(dto, processedAreas.getOrElse(dto.id, 0L)).map(acc => dto.id -> acc)
        )
        .map(_.toMap)
    } yield userAccounts

  def checkAreaLimitReached(
      newProcessingArea: Long,
      user: UserDto,
    ): EitherT[ConnectionIO, ApplicationError, Unit] =
    if (Role(user.role).hasPermission(UnlimitedProcessing)) {
      logger.debug("User has unlimited processing")
      EitherT.rightT[ConnectionIO, ApplicationError] {}
    }
    else
      for {
        processed <- EitherT.right[ApplicationError](
          ProcessedAreaRepo.getProcessedAreaByUserId(user.id)
        )
        userDto <- UserRepo.getUser(user.id)
        isOk = processed + newProcessingArea <= userDto.areaLimit
        _ = logger.debug(s"User has ${userDto.areaLimit} limit and $processed processed area")
        _ <- EitherT.cond[ConnectionIO](
          isOk,
          {},
          AreaLimitExceeded(processed + newProcessingArea, userDto.areaLimit): ApplicationError,
        )
      } yield {}

  def listTeamMembers(
      teamId: UUID
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, List[TeamMemberJson]] =
    for {
      team <- teamService.getTeam(teamId)(user)
      userIds = team.members.map(_.userId)
      accounts <- EitherT.right[ApplicationError](getUserAccounts(userIds))
    } yield team.members.map { m =>
      val acc = accounts(m.userId)
      TeamMemberJson(
        m,
        acc.processedArea,
        acc.remainingArea,
        acc.areaLimit,
        acc.creditsUsed,
        acc.creditsLeft,
        m.creditsLimit.getOrElse(0),
      )
    }
}

object BillingService {
  def apply(teamService: TeamService): BillingService = new BillingService(teamService)
}
