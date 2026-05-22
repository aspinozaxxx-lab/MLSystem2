package io.geoalert.mapflow.service

import java.util.UUID

import scala.concurrent.ExecutionContext
import scala.concurrent.ExecutionContextExecutor

import cats.data.EitherT
import cats.syntax.bifunctor._
import cats.syntax.option._
import cats.syntax.traverse._
import com.typesafe.scalalogging.LazyLogging
import doobie.ConnectionIO
import doobie.implicits._
import io.geoalert.mapflow.Config
import io.geoalert.mapflow.exception._
import io.geoalert.mapflow.implicits.CommonOps.SeqOptionToListOption
import io.geoalert.mapflow.model.Permission._
import io.geoalert.mapflow.model._
import io.geoalert.mapflow.repo.DataProviderRepo
import io.geoalert.mapflow.repo.ProcessingRepo
import io.geoalert.mapflow.repo.TeamRepo
import io.geoalert.mapflow.repo.UserDto
import io.geoalert.mapflow.repo.UserProjectsRepo
import io.geoalert.mapflow.repo.UserRepo
import io.geoalert.mapflow.rest.json.DataProviderJson
import io.geoalert.mapflow.rest.json.DataProviderPriceJson
import io.geoalert.mapflow.rest.json.UserJson
import io.geoalert.mapflow.rest.json.UserStatusJson
import io.geoalert.mapflow.rest.json.WorkflowDefJson
import io.geoalert.mapflow.service.billing.BillingService
import io.geoalert.mapflow.util.PasswordUtils

class UserService(
    billingService: BillingService,
    projectService: ProjectService,
    userSyncService: UserSyncService,
    workflowService: WorkflowService,
    workflowDefService: WorkflowDefService,
    costCalculatorService: CostCalculatorService,
  ) extends LazyLogging {
  implicit val ec: ExecutionContextExecutor = ExecutionContext.global

  def authorize(email: String, pass: String): EitherT[ConnectionIO, ApplicationError, User] =
    for {
      dtoOpt <- EitherT.right(UserRepo.getOneWhere(Some(fr"email=$email")))
      _ = dtoOpt.getOrElse(logger.debug(s"User not found $email"))
      dto <- EitherT.fromOption[ConnectionIO](dtoOpt, new AuthenticationError(): ApplicationError)
      userDataProviders <- EitherT.right[ApplicationError](DataProviderRepo.findByUser(dto.id))
      defaultDataProviders <- EitherT.right[ApplicationError](DataProviderRepo.findDefault())
      isMatch = dto.password.exists(PasswordUtils.validatePassword(pass, _))
      _ = if (!isMatch) logger.debug(s"Invalid password for user $email")
      _ <- EitherT.cond[ConnectionIO](isMatch, (), new AuthenticationError(): ApplicationError)
      userAccount <- EitherT.right[ApplicationError](billingService.getUserAccount(dto))
    } yield buildUser(dto, userDataProviders ++ defaultDataProviders, userAccount.processedArea)

  def createUser(input: CreateUserInput)(actor: User): ConnectionIO[User] = {
    logger.info(s"Creating user: $input")

    val areaLimit = input.areaLimit.getOrElse(Config.defaultAreaLimit)
    val aoiAreaLimit = input.aoiAreaLimit.getOrElse(Config.defaultAoiAreaLimit)
    val memoryLimit = input.memoryLimit.getOrElse(Config.defaultMemoryLimit)

    val pwdHash = input.password.map(PasswordUtils.generatePasswordHash)

    val io = for {
      _ <- EitherT(Validations.checkPermission(actor, AddUser))
      email <- EitherT(Validations.loginIsVacant(input.email))
      role <- EitherT(Validations.validRole(input.role, actor))
      dto <- EitherT.right[ApplicationError](
        UserRepo.createUser(
          email,
          pwdHash,
          role.getOrElse(Role.User),
          areaLimit,
          aoiAreaLimit,
          input.billingType,
          memoryLimit,
          input.activeUntil,
          input.reviewWorkflowEnabled.getOrElse(false),
        )
      )
      user <- getUser(dto.id)(actor)
      _ <- EitherT.right[ApplicationError](projectService.getOrCreateDefaultProject(user))
    } yield user

    io.rethrowT
  }

  def updateUser(input: UpdateUserInput)(actor: User): ConnectionIO[User] = {
    logger.info(s"Updating user: $input")

    val io = for {
      userToUpdate <- EitherT.fromOptionF(
        UserRepo.getByEmail(input.email),
        NotFound(s"User `${input.email}` not found"),
      )
      _ <- EitherT(Validations.checkPermission(actor, UpdateUser))
      pwdHash <- EitherT.rightT[ConnectionIO, ApplicationError](
        input.password.map(PasswordUtils.generatePasswordHash)
      )
      role <- EitherT(Validations.validRole(input.role, actor))
      _ <- EitherT.right[ApplicationError](
        UserRepo.updateUser(
          userToUpdate.id,
          pwdHash,
          input.areaLimit,
          input.aoiAreaLimit,
          input.billingType,
          input.memoryLimit,
          input.maxAoisPerProcessing,
          role,
          input.activeUntil,
          input.reviewWorkflowEnabled,
        )
      )
      user <- getUser(userToUpdate.id)(actor).leftWiden[ApplicationError]
    } yield user

    io.rethrowT
  }

  def getUsers(
      ids: Option[Seq[UUID]] = None,
      emails: Option[Seq[String]] = None,
      roles: Option[Seq[Role]] = None,
    )(
      actor: User
    ): ConnectionIO[List[User]] = {
    val io = for {
      users <- EitherT.right[ApplicationError](
        UserRepo.getUsersWithFilter(ids.listOpt, emails.listOpt, roles.listOpt)
      )
      userIds = users.map(_.id)
      userAccounts <- EitherT.right[ApplicationError](
        users.traverse(user => billingService.getUserAccount(user).map((user.id, _))).map(_.toMap)
      )
      dataProviders <- EitherT.right[ApplicationError](DataProviderRepo.findByUsers(userIds))
      defaultDataProviders <- EitherT.right[ApplicationError](DataProviderRepo.findDefault())
    } yield users.map(dto =>
      buildUser(
        dto,
        dataProviders.getOrElse(dto.id, List[DataProvider]()) ++ defaultDataProviders,
        userAccounts(dto.id).processedArea,
      )
    )

    io.rethrowT
  }

  def getUser(id: UUID)(actor: User): EitherT[ConnectionIO, ApplicationError, User] =
    for {
      _ <- EitherT(Validations.checkUserPermission(id, actor))
      dto <- UserRepo.getUser(id).leftWiden[ApplicationError]
      dataProviders <- EitherT.right[ApplicationError](DataProviderRepo.findByUser(dto.id))
      defaultDataProviders <- EitherT.right[ApplicationError](DataProviderRepo.findDefault())
      account <- EitherT.right[ApplicationError](billingService.getUserAccount(dto))
    } yield buildUser(
      dto,
      dataProviders ++ defaultDataProviders,
      account.processedArea,
    )

  def getUsersByDpId(id: UUID): ConnectionIO[List[User]] =
    for {
      dtos <- UserRepo.getByDpId(id)
      userAccounts <- dtos
        .traverse(user => billingService.getUserAccount(user).map((user.id, _)))
        .map(_.toMap)
      dataProviders <- DataProviderRepo.findByUsers(dtos.map(_.id))
      defaultDataProviders <- DataProviderRepo.findDefault()
      users = dtos.map { dto =>
        buildUser(
          dto,
          dataProviders(dto.id) ++ defaultDataProviders,
          userAccounts(dto.id).processedArea,
        )
      }
    } yield users

  def getUserDto(id: UUID)(actor: User): EitherT[ConnectionIO, ApplicationError, UserDto] =
    for {
      _ <- EitherT(Validations.checkUserPermission(id, actor))
      dto <- UserRepo.getUser(id).leftWiden[ApplicationError]
    } yield dto
  def getUserStatus(user: User): ConnectionIO[UserStatusJson] = {
    logger.debug(s"Getting user info about ${user.email}...")

    for {
      dto <- getUserDto(user.id)(user).rethrowT
      userAccount <- billingService.getUserAccount(user)
      wds <- workflowDefService.listWorkflowDefLinkedToUser(user.id)(user)
      teams <- TeamRepo.listTeamMemberships(user.id)
      userDataProviders <- DataProviderRepo.findByUser(user.id)
      defaultDataProviders <- DataProviderRepo.findDefault()
      dataProviders = userDataProviders ++ defaultDataProviders
    } yield UserStatusJson(
      dto.email,
      userAccount.processedArea,
      userAccount.remainingArea,
      userAccount.creditsLeft,
      userAccount.areaLimit,
      dto.maxAoisPerProcessing,
      dto.memoryLimit,
      dto.billingType.repr,
      wds.map(WorkflowDefJson(_)),
      teams,
      dataProviders
        .filter(_.isMosaic)
        .map(dp =>
          DataProviderJson(dp.id, dp.name, dp.displayName, dp.previewUrl)
        ),
      dto.reviewWorkflowEnabled,
      isAdmin = dto.role == Role.Admin.intVal,
    )
  }
  private def dataProviderPrice(dp: DataProvider): Seq[DataProviderPriceJson] =
    costCalculatorService
      .calculateDataProviderCost(dp)
      .map(pair => DataProviderPriceJson(pair._1, pair._2))
      .toSeq

  def deleteUser(id: UUID)(actor: User): ConnectionIO[String] = {
    logger.info(s"Deleting user with id ${id.toString} by ${actor.email}")

    for {
      _ <- EitherT(Validations.userExists(id)).rethrowT
      _ <- EitherT(Validations.checkPermission(actor, DeleteUser)).rethrowT
      projectIds <- UserProjectsRepo.getUserProjects(id).map(_.map(_.projectId))
      processingIds <- ProcessingRepo.getProcessingIdsByProjectIds(projectIds)
      workflowIds <- workflowService.findRunningWorkflowIdsByProcessing(processingIds.flatten)
      _ <- UserRepo.deleteById(id)
      _ <- workflowService.cancelWorkflows(workflowIds)
    } yield "OK"
  }

  def getUserByEmail(email: String)(actor: User): ConnectionIO[User] = {
    logger.info(s"Getting user with email $email by ${actor.email}")

    for {
      users <- getUsers(emails = Seq(email).some)(actor)
    } yield users.headOption.getOrElse(throw NotFound(s"User $email not found"))
  }

  /** Check if user with specified email exists in WM DB and create the user if necessary
    * @param email user email
    * @param role user role
    * @return
    */
  def synchronizeUser(
      email: String,
      role: Role,
      name: Option[String],
      preferredUsername: Option[String],
      avantpostUserId: Option[UUID]
    ): ConnectionIO[User] =
    for {
      dto <- userSyncService.synchronizeUser(
        email,
        role,
        name = name,
        preferredUsername = preferredUsername,
        avantpostUserId = avantpostUserId,
      )
      defaultDataProviders <- DataProviderRepo.findDefault()
      userDataProviders <- DataProviderRepo.findByUser(dto.id)
      userAccount <- billingService.getUserAccount(dto)
    } yield buildUser(dto, userDataProviders ++ defaultDataProviders, userAccount.processedArea)

  def listUsers(
      offsetOpt: Option[Int],
      limit: Option[Int],
    )(
      user: User
    ): ConnectionIO[List[UserJson]] =
    for {
      dtos <- getUsers()(user)
      users = dtos.sortBy(_.email).map(user => UserJson(user, 0L))
    } yield {
      val shifted: List[UserJson] = offsetOpt match {
        case Some(value) => users.splitAt(value)._2
        case None => users
      }

      val limited = limit match {
        case Some(value) => shifted.take(value)
        case None => shifted
      }
      limited
    }

  private def buildUser(
      dto: UserDto,
      dataProviders: List[DataProvider],
      processedArea: Long,
    ): User =
    User(
      dto.id,
      dto.email,
      Role(dto.role),
      dto.areaLimit,
      dto.aoiAreaLimit,
      dto.billingType,
      dto.created,
      dto.updated,
      processedArea,
      dto.memoryLimit,
      dto.maxAoisPerProcessing,
      dataProviders,
      dto.activeUntil,
      dto.reviewWorkflowEnabled,
      dto.name,
      dto.preferredUsername,
      dto.avantpostUserId,
    )
}

object UserService {
  def apply(
      billingService: BillingService,
      projectService: ProjectService,
      userSyncService: UserSyncService,
      workflowService: WorkflowService,
      workflowDefService: WorkflowDefService,
      costCalculatorService: CostCalculatorService,
    ): UserService = new UserService(
    billingService,
    projectService,
    userSyncService,
    workflowService,
    workflowDefService,
    costCalculatorService,
  )
}
