package io.geoalert.mapflow.service

import java.util.UUID

import cats.data.EitherT
import cats.syntax.applicative._
import cats.syntax.option._
import cats.syntax.traverse._
import com.typesafe.scalalogging.LazyLogging
import doobie.ConnectionIO
import doobie.implicits._
import io.geoalert.mapflow.exception._
import io.geoalert.mapflow.model.Permission.ManageWorkflowDefinition
import io.geoalert.mapflow.model._
import io.geoalert.mapflow.repo.ProjectRepo
import io.geoalert.mapflow.repo.UserProjectsRepo
import io.geoalert.mapflow.repo.WorkflowDefRepo
import io.geoalert.mapflow.service.we.WorkflowEngine
import io.geoalert.mapflow.util.WorkflowDefParser

class WorkflowDefService(teamService: TeamService) extends LazyLogging {
  def createWorkflowDef(
      input: CreateWorkflowDefInput,
      file: Option[String],
    )(
      user: User
    ): ConnectionIO[WorkflowDef] = {
    logger.info(s"Creating WorkflowDef: $input by ${user.email}")

    def syncWorkflowDefWithWE(yml: String): EitherT[ConnectionIO, ApplicationError, Long] =
      EitherT.right[ApplicationError](WorkflowEngine().postDef(yml).to[ConnectionIO])

    def createWD(
        input: CreateWorkflowDefInput,
        yml: String,
      )(
        user: User
      ): EitherT[ConnectionIO, ApplicationError, WorkflowDef] = {
      // TODO Do we really need generating random names?
      val weName = UUID.randomUUID().toString
      for {
        conditionedYml <- EitherT.fromEither[ConnectionIO](
          WorkflowDefParser.updateYml(yml, weName, 0)
        )
        weId <- syncWorkflowDefWithWE(conditionedYml)
        id <- EitherT.right[ApplicationError](
          WorkflowDefRepo.createWorkflowDef(input, weId, weName, conditionedYml)
        )
        _ <- input.projectId match {
          case Some(projectId) =>
            EitherT.right[ApplicationError](WorkflowDefRepo.linkToProject(id, projectId))
          case None => EitherT.rightT[ConnectionIO, ApplicationError]("OK")
        }
        wd <- getWorkflowDefEither(id)(user)
      } yield wd
    }

    val yml = (file.toList ++ input.ymlString.toList)
      .headOption
      .getOrElse(throw BadRequest("File or yamlString should be provided explicitly"))

    val io = for {
      _ <- EitherT(Validations.checkPermission(user, ManageWorkflowDefinition))
      summary <- EitherT.fromEither[ConnectionIO](WorkflowDefParser.parseYml(yml))
      validInput = CreateWorkflowDefInput(
        input.projectId,
        input.name,
        input.description,
        None,
        yml.some,
        input.pricePerSqKm.orElse(summary.pricePerSqKm.some),
        input.isDefault,
      )
      wd <- createWD(validInput, yml)(user)
    } yield wd

    io.rethrowT
  }

  def updateWorkflowDef(
      input: UpdateWorkflowDefInput,
      file: Option[String],
    )(
      user: User
    ): ConnectionIO[WorkflowDef] = {
    logger.info(s"Updating WorkflowDef: $input by ${user.email}")

    def extractYml: Either[ApplicationError, Option[String]] = (file, input.ymlString) match {
      case (Some(yml), None) => Right(yml.some)
      case (None, Some(yml)) => Right(yml.some)
      case (None, None) => Right(None)
      case (_, _) => Left(BadRequest("Either file or ymlString is expected"))
    }

    def synchronizeWithWe(
        ymlOpt: Option[String],
        weId: Long,
        weName: String,
      ): EitherT[ConnectionIO, ApplicationError, Option[(String, WorkflowDefSummary)]] =
      ymlOpt match {
        case Some(yml) =>
          for {
            summary <- EitherT.fromEither[ConnectionIO](WorkflowDefParser.parseYml(yml))
            version <- EitherT.right[ApplicationError](
              WorkflowEngine().getLatestDefVersion(weId).to[ConnectionIO]
            )
            conditionedYml <- EitherT.fromEither[ConnectionIO](
              WorkflowDefParser.updateYml(yml, weName, version + 1)
            )
            _ <- EitherT.right[ApplicationError](
              WorkflowEngine().postDef(conditionedYml).to[ConnectionIO]
            )
          } yield (conditionedYml, summary).some
        case None => EitherT.rightT(None)
      }

    val io = for {
      _ <- EitherT(Validations.checkPermission(user, ManageWorkflowDefinition))
      wd <- getWorkflowDefEither(input.id)(user)
      yml <- EitherT.fromEither[ConnectionIO](extractYml)
      wdOpt <- synchronizeWithWe(yml, wd.weId, wd.weName)
      summary = wdOpt.map(_._2)
      conditionedYml = wdOpt.map(_._1)
      validInput = UpdateWorkflowDefInput(
        input.id,
        input.projectId,
        input.name,
        input.description,
        None,
        conditionedYml,
        input.pricePerSqKm.orElse(summary.map(_.pricePerSqKm)),
        input.isDefault,
      )
      _ <- EitherT.right[ApplicationError](
        WorkflowDefRepo.updateWorkflowDef(validInput, conditionedYml)
      )
      updatedWd <- getWorkflowDefEither(validInput.id)(user)
    } yield updatedWd

    io.rethrowT
  }

  def archiveWorkflowDef(id: UUID)(user: User): ConnectionIO[String] = {
    logger.info(s"Archive WorkflowDef $id by ${user.email}")

    val io = for {
      _ <- EitherT(Validations.checkPermission(user, ManageWorkflowDefinition))
      _ <- EitherT.right[ApplicationError](WorkflowDefRepo.archiveWorkflowDef(id))
    } yield "OK"
    io.rethrowT
  }

  def getWorkflowDef(id: UUID)(user: User): ConnectionIO[WorkflowDef] =
    WorkflowDefRepo.getWorkflowDef(id, user.userFilter(Permission.ManageWorkflowDefinition))

  def getWorkflowDefEither(
      id: UUID
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, WorkflowDef] =
    EitherT.right[ApplicationError](getWorkflowDef(id)(user))

  def listWorkflowDefs(
      ids: Option[Seq[UUID]],
      userIds: Option[Seq[UUID]],
      projectIds: Option[Seq[UUID]],
      isDefault: Option[Boolean],
    )(
      user: User
    ): ConnectionIO[List[WorkflowDef]] = {
    val uIds = if (user.role == Role.User) Seq(user.id).some else userIds
    WorkflowDefRepo.listWorkflowDefs(ids, uIds, projectIds, isDefault)
  }

  def listWorkflowDefLinkedToUser(userId: UUID)(user: User): ConnectionIO[List[WorkflowDef]] =
    for {
      _ <- Validations.checkUserPermission(userId, user)
      workflowDefs <- WorkflowDefRepo.listWorkflowDefs(
        userIds = Seq(userId).some,
        isDefault = false.some,
      )
      defaultWorkflowDefs <- WorkflowDefRepo.listDefaultWorkflowDefs
      teamOwner <- teamService.getTeamOwnerForUser(userId).rethrowT
      ownerWds <- teamOwner match {
        case Some(value) =>
          WorkflowDefRepo.listWorkflowDefs(
            userIds = Seq(value.id).some,
            isDefault = false.some,
          )
        case None => List().pure[ConnectionIO]
      }
    } yield workflowDefs ++ defaultWorkflowDefs ++ ownerWds.filter(wd => !workflowDefs.contains(wd))

  def listDefaultWorkflowDefs(): ConnectionIO[List[WorkflowDef]] =
    WorkflowDefRepo.listDefaultWorkflowDefs

  def listWorkflowDefLinkedToProject(projectId: UUID)(user: User): ConnectionIO[List[WorkflowDef]] =
    for {
      _ <- Validations.checkPermission(user, Permission.ViewAnyUser)
      workflowDefs <- WorkflowDefRepo.listWorkflowDefs(projectIds = Seq(projectId).some)
    } yield workflowDefs

  def linkWorkflowDefToUser(workflowDefId: UUID, userId: UUID)(user: User): ConnectionIO[String] =
    for {
      _ <- Validations.checkPermission(user, Permission.ManageWorkflowDefinition)
      _ <- WorkflowDefRepo.linkToUser(workflowDefId, userId)
    } yield "OK"

  def unlinkWorkflowDefFromUser(
      workflowDefId: UUID,
      userId: UUID,
    )(
      user: User
    ): ConnectionIO[String] =
    for {
      _ <- Validations.checkPermission(user, Permission.ManageWorkflowDefinition)
      _ <- WorkflowDefRepo.unlinkFromUser(workflowDefId, userId)
      projectIds <- UserProjectsRepo.getUserProjects(userId).map(_.map(_.projectId))
      _ <- projectIds.traverse(projectId =>
        WorkflowDefRepo.unlinkFromProject(workflowDefId, projectId)
      )
    } yield "OK"

  def linkWorkflowDefToProject(
      workflowDefId: UUID,
      projectId: UUID,
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, String] =
    for {
      _ <- Validations.checkWorkflowDefPermission(workflowDefId, user)
      project <- ProjectRepo.getProject(projectId)
      _ <- EitherT.right[ApplicationError](
        WorkflowDefRepo.linkToUser(workflowDefId, project.userId)
      )
      _ <- EitherT.right[ApplicationError](WorkflowDefRepo.linkToProject(workflowDefId, project.id))
    } yield "OK"

  def unlinkWorkflowDefFromProject(
      workflowDefId: UUID,
      projectId: UUID,
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, String] =
    for {
      _ <- Validations.checkWorkflowDefPermission(workflowDefId, user)
      _ <- EitherT.right[ApplicationError](
        WorkflowDefRepo.unlinkFromProject(workflowDefId, projectId)
      )
    } yield "OK"
}

object WorkflowDefService {
  def apply(teamService: TeamService): WorkflowDefService = new WorkflowDefService(teamService)
}
