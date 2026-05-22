package io.geoalert.mapflow.service

import java.util.UUID

import _root_.io.geoalert.mapflow.service.billing.BillingService
import cats.data.EitherT
import cats.data.NonEmptyList
import cats.implicits.toTraverseOps
import cats.syntax.bifunctor._
import cats.syntax.option._
import com.typesafe.scalalogging.LazyLogging
import doobie.ConnectionIO
import io.geoalert.mapflow.exception.AccessDenied
import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.implicits.CommonOps._
import io.geoalert.mapflow.model.AoiSummary
import io.geoalert.mapflow.model.CreateProjectInput
import io.geoalert.mapflow.model.Permission.ViewAnyProject
import io.geoalert.mapflow.model.Project
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.model.UpdateProjectInput
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.model.UserProject
import io.geoalert.mapflow.model.enums.MemberRole
import io.geoalert.mapflow.repo.AoiRepo
import io.geoalert.mapflow.repo.ProjectRepo
import io.geoalert.mapflow.repo.TeamRepo
import io.geoalert.mapflow.repo.UserProjectsRepo
import io.geoalert.mapflow.repo.UserRepo
import io.geoalert.mapflow.repo.WorkflowDefRepo
import io.geoalert.mapflow.rest.json.AoiJson
import io.geoalert.mapflow.rest.json.ProcessingJson
import io.geoalert.mapflow.rest.json.ProjectJson

class ProjectService(
    processingService: ProcessingService,
    progressService: ProgressService,
    aoiService: AoiService,
    billingService: BillingService,
  ) extends LazyLogging {
  def createProjectAndGet(input: CreateProjectInput)(user: User): ConnectionIO[ProjectJson] =
    for {
      account <- billingService.getUserAccount(user)
      project <- createProject(input)(user)
    } yield ProjectJson(project, user, account.processedArea)
  def createProject(
      input: CreateProjectInput,
      isDefault: Boolean = false,
    )(
      user: User
    ): ConnectionIO[Project] = {
    logger.info(s"Creating project $input by ${user.email}")

    for {
      id <- ProjectRepo.createProject(input, user.id, isDefault)
      admins: List[UUID] <- UserRepo.getAdmins
      userProjects = UserProject(
        userId = user.id,
        projectId = id,
        role = MemberRole.Owner,
      ) ::
        admins

          .filter(_ != user.id)
          .map(owner =>
            UserProject(
              userId = owner,
              projectId = id,
              role = MemberRole.Maintainer,
            )
          )
      _ <- UserProjectsRepo.create(userProjects)
      project <- getProject(id)(user).rethrowT
    } yield project
  }

  def updateProjectAndGet(input: UpdateProjectInput)(user: User): ConnectionIO[ProjectJson] =
    for {
      account <- billingService.getUserAccount(user)
      project <- updateProject(input)(user)
    } yield ProjectJson(project, user, account.processedArea)

  def updateProject(input: UpdateProjectInput)(user: User): ConnectionIO[Project] = {
    logger.info(s"Updating project $input by ${user.email}")

    val io = for {
      validInput <- EitherT(ProjectValidations.projectExist(input.id)(user)).map(_ => input)
      _ <- EitherT.right[ApplicationError](ProjectRepo.updateProject(validInput))
      project <- getProject(validInput.id)(user).leftWiden[ApplicationError]
    } yield project

    io.rethrowT
  }

  def archiveProject(id: UUID)(user: User): EitherT[ConnectionIO, ApplicationError, String] = {
    logger.info(s"Archiving project $id by ${user.email}")

    for {
      _ <- EitherT(ProjectValidations.projectExist(id)(user))
      _ <- EitherT(ProjectValidations.projectNotDefault(id))
      processings <- EitherT.right(processingService.getProcessings(None, Seq(id).some)(user))
      _ <- EitherT.right(ProjectRepo.archive(id))
      _ <- EitherT.right(processingService.archiveProcessingsUnsafe(processings.map(_.id)))
    } yield "OK"
  }

  def shareProject(
      userProject: UserProject
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, String] =
    for {
      userProjects <- EitherT.right(
        UserProjectsRepo.getUserProjects(user.id)
      )
      _ <- EitherT.cond[ConnectionIO](
        user.role == Role.Admin || userProjects.exists(_.projectId == userProject.projectId),
        (),
        AccessDenied("Only owner or admin allowed to share project"): ApplicationError,
      )
      _ <- EitherT.right[ApplicationError](UserProjectsRepo.create(List(userProject)))
    } yield "OK"

  def unshareProject(
      projectId: UUID,
      userId: UUID,
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, String] =
    for {
      userProjects <- EitherT.right(
        UserProjectsRepo.getUserProjects(user.id)
      )
      _ <- EitherT.cond[ConnectionIO](
        user.role == Role.Admin || userProjects.exists(_.projectId == projectId),
        (),
        AccessDenied("Only owner or admin allowed to unshare project"): ApplicationError,
      )
      _ <- EitherT.right[ApplicationError](UserProjectsRepo.unshareProject(userId, projectId))
    } yield "OK"

  def listProjects(user: User): ConnectionIO[List[ProjectJson]] =
    for {
      projects <- getProjects(Seq.empty[UUID])(user)
      account <- billingService.getUserAccount(user)
    } yield projects.map(project => ProjectJson(project, user, account.processedArea))

  def getProjects(projectIds: Seq[UUID])(user: User): ConnectionIO[List[Project]] =
    for {
      usersProjectsIds <- UserProjectsRepo
        .getUserProjectIds(user.id)
        .map { ids =>
          if (projectIds.nonEmpty)
            ids.intersect(projectIds)
          else ids
        }
        .map(NonEmptyList.fromList)
      dtos <-
        if (user.role == Role.Admin)
          ProjectRepo.getProjects(projectIds)
        else
          usersProjectsIds.toList.flatTraverse(ids => ProjectRepo.getProjects(ids.toList))
      userIds = dtos.map(_.userId).distinct
      ids = dtos.map(_.id)
      projectWDs <- WorkflowDefRepo.getLinkedToProjects(ids)
      defaultWDs <- WorkflowDefRepo.listDefaultWorkflowDefs
      userWorkflowDefs <- WorkflowDefRepo.listUserWorkflowDefs(userIds.distinct)
      aoiStats <- AoiRepo.getAoiSummariesByProjects(ids)

      projectProgress <- progressService.getProjectsProgress(dtos)
      projects = for {
        dto <- dtos
        defaultWorkflowDefs = Option.when(dto.defaultWds)(defaultWDs)
        workflowDefs =
          if (dto.isDefault)
            userWorkflowDefs.getOrElse(dto.userId, Nil)
          else projectWDs(dto.id)
        wds = defaultWorkflowDefs.toList.flatten ++ workflowDefs
        AoiSummary(aoiCount, area, _) = aoiStats(dto.id)
        maybeProgress = projectProgress.get(dto.id)
      } yield maybeProgress.map(progress =>
        Project(
          dto.id,
          dto.name,
          dto.description,
          progress,
          aoiCount,
          area,
          dto.userId,
          dto.isDefault,
          dto.created,
          dto.updated,
          wds,
          dto.archived,
          dto.defaultWds,
        )
      )
    } yield projects.flatten

  def getProject(id: UUID)(user: User): EitherT[ConnectionIO, ApplicationError, Project] =
    getProjects(Seq(id))(user)
      .headOrNotFound(id)
      .leftWiden[ApplicationError]
  def getProjectJson(id: UUID)(user: User): EitherT[ConnectionIO, ApplicationError, ProjectJson] =
    for {
      project <- getProject(id)(user)
      account <- EitherT.right[ApplicationError](billingService.getUserAccount(user))
    } yield ProjectJson(project, user, account.processedArea)
  def getProjectProcessings(projectId: UUID)(user: User): ConnectionIO[List[ProcessingJson]] =
    (for {
      project <- getProject(projectId)(user)
      processings <- EitherT.right[ApplicationError](
        processingService.getProcessings(projectIds = Seq(project.id).some)(user)
      )
      jsons <- EitherT.right[ApplicationError](processings.traverse(ProcessingJson(_)))
    } yield jsons).rethrowT

  def getDefaultPrjProcessings(user: User): ConnectionIO[List[ProcessingJson]] =
    for {
      defaultProject <- getOrCreateDefaultProject(user)
      processings <- processingService
        .getProcessings(projectIds = Seq(defaultProject.id).some)(user)

      processingJson <- processings.traverse(ProcessingJson(_))
    } yield processingJson

  def getDefaultProject(user: User): ConnectionIO[ProjectJson] =
    for {
      project <- getOrCreateDefaultProject(user)
      account <- billingService.getUserAccount(user)
    } yield ProjectJson(project, user, account.processedArea)
  def getOrCreateDefaultProject(user: User): ConnectionIO[Project] = {
    val createInput = CreateProjectInput("Default", None, Some(true))

    for {
      userProjects <- UserProjectsRepo.getUserProjects(user.id).map(_.map(_.projectId))
      maybePrj <-
        NonEmptyList.fromList(userProjects).flatTraverse { nonEmptyUserProjects =>
          ProjectRepo
            .getProjects(nonEmptyUserProjects.toList, isDefault = true.some)
            .map(_.headOption)
        }
      project <- maybePrj.fold(createProject(createInput, isDefault = true)(user))(project =>
        getProject(project.id)(user).rethrowT
      )
    } yield project
  }
  def getProcessingAois(prcId: UUID)(user: User): ConnectionIO[List[AoiJson]] = {
    logger.debug(s"Get processing AOIs $prcId by ${user.email}")

    for {
      _ <- Validations.processingExists(prcId, user.userFilter(ViewAnyProject))
      aoiList <- aoiService.getProcessingAois(prcId)(user)
    } yield for {
      aoi <- aoiList
    } yield AoiJson(aoi)
  }
}

object ProjectService {
  def apply(
      processingService: ProcessingService,
      progressService: ProgressService,
      aoiService: AoiService,
      billingService: BillingService,
    ): ProjectService =
    new ProjectService(
      processingService,
      progressService,
      aoiService,
      billingService,
    )
}
