package io.geoalert.mapflow.service

import java.time.Instant
import java.util.UUID

import scala.reflect.ClassTag

import cats.data.EitherT
import cats.effect.IO
import cats.syntax.either._
import cats.syntax.option._
import doobie._
import doobie.implicits._
import doobie.postgres.implicits._

import io.geoalert.mapflow.exception._
import io.geoalert.mapflow.model.Permission.LargeProcessing
import io.geoalert.mapflow.model.Permission.ManageDataProviders
import io.geoalert.mapflow.model.Permission.ManageWorkflowDefinition
import io.geoalert.mapflow.model.Permission.ViewAnyUser
import io.geoalert.mapflow.model._
import io.geoalert.mapflow.repo._

object Validations {
  private def notFound[A: ClassTag](id: UUID): ApplicationError = NotFound[A](id): ApplicationError

  private def notFound[A: ClassTag](ids: List[UUID]): ApplicationError =
    NotFound[A](ids): ApplicationError

  def processingExists(
      id: UUID,
      userId: Option[UUID],
    ): ConnectionIO[Either[ApplicationError, UUID]] = (for {
    prc <- EitherT.fromOptionF(ProcessingRepo.getOneById(id), notFound[Processing](id))
    userFilter = userId.map(u => fr"user_id = $u")
    isLegal <- EitherT.right[ApplicationError](
      ProjectRepo.existsByIdWhere(prc.projectId, userFilter)
    )
    _ <- EitherT.cond[ConnectionIO](isLegal, id, notFound[Processing](id))
  } yield id).value

  def aoiExists(id: UUID): ConnectionIO[Either[ApplicationError, UUID]] = for {
    exists <- AoiRepo.existsById(id)
  } yield Either.cond(exists, id, notFound[Aoi](id))

  def vectorLayerExists(id: UUID): ConnectionIO[Either[ApplicationError, UUID]] = for {
    exists <- VectorLayerRepo.existsById(id)
  } yield Either.cond(exists, id, notFound[VectorLayer](id))

  def rasterLayerExists(id: UUID): ConnectionIO[Either[ApplicationError, UUID]] = for {
    exists <- RasterLayerRepo.existsById(id)
  } yield Either.cond(exists, id, notFound[RasterLayer](id))

  def wdExists(id: UUID): ConnectionIO[Either[ApplicationError, UUID]] = for {
    exists <- WorkflowDefRepo.existsById(id)
  } yield Either.cond(exists, id, notFound[WorkflowDef](id))

  def processingsExist(ids: List[UUID]): ConnectionIO[Either[ApplicationError, List[UUID]]] = for {
    nonExisting <- ProcessingRepo.nonExisting(ids)
  } yield Either.cond(nonExisting.isEmpty, ids, notFound[Processing](nonExisting))

  def aoisExist(ids: List[UUID]): ConnectionIO[Either[ApplicationError, List[UUID]]] = for {
    nonExisting <- AoiRepo.nonExisting(ids)
  } yield Either.cond(nonExisting.isEmpty, ids, notFound[Aoi](nonExisting))

  def userExists(id: UUID): ConnectionIO[Either[ApplicationError, UUID]] = for {
    exists <- UserRepo.existsById(id)
  } yield Either.cond(exists, id, notFound[User](id))

  def aoisReferToOneVectorLayer(ids: List[UUID]): ConnectionIO[Either[ApplicationError, UUID]] =
    for {
      vlIds <- AoiRepo.getAoiVectorLayers(ids)
    } yield referToOneVectorLayer(vlIds, "Aoi")

  def referToOneVectorLayer(
      vlIds: List[UUID],
      entityName: String,
    ): Either[ApplicationError, UUID] = {
    val distinctsVlIds = vlIds.distinct

    if (distinctsVlIds.size == 1)
      vlIds.head.asRight[ApplicationError]
    else {
      val msg =
        s"${entityName}s must refer to exactly one VectorLayer, but they refer to ${distinctsVlIds.size}"
      (BadRequest(msg): ApplicationError).asLeft[UUID]
    }
  }

  def processingIsEmpty(prcId: UUID): ConnectionIO[Either[ApplicationError, UUID]] = for {
    aoiSummary <- AoiRepo.getAoiSummariesByProcessings(List(prcId))
    isEmpty = aoiSummary(prcId).count == 0
  } yield Either.cond(
    isEmpty,
    prcId,
    BadRequest(s"Processing $prcId is not empty."): ApplicationError,
  )

  def loginIsVacant(email: String): ConnectionIO[Either[ApplicationError, String]] = for {
    maybeUser <- UserRepo.getByEmail(email)
  } yield Either.cond(maybeUser.isEmpty, email, LoginTaken(email): ApplicationError)

  def validRole(
      role: Option[Role],
      actor: User,
    ): ConnectionIO[Either[ApplicationError, Option[Role]]] = {
    val isOk = role.contains(Role.User) || actor.role == Role.Admin
    EitherT
      .cond[ConnectionIO](
        isOk,
        role,
        AccessDenied("Only administrator can make user ADMIN"): ApplicationError,
      )
      .value
  }

  def checkPermission(
      user: User,
      permission: Permission,
    ): ConnectionIO[Either[ApplicationError, Unit]] =
    EitherT
      .cond[ConnectionIO](
        user.role.hasPermission(permission),
        (),
        AccessDenied("Access denied"): ApplicationError,
      )
      .value

  def checkProcessingArea(user: User, prcId: UUID): ConnectionIO[Either[ApplicationError, Unit]] =
    if (user.role.hasPermission(LargeProcessing))
      EitherT.rightT[ConnectionIO, ApplicationError](()).value
    else
      for {
        summaries <- AoiRepo.getAoiSummariesByProcessings(List(prcId))
        area = summaries(prcId).area
        userEntity <- UserRepo.getUser(user.id).rethrowT
      } yield Either.cond(
        area <= userEntity.aoiAreaLimit,
        (),
        TooLargeProcessing(area, userEntity.aoiAreaLimit): ApplicationError,
      )

  def checkUserPermission(
      userId: UUID,
      actor: User,
    ): ConnectionIO[Either[ApplicationError, Unit]] = {
    val isOk = actor.role.hasPermission(ViewAnyUser) || actor.id == userId
    EitherT.cond[ConnectionIO](isOk, (), AccessDenied("Access denied"): ApplicationError).value
  }

  /** Check if WD is linked to a user or default
    */
  def checkWorkflowDefPermission(
      workflowDefId: UUID,
      actor: User,
    ): EitherT[ConnectionIO, ApplicationError, Unit] =
    for {
      wds <- EitherT.right[ApplicationError](
        WorkflowDefRepo.listWorkflowDefs(
          ids = Seq(workflowDefId).some,
          userIds = actor.userFilter(ManageWorkflowDefinition).map(List(_)),
        )
      )
      _ <- EitherT.cond[ConnectionIO](
        wds.nonEmpty,
        {},
        NotFound(s"WorkflowDef not found by id $workflowDefId"): ApplicationError,
      )
    } yield {}

  def checkDataProviderAccess(
      dataProviderId: UUID,
      actor: User,
    ): ConnectionIO[Either[ApplicationError, Unit]] = {
    val accessGranted = actor.role.hasPermission(ManageDataProviders) ||
      actor.availableDataProviders.map(_.id).contains(dataProviderId)
    // It would be better co conceal this under 404, but currently this will not work,
    // as we call provider by name, and if ID is returned it is still info
    // todo: make provider call by ID, not name for processing creation and cost calculation
    // todo: How do we explain to the API client that the provider is not available and he has to pay?
    //  Maybe keep 403 for this
    EitherT
      .cond[ConnectionIO](
        accessGranted,
        (),
        AccessDenied(s"Access denied to Data Provider $dataProviderId"): ApplicationError,
      )
      .value
  }

  def canManageTeam(teamId: UUID, actor: User): ConnectionIO[Either[ApplicationError, Unit]] =
    if (actor.role.hasPermission(Permission.ManageTeams)) {
      val either = Either.right[ApplicationError, Unit] {}
      IO.pure(either).to[ConnectionIO]
    }
    else
      for {
        members <- TeamRepo.listMembers(teamId)
        isOwner = members.filter(_.role == TeamMemberRole.OWNER).map(_.userId).contains(actor.id)
      } yield Either.cond(isOwner, (), NotFound(s"Team not found by id $teamId"): ApplicationError)

  def canRunProcessing(actor: User): EitherT[ConnectionIO, ApplicationError, Unit] =
    EitherT.cond[ConnectionIO](
      actor.activeUntil.forall(_.isAfter(Instant.now())),
      {},
      Forbidden(s"User account is suspended ${actor.activeUntil} due a team membership policy"),
    )
}
