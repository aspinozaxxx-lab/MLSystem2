package io.geoalert.mapflow.repo

import java.time.Instant
import java.util.UUID

import cats.data.EitherT
import cats.data.NonEmptyList
import cats.syntax.applicative._
import cats.syntax.option._
import doobie.Fragment._
import doobie.Fragments._
import doobie._
import doobie.implicits._
import doobie.implicits.legacy.instant._
import doobie.postgres.implicits._
import doobie.util.fragment.Fragment
import io.geoalert.mapflow.exception.NotFound
import io.geoalert.mapflow.implicits.CommonOps._
import io.geoalert.mapflow.implicits.Postgis._
import io.geoalert.mapflow.implicits.Postgres._
import io.geoalert.mapflow.model.Status._
import io.geoalert.mapflow.model._

import geotrellis.vector.Geometry
import geotrellis.vector.Projected

case class WorkflowDto(
    id: UUID,
    aoiId: UUID,
    wdId: UUID,
    externalId: Option[String],
    geometry: Projected[Geometry],
    area: Long,
    status: Int,
    statusUpdateDate: Option[Instant],
    createDate: Option[Instant],
    requiredAction: Option[RequiredAction],
    locked: Boolean,
    lockedAt: Option[Instant],
    failedCount: Int,
  )

object WorkflowRepo
    extends GenericRepo[WorkflowDto](
      "workflow",
      Seq(
        "id",
        "aoi_id",
        "workflow_def_id",
        "external_id",
        "geometry",
        "area",
        "status",
        "status_update_date",
        "create_date",
        "required_action",
        "locked",
        "locked_at",
        "failed_count",
      ),
    ) {
  def getWorkflow(id: UUID): EitherT[ConnectionIO, NotFound, Workflow] =
    find(forUpdate = false, None, fr"id = $id".some)()
      .headOrNotFound(id)
  def getWorkflowByAoiId(id: UUID): ConnectionIO[List[Workflow]] =
    find(forUpdate = false, None, fr"aoi_id = $id".some)()

  def getExternalIdsByAoiIds(aoiIds: Seq[UUID]): ConnectionIO[Map[UUID, List[String]]] = {
    def batch(ids: List[UUID]) = NonEmptyList.fromList(ids) match {
      case Some(ids) =>
        val sql =
          const(
            s"SELECT aoi_id, external_id FROM $dbSchema.workflow WHERE NOT external_id IS NULL AND "
          ) ++ in(
            fr"aoi_id",
            ids,
          )
        for {
          externalIdsByAoiIds <- sql.query[(UUID, String)].to[List]
        } yield externalIdsByAoiIds
      case None => List[(UUID, String)]().pure[ConnectionIO]
    }

    for {
      list <- aoiIds.toList.batchTraverse(maxInFrLen)(batch)
    } yield list.groupBy(_._1).view.mapValues(_.map(_._2)).toMap.withDefaultValue(List[String]())
  }

  def find(
      forUpdate: Boolean,
      limit: Option[Int],
      where: Option[Fragment]*
    )(
      orderBy: Option[Fragment] = None
    ): ConnectionIO[List[Workflow]] =
    for {
      dtos <- getAllWhere(forUpdate, where: _*)(limit = limit, orderBy = orderBy)
      wds <- WorkflowDefRepo.getAllByIds(dtos.map(_.wdId).distinct)
      wdsByIds = wds.map(wd => (wd.id, wd)).toMap
    } yield for {
      d <- dtos
    } yield Workflow(
      d.id,
      d.aoiId,
      wdsByIds.getOrElse(d.wdId, throw new NotFound(s"WorkflowDef not found by id ${d.wdId}")),
      d.externalId,
      d.geometry,
      d.area,
      Status(d.status),
      d.statusUpdateDate,
      d.requiredAction,
      d.locked,
      d.lockedAt,
      d.failedCount,
    )

  def setWorkflowsLocked(
      ids: NonEmptyList[UUID],
      bool: Boolean,
      lockedAt: Option[Instant] = None,
    ): ConnectionIO[Unit] = {
    val fields = Seq(
      Some("locked" -> bool),
      lockedAt.map("locked_at" -> _),
    ).flatten.toMap

    update(fields, in(fr"id", ids).some)
  }

  def workflowsAreLockedAndNotNullActions(lockedAt: Option[Instant] = None): ConnectionIO[List[Workflow]] =
    find(
      forUpdate = false,
      100.some,
      fr"required_action IS NOT NULL".some,
      fr"locked = true".some,
      lockedAt.map(v => fr"locked_at <= $v"),
    )()

  def workflowsAreNotNullActions(locked: Boolean): ConnectionIO[List[Workflow]] =
    find(
      forUpdate = false,
      100.some,
      fr"required_action IS NOT NULL".some,
      fr"locked = $locked".some,
    )(orderBy = fr"ORDER BY create_date".some)

  def findRunningWorkflowIdsByProcessing(
      processingIds: NonEmptyList[UUID]
    ): ConnectionIO[List[UUID]] =
    (const(
      s"SELECT w.id FROM $dbSchema.$table w JOIN $dbSchema.aoi a on w.aoi_id = a.id WHERE "
    ) ++ and(
      in(fr"a.processing_id", processingIds),
      fr"w.status = ${Status.InProgress.intVal}",
    ))
      .query[UUID]
      .to[List]

  def getWorkflowsToUpdate: ConnectionIO[List[WorkflowSummary]] = for {
    data <- (const(
      s"SELECT id, aoi_id, external_id, status, status_update_date FROM $dbSchema.$table WHERE"
    ) ++
      fr"required_action IS NULL AND (status = ${Status.InProgress.intVal} OR (status = ${Status.Ok.intVal} AND status_update_date IS NULL)) ORDER BY create_date IS NULL, create_date DESC")
      .query[(UUID, UUID, Option[String], Int, Option[Instant])]
      .to[List]
  } yield data.map {
    case (id, aoiId, extId, status, statusUpdateDate) =>
      WorkflowSummary(id, aoiId, extId, Status(status), statusUpdateDate)
  }

  def createWorkflow(
      aoiId: UUID,
      wdId: UUID,
      geom: Projected[Geometry],
      area: Long,
    ): ConnectionIO[UUID] = {
    val fields = Map[String, Any](
      "aoi_id" -> aoiId,
      "workflow_def_id" -> wdId,
      "geometry" -> geom,
      "area" -> area,
      "status" -> InProgress.intVal,
      "create_date" -> Instant.now(),
      "required_action" -> fr"${RequiredAction.start}",
    )

    create(fields)
  }

  /** Find all failed processing and restart it
    */
  def restartFailedWorkflowsInProcessing(processingId: UUID): ConnectionIO[Int] = {
    val sql =
      const(
        s"UPDATE $dbSchema.workflow"
      ) ++ fr"SET required_action = ${RequiredAction.restart}, status = ${InProgress.intVal} WHERE" ++
        fr"aoi_id in ( " ++
        const(
          s"SELECT a.id FROM $dbSchema.aoi a JOIN $dbSchema.processing p ON a.processing_id = p.id"
        ) ++ fr"WHERE p.archived = FALSE AND p.id = $processingId" ++
        fr") AND (status = ${Failed.intVal} OR status = ${Cancelled.intVal})"

    sql.update.run
  }
  def updateWorkflow(
      id: UUID,
      status: Status,
      statusUpdateDate: Instant,
      externalId: Option[String],
    ): ConnectionIO[Unit] = {
    val fields = Seq(
      ("status" -> status.intVal).some,
      ("status_update_date" -> statusUpdateDate).some,
      externalId.map("external_id" -> _),
    )
      .flatten
      .toMap
    updateById(fields, id)
  }

  def updateWorkflowFailedCount(id: UUID, count: Int): ConnectionIO[Unit] =
    updateById(Map("failed_count" -> count), id)

  def setRequiredAction(
      ids: NonEmptyList[UUID],
      action: Option[RequiredAction],
    ): ConnectionIO[Unit] =
    update(
      Map("required_action" -> action.map(v => fr"$v").getOrElse(fr"NULL")),
      in(fr"id", ids).some,
    )
}
