package io.geoalert.mapflow.repo

import java.time.Instant
import java.util.UUID

import cats.data.NonEmptyList
import cats.data.NonEmptySeq
import cats.implicits.catsSyntaxApplicativeId
import cats.syntax.option._
import doobie.Fragments._
import doobie._
import doobie.implicits._
import doobie.implicits.legacy.instant._
import doobie.postgres.implicits._
import doobie.util.fragment.Fragment.const
import io.geoalert.mapflow.exception.NotFound
import io.geoalert.mapflow.model._

object WorkflowDefRepo
    extends GenericRepo[WorkflowDef](
      "workflow_def",
      Seq(
        "id",
        "name",
        "description",
        "we_id",
        "we_name",
        "yml",
        "created",
        "updated",
        "archived",
        "is_default",
      ),
    ) {
  def getWorkflowDef(id: UUID, userFilter: Option[UUID]): ConnectionIO[WorkflowDef] = {
    val opt = userFilter match {
      case Some(userId) =>
        val sql = const("SELECT " + columns.map("wd." + _).mkString(", ") + " ") ++
          const(
            s"FROM $dbSchema.workflow_def wd LEFT JOIN $dbSchema.user_workflow_def uwd ON uwd.workflow_def_id = wd.id "
          ) ++
          fr"WHERE (uwd.user_id = $userId OR wd.is_default IS TRUE) AND wd.archived = FALSE" ++
          fr"AND wd.id = $id"

        sql.query[WorkflowDef].to[List].map(_.headOption)
      case None =>
        getOneById(id)
    }

    opt.map(_.getOrElse(throw NotFound[WorkflowDef](id)))
  }

  def getLinkedToProjects(projectIds: Seq[UUID]): ConnectionIO[Map[UUID, List[WorkflowDef]]] =
    NonEmptyList.fromList(projectIds.toList) match {
      case Some(ids) =>
        val sql = const("SELECT pwd.project_id, " + columns.map("wd." + _).mkString(", ")) ++
          const(
            s"FROM $dbSchema.workflow_def wd JOIN $dbSchema.project_workflow_def pwd ON pwd.workflow_def_id = wd.id "
          ) ++
          fr"WHERE " ++ in(fr"pwd.project_id", ids) ++ fr" AND wd.archived = FALSE"

        for {
          items <- sql.query[(UUID, WorkflowDef)].to[List]
        } yield items
          .groupBy(_._1)
          .view
          .mapValues(_.map(_._2))
          .toMap
          .withDefaultValue(List())
      case None => Map[UUID, List[WorkflowDef]]().pure[ConnectionIO]
    }

  def listLinkedUsers(workflowDefId: UUID): ConnectionIO[List[UUID]] = {
    val sql = fr"SELECT uwd.user_id " ++
      const(
        s"FROM $dbSchema.workflow_def wd JOIN $dbSchema.user_workflow_def uwd ON uwd.workflow_def_id = wd.id "
      ) ++
      fr"WHERE wd.id = $workflowDefId AND wd.archived = FALSE"

    sql.query[UUID].to[List]
  }

  def listDefaultWorkflowDefs: ConnectionIO[List[WorkflowDef]] =
    getAllWhere(forUpdate = false, fr"is_default = TRUE".some, fr"archived = FALSE".some)()

  def listUserWorkflowDefs(userIds: Seq[UUID]): ConnectionIO[Map[UUID, List[WorkflowDef]]] = {
    val sql = const("SELECT uwd.user_id, " + columns.map("wd." + _).mkString(", ")) ++
      const(s"FROM $dbSchema.workflow_def wd ") ++
      const(s"LEFT JOIN $dbSchema.user_workflow_def uwd ON uwd.workflow_def_id = wd.id ") ++
      fr"WHERE " ++
      andOpt(
        fr"wd.archived = FALSE".some,
        fr"wd.is_default = FALSE".some,
        NonEmptySeq.fromSeq(userIds).map(in(fr"uwd.user_id", _)),
      ) ++
      fr"group by uwd.user_id, wd.id"

    for {
      pairs <- sql
        .query[(UUID, WorkflowDef)]
        .to[List]
    } yield pairs.groupMap(_._1)(_._2)
  }

  def listWorkflowDefs(
      ids: Option[Seq[UUID]] = None,
      userIds: Option[Seq[UUID]] = None,
      projectIds: Option[Seq[UUID]] = None,
      isDefault: Option[Boolean] = None,
    ): ConnectionIO[List[WorkflowDef]] = {
    val sql = const("SELECT " + columns.map("wd." + _).mkString(", ")) ++
      const(s"FROM $dbSchema.workflow_def wd ") ++
      const(s"LEFT JOIN $dbSchema.user_workflow_def uwd ON uwd.workflow_def_id = wd.id ") ++
      const(s"LEFT JOIN $dbSchema.project_workflow_def pwd ON pwd.workflow_def_id = wd.id ") ++
      fr"WHERE " ++
      andOpt(
        fr"wd.archived = FALSE".some,
        ids.flatMap(NonEmptySeq.fromSeq).map(in(fr"wd.id", _)),
        projectIds.flatMap(NonEmptySeq.fromSeq).map(in(fr"pwd.project_id", _)),
        userIds.flatMap(NonEmptySeq.fromSeq).map(in(fr"uwd.user_id", _)),
        isDefault.map(value => fr"is_default = $value"),
      ) ++
      fr"group by wd.id"

    sql.query[WorkflowDef].to[List]
  }

  def createWorkflowDef(
      wd: CreateWorkflowDefInput,
      weId: Long,
      weName: String,
      yml: String,
    ): ConnectionIO[UUID] = {
    val fields = Seq(
      ("name" -> wd.name).some,
      wd.description.map("description" -> _),
      ("we_id" -> weId).some,
      ("we_name" -> weName).some,
      ("yml" -> yml).some,
      ("created" -> Instant.now()).some,
      ("updated" -> Instant.now()).some,
      wd.pricePerSqKm.map("price_per_sq_km" -> _),
      ("archived" -> false).some,
      wd.isDefault.map("is_default" -> _),
    ).flatten.toMap
    create(fields)
  }

  def updateWorkflowDef(wd: UpdateWorkflowDefInput, yml: Option[String]): ConnectionIO[Unit] = {
    val fields = Seq(
      wd.name.map("name" -> _),
      wd.description.map("description" -> _),
      yml.map("yml" -> _),
      Some("updated" -> Instant.now()),
      wd.pricePerSqKm.map("price_per_sq_km" -> _),
      wd.isDefault.map("is_default" -> _),
    ).flatten.toMap
    updateById(fields, wd.id)
  }

  def archiveWorkflowDef(id: UUID): ConnectionIO[Unit] =
    updateById(Map("archived" -> true), id)

  def linkToUser(workflowDefId: UUID, userId: UUID): ConnectionIO[Int] = {
    val sql =
      const(
        s"INSERT INTO $dbSchema.user_workflow_def (user_id, workflow_def_id) "
      ) ++ sql"VALUES ($userId, $workflowDefId) ON CONFLICT DO NOTHING"
    sql.update.run
  }

  def unlinkFromUser(workflowDefId: UUID, userId: UUID): ConnectionIO[Int] = {
    val sql =
      const(
        s"DELETE FROM $dbSchema.user_workflow_def "
      ) ++ sql"WHERE user_id=$userId AND workflow_def_id=$workflowDefId"
    sql.update.run
  }

  def linkToProject(workflowDefId: UUID, projectId: UUID): ConnectionIO[Int] = {
    val sql =
      const(
        s"INSERT INTO $dbSchema.project_workflow_def (project_id, workflow_def_id) "
      ) ++ sql"VALUES ($projectId, $workflowDefId) ON CONFLICT DO NOTHING"
    sql.update.run
  }

  def unlinkFromProject(workflowDefId: UUID, projectId: UUID): ConnectionIO[Int] = {
    val sql =
      const(
        s"DELETE FROM $dbSchema.project_workflow_def"
      ) ++ sql" WHERE project_id=$projectId AND workflow_def_id=$workflowDefId"
    sql.update.run
  }
}
