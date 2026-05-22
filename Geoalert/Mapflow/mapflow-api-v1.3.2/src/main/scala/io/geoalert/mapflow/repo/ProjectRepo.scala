package io.geoalert.mapflow.repo

import java.time.Instant
import java.util.UUID

import cats.data.EitherT
import cats.data.NonEmptyList
import cats.syntax.applicative._
import cats.syntax.option._
import doobie.Fragments._
import doobie._
import doobie.implicits._
import doobie.implicits.legacy.instant._
import doobie.postgres.implicits._
import doobie.util.fragment.Fragment.const
import io.geoalert.mapflow.exception.NotFound
import io.geoalert.mapflow.implicits.CommonOps._
import io.geoalert.mapflow.implicits.Postgres._
import io.geoalert.mapflow.model._
import io.geoalert.mapflow.model.enums.MemberRole

case class ProjectDto(
    id: UUID,
    name: String,
    description: Option[String],
    userId: UUID,
    isDefault: Boolean,
    created: Instant,
    updated: Instant,
    archived: Boolean,
    defaultWds: Boolean,
  )

object ProjectRepo
    extends GenericRepo[ProjectDto](
      "project",
      Seq(
        "id",
        "name",
        "description",
        "user_id",
        "is_default",
        "created",
        "updated",
        "archived",
        "default_wds",
      ),
    ) {
  def filterProjects(userId: Option[UUID], filter: Option[String]): ConnectionIO[List[UUID]] = {
    val sql = const(
      s"SELECT pr.id FROM $dbSchema.$table pr INNER JOIN $dbSchema.user_projects up ON pr.id = up.project_id INNER JOIN $dbSchema.app_user u ON up.user_id = u.id"
    ) ++
      whereAndOpt(
        fr"pr.archived = false and up.role = 'owner'".some,
        userId.map(id => fr"up.user_id = $id"),
        filter
          .map("%" + _.toLowerCase() + "%")
          .map(f =>
            fr"(LOWER(u.email) LIKE $f OR LOWER(u.name) LIKE $f OR LOWER(u.preferred_username) LIKE $f OR LOWER(pr.name) LIKE $f)"
          ),
      )

    sql.query[UUID].to[List]
  }

  def getProjects(
      ids: Seq[UUID],
      includeArchived: Boolean = false,
      isDefault: Option[Boolean] = None,
      offset: Option[Int] = None,
      limit: Option[Int] = None,
    ): ConnectionIO[List[ProjectDto]] =
    if (ids.nonEmpty)
      getAllByIdsWhere(
        ids,
        Option.unless(includeArchived)(fr"archived = false"),
        isDefault.map(boolean => fr"is_default = $boolean"),
      )
    else
      getAllWhere(
        false,
        Option.unless(includeArchived)(fr"archived = false"),
        isDefault.map(boolean => fr"is_default = $boolean"),
      )(offset, limit)

  def getProject(
      id: UUID,
      includeArchived: Boolean = false,
    ): EitherT[ConnectionIO, NotFound, ProjectDto] =
    getProjects(List(id), includeArchived).headOrNotFound(id)

  def getProjectIdsByProcessings(processingIds: List[UUID]): ConnectionIO[Map[UUID, UUID]] =
    NonEmptyList.fromList(processingIds) match {
      case Some(prcIds) =>
        val sql = fr"SELECT prc.id, prc.project_id" ++
          const(s"FROM $dbSchema.processing prc WHERE ") ++ in(fr"prc.id", prcIds)
        sql.query[(UUID, UUID)].to[List].map(_.toMap)
      case None => Map[UUID, UUID]().pure[ConnectionIO]
    }

  def getProjectProgress(
      projectIds: List[UUID]
    ): ConnectionIO[List[ProjectProgress]] =
    (const(s"SELECT * FROM $dbSchema.project_estimate") ++ whereAndOpt(
      NonEmptyList.fromList(projectIds).map(in(const("project_id"), _))
    )).query[ProjectProgress].to[List]

  def createProject(
      p: CreateProjectInput,
      userId: UUID,
      isDefault: Boolean = false,
    ): ConnectionIO[UUID] = {
    val fields = Seq(
      Some("name" -> p.name),
      p.description.map("description" -> _),
      Some("user_id" -> userId),
      Some("is_default" -> isDefault),
      p.addDefaultWds.map("default_wds" -> _),
      Some("created" -> Instant.now()),
      Some("updated" -> Instant.now()),
    ).flatten.toMap

    create(fields)
  }

  def updateProject(p: UpdateProjectInput): ConnectionIO[Unit] = {
    // TODO consider shapeless to get rid of boilerplate
    val fields = Seq(
      p.name.map("name" -> _),
      p.description.map("description" -> _),
      Some("updated" -> Instant.now()),
    ).flatten.toMap

    updateByIdWhere(fields, p.id)
  }
}
