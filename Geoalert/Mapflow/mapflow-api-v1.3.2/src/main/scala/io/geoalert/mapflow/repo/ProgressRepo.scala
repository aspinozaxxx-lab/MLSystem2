package io.geoalert.mapflow.repo

import java.time.Instant
import java.util.UUID

import cats.data.NonEmptyList
import cats.implicits.catsSyntaxApplicativeError
import cats.syntax.applicative._
import doobie.Fragment._
import doobie.Fragments._
import doobie._
import doobie.implicits._
import doobie.implicits.legacy.instant._
import doobie.postgres.implicits._
import io.geoalert.mapflow.Config.dbSchema
import io.geoalert.mapflow.Config.maxInFrLen
import io.geoalert.mapflow.implicits.CommonOps._
import io.geoalert.mapflow.model._

object ProgressRepo {
  case class AreaDto(
      id: UUID,
      status: Int,
      area: Long,
      completionDate: Option[Instant],
    )
  case class CountDto(
      id: UUID,
      status: Int,
      count: Int,
    )

  def whereFr(idField: String, ids: List[UUID]): Fragment =
    NonEmptyList
      .fromList(ids)
      .map(ids => whereAnd(in(const(idField), ids)))
      .getOrElse(fr"WHERE false")

  def getProgressDetails(
      ids: List[UUID],
      areaDetails: List[UUID] => ConnectionIO[Map[UUID, List[AreaDto]]],
      countDetails: List[UUID] => ConnectionIO[Map[UUID, List[CountDto]]],
    ): ConnectionIO[Map[UUID, List[ProgressDetail]]] = {
    def batch(ids: List[UUID]) = {
      val countDtos = countDetails(ids)
      val areaDtos = areaDetails(ids)

      def buildDetails(areaDtos: List[AreaDto], countDtos: List[CountDto]) = {
        val countByStatuses = countDtos.groupMap(_.status)(_.count)
        val areaByStatuses = areaDtos.groupMap(_.status)(_.area)
        val completionDateByStatuses = areaDtos
          .map(d => (d.status, if (d.status == Status.Ok.intVal) d.completionDate else None))
          .toMap
        val statuses = (areaDtos.map(_.status) ++ countDtos.map(_.status)).distinct

        statuses.map(s =>
          ProgressDetail(
            s,
            countByStatuses.getOrElse(s, Nil).sum,
            areaByStatuses.getOrElse(s, Nil).sum,
            completionDateByStatuses.getOrElse(s, None),
          )
        )
      }

      for {
        areaDtos <- areaDtos
        countDtos <- countDtos
      } yield for {
        id <- ids
      } yield (id, buildDetails(areaDtos(id), countDtos(id)))
    }

    ids match {
      case Nil => Map[UUID, List[ProgressDetail]]().pure[ConnectionIO]
      case _ => ids.batchTraverse(maxInFrLen)(batch).map(_.toMap)
    }
  }

  def getAreaDetails(
      idField: String,
      fromFr: Fragment,
    )(
      ids: List[UUID]
    ): ConnectionIO[Map[UUID, List[AreaDto]]] = {
    val selectFr = const(
      s"SELECT $idField, coalesce(w.status, a.status), sum(coalesce(w.area, a.area)), max(w.status_update_date) as completion_date FROM"
    )
    val joinWorkflowFr = const(s"JOIN $dbSchema.workflow w ON a.id = w.aoi_id")
    val groupByFr = const(s"GROUP BY $idField, coalesce(w.status, a.status)")
    val sql = selectFr ++ fromFr ++ joinWorkflowFr ++ whereFr(idField, ids) ++ groupByFr

    for {
      dtos <- sql.query[AreaDto].to[List]
    } yield dtos.groupBy(_.id).withDefaultValue(List())
  }

  def getCountDetails(
      idField: String,
      fromFr: Fragment,
      byProjectId: Boolean = false,
    )(
      ids: List[UUID]
    ): ConnectionIO[Map[UUID, List[CountDto]]] = {
    val selectFr =
      if (byProjectId)
        const(s"SELECT $idField, a.status, COUNT(DISTINCT prc.id) FROM")
      else
        const(s"SELECT $idField, a.status, sum(1) FROM")
    val groupByFr = const(s"GROUP BY $idField, a.status")
    val sql = selectFr ++ fromFr ++ whereFr(idField, ids) ++ groupByFr

    for {
      dtos <- sql.query[CountDto].to[List]
    } yield dtos.groupBy(_.id).withDefaultValue(List())
  }

  def getAoiCountDetails(ids: List[UUID]): ConnectionIO[Map[UUID, List[CountDto]]] = {
    val selectFr = const(s"SELECT a.id, w.status, sum(1) FROM")
    val fromFr = const(s" $dbSchema.aoi a JOIN $dbSchema.workflow w ON a.id = w.aoi_id")
    val groupByFr = const(s"GROUP BY a.id, w.status")
    val sql = selectFr ++ fromFr ++ whereFr("a.id", ids) ++ groupByFr

    for {
      dtos <- sql.query[CountDto].to[List]
    } yield dtos.groupBy(_.id).withDefaultValue(List())
  }

  def getProjectProgressDetails(ids: List[UUID]): ConnectionIO[Map[UUID, List[ProgressDetail]]] = {
    val fromFr =
      const(s"$dbSchema.project prj JOIN $dbSchema.processing prc ON prc.project_id = prj.id and prc.archived = false") ++
        const(s"JOIN $dbSchema.aoi a ON a.processing_id = prc.id")
    getProgressDetails(
      ids,
      getAreaDetails("prj.id", fromFr ++ fr"LEFT"),
      getCountDetails("prj.id", fromFr, byProjectId = true),
    )
  }

  def getProcessingsProgressDetails(
      ids: List[UUID]
    ): ConnectionIO[Map[UUID, List[ProgressDetail]]] = {
    val fromFr = const(
      s"$dbSchema.processing prc JOIN $dbSchema.aoi a ON a.processing_id = prc.id "
    )
    getProgressDetails(
      ids,
      getAreaDetails("prc.id", fromFr ++ fr"LEFT"),
      getCountDetails("prc.id", fromFr),
    )
  }

  def getAoiProgressDetails(ids: List[UUID]): ConnectionIO[Map[UUID, List[ProgressDetail]]] =
    getProgressDetails(ids, getAreaDetails("a.id", const(s"$dbSchema.aoi a")), getAoiCountDetails)
}
