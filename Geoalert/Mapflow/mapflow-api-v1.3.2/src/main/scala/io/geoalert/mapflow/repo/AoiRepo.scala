package io.geoalert.mapflow.repo

import java.time.Instant
import java.util.UUID

import scala.collection.immutable.ArraySeq

import cats.data.EitherT
import cats.data.NonEmptyList
import cats.data.NonEmptySeq
import cats.implicits.toFunctorOps
import cats.implicits.toTraverseOps
import cats.syntax.applicative._
import cats.syntax.apply._
import cats.syntax.option._
import doobie.ConnectionIO
import doobie.Fragment._
import doobie.Fragments._
import doobie._
import doobie.implicits._
import doobie.implicits.legacy.instant._
import doobie.postgres.implicits._
import io.circe.Json
import io.circe.JsonObject
import io.circe.syntax._
import io.geoalert.mapflow.Config._
import io.geoalert.mapflow.exception.NotFound
import io.geoalert.mapflow.implicits.CommonOps._
import io.geoalert.mapflow.implicits.GeometryOps._
import io.geoalert.mapflow.implicits.Postgis._
import io.geoalert.mapflow.implicits.Postgres._
import io.geoalert.mapflow.model._

import geotrellis.vector._

case class AoiDto(
    id: UUID,
    processingId: UUID,
    geometry: Projected[Geometry],
    area: Long,
    status: Int,
    percentCompleted: Int,
    messages: Option[Json],
    completionDate: Option[Instant],
    vrtUri: Option[String],
    startTime: Option[Instant],
  )

object AoiRepo
    extends GenericRepo[AoiDto](
      "aoi",
      Seq(
        "id",
        "processing_id",
        "geometry",
        "area",
        "status",
        "percent_completed",
        "messages",
        "completion_date",
        "vrt_uri",
        "start_time",
      ),
    ) {
  private val bboxFr = fr"ST_SetSRID(ST_Envelope(ST_Extent(geometry)), 4326)"

  def getAois(ids: Option[Seq[UUID]], userId: Option[UUID]): ConnectionIO[List[AoiDto]] =
    getAoisWithFilter(AoiFilter(ids = ids.listOpt, geometry = None), userId)

  def getAoisWithFilter(
      filter: AoiFilter,
      userId: Option[UUID],
      sort: Option[List[AoiSortEntry]] = None,
      offset: Option[Int] = None,
      limit: Option[Int] = None,
    ): ConnectionIO[List[AoiDto]] = {
    val sortFr = {
      val fields = sort match {
        case Some(es @ _ :: _) =>
          es.map { e =>
            e.field.field + e.desc.filter(d => d).map(_ => " DESC").getOrElse("")
          }
        case _ => List()
      }
      const(s"ORDER BY ${(fields :+ idField).mkString(", ")}")
    }

    val sql = fr"SELECT" ++ const(columns.map(c => s"a.$c").mkString(", ")) ++
      fromWhereWithFilter(filter, userId) ++
      sortFr ++
      limit.map(n => const(s"LIMIT $n")).getOrElse(fr"") ++
      offset.map(n => const(s"OFFSET $n")).getOrElse(fr"")

    sql.query[AoiDto].to[List]
  }

  def getAoiIdsWithFilter(filter: AoiFilter, userId: Option[UUID]): ConnectionIO[List[UUID]] = {
    def batch(filter: AoiFilter) =
      (const(s"SELECT a.$idField") ++ fromWhereWithFilter(filter, userId)).query[UUID].to[List]

    filter match {
      case f @ AoiFilter(Some(allIds), _, _, _) =>
        allIds.batchTraverse(maxInFrLen)(ids => batch(f.copy(ids = ids.some)))
      case f => batch(f)
    }
  }

  private def fromWhereWithFilter(filter: AoiFilter, userId: Option[UUID]): Fragment = {
    val userFr = userId.map(u => fr"up.user_id = $u").getOrElse(fr"true")
    val filterFrArr = filter match {
      case AoiFilter(None, None, None, None) => Array(userFr.some)
      case AoiFilter(maybeIds, maybePrcIds, maybeSs, maybeGeom) =>
        Array(
          maybeIds.map(listToFr(s"a.$idField")),
          maybePrcIds.map(listToFr("a.processing_id")),
          maybeSs.map(ss => listToFr("a.status")(ss.map(_.intVal))),
          maybeGeom.map(g => fr"ST_Intersects($g, a.geometry)"),
        )
    }
    const(s"FROM $dbSchema.aoi a JOIN $dbSchema.processing prc ON prc.id = a.processing_id ") ++
      const(
        s"JOIN (SELECT DISTINCT ON(project_id) * FROM $dbSchema.user_projects up WHERE "
      ) ++ userFr ++ fr") up ON prc.project_id = up.project_id" ++
      whereAndOpt(ArraySeq.unsafeWrapArray(filterFrArr): _*)
  }

  def getAoi(id: UUID, userId: Option[UUID]): EitherT[ConnectionIO, NotFound, AoiDto] =
    getAois(List(id).some, userId).headOrNotFound(id)

  def getProcessingAois(processingId: UUID, userId: Option[UUID]): ConnectionIO[List[AoiDto]] = {
    val userFr = userId.map(u => fr"up.user_id = $u ")
    val processingFr = fr"a.processing_id = $processingId "

    val sql = fr"SELECT" ++ const(columns.map(c => s"a.$c").mkString(", ")) ++
      const(
        s"FROM $dbSchema.aoi a " +
          s"JOIN $dbSchema.processing prc ON prc.id = a.processing_id " +
          s"JOIN $dbSchema.user_projects up ON prc.project_id = up.project_id"
      ) ++
      whereAndOpt(userFr, processingFr.some) ++
      fr"GROUP BY a.id"

    sql.query[AoiDto].to[List]
  }

  def getAoiGeom(aoiIds: List[UUID]): ConnectionIO[List[Projected[Geometry]]] =
    NonEmptyList.fromList(aoiIds) match {
      case Some(ids) =>
        (const(s"SELECT geometry FROM $dbSchema.$table a WHERE ") ++
          in(fr"a.id", ids)).query[Projected[Geometry]].to[List]
      case None => List[Projected[Geometry]]().pure[ConnectionIO]
    }

  def getAoiVectorLayers(aoiIds: List[UUID]): ConnectionIO[List[UUID]] = {
    def getBatch(aoiIds: List[UUID]): ConnectionIO[List[UUID]] =
      NonEmptyList.fromList(aoiIds) match {
        case Some(ids) =>
          (const(s"SELECT DISTINCT p.vector_layer_id FROM $dbSchema.$table a") ++
            const(s"JOIN $dbSchema.processing p ON a.processing_id = p.id WHERE") ++
            in(fr"a.id", ids))
            .query[UUID]
            .to[List]
        case None => List[UUID]().pure[ConnectionIO]
      }
    aoiIds.batchTraverse(maxInFrLen)(getBatch).map(_.distinct)
  }

  def getAoisProcessings(aoiIds: List[UUID]): ConnectionIO[Map[UUID, UUID]] = {
    def getBatch(aoiIds: List[UUID]): ConnectionIO[List[(UUID, UUID)]] =
      NonEmptyList.fromList(aoiIds) match {
        case Some(ids) =>
          val sql =
            const(s"SELECT $idField, processing_id FROM $dbSchema.$table WHERE") ++ in(
              const(idField),
              ids,
            )
          for {
            list <- sql.query[(UUID, UUID)].to[List]
          } yield list
        case None => List[(UUID, UUID)]().pure[ConnectionIO]
      }
    aoiIds.batchTraverse(maxInFrLen)(getBatch).map(_.toMap)
  }

  def getAoiProcessing(aoiId: UUID): ConnectionIO[UUID] = {
    val processings = getAoisProcessings(List(aoiId))
    processings.map(
      _.getOrElse(
        aoiId,
        throw new IllegalStateException(s"DB inconsistency, cannot find processing for AOI $aoiId"),
      )
    )
  }

  def getAoiSummaryWithFilter(filter: AoiFilter, userId: Option[UUID]): ConnectionIO[AoiSummary] = {
    case class Dto(
        count: Option[Int],
        area: Option[Long],
        bbox: Option[Projected[Geometry]],
      )

    def batch(filter: AoiFilter): ConnectionIO[List[AoiSummary]] = for {
      dtos <- (const("SELECT COUNT(a.id), SUM(a.area), ") ++ bboxFr ++ fr" FROM (" ++
        const(
          s"SELECT DISTINCT ON (a.id) a.*"
        ) ++ fromWhereWithFilter(
          filter,
          userId,
        ) ++ fr") as a GROUP BY a.id")
        .query[Dto]
        .to[List]
    } yield dtos
      .flatMap { d =>
        (d.count, d.area, d.bbox.map(_.geom.extent)).mapN(AoiSummary)
      }

    filter match {
      case f @ AoiFilter(Some(allIds), _, _, _) =>
        for {
          aoiSums <- allIds.batchTraverse(maxInFrLen)(ids => batch(f.copy(ids = ids.some)))
          res = aoiSums match {
            case head :: tail =>
              tail.foldLeft(head)((acc, a) =>
                AoiSummary(acc.count + a.count, acc.area + a.area, acc.bbox combine a.bbox)
              )
            case _ => AoiSummary(0, 0, Extent(0.0, 0.0, 0.0, 0.0))
          }
        } yield res
      case f => batch(f).map(_.headOption.getOrElse(AoiSummary(0, 0, Extent(0.0, 0.0, 0.0, 0.0))))
    }
  }

  def getAoiSummariesByProcessings(
      processingIds: Seq[UUID]
    ): ConnectionIO[Map[UUID, AoiSummary]] = {
    case class Dto(
        count: Int,
        area: Long,
        bbox: Projected[Geometry],
      )

    def getBatch(parentIds: List[UUID]): ConnectionIO[List[(UUID, AoiSummary)]] =
      NonEmptyList.fromList(parentIds) match {
        case Some(ids) =>
          val sql = (const(s"SELECT processing_id, SUM(1), SUM(area),") ++ bboxFr ++ const(
            s"FROM $dbSchema.$table WHERE"
          )
            ++ in(const("processing_id"), ids) ++ const(s"GROUP BY processing_id"))
          for {
            list <- sql.query[(UUID, Dto)].to[List]
          } yield list.map {
            case (id, Dto(count, area, bbox)) => (id, AoiSummary(count, area, bbox.geom.extent))
          }
        case None => List[(UUID, AoiSummary)]().pure[ConnectionIO]
      }

    processingIds
      .toList
      .batchTraverse(maxInFrLen)(getBatch)
      .map(_.toMap.withDefaultValue(AoiSummary(0, 0L, Extent(0.0, 0.0, 0.0, 0.0))))
  }

  def getAoiSummariesByProjects(projectIds: Seq[UUID]): ConnectionIO[Map[UUID, AoiSummary]] = {
    case class Dto(
        count: Int,
        area: Long,
        bbox: Projected[Geometry],
      )

    NonEmptyList.fromList(projectIds.toList) match {
      case Some(ids) =>
        val sql =
          (const(s"SELECT p.project_id, SUM(1), SUM(area),") ++ bboxFr ++ const(
            s"FROM $dbSchema.$table a"
          )
            ++ const(
              s"JOIN $dbSchema.processing p ON p.id = a.processing_id and p.archived = false WHERE"
            )
            ++ in(fr"p.project_id", ids) ++ const(s"GROUP BY p.project_id"))
        for {
          list <- sql.query[(UUID, Dto)].to[List]
          summaries = list.map {
            case (id, Dto(count, area, bbox)) => (id, AoiSummary(count, area, bbox.geom.extent))
          }
        } yield summaries.toMap.withDefaultValue(AoiSummary(0, 0L, Extent(0.0, 0.0, 0.0, 0.0)))
      case None => Map[UUID, AoiSummary]().pure[ConnectionIO]
    }
  }

  def getAoiSummary(ids: List[UUID]): ConnectionIO[Map[UUID, AoiSummary]] = {
    case class Dto(
        count: Int,
        area: Long,
        bbox: Projected[Geometry],
      )

    NonEmptyList.fromList(ids) match {
      case Some(value) =>
        val sql =
          fr"SELECT a.id, SUM(1), SUM(a.area)," ++ bboxFr ++ const(
            s"FROM $dbSchema.$table a WHERE"
          ) ++
            in(fr"a.id", value) ++
            fr"GROUP BY a.id"

        for {
          list <- sql.query[(UUID, Dto)].to[List]
          summaries = list.map {
            case (id, Dto(count, area, bbox)) => (id, AoiSummary(count, area, bbox.geom.extent))
          }
        } yield summaries.toMap.withDefaultValue(AoiSummary(0, 0L, Extent(0.0, 0.0, 0.0, 0.0)))
      case None => Map[UUID, AoiSummary]().pure[ConnectionIO]
    }
  }

  def getGeojsonLayer(
      processingId: UUID,
      bbox: Extent,
      xRes: Int,
      yRes: Int,
    ): ConnectionIO[String] = {
    case class Dto(
        id: UUID,
        status: Int,
        percentCompleted: Int,
        radius: Option[Int],
        geometry: Option[String],
      )

    val (xBbox, yBbox) = (bbox.width, bbox.height)

    val widthFr = fr"(ST_XMax(a.geometry) - ST_XMin(a.geometry))"
    val heightFr = fr"(ST_YMax(a.geometry) - ST_YMin(a.geometry))"

    val maxDimFr = {
      val xDimFr = widthFr ++ fr" * $xRes / $xBbox"
      val yDimFr = heightFr ++ fr" * $yRes / $yBbox"
      fr"GREATEST(" ++ xDimFr ++ fr"," ++ yDimFr ++ fr")"
    }

    val pointCondFr = maxDimFr ++ fr" < $pointThreshold"

    val geomFr = {
      val smpl = geojsonLayerSmplMax min bbox.width / geojsonLayerSmplFactor
      fr"ST_AsGeoJSON(CASE WHEN" ++ pointCondFr ++
        fr"THEN ST_Centroid(ST_Envelope(a.geometry))" ++
        fr"ELSE ST_Simplify(a.geometry, $smpl) END, 8)"
    }

    val radiusFr =
      fr"CASE WHEN" ++ pointCondFr ++ fr"THEN GREATEST(1," ++ maxDimFr ++ fr") ELSE NULL END AS radius"

    val sql = const(
      s"SELECT a.$idField, a.status, a.percent_completed,"
    ) ++ radiusFr ++ fr"," ++ geomFr ++
      const(s"FROM $dbSchema.$table a WHERE") ++
      fr"a.processing_id = $processingId AND ST_Intersects(a.geometry, ${bbox.toPolygon().withSRID(4326)})" ++
      fr"ORDER BY a.area DESC LIMIT 8000"

    val features = for {
      dtos <- sql.query[Dto].to[List]
    } yield for {
      d <- dtos if d.geometry.isDefined
      props = JsonObject(
        "id" -> Json.fromString(d.id.toString),
        "status" -> Json.fromString(Status(d.status).repr),
        "percentCompleted" -> Json.fromInt(d.percentCompleted),
        "radius" -> d.radius.map(Json.fromInt).getOrElse(Json.Null),
      )
    } yield s"""{"type": "Feature", "geometry": ${d.geometry.get}, "properties": ${props.asJson.noSpaces}}"""

    features.map(fs => s"""{"type": "FeatureCollection", "features": [${fs.mkString(", ")}]}""")
  }

  def getMvtLayer(processingId: UUID, bbox: Extent): ConnectionIO[Array[Byte]] = {
    val smpl = mvtLayerSmplMax min bbox.width / mvtLayerSmplFactor
    val bboxPoly = bbox.toPolygon().withSRID(4326)

    val sql = const(s"SELECT ST_AsMVT(frows, '$dbSchema.$table', 4096, 'geom') FROM") ++
      const("(SELECT a.id, a.status, a.percent_completed,") ++
      fr"ST_AsMVTGeom(ST_Simplify(a.geometry, $smpl), $bboxPoly, 4096, 256, true) AS geom" ++
      const(s"FROM $dbSchema.$table a") ++
      fr"WHERE a.processing_id = $processingId AND ST_Intersects(a.geometry, $bboxPoly) = true) AS frows"

    sql.query[Array[Byte]].unique
  }

  def createAoi(processingId: UUID, polygon: Projected[Geometry]): ConnectionIO[(UUID, Long)] = {
    val area = polygon.areaInMeters()
    val fields = Seq(
      Some("processing_id" -> processingId),
      Some("geometry" -> polygon),
      Some("area" -> area),
      Some("status" -> Status.Unprocessed.intVal),
      Some("percent_completed" -> 0),
    ).flatten.toMap

    for {
      id <- create(fields)
    } yield (id, area)
  }

  def updateAoiProgress(
      id: UUID,
      area: Long,
      prgDetails: List[ProgressDetail],
      processingCreatedAt: Option[Instant],
    ): ConnectionIO[Unit] = {
    val progress = Progress(prgDetails, area, processingCreatedAt)
    val fields: Map[String, Any] = Seq(
      Some("status" -> progress.status.intVal),
      Some("percent_completed" -> progress.percentCompleted),
      progress.completionDate.map("completion_date" -> _),
    ).flatten.toMap
    updateById(fields, id)
  }

  def updateAoiMessages(id: UUID, messages: Json): ConnectionIO[Unit] = {
    val fields = Map(
      "messages" -> messages
    )

    updateById(fields, id)
  }

  def updateAoiStatusAndVrt(
      ids: List[UUID],
      status: Status,
      vrtUri: Option[String],
    ): ConnectionIO[Unit] =
    NonEmptyList.fromList(ids) match {
      case Some(value) =>
        val fields: Map[String, Any] = Seq(
          Some("status" -> status.intVal),
          vrtUri.map("vrt_uri" -> _),
        ).flatten.toMap
        update(fields, in(fr"id", value).some)
      case None => {}.pure[ConnectionIO]
    }

  def updateAoiStartTime(ids: List[UUID], startTime: Instant): ConnectionIO[Unit] =
    NonEmptyList
      .fromList(ids)
      .traverse { value =>
        update(
          Map("start_time" -> startTime),
          in(fr"id", value).some,
        )
      }
      .void

  def updateAoiStatusByProcessings(processingIds: Seq[UUID], status: Status): ConnectionIO[Unit] = {
    val sql = const(s"UPDATE $dbSchema.$table ") ++
      fr"SET status = ${status.intVal} WHERE " ++
      (NonEmptySeq.fromSeq(processingIds) match {
        case Some(value) => in(fr"processing_id", value)
        case None => fr"FALSE"
      })

    sql.update.run.map { _ => }
  }

  def cancelAoiByProcessingIds(processingIds: Seq[UUID]): ConnectionIO[Unit] = {
    val sql = const(s"UPDATE $dbSchema.$table ") ++
      fr"SET status = ${Status.Cancelled.intVal} WHERE status = ${Status.InProgress.intVal} AND " ++
      (NonEmptySeq.fromSeq(processingIds) match {
        case Some(value) => in(fr"processing_id", value)
        case None => fr"FALSE"
      })

    sql.update.run.map { _ => }
  }

  def getMessagesByProcessings(
      processingIds: List[UUID]
    ): ConnectionIO[Map[UUID, List[Message]]] = {
    case class Dto(processingId: UUID, messages: Option[Json])

    NonEmptyList.fromList(processingIds) match {
      case Some(ids) =>
        for {
          // Processing may have no AOIs. Left Join is used to guarantee processingId presence in the result map
          lists <- (const(
            s"SELECT p.id, a.messages FROM $dbSchema.processing p LEFT JOIN $dbSchema.$table a ON a.processing_id = p.id WHERE "
          ) ++ in(const("p.id"), ids))
            .query[Dto]
            .to[List]
          pairs = lists.map(dto => dto.processingId -> Message.fromJson(dto.messages))
          map = pairs.groupMap(_._1)(_._2).view.mapValues(list => list.flatten).toMap
        } yield map
      case None => Map[UUID, List[Message]]().pure[ConnectionIO]
    }
  }
}
