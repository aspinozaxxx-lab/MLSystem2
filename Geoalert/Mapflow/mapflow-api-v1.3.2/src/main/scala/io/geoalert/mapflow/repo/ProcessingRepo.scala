package io.geoalert.mapflow.repo

import java.time.Instant
import java.util.UUID

import cats.data.EitherT
import cats.data.NonEmptyList
import cats.syntax.applicative._
import cats.syntax.option._
import doobie.ConnectionIO
import doobie.Fragment
import doobie.Fragment._
import doobie.implicits._
import doobie.implicits.legacy.instant._
import doobie.postgres.circe.json.implicits._
import doobie.postgres.implicits._
import doobie.util.fragments.in
import doobie.util.fragments.whereAndOpt
import io.circe.Decoder
import io.circe.Encoder
import io.circe.Json
import io.circe.generic.semiauto
import io.circe.syntax._
import io.geoalert.mapflow.exception.NotFound
import io.geoalert.mapflow.graphql.args.processing.ProcessingFilters
import io.geoalert.mapflow.implicits.CommonOps._
import io.geoalert.mapflow.implicits.Postgres._
import io.geoalert.mapflow.model.CreateProcessingInput
import io.geoalert.mapflow.model.ProcessingMeta
import io.geoalert.mapflow.model.ProcessingProgress
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.model.UpdateProcessingInput
import io.geoalert.mapflow.model.enums.ProcessingSortOrder
import io.geoalert.mapflow.model.enums.ProcessingSortOrder._
import io.geoalert.mapflow.model.enums.SortBy
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.service.PagedResponse

case class BlockParameters(
    name: String,
    enabled: Boolean,
    displayName: Option[String],
  )

object BlockParameters {
  def apply(json: Json): Seq[BlockParameters] =
    json.as[Seq[BlockParameters]].toTry.get

  implicit val blockParametersDecoder: Decoder[BlockParameters] =
    semiauto.deriveDecoder[BlockParameters]
  implicit val blockParametersEncoder: Encoder[BlockParameters] =
    semiauto.deriveEncoder[BlockParameters]
}

case class ProcessingDto(
    id: UUID,
    projectId: UUID,
    vectorLayerId: UUID,
    rasterLayerId: UUID,
    workflowDefId: Option[UUID],
    dataProviderId: Option[UUID],
    name: String,
    description: Option[String],
    params: Map[String, String],
    blocks: Option[Json],
    meta: Map[String, String],
    created: Instant,
    updated: Instant,
    archived: Boolean,
    sourceType: Option[String],
    source: Option[String],
    cost: Option[Long],
  )
object ProcessingRepo
    extends GenericRepo[ProcessingDto](
      "processing",
      Seq(
        "id",
        "project_id",
        "vector_layer_id",
        "raster_layer_id",
        "workflow_def_id",
        "data_provider_id",
        "name",
        "description",
        "params",
        "blocks",
        "meta",
        "created",
        "updated",
        "archived",
        "source_type",
        "source",
        "cost",
      ),
    ) {
  def getProcessingsWithFilter(
      ids: Option[List[UUID]],
      projectIds: Option[List[UUID]],
      wdIds: Option[List[UUID]],
      userId: Option[UUID],
      includeArchived: Boolean = false,
    ): ConnectionIO[List[ProcessingDto]] = {
    def dtos(projectIds: List[UUID]): ConnectionIO[List[ProcessingDto]] =
      (ids, projectIds, wdIds) match {
        case (Some(Nil), _, _) | (_, Nil, _) | (_, _, Some(Nil)) =>
          List[ProcessingDto]().pure[ConnectionIO]
        case _ =>
          getAllWhere(
            forUpdate = false,
            ids.flatMap(l => NonEmptyList.fromList(l)).map(in(const(idField), _)),
            NonEmptyList.fromList(projectIds).map(in(const("project_id"), _)),
            wdIds.flatMap(l => NonEmptyList.fromList(l)).map(in(const("workflow_def_id"), _)),
            if (includeArchived) None else fr"archived = false".some,
          )()
      }

    for {
      userPrjIds <- UserProjectsRepo
        .getAllIdsWhere(false, userId.map(u => fr"user_id = $u"))
        .map(_.distinct)
      dtos <- dtos(projectIds.map(_.filter(userPrjIds.contains)).getOrElse(userPrjIds))
    } yield dtos
  }

  def get(filters: ProcessingFilters): ConnectionIO[PagedResponse[ProcessingDto]] = {
    def sortFunction(
        sortOrder: ProcessingSortOrder,
        sortBy: SortBy,
      ): Fragment =
      sortOrder match {
        case Scenario => const(s"ORDER BY wd.name ${sortBy.entryName}")
        case Name => const(s"ORDER BY p.name ${sortBy.entryName}")
        case ProjectName => const(s"ORDER BY pr.name ${sortBy.entryName}")
        case Email => const(s"ORDER BY u.email ${sortBy.entryName}")
        case Created => const(s"ORDER BY p.created ${sortBy.entryName}")
        case ProcessingSortOrder.Status =>
          const(s"ORDER BY pe.processing_status ${sortBy.entryName}")
        case Progress => const(s"ORDER BY pe.percent_completed ${sortBy.entryName}")
        case _ => fr""
      }

      def statusesFilter(statuses: Option[Seq[Status]]): Option[Fragment] =
      statuses.flatMap { statusList =>
        NonEmptyList
          .fromList(statusList.map(_.entryName).toList)
          .map(in(const("pe.processing_status"), _))
      }

      val sql =
      const(
        s"""SELECT ${columns.map("p." + _).mkString(", ")}, COUNT(*) OVER() as total FROM $dbSchema.$table p
           INNER JOIN $dbSchema.workflow_def wd ON p.workflow_def_id = wd.id
           INNER JOIN $dbSchema.project pr ON p.project_id = pr.id
           INNER JOIN $dbSchema.app_user u ON pr.user_id = u.id
           INNER JOIN $dbSchema.processing_estimate pe ON pe.processing_id = p.id"""
      ) ++
        whereAndOpt(
          fr"p.archived = false".some,
          filters.dateFrom.map(date => fr"p.created >= $date"),
          filters.dateTo.map(date => fr"p.created <= $date"),
          statusesFilter(filters.statuses),
          filters
            .terms
            .map("%" + _.toLowerCase() + "%")
            .map(terms =>
              fr"(LOWER(p.name) LIKE $terms OR LOWER(wd.name) LIKE $terms OR LOWER(u.email) LIKE $terms OR LOWER(u.name) LIKE $terms OR LOWER(u.preferred_username) LIKE $terms OR LOWER(pr.name) LIKE $terms)"
            ),
        ) ++
        filters
          .sortOrder
          .map(sortOrder => sortFunction(sortOrder, filters.sortBy.getOrElse(SortBy.ASC)))
          .getOrElse(Fragment.empty) ++ List(
          filters.offset.map(n => fr" OFFSET $n "),
          filters.limit.map(n => fr" LIMIT $n "),
        ).flatten.reduceOption(_ ++ _).getOrElse(Fragment.empty)

    sql.query[(ProcessingDto, Long)].to[List].map { processing =>
      val processingList = processing.map(_._1)
      PagedResponse(
        processingList,
        processing.map(_._2.toInt).headOption.getOrElse(0),
        processingList.length,
      )
    }

  }

  def getByProcessingIds(processingIds: List[UUID]): ConnectionIO[List[ProcessingDto]] =
    getAllByIdsWhere(
      processingIds,
      fr"archived = false".some,
    )

  def getByAoiIds(aoiIds: List[UUID]): ConnectionIO[List[ProcessingDto]] = {
    val sql = const(
      s"""SELECT ${columns.map("p." + _).mkString(", ")} FROM $dbSchema.$table p
           INNER JOIN $dbSchema.aoi a ON p.id = a.processing_id"""
    ) ++
      whereAndOpt(
        fr"p.archived = false".some,
        NonEmptyList.fromList(aoiIds).map(in(const("a.id"), _)),
      )
    sql.query[ProcessingDto].to[List].map(_.distinct)
  }

  def getProcessings(
      ids: Option[Seq[UUID]],
      userId: Option[UUID],
      includeArchived: Boolean = false,
    ): ConnectionIO[List[ProcessingDto]] =
    getProcessingsWithFilter(ids.listOpt, None, None, userId, includeArchived)

  def getProcessing(
      id: UUID,
      userId: Option[UUID] = None,
      includeArchived: Boolean = false,
    ): EitherT[ConnectionIO, NotFound, ProcessingDto] =
    getProcessings(List(id).some, userId, includeArchived).headOrNotFound(id)

  def getProcessingIdsByProjectIds(projectIds: List[UUID]): ConnectionIO[List[List[UUID]]] = {
    val listOfProcessings = for {
      prcIds <- projectIds.map(prjId => getAllIdsWhere(false, fr"project_id = $prjId".some))
    } yield prcIds
    import cats.implicits._

    listOfProcessings.sequence
  }

  def getProcessingNamesByProjectId(projectId: UUID): ConnectionIO[List[String]] =
    (const(s"SELECT name FROM $dbSchema.$table WHERE project_id =") ++ fr"$projectId")
      .query[String]
      .to[List]
  def getProcessingProgress(
      processingIds: List[UUID]
    ): ConnectionIO[List[ProcessingProgress]] =
    (const(s"SELECT * FROM $dbSchema.processing_estimate") ++ whereAndOpt(
      NonEmptyList.fromList(processingIds).map(in(const("processing_id"), _))
    )).query[ProcessingProgress].to[List]

  def createProcessing(p: CreateProcessingInput): ConnectionIO[UUID] =
    for {
      meta <- ProcessingMeta.parseJson[ConnectionIO](p.meta)
      rawParams = p.params.map(_.toMap).getOrElse(Map())
      params = p
        .partitionSize
        .map(ps => rawParams.updated("partition_size", ps.toString))
        .getOrElse(rawParams)
      fields = Seq(
        Some("project_id" -> p.projectId),
        p.name.map("name" -> _),
        p.description.map("description" -> _),
        p.vectorLayerId.map("vector_layer_id" -> _),
        p.rasterLayerId.map("raster_layer_id" -> _),
        p.dataProviderId.map("data_provider_id" -> _),
        Some("workflow_def_id" -> p.workflowDefId),
        Some("params" -> params),
        Some("meta" -> meta.fold(Map.empty[String, String])(_.toMap)),
        Some("created" -> Instant.now()),
        Some("updated" -> Instant.now()),
        p.source.map("source" -> _.toString),
        p.sourceType.map("source_type" -> _.toString),
        Some("cost" -> p.cost),
        p.blocks.map("blocks" -> _.asJson),
      ).flatten.toMap
      id <- create(fields)
    } yield id

  def updateProcessing(p: UpdateProcessingInput): ConnectionIO[Unit] =
    for {
      meta <- ProcessingMeta.parseJson[ConnectionIO](p.meta)
      fields = Seq(
        p.projectId.map("project_id" -> _),
        p.vectorLayerId.map("vector_layer_id" -> _),
        p.rasterLayerId.map("raster_layer_id" -> _),
        p.workflowDefId.map("workflow_def_id" -> _),
        p.name.map("name" -> _),
        p.description.map("description" -> _),
        Some("updated" -> Instant.now()),
        p.cost.map("cost" -> _),
        Some("meta" -> meta.fold(Map.empty[String, String])(_.toMap)),
      ).flatten.toMap
      _ <- updateById(fields, p.processingId)
    } yield ()
}
