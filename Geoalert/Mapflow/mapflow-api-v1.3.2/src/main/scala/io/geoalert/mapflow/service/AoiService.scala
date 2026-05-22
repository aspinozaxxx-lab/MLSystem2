package io.geoalert.mapflow.service

import java.util.UUID

import cats.data.EitherT
import cats.effect.Sync
import cats.instances.list._
import cats.syntax.either._
import cats.syntax.option._
import cats.syntax.traverse._
import com.typesafe.scalalogging.LazyLogging
import doobie.ConnectionIO
import doobie.free.connection.AsyncConnectionIO
import io.circe.Decoder
import io.circe.Encoder
import io.circe.generic.semiauto.deriveDecoder
import io.circe.generic.semiauto.deriveEncoder
import io.circe.syntax._
import io.geoalert.mapflow.Config._
import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.exception._
import io.geoalert.mapflow.graphql.GraphQLContext
import io.geoalert.mapflow.implicits.CommonOps._
import io.geoalert.mapflow.implicits.GeometryOps._
import io.geoalert.mapflow.model.Aoi
import io.geoalert.mapflow.model.AoiFilter
import io.geoalert.mapflow.model.AoiList
import io.geoalert.mapflow.model.AoiSortEntry
import io.geoalert.mapflow.model.AoiStats
import io.geoalert.mapflow.model.CreateAoisFromFileInput
import io.geoalert.mapflow.model.CreateAoisFromGeometryInput
import io.geoalert.mapflow.model.MergeStrategy
import io.geoalert.mapflow.model.Message
import io.geoalert.mapflow.model.MessageParameter
import io.geoalert.mapflow.model.Permission.ViewAnyProject
import io.geoalert.mapflow.model.Processing
import io.geoalert.mapflow.model.Progress
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.repo.AoiDto
import io.geoalert.mapflow.repo.AoiRepo
import io.geoalert.mapflow.repo.ProjectRepo
import io.geoalert.mapflow.service.importer.GeoJsonImporter
import io.geoalert.mapflow.service.importer.GeojsonParser
import io.geoalert.mapflow.service.importer.ImportResult
import io.geoalert.mapflow.util.BatchingService

import geotrellis.proj4.LatLng
import geotrellis.vector._

class AoiService(processingService: ProcessingService, progressService: ProgressService)
    extends LazyLogging {
  implicit val messageParamEncoder: Encoder[MessageParameter] = deriveEncoder[MessageParameter]
  implicit val messageEncoder: Encoder[Message] = deriveEncoder[Message]

  implicit val messageParamDecoder: Decoder[MessageParameter] = deriveDecoder[MessageParameter]
  implicit val messageDecoder: Decoder[Message] = deriveDecoder[Message]

  val geoJsonImporter: GeoJsonImporter = new GeoJsonImporter(this)

  def createAois(
      processing: Processing,
      geometry: Projected[Geometry],
    )(
      user: User
    ): ConnectionIO[AoiStats] = {
    val geometries = BatchingService
      .mergeGeometriesToBatches(
        geometry.geom.toPolygons,
        BatchingService.estimatePartitionSize(processing),
      )
      .map(_.asGeometry.withSRID(geometry.srid))

    val io = for {
      result <- createAoisFromStream(geometries, processing)(user)
      (inserted, _) = result
      _ <- processingService.updateProcessingCost(processing)
      filter = AoiFilter(inserted.toList.some, None, None, None)
      summary <- EitherT.right[ApplicationError](AoiRepo.getAoiSummaryWithFilter(filter, None))
      _ = progressService.invalidateCache(List(processing.projectId), List(processing.id), List())
    } yield AoiStats(summary)

    io.rethrowT
  }

  def createAoisFromGeometry(
      input: CreateAoisFromGeometryInput
    )(
      user: User
    ): ConnectionIO[AoiStats] =
    (for {
      processing <- processingService.getProcessing(input.processingId)(user)
      aois <- EitherT.right[ApplicationError](createAois(processing, input.geometry)(user))
      _ = progressService.invalidateCache(List(processing.projectId), List(processing.id), List())
    } yield aois).rethrowT

  def createAoisFromFile(
      input: CreateAoisFromFileInput,
      ctx: GraphQLContext,
    )(
      user: User
    ): ConnectionIO[AoiStats] = {
    val inputStream = input
      .file
      .streamOption(ctx)
      .getOrElse(throw BadRequest("Cannot create input stream from file input"))
    val polygons = GeojsonParser.polygonStream(inputStream).toList
    val geometry = GeometryCollection(polygons).withSRID(LatLng.epsgCode.get)

    (for {
      processing <- processingService.getProcessing(input.processingId)(user)
      aois <- EitherT.right[ApplicationError](createAois(processing, geometry)(user))
      _ = progressService.invalidateCache(List(processing.projectId), List(processing.id), List())
    } yield aois).rethrowT
  }

  private def createAoisFromStream(
      ps: Seq[Projected[Geometry]],
      processing: Processing,
    )(
      user: User
    ): EitherT[doobie.ConnectionIO, ApplicationError, (Iterable[UUID], Iterable[UUID])] = {
    logger.debug(s"Starting to import AOIs")

    // TODO blocking vs nonblocking?
    val stream = fs2.Stream.fromIterator(ps.zipWithIndex.iterator)(implicitly[Sync[ConnectionIO]])

    def importPolygon(
        i: Int,
        p: Projected[Geometry],
        processingId: UUID,
      ): doobie.ConnectionIO[Either[AoiImportError, ImportResult]] = {
      if (i % 1000 == 0) logger.debug(s"Importing polygon #${i + 1}")
      geoJsonImporter.importPolygon(p, processingId, MergeStrategy.Union)(user).value
    }

    def results(
        processingId: UUID
      ): ConnectionIO[Either[AccumulativeAoiImportError, List[ImportResult]]] =
      stream
        .evalMap {
          case (p, i) => importPolygon(i, p, processingId)
        }
        .filter {
          case Right(_) => true
          case Left(_: Critical) => true
          case Left(_: NonCritical) => false
        }
        .map(_.toValidatedNec)
        .compile
        .toList
        .map(_.sequence.toEither.leftMap(AccumulativeAoiImportError(_)))

    def insertedDeleted(result: ImportResult): (Iterable[UUID], Iterable[UUID]) = {
      val grouped = result
        .value
        .groupBy {
          case (_, (operation, _)) => operation
        }
        .view
        .mapValues(_.map {
          case (id, (_, _)) => id
        })
        .toMap
        .withDefaultValue(List())

      (grouped(1), grouped(-1))
    }

    for {
      results <- EitherT(results(processing.id))
      result = ImportResult.combineAll(results)
      (inserted, deleted) = insertedDeleted(result)
      msg =
        s"Imported ${results.size} polygons into ${inserted.size} aois; ${deleted.size} old aois deleted."
      _ = logger.debug(msg)
      _ <- EitherT(Validations.checkProcessingArea(user, processing.id))
    } yield (inserted, deleted)
  }

  def getAoiStats(filter: AoiFilter)(user: User): ConnectionIO[AoiStats] =
    AoiRepo
      .getAoiSummaryWithFilter(filter, user.userFilter(ViewAnyProject))
      .map(AoiStats(_))

  def getAoi(id: UUID)(user: User): EitherT[ConnectionIO, ApplicationError, Aoi] =
    for {
      dto <- AoiRepo.getAoi(id, user.userFilter(ViewAnyProject))
      progress <- EitherT.right[ApplicationError](progressService.getAoiProgress(dto))
    } yield Aoi(
      dto.id,
      dto.processingId,
      progress,
      Message.fromJson(dto.messages),
      dto.geometry,
      dto.area,
      dto.completionDate,
      dto.vrtUri,
    )

  def getProcessingAois(processingId: UUID)(user: User): ConnectionIO[List[Aoi]] =
    for {
      dtos <- AoiRepo.getProcessingAois(processingId, user.userFilter(ViewAnyProject))
      progresses <- progressService.getAoisProgress(dtos)
      _ = logger.debug(s"Get Processing Aois, progresses: ${progresses.size}")
    } yield dtosToList(dtos, progresses)

  private def dtosToList(dtos: List[AoiDto], progresses: Map[UUID, Progress]): List[Aoi] = for {
    dto <- dtos
  } yield Aoi(
    dto.id,
    dto.processingId,
    progresses(dto.id),
    Message.fromJson(dto.messages),
    dto.geometry,
    dto.area,
    dto.completionDate,
    dto.vrtUri,
  )

  def getAois(
      filter: AoiFilter,
      sort: Option[Seq[AoiSortEntry]],
      offset: Option[Int],
      limit: Option[Int],
    )(
      user: User
    ): ConnectionIO[AoiList] = {
    val userFilter = user.userFilter(ViewAnyProject)
    val limitPlusOne = limit.map(_ + 1)

    for {
      aoisPlusOne <- AoiRepo.getAoisWithFilter(
        filter,
        userFilter,
        sort.listOpt,
        offset,
        limitPlusOne,
      )
      progresses <- progressService.getAoisProgress(aoisPlusOne)
    } yield (aoisPlusOne, limit) match {
      case (apo, lim) if apo.size.min(lim getOrElse Int.MaxValue) > maxAoiFetch =>
        throw BadRequest(s"Too many aois to render (max is $maxAoiFetch)")
      case (apo, Some(lim)) if lim < aoisPlusOne.size =>
        val dtos = apo take lim
        AoiList(dtosToList(dtos, progresses), hasMore = true)
      case (apo, _) => AoiList(dtosToList(apo, progresses), hasMore = false)
    }
  }

  def getAoiIds(filter: AoiFilter)(user: User): ConnectionIO[List[UUID]] =
    AoiRepo.getAoiIdsWithFilter(filter, user.userFilter(ViewAnyProject))

  def deleteAois(filter: AoiFilter)(user: User): ConnectionIO[Int] = {
    def prcAndPrjIds: ConnectionIO[(List[UUID], List[UUID])] = for {
      ids <- AoiRepo.getAoiIdsWithFilter(filter, user.userFilter(ViewAnyProject))
      prcIds <- AoiRepo.getAoisProcessings(ids).map(_.values.toList.distinct)
      prjIds <- ProjectRepo.getProjectIdsByProcessings(prcIds).map(_.values.toList.distinct)
    } yield (prcIds, prjIds)

    def deleteUnprocessed(): ConnectionIO[List[UUID]] = for {
      ids <- AoiRepo.getAoiIdsWithFilter(filter, user.userFilter(ViewAnyProject))
      _ <- AoiRepo.deleteByIds(ids)
    } yield ids

    for {
      prcAndPrjIds <- prcAndPrjIds
      (processingIds, projectIds) = prcAndPrjIds
      aoiIds <- deleteUnprocessed()
      _ = progressService.invalidateCache(projectIds, processingIds, aoiIds)
    } yield processingIds.size
  }

  def updateAoiMessages(aoiId: UUID, messages: List[Message]): ConnectionIO[Unit] =
    for {
      aoi <- AoiRepo.getOneById(aoiId)
      oldMessagesJson = aoi.flatMap(_.messages)
      oldMessages = oldMessagesJson.map(_.as[List[Message]].getOrElse(List[Message]()))
      oldMessagesSet = oldMessages.getOrElse(List()).toSet
      newMessagesSet = messages.toSet
      m = oldMessagesSet ++ newMessagesSet
      _ <- AoiRepo.updateAoiMessages(aoiId, m.toList.asJson)
    } yield ()

  def updateAoiStatusAndVrt(
      ids: List[UUID],
      status: Status,
      vrtUri: Option[String],
    ): ConnectionIO[Unit] =
    AoiRepo.updateAoiStatusAndVrt(ids, status, vrtUri)
}

object AoiService {
  def apply(processingService: ProcessingService, progressService: ProgressService): AoiService =
    new AoiService(processingService, progressService)
}
