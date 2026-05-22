package io.geoalert.mapflow.service

import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

import akka.http.scaladsl.model.HttpEntity
import akka.stream.scaladsl.Source
import cats.data.EitherT
import cats.implicits.catsSyntaxOptionId
import cats.syntax.bifunctor._
import com.typesafe.scalalogging.LazyLogging
import doobie.ConnectionIO
import doobie.implicits._
import io.geoalert.mapflow.exception._
import io.geoalert.mapflow.model.Permission.ViewAnyProject
import io.geoalert.mapflow.model._
import io.geoalert.mapflow.repo._

object Cache {
  val results = new ConcurrentHashMap[UUID, AoiFilter]()
}

class ResultService(exportGeojsonService: ExportGeojsonService) extends LazyLogging {

  def downloadResult(prcId: UUID,
                      aoiIdOpt: Option[UUID]
                    ): ConnectionIO[Source[HttpEntity.ChunkStreamPart, Any]] = {
    aoiIdOpt match {
      case Some(aoiId) => downloadResultByAoi(aoiId)
      case None => downloadResultByProcessing(prcId)
    }
  }

  def downloadResultByProcessing(
                                  prcId: UUID
                                ): ConnectionIO[Source[HttpEntity.ChunkStreamPart, Any]] = {
    logger.debug(s"Download processing $prcId results")
    (for {
      prc <- ProcessingRepo.getProcessing(prcId, None)
      vectorLayer <- VectorLayerRepo.getVectorLayer(prc.vectorLayerId)
      externalLayerId = vectorLayer.externalId
      _ = logger.debug(
        s"Downloading results for processing $prcId AOIs; externalLayerId = $externalLayerId"
      )
      source = exportGeojsonService.exportLayer(externalLayerId)
    } yield source).rethrowT
  }

  def downloadResultByAoi(
                           aoiId: UUID
                         ): ConnectionIO[Source[HttpEntity.ChunkStreamPart, Any]] = {
    val inputs = for {
      aoiIds <- EitherT.right[ApplicationError](
        AoiRepo.getAoiIdsWithFilter(AoiFilter(ids = List(aoiId).some, geometry = None), None)
      )
      vlId <- EitherT(Validations.aoisReferToOneVectorLayer(aoiIds))
      vectorLayer <- VectorLayerRepo.getVectorLayer(vlId).leftWiden[ApplicationError]
      geoms <- EitherT.right[ApplicationError](AoiRepo.getAoiGeom(aoiIds))
    } yield (aoiIds, vectorLayer.externalId, geoms)

    for {
      inputs <- inputs.rethrowT
      (aoiIds, externalLayerId, geoms) = inputs
      _ = logger.debug(
        s"Downloading results for ${aoiIds.size} AOIs; externalLayerId = $externalLayerId"
      )
      aoiGeomsSource = Source.fromIterator(() => geoms.iterator)
      source = exportGeojsonService.exportLayer(externalLayerId, Some(aoiGeomsSource))
    } yield source
  }
}

object ResultService {
  def apply(exportGeojsonService: ExportGeojsonService): ResultService = new ResultService(
    exportGeojsonService
  )
}
