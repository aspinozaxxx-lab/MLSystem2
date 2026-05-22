package io.geoalert.mapflow.rest.json

import java.time.Instant
import java.util.UUID

import cats.implicits.catsSyntaxOptionId
import doobie.free.connection.ConnectionIO
import io.geoalert.mapflow.model.Processing
import io.geoalert.mapflow.model.ProcessingMeta
import io.geoalert.mapflow.model.Rating
import io.geoalert.mapflow.model.Status

import geotrellis.vector.Geometry

case class ProcessingJson(
    id: UUID,
    name: String,
    description: Option[String],
    projectId: UUID,
    vectorLayer: VectorLayerJson,
    rasterLayer: RasterLayerJson,
    workflowDef: WorkflowDefJson,
    // Deprecated. Will be removed in API 2
    aoiCount: Int,
    // Deprecated.  Use area instead. Will be removed in API 2
    aoiArea: Long,
    area: Long,
    status: Status,
    reviewStatus: Option[ProcessingReviewJson],
    rating: Option[Rating],
    percentCompleted: Int,
    params: Map[String, String],
    blocks: Seq[BlockParametersJson],
    meta: Map[String, String],
    messages: List[MessageJson],
    created: Instant,
    updated: Instant,
  )

case class BlockParametersJson(
    name: String,
    enabled: Boolean,
    displayName: Option[String],
  )

case class CreateAndRunProcessingJson(
    name: Option[String],
    description: Option[String],
    projectId: Option[UUID],
    wdName: Option[String],
    wdId: Option[UUID],
    geometry: Geometry,
    params: Option[Map[String, String]],
    meta: Option[Map[String, String]],
    blocks: Option[Seq[BlockParametersJson]],
  )

case class UpdateProcessingInputJson(
    name: Option[String],
    description: Option[String],
    projectId: Option[UUID],
    meta: Option[Map[String, String]] = None,
  )

case class CreateProcessingRatingJson(rating: Int, feedback: Option[String])

object ProcessingJson {
  def apply(
      processing: Processing
    ): ConnectionIO[ProcessingJson] =
    ProcessingMeta.parseJson[ConnectionIO](processing.meta.some).map { meta =>
      ProcessingJson(
        processing.id,
        processing.name,
        processing.description,
        processing.projectId,
        VectorLayerJson(processing.vectorLayer),
        RasterLayerJson(processing.rasterLayer),
        WorkflowDefJson(processing.workflowDef),
        processing.aoiCount,
        processing.area,
        processing.area,
        // Hotfix because plugin cannot handle UNPROCESSED and CANCELLED statuses
        processing.progress.status match {
          case Status.Unprocessed => Status.InProgress
          case Status.Cancelled => Status.Failed
          case status: Status => status
        },
        processing.reviewStatus.map(ProcessingReviewJson(_)),
        processing.rating,
        processing.progress.percentCompleted,
        processing.params.toMap - "raster_login" - "raster_password",
        processing.blocks.map(bp => BlockParametersJson(bp.name, bp.enabled, bp.displayName)),
        meta.fold(Map.empty[String, String])(_.toMap),
        processing
          .messages
          .map(m => MessageJson(m.code, m.parameters.map(mp => (mp.key, mp.value)).toMap))
          .distinct,
        processing.created,
        processing.updated,
      )
    }
}
