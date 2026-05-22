package io.geoalert.mapflow.service.we.model

import io.geoalert.mapflow.DefaultExternalSystemConfig
import io.geoalert.mapflow.DefaultWeConfig

import geotrellis.vector._

case class AreaOfInterest(geometry: Geometry)

case class BlockParameters(name: String, enabled: Boolean)
case class PostWorkflowRequest(
    areasOfInterest: List[AreaOfInterest],
    workflowDefinitionId: Long,
    system: String,
    processingId: String,
    params: Map[String, String],
    blocks: Seq[BlockParameters],
  )

object PostWorkflowRequest extends DefaultExternalSystemConfig with DefaultWeConfig {
  def apply(summary: RunWorkflowSummary): PostWorkflowRequest = {
    val params =
      if (summary.params contains "url")
        if (
            summary
              .params("url")
              .startsWith("s3://") || summary.params.get("source_type").contains("tif")
        )
          summary.params.updated("source_type", "local")
        else
          summary.params
      else summary.params

    PostWorkflowRequest(
      List(AreaOfInterest(summary.wf.geometry.geom)),
      summary.wf.workflowDef.weId,
      systemId,
      summary.processingId.toString,
      params ++ Map(
        "priority" -> getPriority(summary.areaInProgress).toString,
        "vector-layer-id" -> summary.vl.externalId,
        "raster-layer-uri" -> summary.rl.uri,
      ),
      summary.blocks.map(bp => BlockParameters(bp.name, bp.enabled)),
    )
  }

  private def getPriority(area: Long) = {
    val priority0to10 = 1e9 / (area.toDouble + 1e8)
    val normalizedPriority = priority0to10 * (maxPriority - minPriority) / 10
    (normalizedPriority.round.toInt min maxPriority) max minPriority
  }
}
