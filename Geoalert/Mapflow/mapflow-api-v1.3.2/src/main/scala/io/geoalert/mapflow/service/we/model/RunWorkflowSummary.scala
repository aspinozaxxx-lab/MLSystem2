package io.geoalert.mapflow.service.we.model

import java.util.UUID

import io.geoalert.mapflow.model.RasterLayer
import io.geoalert.mapflow.model.VectorLayer
import io.geoalert.mapflow.model.Workflow
import io.geoalert.mapflow.repo.BlockParameters

case class RunWorkflowSummary(
    wf: Workflow,
    vl: VectorLayer,
    rl: RasterLayer,
    processingId: UUID,
    params: Map[String, String],
    areaInProgress: Long,
    blocks: Seq[BlockParameters],
  )
