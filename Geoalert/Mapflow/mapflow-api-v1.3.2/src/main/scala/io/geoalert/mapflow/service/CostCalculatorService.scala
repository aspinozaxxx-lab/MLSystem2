package io.geoalert.mapflow.service

import _root_.io.geoalert.mapflow.rest.utils.ControllerConstants.ProcessingControllerConstants.SRID
import cats.data.EitherT
import cats.syntax.option._
import com.typesafe.scalalogging.LazyLogging
import doobie.ConnectionIO
import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.exception.BadRequest
import io.geoalert.mapflow.implicits.GeometryOps.ProjectedGeometryOps
import io.geoalert.mapflow.model.BlockConfig
import io.geoalert.mapflow.model.DataProvider
import io.geoalert.mapflow.model.ProcessingMeta
import io.geoalert.mapflow.model.ProcessingParams
import io.geoalert.mapflow.model.SourceType
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.model.WorkflowDef
import io.geoalert.mapflow.model.WorkflowDefSummary
import io.geoalert.mapflow.repo.BlockParameters
import io.geoalert.mapflow.rest.json.BlockParametersJson
import io.geoalert.mapflow.rest.json.CalculateCostInput

import geotrellis.vector._

class CostCalculatorService(
    maxarService: MaxarService,
    workflowDefService: WorkflowDefService,
    dataProviderService: DataProviderService,
  ) extends LazyLogging {
  def metersPerPxEst(zoom: Int): Double = 156543.03 /* meters/pixel at Equator */ *
    Math.cos(Math.toRadians(55) /* latitude of Moscow */ ) / Math.pow(2, zoom.toDouble)

  def megapixelsInSqKm(zoom: Int): Double = Math.pow(1 / metersPerPxEst(zoom), 2)

  /** Takes workflow definition and execution params and return cost in credits
    */

  def estimateCost(input: CalculateCostInput)(user: User): ConnectionIO[Long] = {
    val params = ProcessingParams(input.params.getOrElse(Map()))
    val meta = ProcessingMeta(input.meta.getOrElse(Map()))
    for {
      wd <- workflowDefService.getWorkflowDef(input.wdId)(user)
      params <- useMaxarCredentialsIfNeeded(params, meta, wd.workflowDefSummary)(user).rethrowT
      urlOpt = (params.url ++ wd.workflowDefSummary.url).headOption
      dataProvider <- dataProviderService.retrieveDataProvider(params, urlOpt)
    } yield estimateCostByWd(wd, params, input.geometry, input.areaSqKm, dataProvider, input.blocks)
  }

  private def estimateCostByWd(
      wd: WorkflowDef,
      params: ProcessingParams,
      geometryOpt: Option[Geometry],
      areaSqKmOpt: Option[Double],
      dataProvider: Option[DataProvider],
      blocks: Option[Seq[BlockParametersJson]],
    ): Long = {
    val blockParams =
      blocks.getOrElse(Seq()).map(bp => BlockParameters(bp.name, bp.enabled, bp.displayName))
    (geometryOpt, areaSqKmOpt) match {
      case (Some(geometry), _) =>
        estimateCost(
          wd.workflowDefSummary,
          params,
          geometry.withSRID(SRID),
          dataProvider,
          blockParams,
        )
      case (_, Some(area)) =>
        estimateCost(
          wd.workflowDefSummary,
          params,
          area,
          dataProvider,
          blockParams,
        )
      case _ => throw BadRequest("Either geometry or areaSqKm is required")
    }
  }

  def useMaxarCredentialsIfNeeded(
      params: ProcessingParams,
      meta: ProcessingMeta,
      wd: WorkflowDefSummary,
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, ProcessingParams] =
    // Backward compatibility:
    // If there is a Maxar image AND user didn't provided Maxar Credentials, AND user has access to Maxar premium,
    // use our credentials
    if (meta.source.contains("maxar") && !params.isCredentialsSpecified)
      maxarService.addMaxarCredentialsToParams(params, meta, wd)(user)
    else
      EitherT.rightT[ConnectionIO, ApplicationError](params)
  def estimateCost(
      wd: WorkflowDefSummary,
      params: ProcessingParams,
      geometry: Projected[Geometry],
      dataProvider: Option[DataProvider],
      blocks: Seq[BlockParameters],
    ): Long = {
    logger.debug(s"calculating cost for Workflow Definition $wd with $params on $geometry")

    val areaInSqKm: Double = geometry.areaInMeters() / 1_000_000.0

    estimateCost(wd, params, areaInSqKm, dataProvider, blocks)
  }

  private def getEnabledBlocks(
      config: Seq[BlockConfig],
      params: Seq[BlockParameters],
    ): Set[String] = {
    val enabled = for {
      bc <- config
      bp = params.find(_.name == bc.name)
      enabled = bp.map(_.enabled).getOrElse(bc.defaultEnabled)
    } yield if (enabled) Seq(bc.name) else Seq()

    enabled.flatten.toSet
  }
  /** Takes workflow definition and execution params and return cost in credits
    */
  def estimateCost(
      wd: WorkflowDefSummary,
      params: ProcessingParams,
      areaInSqKm: Double,
      dataProvider: Option[DataProvider],
      blocks: Seq[BlockParameters],
    ): Long = {

    logger.debug(
      s"calculating cost for Workflow Definition $wd with $params for area of $areaInSqKm sq. km"
    )

    // Round area up to 1 sq. km. because small processings cause uneven collateral expenses (mostly download buffers)
    val roundedArea = if (areaInSqKm < 1) {
      logger.debug(s"Area $areaInSqKm rounded up to 1.0 sq.km")
      1.0
    }
    else
      areaInSqKm

    val dataProviderPricePerMp = dataProvider.map(_.pricePerMp).toList
    val source = ProcessingService.extractDataSource(
      params.sourceType.flatMap(name => SourceType.find(name)),
      wd,
      params.some,
    )
    val wdSourcePricePerMp = source.map(_.pricePerMp).toList
    val dataSourcePricePerMp =
      (dataProviderPricePerMp ++ wdSourcePricePerMp).headOption.getOrElse(0.0)

    val areaInMp = estimateAreaMp(wd, params, roundedArea)
    val wdPrice = if (wd.blocks.nonEmpty) {
      val enabledBlocks = getEnabledBlocks(wd.blocks, blocks)
      val pr = wd.blocks.filter(b => !b.optional || enabledBlocks.contains(b.name)).map(_.price).sum
      logger.debug(
        s"Using WD price from blocks config. Enabled blocks: [${enabledBlocks.mkString(", ")}], price $pr"
      )
      pr
    }
    else
      wd.pricePerSqKm
    val cost = (wdPrice * roundedArea) + (dataSourcePricePerMp * areaInMp)
    logger.debug(
      s"Cost calculation explanation: ${wd.pricePerSqKm} (WD price) * $roundedArea (area in sq.km) + $dataSourcePricePerMp (Data Provider price) * $areaInMp (area in MP)"
    )
    cost.ceil.toLong
  }

  def estimateAreaMp(
      workflowDefSummary: WorkflowDefSummary,
      params: ProcessingParams,
      areaInSqKm: Double,
    ): Double = {
    val zoom = (params.zoom.map(_.toInt) ++ workflowDefSummary.zoom).headOption
    val sourceType = ProcessingService.extractSourceType(None, workflowDefSummary, params)

    (zoom, sourceType) match {
      case (_, Some(SourceType.sentinel_l2a)) => 1 /* We charge Sentinel L2A imagery per granule rather then per MP */
      case (_, Some(SourceType.local)) => 0 /* We don't charge for user uploaded data sources */
      case (Some(z), _) => megapixelsInSqKm(z) * areaInSqKm
      case (_, _) =>
        throw new IllegalStateException(
          s"Zoom is expected for the data source: $workflowDefSummary"
        )
    }
  }

  def calculateDataProviderCost(dp: DataProvider): Map[Int, Double] = {
    val seq = for {
      zoom <- 13 to 22
    } yield zoom -> (dp.pricePerMp * megapixelsInSqKm(zoom))

    seq.toMap
  }
}

object CostCalculatorService {
  def apply(
      maxarService: MaxarService,
      workflowDefService: WorkflowDefService,
      dataProviderService: DataProviderService,
    ): CostCalculatorService =
    new CostCalculatorService(maxarService, workflowDefService, dataProviderService)
}
