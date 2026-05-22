package io.geoalert.mapflow.service

import java.util.UUID

import scala.util.Try

import cats.data.EitherT
import cats.syntax.applicative._
import cats.syntax.bifunctor._
import cats.syntax.option._
import com.typesafe.scalalogging.LazyLogging
import doobie.ConnectionIO
import doobie.implicits._
import io.circe.syntax.EncoderOps
import io.geoalert.mapflow.Config._
import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.exception.BadRequest
import io.geoalert.mapflow.exception.InternalServerError
import io.geoalert.mapflow.exception.NotFound
import io.geoalert.mapflow.graphql.args.processing.ProcessingFilters
import io.geoalert.mapflow.implicits.CommonOps._
import io.geoalert.mapflow.model.AoiSummary
import io.geoalert.mapflow.model.BlockConfig
import io.geoalert.mapflow.model.CreateProcessingInput
import io.geoalert.mapflow.model.DataProvider
import io.geoalert.mapflow.model.DataSource
import io.geoalert.mapflow.model.DataSource.DataSource
import io.geoalert.mapflow.model.Message
import io.geoalert.mapflow.model.Permission.ViewAnyProject
import io.geoalert.mapflow.model.Processing
import io.geoalert.mapflow.model.ProcessingMeta
import io.geoalert.mapflow.model.ProcessingParams
import io.geoalert.mapflow.model.Progress
import io.geoalert.mapflow.model.RasterLayer
import io.geoalert.mapflow.model.Rating
import io.geoalert.mapflow.model.SourceType
import io.geoalert.mapflow.model.SourceType.SourceType
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.model.Status.Unprocessed
import io.geoalert.mapflow.model.UpdateProcessingInput
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.model.UserBrief
import io.geoalert.mapflow.model.VectorLayer
import io.geoalert.mapflow.model.WorkflowDef
import io.geoalert.mapflow.model.WorkflowDefSummary
import io.geoalert.mapflow.repo.AoiRepo
import io.geoalert.mapflow.repo.BlockParameters
import io.geoalert.mapflow.repo.DataProviderRepo
import io.geoalert.mapflow.repo.ProcessingDto
import io.geoalert.mapflow.repo.ProcessingRateRepo
import io.geoalert.mapflow.repo.ProcessingRepo
import io.geoalert.mapflow.repo.ProcessingReviewDto
import io.geoalert.mapflow.repo.ProcessingReviewRepo
import io.geoalert.mapflow.repo.ProjectDto
import io.geoalert.mapflow.repo.ProjectRepo
import io.geoalert.mapflow.repo.RasterLayerRepo
import io.geoalert.mapflow.repo.RatingDto
import io.geoalert.mapflow.repo.UserDto
import io.geoalert.mapflow.repo.UserRepo
import io.geoalert.mapflow.repo.VectorLayerRepo
import io.geoalert.mapflow.repo.WorkflowDefRepo
import io.geoalert.mapflow.service.ProcessingService.extractDataSourceWithoutParams
import io.geoalert.mapflow.service.ProcessingService.extractSourceType
import io.geoalert.mapflow.service.ProcessingService.updateParamsWithUrl
import io.geoalert.mapflow.service.notification.NotificationService
import io.geoalert.mapflow.util.WorkflowDefParser

import geotrellis.vector.Extent

class ProcessingService(
    costCalculatorService: CostCalculatorService,
    dataProviderService: DataProviderService,
    notificationService: NotificationService,
    progressService: ProgressService,
    rasterService: RasterService,
    workflowDefService: WorkflowDefService,
    workflowService: WorkflowService,
  ) extends LazyLogging {
  private def generateProcessingName(
      name: Option[String],
      wd: WorkflowDef,
      projectId: UUID,
    ): ConnectionIO[String] = {
    def defaultName(wdName: String): ConnectionIO[String] = {
      def uniqueName(desired: String, existing: List[String]) =
        if (existing.contains(desired)) {
          val numbers = existing
            .filter(_.contains(wdName))
            .map(_.replace(wdName, "").trim)
            .flatMap(n => Try(n.toInt).toOption)
            .toSet
          val number = (2 to numbers.size + 2).to(LazyList).filter(!numbers.contains(_)).head
          desired + number
        }
        else desired

      for {
        existing <- ProcessingRepo.getProcessingNamesByProjectId(projectId)
      } yield uniqueName(wdName, existing)
    }

    name match {
      case Some(n) => n.pure[ConnectionIO]
      case None =>
        for {
          name <- defaultName(wd.name)
        } yield name
    }
  }

  def createProcessing(
      input: CreateProcessingInput
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, Processing] = {
    logger.info(s"Creating processing: $input by ${user.email}")

    // TODO: Create VectorLayerService
    def obtainVectorLayer(name: String): EitherT[ConnectionIO, ApplicationError, VectorLayer] =
      input.vectorLayerId match {
        case Some(id) => VectorLayerRepo.getVectorLayer(id).leftWiden[ApplicationError]
        case None => EitherT.right(VectorLayerRepo.createVectorLayer(name, UUID.randomUUID()))
      }

    // TODO: Create RasterLayerService
    def obtainRasterLayer(): EitherT[ConnectionIO, ApplicationError, RasterLayer] =
      input.rasterLayerId match {
        case Some(id) => RasterLayerRepo.getRasterLayer(id).leftWiden[ApplicationError]
        case None =>
          val uri = s"s3://$rastersBucket/${UUID.randomUUID()}"
          EitherT.right(RasterLayerRepo.createRasterLayer(uri))
      }

    def obtainDataProvider(): EitherT[ConnectionIO, ApplicationError, Option[DataProvider]] =
      input.dataProviderId match {
        case Some(id) => dataProviderService.getDataProvider(id)(user).map(_.some)
        case None => EitherT.rightT[ConnectionIO, ApplicationError](None)
      }

    for {
      _ <- EitherT(ProjectValidations.projectExist(input.projectId)(user))
      project <- ProjectRepo.getProject(input.projectId)
      userWorkflowDefs <- EitherT.right[ApplicationError](
        workflowDefService.listWorkflowDefLinkedToUser(project.userId)(user)
      )
      defaultWorkflowDefs <- EitherT.right[ApplicationError](
        workflowDefService.listDefaultWorkflowDefs()
      )
      wdOpt = (defaultWorkflowDefs ++ userWorkflowDefs).find(_.id == input.workflowDefId)
      wd <- EitherT.fromOption[ConnectionIO](
        wdOpt,
        NotFound(s"WorkflowDef not found by id ${input.workflowDefId}"): ApplicationError,
      )
      name <- EitherT.right(generateProcessingName(input.name, wd, input.projectId))
      vectorLayer <- obtainVectorLayer(name)
      rasterLayer <- obtainRasterLayer()
      dataProvider <- obtainDataProvider()
      sourceType = extractSourceType(input.sourceType, wd.workflowDefSummary, input.params)
      dataSource = extractDataSourceWithoutParams(sourceType, wd.workflowDefSummary)
      params = updateParamsWithUrl(
        input.params,
        input.url,
        dataProvider.flatMap(_.urlTemplate),
        sourceType,
      )
      updatedInput = CreateProcessingInput(
        input.projectId,
        vectorLayer.id.some,
        rasterLayer.id.some,
        wd.id,
        dataProvider.map(_.id),
        name.some,
        input.description,
        input.cost,
        input.partitionSize,
        params,
        input.meta,
        dataSource,
        sourceType,
        input.blocks,
        None,
      )
      processingId <- EitherT.right(ProcessingRepo.createProcessing(updatedInput))
      processing <- getProcessing(processingId)(user).leftWiden[ApplicationError]
    } yield processing
  }

  def updateProcessing(input: UpdateProcessingInput)(user: User): ConnectionIO[Processing] = {
    logger.info(s"Updating processing: $input by ${user.email}")

    def obtainProcessing: EitherT[ConnectionIO, ApplicationError, Processing] =
      getProcessingOrForbidden(input.processingId)(user)
        .leftWiden[ApplicationError]

    def obtainWorkflowDef(
        processing: Processing
      ): EitherT[ConnectionIO, ApplicationError, WorkflowDef] =
      (processing.progress.status, input.workflowDefId) match {
        case (Unprocessed, Some(id)) => EitherT.right(workflowDefService.getWorkflowDef(id)(user))
        // TODO: We dont check the new WD belongs to the same project
        case (_, None) => EitherT.rightT(processing.workflowDef)
        case (s, Some(_)) =>
          EitherT.leftT(BadRequest(s"Cannot change WorkflowDef for $s processings"))
      }

    // TODO: Do we really need this feature
    def obtainVectorLayer(
        processing: Processing
      ): EitherT[ConnectionIO, ApplicationError, VectorLayer] =
      (processing.progress.status, input.vectorLayerId) match {
        case (Unprocessed, Some(id)) =>
          VectorLayerRepo.getVectorLayer(id).leftWiden[ApplicationError]
        case (_, None) => EitherT.rightT(processing.vectorLayer)
        case (s, _) => EitherT.leftT(BadRequest(s"Cannot change VectorLayer for $s processings"))
      }

    // TODO: Do we really need this feature
    def obtainRasterLayer(
        processing: Processing
      ): EitherT[ConnectionIO, ApplicationError, RasterLayer] =
      (processing.progress.status, input.rasterLayerId) match {
        case (Unprocessed, Some(id)) =>
          RasterLayerRepo.getRasterLayer(id).leftWiden[ApplicationError]
        case (_, None) => EitherT.rightT(processing.rasterLayer)
        case (s, _) => EitherT.leftT(BadRequest(s"Cannot change RasterLayer for $s processings"))
      }

    (for {
      processing <- obtainProcessing
      projectId <- EitherT(
        ProjectValidations.projectExist(input.projectId.getOrElse(processing.projectId))(user)
      )
      wd <- obtainWorkflowDef(processing)
      vectorLayer <- obtainVectorLayer(processing)
      rasterLayer <- obtainRasterLayer(processing)
      validInput = UpdateProcessingInput(
        processing.id,
        projectId.some,
        vectorLayer.id.some,
        rasterLayer.id.some,
        wd.id.some,
        input.name,
        input.description,
        input.cost,
        input.meta,
      )
      _ <- EitherT.right[ApplicationError](ProcessingRepo.updateProcessing(validInput))
      result <- getProcessing(processing.id)(user).leftWiden[ApplicationError]
    } yield result).rethrowT
  }

  def archiveProcessing(id: UUID)(user: User): ConnectionIO[String] = {
    logger.info(s"Deleting processing $id by ${user.email}")

    for {
      _ <- EitherT(Validations.processingExists(id, user.userFilter(ViewAnyProject))).rethrowT
      _ <- archiveProcessingsUnsafe(List(id))
    } yield "OK"
  }

  def archiveProcessingsUnsafe(ids: List[UUID]): ConnectionIO[String] =
    for {
      processingProjects <- ProjectRepo.getProjectIdsByProcessings(ids)
      workflowIds <- workflowService.findRunningWorkflowIdsByProcessing(ids)
      _ <- ProcessingRepo.archive(ids)
      _ = progressService.invalidateCache(
        processingProjects.values.toList.distinct,
        processingProjects.keys.toList,
        List(),
      )
      _ = ids.map(notificationService.removeRestartingProcessing)
      _ <- AoiRepo.updateAoiStatusByProcessings(ids, Status.Cancelled)
      _ <- workflowService.cancelWorkflows(workflowIds)
    } yield "OK"

  def cancelProcessing(id: UUID)(user: User): ConnectionIO[String] =
    for {
      _ <- EitherT(Validations.processingExists(id, user.userFilter(ViewAnyProject))).rethrowT
      processingIds = List(id)
      processingProjects <- ProjectRepo.getProjectIdsByProcessings(processingIds)
      aois <- AoiRepo.getProcessingAois(id, None)
      workflowIds <- workflowService.findRunningWorkflowIdsByProcessing(processingIds)
      _ = progressService.invalidateCache(
        processingProjects.values.toList.distinct,
        processingProjects.keys.toList,
        aois.map(_.id),
      )
      _ = notificationService.removeRestartingProcessing(id)
      _ <- AoiRepo.cancelAoiByProcessingIds(processingIds)
      _ <- workflowService.cancelWorkflows(workflowIds)
    } yield "OK"

  def getProcessing(id: UUID)(user: User): EitherT[ConnectionIO, NotFound, Processing] =
    getProcessings(Seq(id).some, None)(user)
      .headOrNotFound(id)

  def getProcessingOrForbidden(
      id: UUID
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, Processing] =
    getProcessings(Seq(id).some, None)(user).headOrForbidden

  def getProcessings(
      filters: ProcessingFilters
    ): ConnectionIO[PagedResponse[Processing]] =
    for {
      dtos <- ProcessingRepo.get(filters)
      processings <- fillProcessingFromDto(dtos.results)
    } yield dtos.copy(results = processings)
  def getProcessingsPaged(
      filters: ProcessingFilters
    ): ConnectionIO[PagedResponse[Processing]] =
    getProcessings(
      filters.copy(limit = filters.limit.orElse(25.some), offset = filters.offset.orElse(3.some))
    )

  def getProcessings(
      ids: Option[Seq[UUID]] = None,
      projectIds: Option[Seq[UUID]] = None,
      includeArchived: Boolean = false,
    )(
      user: User
    ): ConnectionIO[List[Processing]] =
    for {
      dtos <- ProcessingRepo.getProcessingsWithFilter(
        ids.listOpt,
        projectIds.listOpt,
        None,
        user.userFilter(ViewAnyProject),
        includeArchived,
      )
      processings <- fillProcessingFromDto(dtos)
    } yield processings

  def getProcessingsWithArchived(ids: List[UUID])(user: User): ConnectionIO[List[Processing]] =
    getProcessings(ids.some, None, includeArchived = true)(user)

  def updateProcessingCost(
      processingId: UUID
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, Long] =
    for {
      processing <- getProcessing(processingId)(user)
      cost <- updateProcessingCost(processing)
    } yield cost

  def updateProcessingCost(processing: Processing): EitherT[ConnectionIO, ApplicationError, Long] =
    for {
      aois <- EitherT.right[ApplicationError](AoiRepo.getProcessingAois(processing.id, None))
      area = aois.map(_.area).sum
      cost = costCalculatorService.estimateCost(
        processing.workflowDef.workflowDefSummary,
        processing.params,
        area.toDouble / 1_000_000,
        processing.dataProvider,
        processing.blocks,
      )
      _ <- EitherT.right[ApplicationError](
        ProcessingRepo.updateById(Map("cost" -> cost), processing.id)
      )
    } yield cost

  def rateProcessing(
      processingId: UUID,
      rating: Int,
      feedback: Option[String],
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, Unit] =
    for {
      processing <- getProcessing(processingId)(user)
      rateOpt <- EitherT.right[ApplicationError](ProcessingRateRepo.getRate(processing.id))
      _ <- rateOpt match {
        case Some(_) =>
          EitherT.right[ApplicationError](
            ProcessingRateRepo.update(processing.id, rating.some, feedback)
          )
        case None =>
          EitherT.right[ApplicationError](
            ProcessingRateRepo.create(processing.id, rating, feedback)
          )
      }
    } yield {}
  private def prepareBlockParameters(
      config: Seq[BlockConfig],
      params: Seq[BlockParameters],
    ): Seq[BlockParameters] =
    for {
      bp <- params
      bc <- config.find(_.name == bp.name)
    } yield bp.copy(displayName = bc.displayName.some)

  private def makeProcessing(
      processings: List[ProcessingDto],
      wdById: Map[UUID, WorkflowDef],
      vlById: Map[UUID, VectorLayer],
      rlById: Map[UUID, RasterLayer],
      projectById: Map[UUID, ProjectDto],
      dataProviderById: Map[UUID, DataProvider],
      aoiStats: Map[UUID, AoiSummary],
      progressByProcessingId: Map[UUID, Progress],
      reviewByProcessingId: Map[UUID, ProcessingReviewDto],
      messagesProcessingId: Map[UUID, List[Message]],
      ratingProcessingId: Map[UUID, RatingDto],
      userByProjectId: Map[UUID, UserDto],
    ): List[Processing] =
    for {
      processing <- processings
      notFoundError = (entity: String) =>
        throw InternalServerError(s"$entity, referenced by ${processing.id}, doesn't exist.")
      workflowDef = processing
        .workflowDefId
        .map(wdById.getOrElse(_, notFoundError("WorkflowDef")))
        .get
      vectorLayer = vlById.getOrElse(processing.vectorLayerId, notFoundError("VectorLayer"))
      rasterLayer = rlById.getOrElse(processing.rasterLayerId, notFoundError("RasterLayer"))
      dataProvider = processing.dataProviderId.flatMap(dataProviderById.get)
      sourceType = processing.sourceType.map(SourceType.withName)
      dataSource = processing.source.map(DataSource.withName)
      AoiSummary(aoiCount, area, bbox) = aoiStats(processing.id)
      bboxOrDefault = if (bbox == Extent(0.0, 0.0, 0.0, 0.0)) Extent(-150, -50, 170, 80) else bbox
    } yield Processing(
      id = processing.id,
      projectId = processing.projectId,
      vectorLayer = vectorLayer,
      rasterLayer = rasterLayer,
      dataProvider = dataProvider,
      workflowDef = workflowDef,
      name = processing.name,
      description = processing.description,
      bbox = bboxOrDefault,
      aoiCount = aoiCount,
      area = area,
      progress = progressByProcessingId(processing.id),
      reviewStatus = reviewByProcessingId.get(processing.id),
      params = ProcessingParams(processing.params),
      blocks = processing.blocks.fold[Seq[BlockParameters]](Seq.empty) { json =>
        prepareBlockParameters(
          workflowDef.workflowDefSummary.blocks,
          BlockParameters(json),
        )
      },
      meta = ProcessingMeta(processing.meta).asJson,
      messages = messagesProcessingId(processing.id),
      sourceType = sourceType,
      source = dataSource,
      cost = processing.cost,
      rating = ratingProcessingId.get(processing.id).map(Rating(_)),
      created = processing.created,
      updated = processing.updated,
      archived = processing.archived,
      projectName = projectById.get(processing.projectId).map(_.name),
      email = userByProjectId(processing.projectId).email,
      user = UserBrief(
        userByProjectId(processing.projectId).id,
        userByProjectId(processing.projectId).email,
        userByProjectId(processing.projectId).name,
        userByProjectId(processing.projectId).preferredUsername,
        userByProjectId(processing.projectId).avantpostUserId,
      ),
    )

  private def fillProcessingFromDto(
      processingDtos: List[ProcessingDto]
    ): ConnectionIO[List[Processing]] = {
    val ids = processingDtos.map(_.id)
    val wdIds = processingDtos.flatMap(_.workflowDefId).distinct
    val vlIds = processingDtos.map(_.vectorLayerId).distinct
    val rlIds = processingDtos.map(_.rasterLayerId).distinct
    val projectIds = processingDtos.map(_.projectId).distinct
    val dataProvidersIds = processingDtos.flatMap(_.dataProviderId).distinct
    for {
      wdById <- WorkflowDefRepo.getAllByIds(wdIds).map(_.map(wd => wd.id -> wd).toMap)
      vlById <- VectorLayerRepo.getAllByIds(vlIds).map(_.map(vl => vl.id -> vl).toMap)
      rlById <- RasterLayerRepo.getAllByIds(rlIds).map(_.map(rl => rl.id -> rl).toMap)
      projectById <- ProjectRepo.getAllByIds(projectIds).map(_.map(p => p.id -> p).toMap)
      userByProjectId <- UserRepo.getByProjectId(projectIds)
      aoiStats <- AoiRepo.getAoiSummariesByProcessings(ids)
      messagesProcessingId <- AoiRepo.getMessagesByProcessings(ids)
      dataProviderById <- DataProviderRepo
        .getAllByIds(dataProvidersIds)
        .map(_.map(dp => dp.id -> dp.toDomain).toMap)
      progressByProcessingId <- progressService.getProcessingsProgress(processingDtos)
      ratingProcessingId <- ProcessingRateRepo.listRatings(ids)
      reviewByProcessingId <- ProcessingReviewRepo.listReviewStatuses(ids)
      processings = makeProcessing(
        processings = processingDtos,
        wdById = wdById,
        vlById = vlById,
        rlById = rlById,
        projectById = projectById,
        dataProviderById = dataProviderById,
        aoiStats = aoiStats,
        progressByProcessingId = progressByProcessingId,
        reviewByProcessingId = reviewByProcessingId,
        messagesProcessingId = messagesProcessingId,
        ratingProcessingId = ratingProcessingId,
        userByProjectId = userByProjectId,
      )
    } yield processings
  }
}

object ProcessingService {
  def apply(
      costCalculatorService: CostCalculatorService,
      dataProviderService: DataProviderService,
      notificationService: NotificationService,
      progressService: ProgressService,
      rasterService: RasterService,
      workflowDefService: WorkflowDefService,
      workflowService: WorkflowService,
    ): ProcessingService = new ProcessingService(
    costCalculatorService,
    dataProviderService,
    notificationService,
    progressService,
    rasterService,
    workflowDefService,
    workflowService,
  )
  def extractDataSource(
      sourceTypeOpt: Option[SourceType],
      wd: WorkflowDefSummary,
      params: Option[ProcessingParams],
    ): Option[DataSource] = {
    val paramOpt = for {
      p <- params
      url <- p.url
      sourceType <- sourceTypeOpt
      ds = WorkflowDefParser.extractDataSourceFromUrl(sourceType, url)
    } yield ds
    (paramOpt ++ wd.source).headOption
  }
  def extractDataSourceWithoutParams(
      sourceTypeOpt: Option[SourceType],
      wd: WorkflowDefSummary,
    ): Option[DataSource] = {
    val paramOpt = for {
      sourceType <- sourceTypeOpt
      ds = WorkflowDefParser.extractDataSource(sourceType)
    } yield ds
    (paramOpt ++ wd.source).headOption
  }
  def updateParamsWithUrl(
      params: Option[ProcessingParams],
      inputUrl: Option[String],
      urlTemplate: Option[String],
      maybeSourceType: Option[SourceType],
    ): Option[ProcessingParams] = {
    val maybeUrl = inputUrl.orElse(urlTemplate)
    val processingParams = List(
      maybeUrl.map(url => Map("url" -> url)),
      maybeSourceType.map(sourceType => Map("sourceType" -> sourceType.toString)),
    ).flatten.reduceOption(_ ++ _).map(ProcessingParams.apply)
    params.fold(processingParams)(pp =>
      (maybeUrl, maybeSourceType) match {
        case Some(url) -> Some(sourceType) =>
          pp.copy(url = url.some, sourceType = sourceType.toString.some).some
        case None -> Some(sourceType) => pp.copy(sourceType = sourceType.toString.some).some
        case Some(url) -> None => pp.copy(url = url.some).some
        case None -> None => pp.some
      }
    )
  }
  def extractSourceType(
      sourceTypeOpt: Option[SourceType],
      wd: WorkflowDefSummary,
      paramsOpt: Option[ProcessingParams],
    ): Option[SourceType] =
    (sourceTypeOpt
      ++ paramsOpt.flatMap(_.sourceType).flatMap {
        case "tif" => SourceType.local.some
        case "tiff" => SourceType.local.some
        case name =>
          Some(
            SourceType.find(name).getOrElse(throw BadRequest(s"Illegal source type $name provided"))
          )
      } ++ wd.sourceType).headOption

  def extractSourceType(
      sourceTypeOpt: Option[SourceType],
      wd: WorkflowDefSummary,
      params: ProcessingParams,
    ): Option[SourceType] =
    (sourceTypeOpt
      ++ params.sourceType.flatMap {
        case "tif" => SourceType.local.some
        case "tiff" => SourceType.local.some
        case name =>
          Some(
            SourceType.find(name).getOrElse(throw BadRequest(s"Illegal source type $name provided"))
          )
      } ++ wd.sourceType).headOption
}
