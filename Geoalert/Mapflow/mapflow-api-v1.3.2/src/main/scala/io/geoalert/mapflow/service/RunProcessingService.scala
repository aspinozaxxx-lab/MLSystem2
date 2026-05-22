package io.geoalert.mapflow.service

import java.util.UUID

import _root_.io.circe.Json
import _root_.io.circe.syntax._
import cats.data.EitherT
import cats.implicits.catsSyntaxApplicativeId
import cats.implicits.catsSyntaxOptionId
import cats.implicits.toBifunctorOps
import cats.implicits.toTraverseOps
import com.typesafe.scalalogging.LazyLogging
import doobie.ConnectionIO
import doobie.implicits._
import io.geoalert.mapflow.exception.AccessDenied
import io.geoalert.mapflow.exception.ApplicationError
import io.geoalert.mapflow.exception.BadRequest
import io.geoalert.mapflow.model.Aoi
import io.geoalert.mapflow.model.BlockParametersInput
import io.geoalert.mapflow.model.CreateProcessingInput
import io.geoalert.mapflow.model.DataProvider
import io.geoalert.mapflow.model.Processing
import io.geoalert.mapflow.model.ProcessingMeta
import io.geoalert.mapflow.model.ProcessingParams
import io.geoalert.mapflow.model.Project
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.model.Status
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.model.UserCredentials
import io.geoalert.mapflow.model.WorkflowDef
import io.geoalert.mapflow.model.WorkflowDefSummary
import io.geoalert.mapflow.repo.BlockParameters
import io.geoalert.mapflow.rest.json.CreateAndRunProcessingJson
import io.geoalert.mapflow.rest.json.ProcessingJson
import io.geoalert.mapflow.rest.utils.ControllerConstants.ProcessingControllerConstants.SRID
import io.geoalert.mapflow.rest.utils.ControllerConstants.ProcessingControllerConstants.sourceType
import io.geoalert.mapflow.rest.utils.ControllerConstants.ProcessingControllerConstants.urlParam
import io.geoalert.mapflow.service.billing.BillingService
import io.geoalert.mapflow.service.nspd.NspdClient
import io.geoalert.mapflow.util.HttpUtils

import geotrellis.vector._

class RunProcessingService(
    aoiService: AoiService,
    billingService: BillingService,
    processingService: ProcessingService,
    progressService: ProgressService,
    projectService: ProjectService,
    userService: UserService,
    workflowService: WorkflowService,
    nspdClient: NspdClient,
    dataProviderService: DataProviderService,
    maxarService: MaxarService,
    workflowDefService: WorkflowDefService,
    costCalculatorService: CostCalculatorService,
  ) extends LazyLogging {
  def runProcessing(
      processingId: UUID
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, String] =
    for {
      processing <- processingService.getProcessingOrForbidden(processingId)(user)
      msg <- runProcessing(processing)(user)
    } yield msg

  def runAoi(aoiId: UUID)(user: User): EitherT[ConnectionIO, ApplicationError, String] =
    for {
      aoi <- aoiService.getAoi(aoiId)(user)
      processing <- processingService.getProcessing(aoi.processingId)(user)
      _ <- EitherT.cond[ConnectionIO](
        aoi.progress.status == Status.Unprocessed,
        {},
        BadRequest("AOI already started"): ApplicationError,
      )
      _ <- runAois(List(aoi), processing)
    } yield "OK"

  def runProcessing(
      processing: Processing
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, String] =
    for {
      _ <- billingService.hold(processing)(user)
      aois <- EitherT.right[ApplicationError](aoiService.getProcessingAois(processing.id)(user))
      _ = logger.debug(s"Run Processing, aois: ${aois.size}")
      unprocessedAois = aois.filter(_.progress.status == Status.Unprocessed)
      _ <- runAois(unprocessedAois, processing)
    } yield "OK"
  def createAndRun(
      input: CreateAndRunProcessingJson
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, ProcessingJson] = {
    logger.debug(s"Create processing $input")

    def retrieveProject(projectId: Option[UUID]): EitherT[ConnectionIO, ApplicationError, Project] =
      projectId match {
        case Some(id) => projectService.getProject(id)(user)
        case None => EitherT.right(projectService.getOrCreateDefaultProject(user))
      }

    def retrieveWd(
        wds: List[WorkflowDef],
        wdIdOpt: Option[UUID],
        wdNameOpt: Option[String],
      ): Option[WorkflowDef] =
      (wdIdOpt, wdNameOpt) match {
        case (Some(id), None) => wds.find(_.id == id)
        case (None, Some(name)) => wds.find(_.name == name)
        case (Some(_), Some(_)) =>
          throw BadRequest("Only one of 'wdId' and 'wdName' may be specified.")
        case (None, None) =>
          throw BadRequest("One of 'wdId' and 'wdName' should be specified.")
      }

    def retrieveDataProvider(
        params: ProcessingParams,
        wd: WorkflowDefSummary,
      ): EitherT[ConnectionIO, ApplicationError, Option[DataProvider]] = {
      val io = params.dataProvider match {
        case Some(value) =>
          for {
            providers <- dataProviderService.listDataProvidersByName(value)
          } yield providers.headOption

        case None =>
          val url = (params.url ++ wd.url).headOption
          getDataProvider(url)
      }

      EitherT.right[ApplicationError](io)
    }

    input
      .params
      .foreach(ps =>
        if (ps.contains(sourceType) && (!ps.contains(urlParam) || ps.contains("data_provider")))
          throw BadRequest(
            s"params.url is required for source_type=${ps(sourceType)}"
          )
      )

    def processingInput(
        project: Project,
        wd: WorkflowDef,
        params: ProcessingParams,
        metaJson: Json,
        cost: Long,
        dataProvider: Option[DataProvider],
      ) =
      CreateProcessingInput(
        project.id,
        None,
        None,
        wd.id,
        dataProvider.map(_.id),
        input.name,
        input.description,
        cost,
        params = params.some,
        meta = metaJson.some,
        blocks = input.blocks.map(_.map(bp => BlockParametersInput(bp.name, bp.enabled))),
        url=params.url,
      )

    def getDataProvider(url: Option[String]): ConnectionIO[Option[DataProvider]] =
      url match {
        case Some(value) => dataProviderService.extractDataProviderFromUrl(value)
        case None => (None: Option[DataProvider]).pure[ConnectionIO]
      }

    def addGtiffDataProvider(params: ProcessingParams): EitherT[ConnectionIO, ApplicationError, ProcessingParams] = {
      // if data provider is not specified and source_type is "local", "tif", "tiff" =>
      // add data_provider="GTIFF" for better display at web
      val updatedParams = params.sourceType match {
        case Some("local") | Some("tif") | Some("tiff") => ProcessingParams(params.toMap ++ Map("data_provider" -> "GTIFF"))
        case _ => params
      }

      EitherT.rightT[ConnectionIO, ApplicationError](updatedParams)
    }

    if (input.geometry.getNumGeometries > user.maxAoisPerProcessing && user.role != Role.Admin)
      throw BadRequest(
        s"Too many AOI geometries. One processing can have up to ${user.maxAoisPerProcessing} geometries"
      )

    val meta = ProcessingMeta(input.meta.getOrElse(Map()))
    val params = ProcessingParams(input.params.getOrElse(Map()))

    val blocks =
      input.blocks.getOrElse(Seq()).map(bp => BlockParameters(bp.name, bp.enabled, bp.displayName))

    for {
      _ <- Validations.canRunProcessing(user)
      project <- retrieveProject(input.projectId)
      userWds <- EitherT.right[ApplicationError](
        workflowDefService.listWorkflowDefLinkedToUser(user.id)(user)
      )
      wd <- EitherT.fromOption[ConnectionIO](
        retrieveWd(userWds, input.wdId, input.wdName),
        BadRequest("Workflow Definition not found"),
      )
      params <- costCalculatorService.useMaxarCredentialsIfNeeded(
        params,
        meta,
        wd.workflowDefSummary,
      )(
        user
      )
      // Update parameters first in order to choose data provider and set proper connection ID
      params <- addGtiffDataProvider(params)
      dataProvider <- retrieveDataProvider(params, wd.workflowDefSummary)
      params <- EitherT.rightT[ConnectionIO, ApplicationError](RunProcessingService.overrideDataProviderUrl(params, dataProvider))
      params <- loadImageMetadataIfNeeded(
        params,
        dataProvider,
        wd.workflowDefSummary,
      )(user)
      cost = costCalculatorService
        .estimateCost(
          wd.workflowDefSummary,
          params,
          input.geometry.withSRID(SRID),
          dataProvider,
          blocks,
        )
      processing <- processingService.createProcessing(
        processingInput(project, wd, params, meta.asJson, cost, dataProvider)
      )(user)
      _ <- EitherT.right[ApplicationError](
        aoiService.createAois(processing, input.geometry.withSRID(SRID))(user)
      )
      // Get processing with updated area
      processing <- processingService.getProcessing(processing.id)(user)
      _ <- runProcessing(processing)(user)
      processingJson <- processingService
        .getProcessing(processing.id)(user)
        .leftWiden[ApplicationError]
        .semiflatMap(ProcessingJson(_))
    } yield processingJson
  }
  def runAois(
      aois: List[Aoi],
      processing: Processing,
    ): EitherT[ConnectionIO, ApplicationError, Unit] =
    for {
      mapConfig <- EitherT.right(
        processing
          .dataProvider
          .flatMap(_.mapfileUri)
          .traverse(uri => nspdClient.getMapfile(uri).to[ConnectionIO])
      )
      _ <- EitherT.right[ApplicationError](workflowService.createWorkflows(aois, processing))
      aoiIds = aois.map(_.id)
      _ <- EitherT.right[ApplicationError](
        aoiService.updateAoiStatusAndVrt(
          aoiIds,
          Status.InProgress,
          mapConfig.map(_.layer.datasource.file),
        )
      )
      _ = progressService.invalidateCache(List(processing.projectId), List(processing.id), aoiIds)
      _ = logger.info(s"Created workflows for processing ${processing.id}")
    } yield {}

  def restartAoi(aoiId: UUID)(user: User): EitherT[ConnectionIO, ApplicationError, Int] =
    for {
      aoi <- aoiService.getAoi(aoiId)(user)
      _ <- EitherT.cond[ConnectionIO](
        aoi.progress.status == Status.Failed,
        {},
        BadRequest("Only FAILED AOIs can be restarted"): ApplicationError,
      )
      processing <- processingService.getProcessing(aoi.processingId)(user)
      _ <- EitherT.rightT[ConnectionIO, ApplicationError](
        progressService.invalidateCache(
          List(processing.projectId),
          List(processing.id),
          List(aoi.id),
        )
      )
      count <- EitherT.right[ApplicationError](
        workflowService.restartFailedWorkflowsInProcessing(aoi.id)
      )
      _ <- EitherT.right[ApplicationError](
        aoiService.updateAoiStatusAndVrt(List(aoi.id), Status.InProgress, None)
      )
    } yield count

  def restartProcessing(
      processingId: UUID
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, Int] =
    // Restart all FAILED AOIS in the processing
    // If any, hold credits and invalidate caches
    for {
      processing <- processingService.getProcessing(processingId)(user)
      aois <- EitherT.right[ApplicationError](
        aoiService.getProcessingAois(processing.id)(user)
      )
      project <- projectService.getProject(processing.projectId)(user)
      owner <- userService.getUser(project.userId)(user)
      _ <- billingService.hold(processing)(owner)
      count <- EitherT.right[ApplicationError](
        workflowService.restartFailedWorkflowsInProcessing(processing.id)
      )

      _ <- EitherT.rightT[ConnectionIO, ApplicationError](
        progressService.invalidateCache(
          List(processing.projectId),
          List(processing.id),
          aois.map(_.id),
        )
      )
    } yield count

  def loadImageMetadataIfNeeded(
      params: ProcessingParams,
      dataProvider: Option[DataProvider],
      wd: WorkflowDefSummary,
    )(
      user: User
    ): EitherT[ConnectionIO, ApplicationError, ProcessingParams] = {

    logger.debug(s"Loading metadata for an image: $params, ${dataProvider.map(_.displayName)}, $wd")

    val hasAccessToDataProvider =
      dataProvider.exists(dp => user.availableDataProviders.map(_.id).contains(dp.id))
    val maxarMetadataRequired =
      (dataProvider.map(_.name).contains("securewatch") || params.url.exists { url =>
        url.toLowerCase.contains("securewatch.digitalglobe.com")
      }) && wd.userInputBucket.isDefined
    val maybeUserCred = params.url.map { url =>
      val queryParams = HttpUtils.parseUriParameters(url.toLowerCase())
      UserCredentials(params.rasterLogin, params.rasterPassword, queryParams.get("connectid"))
    }
    val userHasAccessToDataProvider =
      params.url.exists { url =>
        url.toLowerCase.contains("securewatch.digitalglobe.com")
      } && maybeUserCred.exists(uc =>
        uc.token.nonEmpty && uc.username.nonEmpty && uc.password.nonEmpty
      )
    (maxarMetadataRequired, hasAccessToDataProvider, userHasAccessToDataProvider) match {
      case (true, true, _) =>
        maxarService.addImageMetadataToParams(params, dataProvider, maybeUserCred, wd).recover {
          case e: ApplicationError =>
            logger.error("Unable to load image metadata for processing", e)
            params
        }
      case (true, false, true) =>
        maxarService.addImageMetadataToParams(params, dataProvider, maybeUserCred, wd).recover {
          case e: ApplicationError =>
            logger.error("Unable to load image metadata for processing", e)
            params
        }
      case (true, false, false) =>
        EitherT.leftT(
          AccessDenied(
            s"User has no access to Data Provider ${dataProvider.map(_.displayName)}"
          ): ApplicationError
        )
      case _ =>
        EitherT.rightT[ConnectionIO, ApplicationError](params)
    }
  }
}

object RunProcessingService {
  def apply(
      aoiService: AoiService,
      billingService: BillingService,
      processingService: ProcessingService,
      progressService: ProgressService,
      projectService: ProjectService,
      userService: UserService,
      workflowService: WorkflowService,
      nspdClient: NspdClient,
      dataProviderService: DataProviderService,
      maxarService: MaxarService,
      workflowDefService: WorkflowDefService,
      costCalculatorService: CostCalculatorService,
    ): RunProcessingService = new RunProcessingService(
    aoiService,
    billingService,
    processingService,
    progressService,
    projectService,
    userService,
    workflowService,
    nspdClient,
    dataProviderService,
    maxarService,
    workflowDefService,
    costCalculatorService,
  )
  def overrideDataProviderUrl(
                               params: ProcessingParams,
                               dataProvider: Option[DataProvider],
                             ): ProcessingParams = {
    dataProvider match {
      case Some(provider) if provider.name == "GTIFF" =>
        params
      case Some(provider) =>
        ProcessingParams(params.toMap ++ provider.urlTemplate.map("url" -> _).toMap)
      case None => params
    }
  }
}
