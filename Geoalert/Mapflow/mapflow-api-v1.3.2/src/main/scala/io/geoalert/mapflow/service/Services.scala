package io.geoalert.mapflow.service

import io.geoalert.mapflow.providers.maxar.MaxarCatalogClient
import io.geoalert.mapflow.providers.maxar.MaxarTilesProxy
import io.geoalert.mapflow.service.avanpost.AvanpostClient
import io.geoalert.mapflow.service.billing.BillingReportService
import io.geoalert.mapflow.service.billing.BillingService
import io.geoalert.mapflow.service.notification.NotificationService
import io.geoalert.mapflow.service.nspd.NspdClient
import io.geoalert.mapflow.service.we.WorkflowEngine

trait Services {
  val healthCheckService: HealthCheckService = HealthCheckService()

  val rasterService: RasterService = RasterService()

  val notificationService: NotificationService = NotificationService(rasterService)

  val userSyncService: UserSyncService = new UserSyncService()

  val teamService: TeamService = TeamService(userSyncService)

  val billingService: BillingService = BillingService(teamService)

  val progressService: ProgressService = ProgressService()

  val reviewService = new ReviewService(billingService)
  val workflowService: WorkflowService =
    WorkflowService(billingService, notificationService, progressService, reviewService)

  val workflowDefService: WorkflowDefService = WorkflowDefService(teamService)
  val dataProviderService: DataProviderService = DataProviderService()
  val maxarService: MaxarService =
    MaxarService(MaxarCatalogClient(), MaxarTilesProxy(), dataProviderService)

  val costCalculatorService: CostCalculatorService =
    CostCalculatorService(maxarService, workflowDefService, dataProviderService)

  val processingService: ProcessingService = ProcessingService(
    costCalculatorService,
    dataProviderService,
    notificationService,
    progressService,
    rasterService,
    workflowDefService,
    workflowService,
  )

  val aoiService: AoiService = AoiService(processingService, progressService)

  val projectService: ProjectService =
    ProjectService(
      processingService,
      progressService,
      aoiService,
      billingService,
    )

  val userService: UserService =
    UserService(
      billingService,
      projectService,
      userSyncService,
      workflowService,
      workflowDefService,
      costCalculatorService,
    )

  val billingReportService: BillingReportService = BillingReportService(teamService, userService)

  val authorizationService: AuthorizationService = AuthorizationService(userService)
  val avanpostClient: AvanpostClient = AvanpostClient.make
  val nspdClient: NspdClient = NspdClient.make

  val workflowEngineService: WorkflowEngineService = WorkflowEngineService(
    progressService,
    WorkflowEngine(),
    workflowService,
    notificationService,
  )

  val exportGeojsonService: ExportGeojsonService = ExportGeojsonService()

  val resultService: ResultService = ResultService(exportGeojsonService)

  val layerService: LayerService = LayerService()

  val runProcessingService: RunProcessingService = RunProcessingService(
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

  val headService: HeadService = HeadService(dataProviderService)

  val catalogService: CatalogService = CatalogService(maxarService, headService)
}
