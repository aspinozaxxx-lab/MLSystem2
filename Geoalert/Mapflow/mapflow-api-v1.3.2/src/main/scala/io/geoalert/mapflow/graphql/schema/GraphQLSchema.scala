package io.geoalert.mapflow.graphql.schema

import java.nio.charset.StandardCharsets
import java.util.UUID

import scala.concurrent.ExecutionContext
import scala.concurrent.Future

import cats.data.EitherT
import doobie.ConnectionIO
import doobie.implicits._
import io.geoalert.mapflow.graphql.Authorized
import io.geoalert.mapflow.graphql.GraphQLContext
import io.geoalert.mapflow.graphql.GraphQLController
import io.geoalert.mapflow.graphql.PrivilegeRequired
import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.model.Upload
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.service._
import sangria.schema._
import schema._

object GraphQLSchema extends Services {
  implicit val ec: ExecutionContext = ExecutionContext.global

  def toFuture[A, B <: Throwable](io: EitherT[ConnectionIO, B, A]): Future[A] =
    io.rethrowT.transact(xa).unsafeToFuture()

  def toFuture[A, B <: Throwable](io: ConnectionIO[A]): Future[A] =
    io.transact(xa).unsafeToFuture()

  def uploadFile(upload: Option[Upload], ctx: GraphQLContext): Option[String] =
    for {
      upload <- upload
      is <- upload.streamOption(ctx)
      yml = new String(is.readAllBytes(), StandardCharsets.UTF_8)
    } yield yml

  val Query: ObjectType[GraphQLContext, Unit] = ObjectType(
    "Query",
    fields[GraphQLContext, Unit](
      Field(
        "project",
        ProjectType,
        arguments = IdArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(projectService.getProject(c arg IdArg)(c.ctx.user)),
      ),
      Field(
        "projects",
        ListType(ProjectType),
        arguments = IdsArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c =>
          toFuture(
            projectService.getProjects((c arg IdsArg).getOrElse(Seq.empty[UUID]))(c.ctx.user)
          ),
      ),
      Field(
        "projectsPaged",
        ProjectsPagedResponseType,
        arguments = PagedRequestArg :: Nil,
        tags = Authorized :: Nil,
        resolve =
          c => toFuture(GraphQLController.listProjectsPaged(c arg PagedRequestArg)(c.ctx.user)),
      ),
      Field(
        "processing",
        ProcessingType,
        arguments = IdArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(processingService.getProcessing(c arg IdArg)(c.ctx.user)),
      ),
      Field(
        "processingsPaged",
        ProcessingPagedResponseType,
        arguments = ProcessingFiltersArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c => toFuture(processingService.getProcessingsPaged(c arg ProcessingFiltersArg)),
      ),
      Field(
        "processings",
        ListType(ProcessingType),
        arguments = IdsArg :: ProjectIdsArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c =>
          toFuture(
            processingService.getProcessings(c.args arg IdsArg, c.args arg ProjectIdsArg)(
              c.ctx.user
            )
          ),
      ),
      Field(
        "aoi",
        AoiType,
        arguments = IdArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(aoiService.getAoi(c arg IdArg)(c.ctx.user)),
      ),
      Field(
        "aois",
        AoiListType,
        arguments = AoiFilterArg :: AoiSortArg :: OffsetArg :: LimitArg :: Nil,
        resolve = c =>
          toFuture(
            aoiService.getAois(
              c arg AoiFilterArg,
              c arg AoiSortArg,
              c arg OffsetArg,
              c arg LimitArg,
            )(c.ctx.user)
          ),
        tags = Authorized :: Nil,
      ),
      Field(
        "aoiIds",
        ListType(UuidIdType),
        arguments = AoiFilterArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(aoiService.getAoiIds(c arg AoiFilterArg)(c.ctx.user)),
      ),
      Field(
        "aoiStats",
        AoiStatsType,
        arguments = AoiFilterArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(aoiService.getAoiStats(c arg AoiFilterArg)(c.ctx.user)),
      ),
      Field(
        "aoiLayer",
        StringType,
        arguments = ProcessingIdArg :: BboxArg :: XResArg :: YResArg :: Nil,
        resolve = c => toFuture(layerService.getGeojsonLayer(c.args)(c.ctx.user)),
        tags = Authorized :: Nil,
      ),
      Field(
        "currentUser",
        UserType,
        arguments = Nil,
        resolve = c => Future.successful(c.ctx.user),
        tags = Authorized :: Nil,
      ),
      Field(
        "users",
        ListType(UserType),
        arguments = IdsArg :: UserEmailsArg :: UserRolesArg :: Nil,
        resolve = c =>
          toFuture(
            userService.getUsers(
              c arg IdsArg,
              c arg UserEmailsArg,
              c arg UserRolesArg,
            )(c.ctx.user)
          ),
        tags = PrivilegeRequired(Role.Admin) :: Nil,
      ),
      Field(
        "dataProviderUsers",
        ListType(UserType),
        arguments = IdArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c => toFuture(userService.getUsersByDpId(c arg IdArg)),
      ),
      Field(
        "billing",
        BillingReportType,
        arguments = EmailArg :: StartDateArg :: EndDateArg :: Nil,
        resolve = c =>
          toFuture(
            billingReportService.generateBillingReport(
              c arg EmailArg,
              c arg StartDateArg,
              c arg EndDateArg,
            )(c.ctx.user)
          ),
        tags = Authorized :: Nil,
      ),
      Field(
        "basemapDataProviders",
        ListType(DataProviderType),
        arguments = IdsArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(dataProviderService.listDataProviders(c arg IdsArg)(c.ctx.user)),
      ),
      Field(
        "dataProviders",
        ListType(DataProviderType),
        arguments = IdsArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(dataProviderService.listDataProviders(c arg IdsArg)(c.ctx.user)),
      ),
      Field(
        "teams",
        ListType(TeamType),
        arguments = IdsArg :: UserIdsArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(teamService.listTeams(c arg IdsArg, c arg UserIdsArg)(c.ctx.user)),
      ),
      Field(
        "team",
        TeamWithMembersType,
        arguments = IdArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(teamService.getTeam(c arg IdArg)(c.ctx.user)),
      ),
      Field(
        "workflowDef",
        WorkflowDefType,
        arguments = IdArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(workflowDefService.getWorkflowDef(c arg IdArg)(c.ctx.user)),
      ),
      Field(
        "workflowDefs",
        ListType(WorkflowDefType),
        arguments = IdsArg :: UserIdsArg :: ProjectIdsArg :: IsDefaultArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c =>
          toFuture(
            workflowDefService.listWorkflowDefs(
              c arg IdsArg,
              c arg UserIdsArg,
              c arg ProjectIdsArg,
              c arg IsDefaultArg,
            )(c.ctx.user)
          ),
      ),
      Field(
        "workflowDefUsers",
        ListType(UserType),
        arguments = IdArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c =>
          toFuture(
            GraphQLController.listWorkflowDefUsers(
              c arg IdArg
            )(c.ctx.user)
          ),
      ),
    ),
  )

  val Mutation: ObjectType[GraphQLContext, Unit] = ObjectType(
    "Mutation",
    fields[GraphQLContext, Unit](
      // Projects
      Field(
        "createProject",
        ProjectType,
        arguments = CreateProjectArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(projectService.createProject(c arg CreateProjectArg)(c.ctx.user)),
      ),
      Field(
        "updateProject",
        ProjectType,
        arguments = UpdateProjectArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(projectService.updateProject(c arg UpdateProjectArg)(c.ctx.user)),
      ),
      Field(
        "deleteProject",
        StringType,
        arguments = IdArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(projectService.archiveProject(c arg IdArg)(c.ctx.user)),
      ),
      Field(
        "shareProject",
        StringType,
        arguments = ShareProjectArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(projectService.shareProject(c arg ShareProjectArg)(c.ctx.user)),
      ),
      Field(
        "unshareProject",
        StringType,
        arguments = ProjectIdArg :: UserIdArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c =>
          toFuture(projectService.unshareProject(c arg ProjectIdArg, c arg UserIdArg)(c.ctx.user)),
      ),

      // WorkflowDefs
      Field(
        "createWorkflowDef",
        WorkflowDefType,
        arguments = CreateWorkflowDefArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c =>
          toFuture(
            workflowDefService.createWorkflowDef(
              c arg CreateWorkflowDefArg,
              uploadFile((c arg CreateWorkflowDefArg).file, c.ctx),
            )(c.ctx.user)
          ),
      ),
      Field(
        "updateWorkflowDef",
        WorkflowDefType,
        arguments = UpdateWorkflowDefArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c =>
          toFuture(
            workflowDefService.updateWorkflowDef(
              c arg UpdateWorkflowDefArg,
              uploadFile((c arg UpdateWorkflowDefArg).file, c.ctx),
            )(c.ctx.user)
          ),
      ),
      Field(
        "deleteWorkflowDef",
        StringType,
        arguments = IdArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c => toFuture(workflowDefService.archiveWorkflowDef(c arg IdArg)(c.ctx.user)),
      ),
      Field(
        "linkWorkflowDefToUser",
        StringType,
        arguments = WorkflowDefIdArg :: UserIdArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c =>
          toFuture(
            workflowDefService.linkWorkflowDefToUser(c arg WorkflowDefIdArg, c arg UserIdArg)(
              c.ctx.user
            )
          ),
      ),
      Field(
        "unlinkWorkflowDefFromUser",
        StringType,
        arguments = WorkflowDefIdArg :: UserIdArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c =>
          toFuture(
            workflowDefService.unlinkWorkflowDefFromUser(c arg WorkflowDefIdArg, c arg UserIdArg)(
              c.ctx.user
            )
          ),
      ),
      Field(
        "linkWorkflowDefToProject",
        StringType,
        arguments = WorkflowDefIdArg :: ProjectIdArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c =>
          toFuture(
            workflowDefService.linkWorkflowDefToProject(c arg WorkflowDefIdArg, c arg ProjectIdArg)(
              c.ctx.user
            )
          ),
      ),
      Field(
        "unlinkWorkflowDefFromProject",
        StringType,
        arguments = WorkflowDefIdArg :: ProjectIdArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c =>
          toFuture(
            workflowDefService.unlinkWorkflowDefFromProject(
              c arg WorkflowDefIdArg,
              c arg ProjectIdArg,
            )(c.ctx.user)
          ),
      ),

      // Processings
      Field(
        "createProcessing",
        ProcessingType,
        arguments = CreateProcessingArg :: Nil,
        tags = Authorized :: Nil,
        resolve =
          c => toFuture(processingService.createProcessing(c arg CreateProcessingArg)(c.ctx.user)),
      ),
      Field(
        "updateProcessing",
        ProcessingType,
        arguments = UpdateProcessingArg :: Nil,
        tags = Authorized :: Nil,
        resolve =
          c => toFuture(processingService.updateProcessing(c arg UpdateProcessingArg)(c.ctx.user)),
      ),
      Field(
        "deleteProcessing",
        StringType,
        arguments = IdArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(processingService.archiveProcessing(c arg IdArg)(c.ctx.user)),
      ),
      Field(
        "cancelProcessing",
        StringType,
        arguments = IdArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(processingService.cancelProcessing(c arg IdArg)(c.ctx.user)),
      ),

      // AOIs
      Field(
        "createAoisFromGeometry",
        AoiStatsType,
        arguments = CreateAoisFromGeometryArg :: Nil,
        resolve = c =>
          toFuture(aoiService.createAoisFromGeometry(c arg CreateAoisFromGeometryArg)(c.ctx.user)),
        tags = Authorized :: Nil,
      ),
      Field(
        "createAoisFromFile",
        AoiStatsType,
        arguments = CreateAoisFromFileArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c =>
          toFuture(aoiService.createAoisFromFile(c arg CreateAoisFromFileArg, c.ctx)(c.ctx.user)),
      ),
      Field(
        "deleteAois",
        IntType,
        arguments = AoiFilterArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(aoiService.deleteAois(c arg AoiFilterArg)(c.ctx.user)),
      ),
      Field(
        "runProcessing",
        StringType,
        arguments = ProcessingIdArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c =>
          toFuture(runProcessingService.runProcessing(c arg ProcessingIdArg)(c.ctx.user).rethrowT),
      ),
      Field(
        "runAoi",
        StringType,
        arguments = AoiIdArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(runProcessingService.runAoi(c arg AoiIdArg)(c.ctx.user).rethrowT),
      ),
      Field(
        "restartProcessing",
        IntType,
        arguments = ProcessingIdArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c =>
          toFuture(
            runProcessingService.restartProcessing(c arg ProcessingIdArg)(c.ctx.user).rethrowT
          ),
      ),
      Field(
        "restartAoi",
        IntType,
        arguments = AoiIdArg :: Nil,
        tags = Authorized :: Nil,
        resolve =
          c => toFuture(runProcessingService.restartAoi(c arg AoiIdArg)(c.ctx.user).rethrowT),
      ),
      Field(
        "updateProcessingCost",
        LongType,
        arguments = IdArg :: Nil,
        tags = Authorized :: Nil,
        resolve = c => toFuture(processingService.updateProcessingCost(c arg IdArg)(c.ctx.user)),
      ),

      // Users
      Field(
        "createUser",
        UserType,
        arguments = CreateUserArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c => toFuture(userService.createUser(c arg CreateUserArg)(c.ctx.user)),
      ),
      Field(
        "updateUser",
        UserType,
        arguments = UpdateUserArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c => toFuture(userService.updateUser(c arg UpdateUserArg)(c.ctx.user)),
      ),
      Field(
        "deleteUser",
        StringType,
        arguments = IdArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c => toFuture(userService.deleteUser(c arg IdArg)(c.ctx.user)),
      ),

      // Data providers
      Field(
        "createDataProvider",
        DataProviderType,
        arguments = CreateDataProviderArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c =>
          toFuture(dataProviderService.createDataProvider(c arg CreateDataProviderArg)(c.ctx.user)),
      ),
      Field(
        "updateDataProvider",
        DataProviderType,
        arguments = UpdateDataProviderArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c =>
          toFuture(dataProviderService.updateDataProvider(c arg UpdateDataProviderArg)(c.ctx.user)),
      ),
      Field(
        "deleteDataProvider",
        StringType,
        arguments = IdArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c => toFuture(dataProviderService.deleteDataProvider(c arg IdArg)(c.ctx.user)),
      ),
      Field(
        "linkDataProvider",
        StringType,
        arguments = UserIdArg :: DataProviderIdArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c =>
          toFuture(
            dataProviderService.linkDataProvider(c arg UserIdArg, c arg DataProviderIdArg)(
              c.ctx.user
            )
          ),
      ),
      Field(
        "unlinkDataProvider",
        StringType,
        arguments = UserIdArg :: DataProviderIdArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c =>
          toFuture(
            dataProviderService.unlinkDataProvider(c arg UserIdArg, c arg DataProviderIdArg)(
              c.ctx.user
            )
          ),
      ),

      // Teams
      Field(
        "createTeam",
        TeamType,
        arguments = CreateTeamArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c => toFuture(teamService.createTeam(c arg CreateTeamArg)(c.ctx.user)),
      ),
      Field(
        "updateTeam",
        TeamType,
        arguments = UpdateTeamArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c => toFuture(teamService.updateTeam(c arg UpdateTeamArg)(c.ctx.user)),
      ),
      Field(
        "deleteTeam",
        StringType,
        arguments = IdArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c => toFuture(teamService.archiveTeam(c arg IdArg)(c.ctx.user)),
      ),
      Field(
        "linkUserToTeam",
        StringType,
        arguments =
          TeamIdArg :: EmailArg :: TeamMemberRoleArg :: ActiveUntilArg :: AreaLimitArg :: CreditsLimitArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c =>
          toFuture(
            teamService.linkUserToTeam(
              c arg TeamIdArg,
              c arg EmailArg,
              c arg TeamMemberRoleArg,
              c arg ActiveUntilArg,
              c arg AreaLimitArg,
              c arg CreditsLimitArg,
              failToLinkExistingUser = false,
            )(c.ctx.user)
          ),
      ),
      Field(
        "unlinkUserFromTeam",
        StringType,
        arguments = TeamIdArg :: EmailArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c =>
          toFuture(
            teamService.unlinkUserFromTeam(
              c arg TeamIdArg,
              c arg EmailArg,
            )(c.ctx.user)
          ),
      ),
      Field(
        "acceptProcessing",
        StringType,
        arguments = ProcessingIdArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c =>
          toFuture(
            reviewService.acceptProcessing(c arg ProcessingIdArg)(c.ctx.user)
          ),
      ),
      Field(
        "refundProcessing",
        StringType,
        arguments = ProcessingIdArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c =>
          toFuture(
            reviewService.refund(c arg ProcessingIdArg)(c.ctx.user)
          ),
      ),
      Field(
        "returnProcessingToInReview",
        StringType,
        arguments = ProcessingIdArg :: Nil,
        tags = PrivilegeRequired(Role.Admin) :: Nil,
        resolve = c =>
          toFuture(
            reviewService.returnToReview(c arg ProcessingIdArg)(c.ctx.user)
          ),
      ),
    ),
  )

  val schema: Schema[GraphQLContext, Unit] = Schema(Query, Some(Mutation))
}
