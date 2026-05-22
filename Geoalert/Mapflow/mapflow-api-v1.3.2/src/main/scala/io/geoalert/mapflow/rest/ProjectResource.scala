package io.geoalert.mapflow.rest

import akka.http.scaladsl.server.Directives
import akka.http.scaladsl.server.Route
import de.heikoseeberger.akkahttpcirce.ErrorAccumulatingCirceSupport._
import io.circe.generic.auto.exportDecoder
import io.geoalert.mapflow.model.CreateProjectInput
import io.geoalert.mapflow.model.UpdateProjectInput
import io.geoalert.mapflow.model.UserProject
import io.geoalert.mapflow.rest.json.CreateProjectInputJson
import io.geoalert.mapflow.rest.json.UpdateProjectInputJson
import io.geoalert.mapflow.rest.json.WorkflowDefJson

object ProjectResource extends Directives with Authorization with RestImplicits {
  def retrieveProject: Route = (path("projects" / JavaUUID) & get & authorized) { (id, user) =>
    toComplete(projectService.getProjectJson(id)(user))
  }

  def listProjects: Route = (path("projects") & get & authorized) { user =>
    toComplete(projectService.listProjects(user))
  }

  def retrieveDefaultProject: Route = (path("projects" / "default") & get & authorized) { user =>
    toComplete(projectService.getDefaultProject(user))
  }

  def createProject: Route =
    (path("projects") & post & authorized & entity(as[CreateProjectInputJson])) { (user, input) =>
      toComplete(
        projectService.createProjectAndGet(
          CreateProjectInput(input.name, input.description, Some(true))
        )(user)
      )
    }

  def updateProject: Route =
    (path("projects" / JavaUUID) & put & authorized & entity(as[UpdateProjectInputJson])) {
      (projectId, user, input) =>
        toComplete(
          projectService.updateProjectAndGet(
            UpdateProjectInput(projectId, input.name, input.description)
          )(user)
        )
    }
  def shareProject: Route =
    (path("projects" / "share") & post & authorized & entity(as[UserProject])) {
      (user, userProject) =>
        toComplete(
          projectService.shareProject(userProject)(user)
        )
    }
  def unshareProject: Route =
    (path("projects" / JavaUUID / "users" / JavaUUID) & delete & authorized) {
      (projectId, userId, user) =>
        toComplete(
          projectService.unshareProject(projectId, userId)(user)
        )
    }

  def archiveProject: Route = (path("projects" / JavaUUID) & delete & authorized) { (id, user) =>
    toComplete(projectService.archiveProject(id)(user))
  }

  def listWorkflowDef: Route = (path("projects" / JavaUUID / "models") & get & authorized) {
    (projectId, user) =>
      toComplete(
        workflowDefService
          .listWorkflowDefLinkedToProject(projectId)(user)
          .map(_.map(WorkflowDefJson(_)))
      )
  }

  def linkWorkflowDef: Route =
    (path("projects" / JavaUUID / "models" / JavaUUID) & post & authorized) {
      (projectId, workflowDefId, user) =>
        toComplete(workflowDefService.linkWorkflowDefToProject(workflowDefId, projectId)(user))
    }

  def unlinkWorkflowDef: Route =
    (path("projects" / JavaUUID / "models" / JavaUUID) & delete & authorized) {
      (projectId, workflowDefId, user) =>
        toComplete(workflowDefService.unlinkWorkflowDefFromProject(workflowDefId, projectId)(user))
    }

  val routes: Route = concat(
    retrieveProject,
    retrieveDefaultProject,
    listProjects,
    createProject,
    archiveProject,
    updateProject,
    listWorkflowDef,
    linkWorkflowDef,
    shareProject,
    unshareProject,
    unlinkWorkflowDef,
  )
}
