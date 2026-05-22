package io.geoalert.mapflow.rest

import akka.http.scaladsl.server.Directives
import akka.http.scaladsl.server.Route
import io.geoalert.mapflow.rest.json.WorkflowDefJson
import io.geoalert.mapflow.service.Services

object UserResource extends Directives with Authorization with RestImplicits with Services {
  def getUserStatus: Route = (path("user" / "status") & get & authorized) { user =>
    toComplete(userService.getUserStatus(user))
  }

  def listWorkflowDef: Route = (path("users" / JavaUUID / "models") & get & authorized) {
    (userId, user) =>
      toComplete(
        workflowDefService.listWorkflowDefLinkedToUser(userId)(user).map(_.map(WorkflowDefJson(_)))
      )
  }

  def linkWorkflowDef: Route =
    (path("users" / JavaUUID / "models" / JavaUUID) & post & authorized) {
      (userId, workflowDefId, user) =>
        toComplete(workflowDefService.linkWorkflowDefToUser(workflowDefId, userId)(user))
    }

  def unlinkWorkflowDef: Route =
    (path("users" / JavaUUID / "models" / JavaUUID) & delete & authorized) {
      (userId, workflowDefId, user) =>
        toComplete(workflowDefService.unlinkWorkflowDefFromUser(workflowDefId, userId)(user))
    }

  val routes: Route = concat(
    getUserStatus,
    listWorkflowDef,
    linkWorkflowDef,
    unlinkWorkflowDef,
  )
}
