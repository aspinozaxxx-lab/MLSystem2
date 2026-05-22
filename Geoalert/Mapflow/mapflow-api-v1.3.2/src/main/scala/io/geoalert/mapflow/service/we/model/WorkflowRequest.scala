package io.geoalert.mapflow.service.we.model

case class WorkflowRequest(filter: WorkflowFilter, limit: Int)

case class WorkflowFilter(workflowIds: List[String])
