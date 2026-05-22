package io.geoalert.mapflow.service.we.model

case class LatestVersion(version: Int)
case class WorkflowDefResponse(id: Long, latestVersion: LatestVersion)
