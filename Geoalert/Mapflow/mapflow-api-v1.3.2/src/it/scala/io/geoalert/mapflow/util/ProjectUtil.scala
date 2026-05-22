package io.geoalert.mapflow.util

import doobie.implicits._

import io.geoalert.mapflow.model.CreateProjectInput
import io.geoalert.mapflow.model.Project
import io.geoalert.mapflow.model.User
import io.geoalert.mapflow.repo.Db.xa
import io.geoalert.mapflow.service.Services

object ProjectUtil extends Services {
  def createProject(user: User): Project =
    createProject(CreateProjectInput(s"Test project for ${user.email}", None, Some(true)))(user)

  def createProject(input: CreateProjectInput)(user: User): Project =
    projectService
      .createProject(input)(user)
      .transact(xa)
      .unsafeRunSync()

  def defaultProject(user: User): Project =
    projectService
      .getOrCreateDefaultProject(user)
      .transact(xa)
      .unsafeRunSync()
}
