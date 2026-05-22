package io.geoalert.mapflow.rest.json

import java.time.Instant
import java.util.UUID

import io.geoalert.mapflow.model.Role
import io.geoalert.mapflow.model.User

case class CreateUserInputJson(
    email: String,
    password: ],
    areaLimit: Option[Long],
    aoiAreaLimit: Option[Long],
    memoryLimit: Option[Long],
  )
case class UpdateUserInputJson(
    email: String,
    password: ],
    areaLimit: Option[Long],
    aoiAreaLimit: Option[Long],
    memoryLimit: Option[Long],
  )

case class UserJson(
    id: UUID,
    email: String,
    role: Role,
    areaLimit: Long,
    aoiAreaLimit: Long,
    processedArea: Long,
    created: Instant,
    updated: Instant,
    isPremium: Boolean,
    reviewWorkflowEnabled: Boolean,
  )

object UserJson {
  def apply(user: User, processedArea: Long): UserJson = new UserJson(
    user.id,
    user.email,
    user.role,
    user.areaLimit,
    user.aoiAreaLimit,
    processedArea,
    user.created,
    user.updated,
    user
      .availableDataProviders
      .map(!_.isDefault)
      .reduceLeftOption(_ || _)
      .getOrElse(false),
    user.reviewWorkflowEnabled,
  )
}
