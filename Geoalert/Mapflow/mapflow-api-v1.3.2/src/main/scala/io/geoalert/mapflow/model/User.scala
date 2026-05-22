package io.geoalert.mapflow.model

import java.time.Instant
import java.util.UUID

case class User(
    id: UUID,
    email: String,
    role: Role,
    areaLimit: Long,
    aoiAreaLimit: Long,
    billingType: BillingType,
    created: Instant,
    updated: Instant,
    processedArea: Long,
    memoryLimit: Long,
    maxAoisPerProcessing: Int,
    availableDataProviders: List[DataProvider],
    activeUntil: Option[Instant],
    reviewWorkflowEnabled: Boolean,
    name: Option[String] = None,
    preferredUsername: Option[String] = None,
    avantpostUserId: Option[UUID] = None,
  ) {
  def userFilter(permission: Permission): Option[UUID] =
    if (role.hasPermission(permission)) None else Some(id)
}

sealed abstract class BillingType(val repr: String)

object BillingType {
  case object Area extends BillingType("AREA")
  case object Credits extends BillingType("CREDITS")

  case object None extends BillingType("NONE")

  def fromString(name: String): BillingType = name match {
    case "AREA" => Area
    case "CREDITS" => Credits
    case "NONE" => None
    case _ => sys.error(s"Invalid role code: $name")
  }
}

case class JwtUser(email: String, userRole: String) {
  lazy val role: Role = Role.fromString(userRole)
}

object JwtUser {
  def apply(user: User): JwtUser = new JwtUser(user.email, user.role.repr)
}

case class CreateUserInput(
    email: String,
    role: Option[Role],
    password: ],
    areaLimit: Option[Long],
    aoiAreaLimit: Option[Long],
    billingType: BillingType,
    memoryLimit: Option[Long],
    activeUntil: Option[Instant],
    reviewWorkflowEnabled: Option[Boolean],
  ) {
  override def toString: String =
    s"CreateUserInput(email=$email, areaLimit=$areaLimit, aoiAreaLimit=$aoiAreaLimit,  memoryLimit=$memoryLimit)"
}

case class UpdateUserInput(
    email: String,
    role: Option[Role],
    password: ],
    areaLimit: Option[Long],
    aoiAreaLimit: Option[Long],
    billingType: Option[BillingType],
    memoryLimit: Option[Long],
    maxAoisPerProcessing: Option[Int],
    activeUntil: Option[Instant],
    reviewWorkflowEnabled: Option[Boolean],
  ) {
  override def toString: String =
    s"UpdateUserInput(email=$email, password= => "***")}, " +
      s" areaLimit=$areaLimit aoiAreaLimit=$aoiAreaLimit)" + s" memoryLimit=$memoryLimit"
}
