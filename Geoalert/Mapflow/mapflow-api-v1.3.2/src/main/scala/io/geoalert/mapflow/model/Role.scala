package io.geoalert.mapflow.model

import Permission._

sealed abstract class Role(
    val repr: String,
    val intVal: Int,
    val permissions: Set[Permission],
  ) {
  override def toString: String = repr
  def hasPermission(permission: Permission): Boolean = permissions.contains(permission)
  def isAdmin: Boolean = Role.Admin.intVal == intVal
}

object Role {
  private val userPermissions = Set[Permission](
    ViewOwnProject
  )

  private val adminPermissions = userPermissions ++ Set[Permission](
    ViewAnyProject,
    ViewAnyUser,
    AddUser,
    UpdateUser,
    DeleteUser,
    LargeProcessing,
    UnlimitedProcessing,
    NoZoomRestrictionsForMaxar,
    ManageDataProviders,
    ManageWorkflowDefinition,
    ManageTeams,
    ConfirmProcessingReview,
  )

  case object Admin extends Role("ADMIN", 0, adminPermissions)

  case object User extends Role("USER", 1, userPermissions)

  def fromString(role: String): Role = role match {
    case Admin.repr => Admin
    case User.repr => User
    case _ => sys.error(s"Invalid role string: $role")
  }

  def apply(intVal: Int): Role = intVal match {
    case Admin.intVal => Admin
    case User.intVal => User
    case _ => sys.error(s"Invalid role code: $intVal")
  }
}
